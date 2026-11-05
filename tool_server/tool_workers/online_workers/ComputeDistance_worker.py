import argparse
import uuid
import os
import sys
import time
import signal

_PROJ_DATA_CANDIDATES = [
    os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "rasterio",
        "proj_data",
    ),
    os.path.join(sys.prefix, "share", "proj"),
]
for _proj_data in _PROJ_DATA_CANDIDATES:
    if os.path.isdir(_proj_data) and os.path.exists(os.path.join(_proj_data, "proj.db")):
        os.environ["PROJ_DATA"] = _proj_data
        os.environ["PROJ_LIB"] = _proj_data
        break

import geopandas as gpd
import fiona
from shapely.geometry import LineString
from pyproj import datadir as pyproj_datadir

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger

CRS = "EPSG:4326"
MAX_ALL_PAIRS = 5000
MAX_OUTPUT_LINES = 80

worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ComputeDistance_worker_{worker_id}.log")
if os.environ.get("PROJ_DATA"):
    pyproj_datadir.set_data_dir(os.environ["PROJ_DATA"])


def get_utm_epsg_from_gpkg(gpkg):
    """Return UTM EPSG code based on the centroid of the area boundary layer in the GeoPackage."""
    area_gdf = gpd.read_file(gpkg, layer="area_boundary")
    if area_gdf.crs is None:
        area_gdf = area_gdf.set_crs(CRS)
    area = area_gdf.to_crs(CRS).geometry.iloc[0]
    lon, lat = area.centroid.x, area.centroid.y
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

def get_name(row, index):
    """Return best display name for a GeoDataFrame row."""
    for key in ("display_name", "name:en", "name"):
        if key in row and row[key] is not None and str(row[key]).strip():
            return str(row[key])
    for key, value in row.items():
        if str(key).startswith("name") and value is not None and str(value).strip():
            return str(value)
    return f"feature_{index}"


def normalize_top(top):
    if top is None or top == "":
        return None
    try:
        top = int(top)
    except Exception as error:
        raise ValueError(f"top must be an integer, got {top!r}") from error
    if top <= 0:
        raise ValueError("top must be a positive integer")
    return top


def validate_layers(gpkg, src_layer, tar_layer):
    available = fiona.listlayers(gpkg)
    missing = [layer for layer in (src_layer, tar_layer, "area_boundary") if layer not in available]
    if missing:
        raise ValueError(f"Missing layer(s): {missing}. Available layers: {available}")

def compute_distances(gpkg, src_layer, tar_layer, top=None):
    """
    Compute pairwise distances between features of two layers in a GeoPackage.
    Saves results as new line layer "{src_layer}_to_{tar_layer}_distances" in GeoPackage.
    Optionally, only the k closest target per source can be retained ( top = k). If top = 1 : single closest target
    
    Parameters
    ----------
    gpkg : str
        Path to GeoPackage file.
    src_layer : str
        Source layer name prefix (e.g., "schools").
    tar_layer : str
        Target layer name prefix (e.g., "restaurants").
    top : bool, optional (default=None)

    Returns
    -------
    str
        Lines of "src tar distance=dist"
    """
    validate_layers(gpkg, src_layer, tar_layer)
    top = normalize_top(top)

    src = gpd.read_file(gpkg, layer=src_layer)
    tar = gpd.read_file(gpkg, layer=tar_layer)
    if src.empty or tar.empty:
        raise ValueError(f"Input layer is empty: {src_layer}={len(src)}, {tar_layer}={len(tar)}")
    if src.crs is None:
        src = src.set_crs(CRS)
    if tar.crs is None:
        tar = tar.set_crs(CRS)

    utm_epsg = get_utm_epsg_from_gpkg(gpkg)
    projected_crs = f"EPSG:{utm_epsg}"
    src_wgs = src.to_crs(CRS)
    tar_wgs = tar.to_crs(CRS)
    src_proj = src.to_crs(projected_crs)
    tar_proj = tar.to_crs(projected_crs)

    pair_count = len(src_proj) * len(tar_proj)
    limited_large_pairs = False
    if top is None and pair_count > MAX_ALL_PAIRS:
        top = 1
        limited_large_pairs = True

    results = []
    for src_pos, (src_idx, src_row) in enumerate(src_proj.iterrows()):
        src_geom = src_row.geometry
        if src_geom is None or src_geom.is_empty:
            continue
        src_name = get_name(src.loc[src_idx], src_pos)
        distances = tar_proj.geometry.distance(src_geom)
        row_results = []
        for tar_idx, distance in distances.items():
            if distance is None or distance != distance:
                continue
            if src_layer == tar_layer and src_idx == tar_idx:
                continue
            tar_geom = tar_proj.loc[tar_idx].geometry
            if tar_geom is None or tar_geom.is_empty:
                continue
            tar_name = get_name(tar.loc[tar_idx], tar_idx)
            src_point = src_wgs.loc[src_idx].geometry.representative_point()
            tar_point = tar_wgs.loc[tar_idx].geometry.representative_point()
            row_results.append({
                src_layer: src_name,
                tar_layer: tar_name,
                "distance_m": float(distance),
                "geometry": LineString([(src_point.x, src_point.y), (tar_point.x, tar_point.y)]),
            })
        row_results.sort(key=lambda r: r["distance_m"])
        if top is None:
            results.extend(row_results)
        else:
            results.extend(row_results[:top])

    if not results:
        raise ValueError(f"No valid feature pairs found for {src_layer} and {tar_layer}")

    gpd.GeoDataFrame(results, geometry="geometry", crs="EPSG:4326") \
        .to_file(gpkg, layer=f"{src_layer}_to_{tar_layer}_distances", driver="GPKG")
    out_lines = []
    distances = []
    out_lines.append(f"Distances (in meters) saved to line layer: '{src_layer}_to_{tar_layer}_distances': ")
    if limited_large_pairs:
        out_lines.append(
            f"Large layer pair ({pair_count} pairs); returned nearest {top} target(s) per source."
        )
    visible_results = results[:MAX_OUTPUT_LINES]
    for r in visible_results:
        txt = f"{r[src_layer]} , {r[tar_layer]}, distance={r['distance_m']:.2f} m"
        out_lines.append(txt)
        distances.append(round(r['distance_m'], 2))
    if len(results) > MAX_OUTPUT_LINES:
        out_lines.append(f"... ({len(results) - MAX_OUTPUT_LINES} more distances omitted)")
    out_lines.append(f"distances = {distances}")

    output_str = "\n".join(out_lines)
    return output_str


class ComputeDistanceWorker(BaseToolWorker):
    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="ComputeDistance",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=1,
                 model_semaphore=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            model_name,
            False,
            False,
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info("ComputeDistanceWorker does not need a model. Ready to run.")

    def generate(self, params):
        required_keys = ("gpkg", "src_layer", "tar_layer")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        gpkg = params.get("gpkg")
        src_layer = params.get("src_layer")
        tar_layer = params.get("tar_layer")
        top = params.get("top", None)
        start = time.perf_counter()
        
        if not os.path.exists(gpkg):
                txt_e = f"GeoPackage not found"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}
        
        try:
            def _handle_timeout(signum, frame):
                raise TimeoutError("ComputeDistance execution timed out")

            previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(int(self.task_timeout))
            output_str = compute_distances(gpkg, src_layer, tar_layer, top)
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)
            logger.info(
                f"ComputeDistance timing | src={src_layer} | tar={tar_layer} | "
                f"elapsed={time.perf_counter() - start:.3f}s"
            )
            return {"text": output_str, "error_code": 0}

        except Exception as e:
            try:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_handler)
            except Exception:
                pass
            txt_e = f"Error in ComputeDistance: {e}"
            logger.error(f"{txt_e} | elapsed={time.perf_counter() - start:.3f}s")
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        instruction = {
            "type": "function",
            "function": {
                "name": "ComputeDistance",
                "description": "Computes projected straight-line distances between features of two layers in a GeoPackage and saves results to a new line layer. When src_layer and tar_layer are the same, self matches are skipped.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpkg": {"type": "string", "description": "Path to the GeoPackage."},
                        "src_layer": {"type": "string", "description": "Source layer name."},
                        "tar_layer": {"type": "string", "description": "Target layer name."},
                        "top": {"type": "integer", "description": "Optional: number of closest targets per source to keep."},
                    },
                    "required": ["gpkg", "src_layer", "tar_layer"]
                }
            }
        }
        return instruction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20019)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ComputeDistance")
    parser.add_argument("--limit-model-concurrency", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ComputeDistanceWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
