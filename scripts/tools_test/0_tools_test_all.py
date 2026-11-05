import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tool_server.tool_workers.skills.registry import SKILL_REGISTRY
from tool_server.tool_workers.tool_manager.base_manager import EXPECTED_ONLINE_TOOLS, ToolManager
from tool_server.utils.paths import env_path

"""
Registered tool and skill test script.

It defines one suitable test case for each expected online tool and manifest-backed
skill, following the registration order in EXPECTED_ONLINE_TOOLS. Offline-only tools
are intentionally not included here.
"""


ToolCase = Tuple[str, Dict, List[str]]
SetupHook = Callable[[ToolManager, Dict[str, str]], None]
DISABLED_CASE_NAMES = {
    "OCR",
    "DisplayOnMap",
    "AddIndexLayer",
    "ComputeIndexChange",
    "ShowIndexLayer",
    "SliceGeoTIFF",
    "SARToRGB",
    "CloudRemoval",
}
NON_SKILL_CASE_NAMES = {
    "Calculator",
    "OCR",
    "DrawBox",
    "AddText",
    "GoogleSearch",
    "Plot",
    "Solver",
    "TextToBbox",
    "BaseModel",
    "ImageDescription",
    "RegionAttributeDescription",
    "CountGivenObject",
    "SegmentObjectPixels",
    "ChangeDetection",
    "ObjectDetection",
    "GetAreaBoundary",
    "AddPoisLayer",
    "ComputeDistance",
    "DisplayOnMap",
    "AddIndexLayer",
    "ComputeIndexChange",
    "ShowIndexLayer",
    "SliceGeoTIFF",
    "SARToRGB",
    "CloudRemoval",
    "SmallObjectDetection",
    "DenoDetV2",
    "CrossModalInterpret",
}


def make_case(tool_name: str, params: Dict, required_paths: Optional[List[str]] = None) -> ToolCase:
    return tool_name, params, required_paths or []


def abs_path(rel_path: str) -> str:
    """Convert relative test file paths to absolute paths."""
    return os.path.abspath(os.path.join(SCRIPT_DIR, rel_path))


def repo_path(rel_path: str) -> str:
    """Convert repository-relative paths to absolute paths."""
    return os.path.abspath(os.path.join(REPO_ROOT, rel_path))


def copy_fixture(rel_path: str, scratch_dir: str, filename: Optional[str] = None) -> str:
    """Copy mutable fixtures to scratch space so tests do not alter tracked data."""
    source = abs_path(rel_path)
    target = os.path.join(scratch_dir, filename or os.path.basename(source))
    shutil.copy2(source, target)
    return target


def build_context() -> Dict[str, str]:
    scratch_dir = tempfile.mkdtemp(prefix="single_tool_test_")
    return {
        "scratch_dir": scratch_dir,
        "rgb_image": abs_path("tools_test_images/1.jpg"),
        "rgb_with_text": abs_path("tools_test_images/1_withText.jpg"),
        "cloud_image": abs_path("tools_test_images/56.png"),
        "change_pre": abs_path("tools_test_images/xBD_cls_1.png"),
        "change_post": abs_path("tools_test_images/xBD_cls_2.png"),
        "sar_image": abs_path("tools_test_images/sar_test_1.png"),
        "sar_translate_image": abs_path("tools_test_images/sar_test_3.png"),
        "assets_rgb": repo_path("assets/Singapore_clip_6_9_RGB.tif"),
        "assets_sar": repo_path("assets/Singapore_clip_6_9_SAR.tif"),
        "geotiff": abs_path("tools_test_images/S_07.tif"),
        "aoi_gpkg": copy_fixture("tools_test_images/aoi.gpkg", scratch_dir, "aoi_distance.gpkg"),
        "aoi_index_gpkg": copy_fixture("tools_test_images/aoi_1.gpkg", scratch_dir, "aoi_index.gpkg"),
        "aoi_map_gpkg": copy_fixture("tools_test_images/aoi_1.gpkg", scratch_dir, "aoi_map.gpkg"),
        "aoi_tif_gpkg": copy_fixture("tools_test_images/aoi_tif.gpkg", scratch_dir, "aoi_tif.gpkg"),
    }


def registered_tool_order() -> List[str]:
    return list(dict.fromkeys(EXPECTED_ONLINE_TOOLS))


def registered_case_order(cases: Dict[str, ToolCase], include_disabled: bool = False) -> List[str]:
    active_cases = {
        tool: case
        for tool, case in cases.items()
        if include_disabled or tool not in DISABLED_CASE_NAMES
    }
    ordered = [tool for tool in registered_tool_order() if tool in active_cases]
    remaining = [tool for tool in active_cases if tool not in ordered]
    return ordered + remaining


def list_case_order(include_disabled: bool = False) -> List[str]:
    case_names = NON_SKILL_CASE_NAMES | {"SkillExecutor"} | set(SKILL_REGISTRY.keys())
    if not include_disabled:
        case_names -= DISABLED_CASE_NAMES
    ordered = [tool for tool in registered_tool_order() if tool in case_names]
    remaining = sorted(case_names - set(ordered))
    return ordered + remaining


def missing_required_paths(required_paths: List[str]) -> List[str]:
    return [path for path in required_paths if not path or not os.path.exists(path)]


def build_tool_cases(ctx: Dict[str, str]) -> Dict[str, ToolCase]:
    scratch_dir = ctx["scratch_dir"]
    sar_to_rgb_image = env_path("SAR_TEST_PREPROCESSED_IMAGE", ctx["sar_translate_image"])

    return {
        "Calculator": make_case("Calculator", {"expression": "round(302 / pi, 4)"}),
        "OCR": make_case("OCR", {"image": ctx["rgb_with_text"]}, [ctx["rgb_with_text"]]),
        "DrawBox": make_case(
            "DrawBox",
            {"image": ctx["rgb_image"], "bbox": "(289,645,330,724)", "annotation": "Red Car"},
            [ctx["rgb_image"]],
        ),
        "AddText": make_case(
            "AddText",
            {"image": ctx["rgb_image"], "text": "Red Car", "position": "(289,640)"},
            [ctx["rgb_image"]],
        ),
        "GoogleSearch": make_case("GoogleSearch", {"query": "San Jose State Spartan Stadium location", "k": 3}),
        "Plot": make_case(
            "Plot",
            {
                "command": "```python\nimport matplotlib.pyplot as plt\ndef solution():\n    fig = plt.figure()\n    plt.plot([0, 1], [2, 3])\n    return fig\n```"
            },
        ),
        "Solver": make_case(
            "Solver",
            {
                "command": "```python\nfrom sympy import Eq, solve, symbols\ndef solution():\n    x = symbols('x')\n    return str(solve(Eq(2 * x + 3, 11), x))\n```"
            },
        ),
        "TextToBbox": make_case(
            "TextToBbox",
            {"image": ctx["rgb_image"], "text": "red car", "top1": True},
            [ctx["rgb_image"]],
        ),
        "BaseModel": make_case(
            "BaseModel",
            {"image": ctx["rgb_image"], "query": "Describe this aerial image in one short sentence."},
            [ctx["rgb_image"]],
        ),
        "ImageDescription": make_case("ImageDescription", {"image": ctx["rgb_image"]}, [ctx["rgb_image"]]),
        "RegionAttributeDescription": make_case(
            "RegionAttributeDescription",
            {"image": ctx["rgb_image"], "bbox": "(289,645,330,724)", "attribute": "vehicle color"},
            [ctx["rgb_image"]],
        ),
        "CountGivenObject": make_case("CountGivenObject", {"image": ctx["rgb_image"], "text": "cars"}, [ctx["rgb_image"]]),
        "SegmentObjectPixels": make_case(
            "SegmentObjectPixels",
            {"image": ctx["rgb_image"], "text": "red car", "flag": True},
            [ctx["rgb_image"]],
        ),
        "ChangeDetection": make_case(
            "ChangeDetection",
            {
                "text": "Classify the level of damage experienced by the building at location [0, 8, 49, 53].",
                "pre_image": ctx["change_pre"],
                "post_image": ctx["change_post"],
                "mode": "scene",
            },
            [ctx["change_pre"], ctx["change_post"]],
        ),
        "ObjectDetection": make_case("ObjectDetection", {"image": ctx["rgb_image"], "text":  "car . ship"}, [ctx["rgb_image"]]),
        "GetAreaBoundary": make_case(
            "GetAreaBoundary",
            {"area": "Edinburgh Castle, Edinburgh, Scotland, UK", "buffer_m": 1000},
        ),
        "AddPoisLayer": make_case(
            "AddPoisLayer",
            {"gpkg": ctx["aoi_gpkg"], "query": {"amenity": "pharmacy"}, "layer_name": "pharmacies_test"},
            [ctx["aoi_gpkg"]],
        ),
        "ComputeDistance": make_case(
            "ComputeDistance",
            {"gpkg": ctx["aoi_gpkg"], "src_layer": "schools", "tar_layer": "pharmacies", "top": 1},
            [ctx["aoi_gpkg"]],
        ),
        "DisplayOnMap": make_case(
            "DisplayOnMap",
            {"gpkg": ctx["aoi_map_gpkg"], "layers": ["area_boundary", "dem_contours_20m", "dem_bands_100m"]},
            [ctx["aoi_map_gpkg"]],
        ),
        "AddIndexLayer": make_case(
            "AddIndexLayer",
            {"gpkg": ctx["aoi_index_gpkg"], "index_type": "NDVI", "layer_name": "ndvi_2022", "year": 2022},
            [ctx["aoi_index_gpkg"]],
        ),
        "ComputeIndexChange": make_case(
            "ComputeIndexChange",
            {
                "gpkg": ctx["aoi_index_gpkg"],
                "index_type": "NDVI",
                "layer1_name": "ndvi_2022",
                "layer2_name": "ndvi_2025",
                "diff_layer_name": "delta_ndvi",
            },
            [ctx["aoi_index_gpkg"]],
        ),
        "ShowIndexLayer": make_case(
            "ShowIndexLayer",
            {"gpkg": ctx["aoi_index_gpkg"], "index_type": "NDVI", "layer_name": "delta_ndvi"},
            [ctx["aoi_index_gpkg"]],
        ),
        "SliceGeoTIFF": make_case(
            "SliceGeoTIFF",
            {
                "tif_path": ctx["geotiff"],
                "tile_size": 256,
                "overlap": 0,
                "output_dir": os.path.join(scratch_dir, "slices"),
            },
            [ctx["geotiff"]],
        ),
        "SARToRGB": make_case(
            "SARToRGB",
            {
                "image": sar_to_rgb_image,
                "output_path": os.path.join(scratch_dir, "sar_to_rgb_result.png"),
            },
            [sar_to_rgb_image],
        ),
        "CloudRemoval": make_case(
            "CloudRemoval",
            {"image": ctx["cloud_image"], "output_path": os.path.join(scratch_dir, "cloud_removed_result.tif")},
            [ctx["cloud_image"]],
        ),
        "SmallObjectDetection": make_case(
            "SmallObjectDetection",
            {"image": ctx["rgb_image"], "text": "small vehicle", "score_threshold": 0.25, "max_detections": 5},
            [ctx["rgb_image"]],
        ),
        "DenoDetV2": make_case(
            "DenoDetV2",
            {"image": ctx["sar_image"], "box_threshold": 0.3, "return_json": True},
            [ctx["sar_image"]],
        ),
        "CrossModalInterpret": make_case(
            "CrossModalInterpret",
            {
                "rgb_text": "(10, 20, 50, 80), ship, score 0.73\n(100, 120, 150, 180), ship, score 0.60",
                "sar_text": "(12, 22, 52, 82), ship, score 0.86",
                "target": "ship",
                "iou_threshold": 0.2,
                "output_gpkg": True,
                "output_gpkg_path": os.path.join(scratch_dir, "cross_modal_ship.gpkg"),
                "layer_prefix": "ship",
            },
        ),
        "SkillExecutor": make_case(
            "SkillExecutor",
            {
                "skill_name": "ObjectVisualOutputSkill",
                "image": ctx["rgb_image"],
                "visual_type": "box_and_label",
                "bbox": [289, 645, 330, 724],
                "labels": ["red car"],
                "output_path": os.path.join(scratch_dir, "skill_executor_visual_red_car.png"),
            },
            [ctx["rgb_image"]],
        ),
        "ObjectDetectSelectSkill": make_case(
            "ObjectDetectSelectSkill",
            {
                "image": ctx["rgb_image"],
                "target": "red car",
                "selection_mode": "all",
                "detector": "text_to_bbox",
                "visualize": True,
                "max_draw": 3,
            },
            [ctx["rgb_image"]],
        ),
        "ObjectMeasureSkill": make_case(
            "ObjectMeasureSkill",
            {
                "image": ctx["rgb_image"],
                "target": "red car",
                "measure_type": "area",
                "bbox": [289, 645, 330, 724],
                "gsd_m_per_pixel": 0.3,
                "visualize": True,
            },
            [ctx["rgb_image"]],
        ),
        "ObjectVisualOutputSkill": make_case(
            "ObjectVisualOutputSkill",
            {
                "image": ctx["rgb_image"],
                "visual_type": "box_and_label",
                "bbox": [289, 645, 330, 724],
                "labels": ["red car"],
                "output_path": os.path.join(scratch_dir, "object_visual_red_car.png"),
            },
            [ctx["rgb_image"]],
        ),
        "SAREvidenceExtractSkill": make_case(
            "SAREvidenceExtractSkill",
            {
                "sar_image": ctx["sar_image"],
                "target": "ship",
                "preprocess_mode": "auto",
                "confidence_threshold": 0.25,
                "visualize": True,
                "max_candidates": 20,
            },
            [ctx["sar_image"]],
        ),
        "CrossModalMatchSkill": make_case(
            "CrossModalMatchSkill",
            {
                "rgb_image": ctx["assets_rgb"],
                "sar_image": ctx["assets_sar"],
                "target": "ship",
                "rgb_detections": [{"bbox": [10, 10, 60, 60], "score": 0.9, "label": "ship"}],
                "sar_candidates": [{"bbox": [12, 12, 62, 62], "score": 0.8, "label": "ship"}],
                "match_strategy": "hybrid",
                "score_threshold": 0.3,
                "visualize": True,
                "max_draw": 5,
            },
            [ctx["assets_rgb"], ctx["assets_sar"]],
        ),
        "CrossModalTaskSkill": make_case(
            "CrossModalTaskSkill",
            {
                "rgb_image": ctx["assets_rgb"],
                "sar_image": ctx["assets_sar"],
                "target": "ship",
                "task_type": "optical_sar_verify",
                "question": "Verify with SAR evidence whether the optical ship detections are credible.",
                "matched_pairs": [
                    {
                        "rgb_bbox": [10, 10, 60, 60],
                        "sar_bbox": [12, 12, 62, 62],
                        "match_score": 0.82,
                        "iou": 0.7,
                        "match_type": "strong_match",
                        "rgb_label": "ship",
                        "sar_label": "ship",
                    }
                ],
                "visualize": True,
                "max_draw": 5,
            },
            [ctx["assets_rgb"], ctx["assets_sar"]],
        ),
    }


def build_setup_hooks(ctx: Dict[str, str]) -> Dict[str, SetupHook]:
    def ensure_index_layers(tm: ToolManager, _ctx: Dict[str, str]):
        if "AddIndexLayer" not in tm.available_online_tools:
            return
        for layer_name, year in [("ndvi_2022", 2022), ("ndvi_2025", 2025)]:
            print(f"Preparing prerequisite index layer {layer_name} for index-change tests...")
            result = tm.call_tool(
                "AddIndexLayer",
                {"gpkg": ctx["aoi_index_gpkg"], "index_type": "NDVI", "layer_name": layer_name, "year": year},
            )
            if result.get("error_code") != 0:
                print(f"⚠️  Failed to prepare {layer_name}: {result.get('text')}")

    def ensure_delta_layer(tm: ToolManager, _ctx: Dict[str, str]):
        ensure_index_layers(tm, _ctx)
        if "ComputeIndexChange" not in tm.available_online_tools:
            return
        print("Preparing prerequisite delta_ndvi layer for ShowIndexLayer...")
        result = tm.call_tool(
            "ComputeIndexChange",
            {
                "gpkg": ctx["aoi_index_gpkg"],
                "index_type": "NDVI",
                "layer1_name": "ndvi_2022",
                "layer2_name": "ndvi_2025",
                "diff_layer_name": "delta_ndvi",
            },
        )
        if result.get("error_code") != 0:
            print(f"⚠️  Failed to prepare delta_ndvi: {result.get('text')}")

    return {
        "ComputeIndexChange": ensure_index_layers,
        "ShowIndexLayer": ensure_delta_layer,
    }


def is_registered(tm: ToolManager, tool_name: str) -> bool:
    return tool_name in tm.available_online_tools


def test_single_tool(
    tm: ToolManager,
    tool_name: str,
    params: Dict,
    setup_hooks: Optional[Dict[str, SetupHook]] = None,
    ctx: Optional[Dict[str, str]] = None,
    ) -> bool:
    if not is_registered(tm, tool_name):
        print(f"⚠️  {tool_name} is not registered or unavailable — skipping.\n")
        return False

    if setup_hooks and tool_name in setup_hooks:
        setup_hooks[tool_name](tm, ctx or {})

    print(f"\n🔹 Testing {tool_name} with parameters:")
    print(json.dumps(params, indent=2, ensure_ascii=False))

    try:
        result = tm.call_tool(tool_name, params)
        if result.get("error_code") == 0:
            print(f"\n✅ {tool_name} executed successfully.")
            print(f"Result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
            return True

        print(f"\n❌ {tool_name} returned an error.")
        print(f"Result:\n{json.dumps(result, indent=2, ensure_ascii=False)}\n")
        return False
    except Exception as e:
        print(f"\n❌ Exception while testing {tool_name}: {str(e)}\n")
        return False


def run_case(
    tm: ToolManager,
    tool_name: str,
    params: Dict,
    required_paths: List[str],
    setup_hooks: Optional[Dict[str, SetupHook]] = None,
    ctx: Optional[Dict[str, str]] = None,
) -> Tuple[bool, bool]:
    missing = missing_required_paths(required_paths)
    if missing:
        print(f"⚠️  {tool_name} is skipped because required fixtures are missing:")
        for path in missing:
            print(f"   - {path}")
        print()
        return False, True

    return test_single_tool(tm, tool_name, params, setup_hooks=setup_hooks, ctx=ctx), False


def print_no_online_tools_diagnostic(tm: ToolManager) -> None:
    controller_addr = getattr(tm, "controller_addr", None) or "<not set>"
    init_error = getattr(tm, "online_tools_init_error", None)
    print("\nERROR: ToolManager did not load any online tools from controller.")
    print(f"Controller: {controller_addr}")
    if init_error:
        print(f"Initialization error: {init_error}")
    print("Check that the local tool server is running and that /refresh_all_workers returns before rerunning tooltest.")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Run registered tool and skill test cases.")
    parser.add_argument("--tool", default="Calculator", help="Tool or skill name to test. Use --all to run every registered test case.")
    parser.add_argument("--all", action="store_true", help="Run all registered test cases in registration order.")
    parser.add_argument("--include-disabled", action="store_true", help="Include legacy disabled tool cases in --all and --list output.")
    parser.add_argument("--list", action="store_true", help="List available test cases and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list:
        print("Registered test cases in execution order:")
        for tool in list_case_order(include_disabled=args.include_disabled):
            print(f"- {tool}")
        return

    print("\n=== Initializing Tool Manager ===")
    tm = ToolManager(controller_url_location=None)
    if not tm.available_online_tools:
        print_no_online_tools_diagnostic(tm)
        return

    ctx = build_context()
    cases = build_tool_cases(ctx)
    ordered_tools = registered_case_order(cases, include_disabled=args.include_disabled)
    supported_tools = set(cases.keys())
    setup_hooks = build_setup_hooks(ctx)

    if args.all:
        success = 0
        failed = []
        skipped = []
        for tool in ordered_tools:
            case_tool, params, required_paths = cases[tool]
            if not is_registered(tm, case_tool):
                skipped.append(case_tool)
                print(f"⚠️  {case_tool} is not registered or unavailable — skipping.\n")
                continue
            ok, was_skipped = run_case(
                tm,
                case_tool,
                params,
                required_paths,
                setup_hooks=setup_hooks,
                ctx=ctx,
            )
            if was_skipped:
                skipped.append(case_tool)
            elif ok:
                success += 1
            else:
                failed.append(case_tool)
            print("-" * 100)
            time.sleep(0.5)

        print("\n📊 Summary:")
        print(f"   ✅ Successful : {success}")
        print(f"   ❌ Failed     : {len(failed)}")
        print(f"   ⚠️  Skipped    : {len(skipped)}")
        if failed:
            print(f"Failed tools: {sorted(set(failed))}")
        if skipped:
            print(f"Skipped tools: {sorted(set(skipped))}")
        print(f"Scratch directory: {ctx['scratch_dir']}")
        return

    if args.tool not in cases:
        print(f"Unknown or unregistered test case: {args.tool}")
        print("Use --list to see supported registered tools and skills.")
        return

    if args.tool not in supported_tools:
        print(f"Unsupported test case is excluded from this runner: {args.tool}")
        print("Use --list to see supported registered tools and skills.")
        return

    case_tool, params, required_paths = cases[args.tool]
    run_case(tm, case_tool, params, required_paths, setup_hooks=setup_hooks, ctx=ctx)
    print(f"Scratch directory: {ctx['scratch_dir']}")


if __name__ == "__main__":
    main()
