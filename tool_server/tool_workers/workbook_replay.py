"""Server-controlled property workbook: the target's attributes, and nothing else.

Detection is live.  ``ObjectDetection`` / ``DenoDetV2`` produce the boxes, the
labels and the scores through the normal registry gate; this module contributes
exactly one thing — the frozen per-instance attributes of the requested target
(``subtype`` / ``color`` / ``size_bin`` / ``orientation``, plus the acquisition
``visibility`` context).  No box, no score, no pairing, no change label and no
selection ever comes from here.

Binding rule
------------
``instance_id`` ends with the 0-based position of the box in the raw detector
JSON order for that view (``task4_000001:rgb:0007``).  ``_stage_detections``
keeps that same raw position in ``detection["index"]``, so a live box binds to
the workbook row carrying the same ``(view, index)``.  The row's frozen bbox is
kept for one purpose only: auditing the binding (IoU against the live box), so a
diverged snapshot is visible instead of silently mis-attributing.

Sheets: ``instances`` (attributes per box) and ``samples`` (pair -> images).
``identities`` and any truth workbook are never opened.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from openpyxl import load_workbook

ATTRIBUTES = ("subtype", "color", "size_bin", "orientation")
TARGETS = frozenset({"ship", "aircraft", "vehicle", "tank", "bridge", "harbor"})
DATASETS = {"task4": ("match", ("rgb", "sar")), "task5": ("time", ("time1", "time2"))}
VISIBILITY = frozenset({"clear", "partial", "poor"})


class ReplayError(ValueError):
    """Explicit replay validation failure."""


def _path(root, value):
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ReplayError("Replay paths must be root-relative without traversal")
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ReplayError("Replay path escapes data root")
    return result


def _rows(sheet, required):
    iterator = sheet.iter_rows(values_only=True)
    headers = next(iterator, ())
    if len(set(headers)) != len(headers) or not set(required).issubset(headers):
        raise ReplayError(f"Invalid {sheet.title} headers")
    for number, row in enumerate(iterator, 2):
        if any(v is not None for v in row):
            yield number, dict(zip(headers, row))


def _number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ReplayError("Invalid numeric workbook field") from None
    if not math.isfinite(result):
        raise ReplayError("Nonfinite numeric workbook field")
    return result


def load_property_workbook(dataset_id, family, pair_id, target):
    """Read the attributes of ``target`` for one frozen dataset pair.

    Returns ``{"pair", "attributes", "instance_counts", "provenance"}`` where
    ``attributes`` maps ``view`` -> ``{raw detector index: {"attributes",
    "context", "bbox", "category", "instance_id", "sheet_row"}}``.  Rows of any
    other category are dropped without being validated: they are not attributes
    of the requested target.  ``instance_counts`` is the frozen row count per
    view (all classes) and exists only to audit positional binding.  The
    returned payload never contains a detection list.
    """
    if os.environ.get("ONEID_ENABLE_WORKBOOK_REPLAY") != "1":
        raise ReplayError("Workbook replay requires ONEID_ENABLE_WORKBOOK_REPLAY=1")
    if not isinstance(dataset_id, str) or dataset_id not in DATASETS or DATASETS[dataset_id][0] != family:
        raise ReplayError("Expected task4/match or task5/time")
    if not isinstance(pair_id, str) or not pair_id.startswith(dataset_id + "_") or not pair_id.replace("_", "").isalnum():
        raise ReplayError("Explicit dataset-scoped pair_id required")
    if not isinstance(target, str) or target not in TARGETS:
        raise ReplayError("Explicit supported target required before loading attributes")
    if not os.environ.get("ONEID_DATA_ROOT"):
        raise ReplayError("ONEID_DATA_ROOT required")
    root = Path(os.environ["ONEID_DATA_ROOT"]).expanduser().resolve()
    filename = f"tool_{dataset_id}_property.xlsx"
    if os.environ.get("ONEID_REPLAY_CONFIG"):
        config = json.loads(Path(os.environ["ONEID_REPLAY_CONFIG"]).read_text())
        filename = config.get("datasets", {}).get(dataset_id)
        if not isinstance(filename, str):
            raise ReplayError("Dataset not allowlisted")
    path = _path(root, filename)
    if path.name != f"tool_{dataset_id}_property.xlsx" or not path.is_file():
        raise ReplayError("Allowlisted tool workbook unavailable")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        if not {"instances", "samples"}.issubset(wb.sheetnames):
            raise ReplayError("instances/samples required")
        sample, pair_ids = None, set()
        for _, row in _rows(wb["samples"], ("pair_id", "image_a", "image_b")):
            if row["pair_id"] in pair_ids:
                raise ReplayError("Duplicate sample ID")
            pair_ids.add(row["pair_id"])
            if row["pair_id"] == pair_id:
                sample = row
        if sample is None:
            raise ReplayError("Unknown pair_id")
        images = {k: str(_path(root, sample[c])) for k, c in (("a", "image_a"), ("b", "image_b"))}
        if images["a"] == images["b"] or not all(Path(p).is_file() for p in images.values()):
            raise ReplayError("Replay images missing or identical")
        views = DATASETS[dataset_id][1]
        records, ids, dimensions = {"a": {}, "b": {}}, set(), {}
        snapshot_rows = {"a": 0, "b": 0}
        required = ("instance_id", "pair_id", "view", "image_file", "image_width", "image_height", "x1", "y1", "x2", "y2", "category") + ATTRIBUTES
        for number, row in _rows(wb["instances"], required):
            if row["pair_id"] != pair_id:
                continue
            raw_view = str(row["view"]).lower()
            if raw_view not in views:
                raise ReplayError("Invalid view")
            view = "a" if raw_view == views[0] else "b"
            iid = str(row["instance_id"] or "")
            if not iid.startswith(f"{pair_id}:{raw_view}:") or iid in ids:
                raise ReplayError("Duplicate or inconsistent instance ID")
            ids.add(iid)
            if str(_path(root, row["image_file"])) != images[view]:
                raise ReplayError("Instance/sample image mismatch")
            width, height = (_number(row[c]) for c in ("image_width", "image_height"))
            box = [_number(row[c]) for c in ("x1", "y1", "x2", "y2")]
            if not (width > 0 and height > 0 and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ReplayError("Invalid bbox/dimensions")
            if view in dimensions and dimensions[view] != (width, height):
                raise ReplayError("Inconsistent dimensions")
            dimensions[view] = (width, height)
            snapshot_rows[view] += 1
            label = str(row["category"] or "").strip().lower()
            if label != target:
                # Another class' row is never read: it is not an attribute of the
                # requested target, so its category vocabulary and attribute
                # values are not this mode's business.  Box/dimension checks stay
                # global because they tie the whole sheet to the pair's rasters.
                continue
            suffix = iid.rsplit(":", 1)[-1]
            if not suffix.isdigit():
                raise ReplayError("Instance ID requires a numeric stable index")
            index = int(suffix)
            if index in records[view]:
                raise ReplayError("Duplicate numeric instance index")
            attrs = {a: str(row[a]).strip().lower() for a in ATTRIBUTES if row.get(a) is not None and str(row[a]).strip()}
            context = {}
            if row.get("visibility") is not None and str(row["visibility"]).strip():
                visibility = str(row["visibility"]).strip().lower()
                if visibility not in VISIBILITY:
                    raise ReplayError("Invalid visibility")
                context["visibility"] = visibility
            records[view][index] = {"attributes": attrs, "context": context, "bbox": box, "category": label, "instance_id": iid, "sheet_row": number}
        if not any(records.values()):
            raise ReplayError(f"Workbook carries no attributes for target {target}")
        from PIL import Image
        for view, expected in dimensions.items():
            with Image.open(images[view]) as image:
                if image.size != expected:
                    raise ReplayError("Raster/workbook dimensions differ")
        return {
            "pair": {"base_id": pair_id, "family": family, "images": images},
            "attributes": records,
            # Every instance row of the pair, all classes: the frozen detector
            # output size per view.  Used only to audit positional binding.
            "instance_counts": snapshot_rows,
            "provenance": {
                "source_mode": "workbook_replay",
                "detection_source": "live_detectors",
                "attribute_source": "property_workbook",
                "dataset_id": dataset_id,
                "family": family,
                "pair_id": pair_id,
                "target": target,
                "workbook": str(path),
                "sha256": digest,
                "sheets_read": ["instances", "samples"],
                "truth_data": False,
                "attribute_only": True,
            },
        }
    finally:
        wb.close()

