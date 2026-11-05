"""Parameter normalizer for ObjectMeasureSkill."""
from typing import Any, Dict

from tool_server.tool_workers.skills.normalizers import register_normalizer
from tool_server.tool_workers.skills.utils import (
    normalize_target_text,
    pick_first,
    set_if_missing,
    to_bool,
)


MEASURE_ALIASES = {
    "area": "area",
    "面积": "area",
    "distance": "distance",
    "距离": "distance",
    "像素距离": "distance",
    "实际距离": "distance",
    "between": "pair_distance",
    "pair": "pair_distance",
    "pair distance": "pair_distance",
    "pair_distance": "pair_distance",
    "最小和最大之间": "pair_distance",
    "center": "centroid",
    "centroid": "centroid",
    "center point": "centroid",
    "中心点": "centroid",
    "bbox size": "bbox_size",
    "bbox_size": "bbox_size",
    "size": "bbox_size",
    "尺寸": "bbox_size",
}


SELECT_ALIASES = {
    "first": "first",
    "target": "first",
    "a": "first",
    "second": "second",
    "reference": "second",
    "b": "second",
    "smallest": "smallest",
    "small": "smallest",
    "最小": "smallest",
    "largest": "largest",
    "large": "largest",
    "biggest": "largest",
    "最大": "largest",
    "nearest": "nearest",
    "closest": "nearest",
    "最近": "nearest",
    "farthest": "farthest",
    "furthest": "farthest",
    "最远": "farthest",
    "left": "leftmost",
    "leftmost": "leftmost",
    "最左": "leftmost",
    "right": "rightmost",
    "rightmost": "rightmost",
    "最右": "rightmost",
    "top": "topmost",
    "topmost": "topmost",
    "最上": "topmost",
    "bottom": "bottommost",
    "bottommost": "bottommost",
    "最下": "bottommost",
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _normalize_measure_type(value: Any) -> str:
    text = _normalize_text(value)
    if text in MEASURE_ALIASES:
        return MEASURE_ALIASES[text]
    for key, mode in MEASURE_ALIASES.items():
        if key in text:
            return mode
    return text


def _normalize_selector(value: Any) -> str:
    text = _normalize_text(value)
    if text in SELECT_ALIASES:
        return SELECT_ALIASES[text]
    for key, selector in SELECT_ALIASES.items():
        if key in text:
            return selector
    return text


def _infer_measure_type(params: Dict[str, Any]) -> str:
    raw = pick_first(params, ["measure_type", "measurement_type", "mode", "task_type", "operation", "intent"])
    text = _normalize_text(raw)
    context = _normalize_text(" ".join(str(x) for x in [params.get("query"), params.get("instruction"), params.get("target")] if x))
    combined = " ".join(x for x in [text, context] if x)
    inferred = _normalize_measure_type(combined)
    return inferred or "distance"


def _parse_gsd(value: Any):
    if value in [None, ""]:
        return None, None
    try:
        parsed = float(value)
    except Exception:
        return None, f"Invalid gsd_m_per_pixel: {value}"
    if parsed <= 0:
        return None, f"gsd_m_per_pixel must be positive: {value}"
    return parsed, None


@register_normalizer("object_measure", override=True)
def normalize_object_measure(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []
    raw_context = _normalize_text(
        " ".join(
            str(x)
            for x in [
                p.get("query"),
                p.get("instruction"),
                p.get("measure_type"),
                p.get("measurement_type"),
                p.get("mode"),
                p.get("target"),
            ]
            if x
        )
    )

    set_if_missing(p, "image", pick_first(p, ["image", "img", "input_image", "primary_image"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "measure_type", _infer_measure_type(p))
    if p.get("measure_type") not in [None, ""]:
        p["measure_type"] = _normalize_measure_type(p.get("measure_type"))

    gsd_raw = pick_first(p, ["gsd_m_per_pixel", "gsd", "GSD", "pixel_size", "pixel_size_m", "meter_per_pixel", "meters_per_pixel"])
    gsd_value, gsd_error = _parse_gsd(gsd_raw)
    if gsd_error:
        p["normalizer_error"] = gsd_error
    elif gsd_value is not None:
        p["gsd_m_per_pixel"] = gsd_value

    for key in ["select_a", "selection_a", "source_selection"]:
        if p.get(key) not in [None, ""]:
            p["select_a"] = _normalize_selector(p.get(key))
            break
    for key in ["select_b", "selection_b", "target_selection", "reference_selection"]:
        if p.get(key) not in [None, ""]:
            p["select_b"] = _normalize_selector(p.get(key))
            break

    context = _normalize_text(" ".join(str(x) for x in [raw_context, p.get("query"), p.get("instruction"), p.get("measure_type"), p.get("target")] if x))
    if not p.get("select_a") and "smallest" in context:
        p["select_a"] = "smallest"
    if not p.get("select_a") and "最小" in context:
        p["select_a"] = "smallest"
    if not p.get("select_b") and "largest" in context:
        p["select_b"] = "largest"
    if not p.get("select_b") and "最大" in context:
        p["select_b"] = "largest"

    if p.get("select_a") and p.get("select_b") and p.get("measure_type") == "distance":
        p["measure_type"] = "pair_distance"
    if "between" in context or "之间" in context:
        p["measure_type"] = "pair_distance"

    if "unit" in p and p.get("unit") not in [None, ""]:
        unit = _normalize_text(p.get("unit"))
        p["unit"] = "pixel" if unit in {"pixel", "pixels", "px", "像素"} else "meter"
    if "visualize" in p:
        p["visualize"] = to_bool(p["visualize"], False)

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "failed" if p.get("normalizer_error") else "success",
            "fields": ["image", "target", "measure_type", "gsd_m_per_pixel", "select_a", "select_b"],
            "target": p.get("target"),
            "measure_type": p.get("measure_type"),
            "select_a": p.get("select_a"),
            "select_b": p.get("select_b"),
            "normalizer_error": p.get("normalizer_error"),
        }
    )
    return p, trace
