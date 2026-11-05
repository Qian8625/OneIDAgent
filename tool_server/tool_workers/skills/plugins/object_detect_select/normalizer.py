"""Parameter normalizer for ObjectDetectSelectSkill."""
from typing import Any, Dict

from tool_server.tool_workers.skills.normalizers import register_normalizer
from tool_server.tool_workers.skills.utils import (
    normalize_target_text,
    pick_first,
    set_if_missing,
    to_bool,
    to_float_or_none,
    to_int_or_none,
)


MEASUREMENT_KEYS = {
    "gsd",
    "gsd_m_per_pixel",
    "resolution",
    "meter_per_pixel",
    "meters_per_pixel",
    "pixel_size_m",
    "distance",
    "distance_m",
    "distance_px",
    "area",
    "area_m2",
    "total_area_m2",
    "measure",
    "measurement",
    "reference_target",
}


SELECTION_ALIASES = {
    "all": "all",
    "全部": "all",
    "所有": "all",
    "smallest": "smallest",
    "small": "smallest",
    "area smallest": "smallest",
    "min area": "smallest",
    "minimum": "smallest",
    "最小": "smallest",
    "面积最小": "smallest",
    "largest": "largest",
    "large": "largest",
    "biggest": "largest",
    "max area": "largest",
    "maximum": "largest",
    "最大": "largest",
    "面积最大": "largest",
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
    "upper": "topmost",
    "topmost": "topmost",
    "上方": "topmost",
    "最上": "topmost",
    "bottom": "bottommost",
    "lower": "bottommost",
    "bottommost": "bottommost",
    "下方": "bottommost",
    "最下": "bottommost",
    "nth": "nth",
    "第几个": "nth",
}


def _normalize_selection_mode(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", "-")
    text = " ".join(text.replace("-", " ").split())
    if text in SELECTION_ALIASES:
        return SELECTION_ALIASES[text]

    for key, mode in SELECTION_ALIASES.items():
        if key and key in text:
            return mode
    return text


def _infer_selection_mode(params: Dict[str, Any]) -> str:
    raw = pick_first(params, ["selection_mode", "selection", "mode", "task_type", "operation", "intent"])
    if raw not in [None, ""]:
        return _normalize_selection_mode(raw)

    text = " ".join(
        str(x).lower()
        for x in [params.get("query"), params.get("instruction"), params.get("target"), params.get("text")]
        if x not in [None, ""]
    )
    inferred = _normalize_selection_mode(text)
    return inferred if inferred else "all"


@register_normalizer("object_detect_select", override=True)
def normalize_object_detect_select(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    removed_measurement_keys = []
    for key in list(p.keys()):
        if key in MEASUREMENT_KEYS:
            removed_measurement_keys.append(key)
            p.pop(key, None)

    set_if_missing(p, "image", pick_first(p, ["image", "img", "input_image", "primary_image"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "selection_mode", _infer_selection_mode(p))
    if p.get("selection_mode") not in [None, ""]:
        p["selection_mode"] = _normalize_selection_mode(p.get("selection_mode"))

    if "visualize" in p:
        p["visualize"] = to_bool(p["visualize"], True)
    if "top_k" in p:
        top_k = to_int_or_none(p["top_k"])
        if top_k is not None:
            p["top_k"] = top_k
    if "max_draw" in p:
        max_draw = to_int_or_none(p["max_draw"])
        if max_draw is not None:
            p["max_draw"] = max_draw
    if "nth" in p:
        nth = to_int_or_none(p["nth"])
        if nth is not None:
            p["nth"] = nth
    if "confidence_threshold" in p:
        threshold = to_float_or_none(p["confidence_threshold"])
        if threshold is not None:
            p["confidence_threshold"] = threshold
    if "iou_threshold" in p:
        iou_threshold = to_float_or_none(p["iou_threshold"])
        if iou_threshold is not None:
            p["iou_threshold"] = iou_threshold

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["image", "target", "selection_mode"],
            "target": p.get("target"),
            "selection_mode": p.get("selection_mode"),
            "ignored_measurement_fields": removed_measurement_keys,
        }
    )
    return p, trace
