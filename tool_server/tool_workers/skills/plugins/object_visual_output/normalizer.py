"""Parameter normalizer for ObjectVisualOutputSkill."""
import ast
import re
from typing import Any, Dict, List, Optional

from tool_server.tool_workers.skills.normalizers import register_normalizer
from tool_server.tool_workers.skills.utils import pick_first, set_if_missing, to_int_or_none


VISUAL_ALIASES = {
    "box": "box",
    "draw box": "box",
    "bbox": "box",
    "画框": "box",
    "框出": "box",
    "point": "point",
    "mark": "point",
    "plot point": "point",
    "画点": "point",
    "标点": "point",
    "plot": "plot",
    "visualize": "plot",
    "可视化": "plot",
    "label": "box_and_label",
    "annotate": "box_and_label",
    "box and label": "box_and_label",
    "标注文字": "box_and_label",
    "标注": "box_and_label",
    "measurement overlay": "measurement_overlay",
    "overlay": "measurement_overlay",
    "测量标注": "measurement_overlay",
}


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _normalize_visual_type(value: Any) -> str:
    text = _normalize_text(value)
    if text in VISUAL_ALIASES:
        return VISUAL_ALIASES[text]
    for key, visual_type in VISUAL_ALIASES.items():
        if key in text:
            return visual_type
    return text or "box"


def _literal_or_raw(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return ast.literal_eval(text)
    except Exception:
        return value


def _coerce_bbox(value: Any) -> Optional[List[float]]:
    value = _literal_or_raw(value)
    if isinstance(value, dict):
        value = value.get("bbox") or value.get("box")
    if isinstance(value, str):
        nums = re.findall(r"[-+]?\d*\.?\d+", value)
        if len(nums) >= 4:
            value = nums[:4]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _coerce_point(value: Any) -> Optional[List[float]]:
    value = _literal_or_raw(value)
    if isinstance(value, dict):
        value = value.get("point") or value.get("centroid") or value.get("center")
    if isinstance(value, str):
        nums = re.findall(r"[-+]?\d*\.?\d+", value)
        if len(nums) >= 2:
            value = nums[:2]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = [float(v) for v in value[:2]]
    except Exception:
        return None
    return [x, y]


def _labels(value: Any) -> List[str]:
    value = _literal_or_raw(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    if value not in [None, ""]:
        return [str(value)]
    return []


def _objects_from_bbox_values(value: Any, source: str, labels: List[str]) -> List[Dict[str, Any]]:
    value = _literal_or_raw(value)
    values = value if isinstance(value, list) and value and isinstance(value[0], (list, tuple, dict)) else [value]
    objects = []
    for idx, item in enumerate(values):
        bbox = _coerce_bbox(item)
        if not bbox:
            continue
        label = labels[idx] if idx < len(labels) else None
        if isinstance(item, dict):
            label = item.get("label") or item.get("selection_reason") or label
        objects.append({"type": "bbox", "bbox": bbox, "label": label, "source": source})
    return objects


def _objects_from_points(value: Any, source: str, labels: List[str]) -> List[Dict[str, Any]]:
    value = _literal_or_raw(value)
    values = value if isinstance(value, list) and value and isinstance(value[0], (list, tuple, dict)) else [value]
    objects = []
    for idx, item in enumerate(values):
        point = _coerce_point(item)
        if not point:
            continue
        label = labels[idx] if idx < len(labels) else None
        if isinstance(item, dict):
            label = item.get("label") or item.get("selection_reason") or label
        objects.append({"type": "point", "point": point, "label": label, "source": source})
    return objects


def _objects_from_detections(value: Any, source: str, labels: List[str]) -> List[Dict[str, Any]]:
    value = _literal_or_raw(value)
    if not isinstance(value, list):
        return []
    return _objects_from_bbox_values(value, source, labels)


def _objects_from_measurement(measurement: Any, source: str) -> List[Dict[str, Any]]:
    if not isinstance(measurement, dict):
        return []
    point = _coerce_point(measurement.get("centroid") or measurement.get("point"))
    if point:
        return [{"type": "point", "point": point, "label": "centroid", "source": source}]
    return []


def _objects_from_structured(structured: Any, source: str) -> List[Dict[str, Any]]:
    if not isinstance(structured, dict):
        return []
    objects: List[Dict[str, Any]] = []
    objects.extend(_objects_from_detections(structured.get("selected"), f"{source}.selected", []))
    objects.extend(_objects_from_detections(structured.get("selected_detections"), f"{source}.selected_detections", []))
    objects.extend(_objects_from_detections(structured.get("objects_used"), f"{source}.objects_used", []))
    if not objects:
        objects.extend(_objects_from_detections(structured.get("detections"), f"{source}.detections", []))
    objects.extend(_objects_from_measurement(structured.get("measurement"), f"{source}.measurement"))
    return objects


def _latest_structured(input_registry: Any) -> List[Dict[str, Any]]:
    if not isinstance(input_registry, dict):
        return []
    objects: List[Dict[str, Any]] = []
    latest = input_registry.get("latest_structured")
    if isinstance(latest, dict):
        objects.extend(_objects_from_structured(latest, "input_registry.latest_structured"))
    for key, value in input_registry.items():
        if key != "latest_structured" and isinstance(key, str) and key.endswith("_structured") and isinstance(value, dict):
            objects.extend(_objects_from_structured(value, f"input_registry.{key}"))
    return objects


@register_normalizer("object_visual_output", override=True)
def normalize_object_visual_output(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "image", pick_first(p, ["image", "img", "input_image", "primary_image"]))
    set_if_missing(p, "visual_type", pick_first(p, ["visual_type", "mode", "task_type", "operation", "intent"], "box"))
    p["visual_type"] = _normalize_visual_type(p.get("visual_type"))

    labels = _labels(p.get("labels"))
    visual_objects: List[Dict[str, Any]] = []
    visual_objects.extend(_objects_from_bbox_values(p.get("bbox"), "bbox_input", labels))
    visual_objects.extend(_objects_from_bbox_values(p.get("bboxes"), "bbox_input", labels))
    visual_objects.extend(_objects_from_points(p.get("points"), "points_input", labels))
    visual_objects.extend(_objects_from_detections(p.get("selected_detections"), "selected_detections", labels))
    visual_objects.extend(_objects_from_detections(p.get("detections"), "detections", labels))
    visual_objects.extend(_objects_from_measurement(p.get("measurement"), "measurement"))
    visual_objects.extend(_objects_from_structured(p.get("structured"), "structured"))
    visual_objects.extend(_latest_structured(p.get("input_registry")))

    if visual_objects and not p.get("visual_objects"):
        p["visual_objects"] = visual_objects
    if "max_draw" in p:
        max_draw = to_int_or_none(p.get("max_draw"))
        if max_draw is not None:
            p["max_draw"] = max_draw
    if "thickness" in p:
        thickness = to_int_or_none(p.get("thickness"))
        if thickness is not None:
            p["thickness"] = thickness

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["image", "visual_type", "visual_objects"],
            "visual_type": p.get("visual_type"),
            "num_visual_objects": len(p.get("visual_objects") or []),
        }
    )
    return p, trace
