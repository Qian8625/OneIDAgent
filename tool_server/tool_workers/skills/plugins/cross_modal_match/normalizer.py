"""Parameter normalizer for CrossModalMatchSkill."""
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


VALID_STRATEGIES = {"iou", "center_distance", "hybrid", "global"}


def normalize_match_strategy(value: Any) -> str:
    text = str(value or "hybrid").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "center": "center_distance",
        "distance": "center_distance",
        "center_dist": "center_distance",
        "centre_distance": "center_distance",
        "greedy": "global",
        "global_greedy": "global",
        "score": "hybrid",
        "hybrid_score": "hybrid",
    }
    strategy = aliases.get(text, text)
    return strategy if strategy in VALID_STRATEGIES else "hybrid"


def _set_float_default(params: Dict[str, Any], key: str, default: float):
    value = to_float_or_none(params.get(key))
    params[key] = default if value is None else value


@register_normalizer("cross_modal_match", override=True)
def normalize_cross_modal_match(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "rgb_image", pick_first(p, ["rgb_image", "optical_image", "rgb", "image", "primary_image"]))
    set_if_missing(p, "sar_image", pick_first(p, ["sar_image", "sar", "radar_image", "radar", "secondary_image", "time1_image"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "rgb_detections", pick_first(p, ["rgb_detections", "detections", "rgb_boxes", "boxes"]))
    set_if_missing(p, "sar_candidates", pick_first(p, ["sar_candidates", "sar_detections", "sar_boxes", "radar_boxes"]))
    set_if_missing(p, "match_strategy", pick_first(p, ["match_strategy", "strategy", "mode"], "hybrid"))
    p["match_strategy"] = normalize_match_strategy(p.get("match_strategy"))

    set_if_missing(p, "iou_threshold", pick_first(p, ["iou_threshold", "match_threshold"], 0.2))
    set_if_missing(p, "center_distance_threshold", pick_first(p, ["center_distance_threshold", "max_center_distance"], "auto"))
    set_if_missing(p, "score_threshold", pick_first(p, ["score_threshold", "confidence_threshold", "box_threshold"], 0.3))
    set_if_missing(p, "visualize", True)
    set_if_missing(p, "top1", False)
    set_if_missing(p, "max_draw", 20)

    _set_float_default(p, "iou_threshold", 0.2)
    _set_float_default(p, "score_threshold", 0.3)
    if str(p.get("center_distance_threshold", "")).strip().lower() != "auto":
        value = to_float_or_none(p.get("center_distance_threshold"))
        p["center_distance_threshold"] = "auto" if value is None else value
    else:
        p["center_distance_threshold"] = "auto"

    p["visualize"] = to_bool(p.get("visualize"), True)
    p["top1"] = to_bool(p.get("top1"), False)
    max_draw = to_int_or_none(p.get("max_draw"))
    p["max_draw"] = 20 if max_draw is None else max_draw

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["rgb_image", "sar_image", "target", "rgb_detections", "sar_candidates", "match_strategy"],
            "target": p.get("target"),
            "match_strategy": p.get("match_strategy"),
        }
    )
    return p, trace
