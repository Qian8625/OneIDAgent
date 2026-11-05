"""Parameter normalizer for CrossModalTaskSkill."""
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


TASK_ALIASES = {
    "verify": "optical_sar_verify",
    "confirm": "optical_sar_verify",
    "verify with sar": "optical_sar_verify",
    "confirm with sar": "optical_sar_verify",
    "validate": "optical_sar_verify",
    "cross validate": "optical_sar_verify",
    "cross validation": "optical_sar_verify",
    "crosscheck": "optical_sar_verify",
    "crosscheck_exists": "optical_sar_verify",
    "crosscheck exists": "optical_sar_verify",
    "cross check exists": "optical_sar_verify",
    "exists": "optical_sar_verify",
    "exist": "optical_sar_verify",
    "presence": "optical_sar_verify",
    "largest": "cross_modal_select_largest",
    "select largest": "cross_modal_select_largest",
    "select_largest": "cross_modal_select_largest",
    "highest confidence": "cross_modal_select_highest_score",
    "highest-confidence": "cross_modal_select_highest_score",
    "highest_confidence": "cross_modal_select_highest_score",
    "select highest score": "cross_modal_select_highest_score",
    "select_highest_score": "cross_modal_select_highest_score",
    "highest_score": "cross_modal_select_highest_score",
    "highest_confidence": "cross_modal_select_highest_score",
    "joint_scene": "optical_sar_consistency",
    "distance_largest_smallest": "cross_modal_distance_largest_smallest",
    "largest_smallest_distance": "cross_modal_distance_largest_smallest",
    "select_smallest": "cross_modal_select_largest",
    "select_largest_smallest": "cross_modal_distance_largest_smallest",
    "final position": "cross_modal_select_largest",
    "return final position": "cross_modal_select_largest",
    "sar verify": "optical_sar_verify",
    "sar validation": "optical_sar_verify",
    "用sar确认": "optical_sar_verify",
    "用SAR确认": "optical_sar_verify",
    "sar验证": "optical_sar_verify",
    "SAR验证": "optical_sar_verify",
    "attribute": "sar_optical_attribution",
    "attribution": "sar_optical_attribution",
    "sar attribution": "sar_optical_attribution",
    "what is sar target": "sar_optical_attribution",
    "sar对应什么": "sar_optical_attribution",
    "SAR对应什么": "sar_optical_attribution",
    "sar归因": "sar_optical_attribution",
    "SAR归因": "sar_optical_attribution",
    "recall": "sar_candidate_recall",
    "missed target": "sar_candidate_recall",
    "recall candidate": "sar_candidate_recall",
    "sar recall": "sar_candidate_recall",
    "补漏": "sar_candidate_recall",
    "SAR补漏": "sar_candidate_recall",
    "漏检": "sar_candidate_recall",
    "consistent": "optical_sar_consistency",
    "consistency": "optical_sar_consistency",
    "optical sar consistency": "optical_sar_consistency",
    "一致性": "optical_sar_consistency",
    "是否一致": "optical_sar_consistency",
    "光学sar一致性": "optical_sar_consistency",
    "光学SAR一致性": "optical_sar_consistency",
}


VALID_TASKS = {
    "optical_sar_verify",
    "sar_optical_attribution",
    "sar_candidate_recall",
    "optical_sar_consistency",
    "cross_modal_select_largest",
    "cross_modal_select_highest_score",
    "cross_modal_distance_largest_smallest",
}


def normalize_task_type(value: Any, question: Any = None) -> str:
    raw = str(value or "").strip()
    if raw in VALID_TASKS:
        return raw

    text = raw or str(question or "")
    compact = " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())
    if compact in TASK_ALIASES:
        return TASK_ALIASES[compact]

    compact_raw = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if compact_raw in TASK_ALIASES:
        return TASK_ALIASES[compact_raw]

    mixed = text.strip()
    for alias, task_type in TASK_ALIASES.items():
        if alias.lower() in compact or alias in mixed:
            return task_type
    return raw


@register_normalizer("cross_modal_task", override=True)
def normalize_cross_modal_task(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "rgb_image", pick_first(p, ["rgb_image", "optical_image", "image", "primary_image"]))
    set_if_missing(p, "sar_image", pick_first(p, ["sar_image", "radar_image", "sar", "radar", "secondary_image", "time1_image"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "question", pick_first(p, ["question", "query", "instruction", "prompt"]))
    set_if_missing(p, "task_type", normalize_task_type(pick_first(p, ["task_type", "task", "mode", "intent", "operation"]), p.get("question")))

    if p.get("task_type") not in [None, ""]:
        p["task_type"] = normalize_task_type(p.get("task_type"), p.get("question"))

    set_if_missing(p, "evidence_required", True)
    set_if_missing(p, "confidence_threshold", pick_first(p, ["confidence_threshold", "score_threshold", "box_threshold"], 0.3))
    set_if_missing(p, "visualize", True)
    set_if_missing(p, "max_draw", 20)

    if "evidence_required" in p:
        p["evidence_required"] = to_bool(p["evidence_required"], True)
    if "visualize" in p:
        p["visualize"] = to_bool(p["visualize"], True)
    if "confidence_threshold" in p:
        value = to_float_or_none(p["confidence_threshold"])
        if value is not None:
            p["confidence_threshold"] = value
    for key in ["match_threshold", "iou_threshold"]:
        if key in p:
            value = to_float_or_none(p[key])
            if value is not None:
                p[key] = value
    if "max_draw" in p:
        value = to_int_or_none(p["max_draw"])
        if value is not None:
            p["max_draw"] = value

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["rgb_image", "sar_image", "target", "task_type", "question"],
            "task_type": p.get("task_type"),
            "target": p.get("target"),
        }
    )
    return p, trace
