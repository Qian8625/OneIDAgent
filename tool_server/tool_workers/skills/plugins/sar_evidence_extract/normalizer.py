"""Parameter normalizer for SAREvidenceExtractSkill."""
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


def _normalize_preprocess_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "none": "direct",
        "raw": "direct",
        "passthrough": "direct",
        "pass_through": "direct",
        "sar_preprocess": "sar_to_rgb",
        "sartorgb": "sar_to_rgb",
        "sar2rgb": "sar_to_rgb",
        "rgb": "sar_to_rgb",
    }
    return aliases.get(mode, mode)


def _normalize_detector(value: Any) -> str:
    detector = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "deno": "denodetv2",
        "denodet": "denodetv2",
        "denodet_v2": "denodetv2",
        "sar": "denodetv2",
        "sar_detector": "denodetv2",
    }
    return aliases.get(detector, detector)


@register_normalizer("sar_evidence_extract", override=True)
def normalize_sar_evidence_extract(params: Dict[str, Any]):
    p = dict(params or {})
    trace = []

    set_if_missing(p, "sar_image", pick_first(p, ["sar_image", "image", "input_image", "primary_image"]))
    set_if_missing(p, "target", normalize_target_text(pick_first(p, ["target", "object", "text", "category", "class_name"])))
    set_if_missing(p, "preprocess_mode", pick_first(p, ["preprocess_mode", "sar_preprocess_mode"], "auto"))
    set_if_missing(p, "confidence_threshold", pick_first(p, ["confidence_threshold", "box_threshold"], 0.25))
    set_if_missing(p, "response_threshold", pick_first(p, ["response_threshold", "sar_response_threshold"], "auto"))
    set_if_missing(p, "max_candidates", pick_first(p, ["max_candidates", "top_k", "limit"], 50))
    set_if_missing(p, "detector", pick_first(p, ["detector", "sar_detector"], "auto"))
    set_if_missing(p, "visualize", pick_first(p, ["visualize", "draw", "visual_output"], True))

    p["preprocess_mode"] = _normalize_preprocess_mode(p.get("preprocess_mode"))
    p["detector"] = _normalize_detector(p.get("detector"))

    for key in ["confidence_threshold", "min_area", "max_area"]:
        if key in p:
            value = to_float_or_none(p[key])
            if value is not None:
                p[key] = value

    if str(p.get("response_threshold", "")).strip().lower() != "auto":
        value = to_float_or_none(p.get("response_threshold"))
        if value is not None:
            p["response_threshold"] = value

    for key in ["max_candidates", "max_draw"]:
        if key in p:
            value = to_int_or_none(p[key])
            if value is not None:
                p[key] = value

    if "visualize" in p:
        p["visualize"] = to_bool(p["visualize"], True)

    trace.append(
        {
            "stage": "normalize_payload",
            "status": "success",
            "fields": ["sar_image", "target", "preprocess_mode", "confidence_threshold", "response_threshold"],
            "target": p.get("target"),
            "preprocess_mode": p.get("preprocess_mode"),
            "detector": p.get("detector"),
        }
    )
    return p, trace
