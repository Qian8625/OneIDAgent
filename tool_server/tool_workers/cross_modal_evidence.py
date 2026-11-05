"""RGB/SAR 检测证据管道（唯一实现）。

痛点：检测结果只有经过"登记 → provenance → 传入 CrossModalInterpret"这一条链路，
才允许参与跨模态推理；否则模型只能手抄检测框，或者拿不到上一步工具的输出。

本模块把这条链路收敛成一处实现，app 变体与 eval 侧的 ``agent_runner`` 共用：

* :func:`extract_detections` —— 从工具结构化输出里取检测列表；
* :func:`register_detections` —— 登记检测 + provenance 到 registry；
* :func:`prepare_interpret_args` —— 调 CrossModalInterpret 前把两者注入参数并做门禁。

除本文件外，仓库里不应再有第二份实现。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

#: 提示词里的示例检测框：一旦出现在入参里就说明模型在抄示例，而不是引用真实工具输出。
EXAMPLE_BBOX_PATTERNS = (
    "(10, 20, 50, 80), ship, score 0.73",
    "(12, 22, 52, 82), ship, score 0.86",
)

#: 产出 RGB 检测的工具。
RGB_DETECTION_TOOLS = ("ObjectDetection", "SmallObjectDetection")

#: 产出 SAR 检测的工具。
SAR_DETECTION_TOOLS = ("DenoDetV2",)

#: registry 里检测与 provenance 的键名。
RGB_DETECTIONS_KEY = "latest_rgb_detections"
SAR_DETECTIONS_KEY = "latest_sar_detections"
RGB_PROVENANCE_KEY = "latest_rgb_evidence_provenance"
SAR_PROVENANCE_KEY = "latest_sar_evidence_provenance"

#: 依赖不满足时给模型的下一步提示（避免"重试死循环"）。
EMPTY_DETECTION_HINT = (
    "CrossModalInterpret needs both RGB and SAR detections produced by successful tools. "
    "Run ObjectDetection on the RGB image and DenoDetV2 on the SAR image first, then call "
    "CrossModalInterpret again."
)

EXAMPLE_LEAKAGE_HINT = (
    "The argument contains the prompt's example detection box. Do not copy example boxes; "
    "pass detection output from a previous successful tool call instead."
)


def extract_detections(structured: Any) -> List[Dict[str, Any]]:
    """Pull a detection list out of a tool's structured output."""
    if not isinstance(structured, dict):
        return []
    for key in ("detections", "selected_detections", "objects_used", "rgb_evidence", "sar_evidence"):
        value = structured.get(key)
        if isinstance(value, list) and value:
            return [item for item in value if isinstance(item, dict)]
    evidence = structured.get("evidence")
    if isinstance(evidence, dict):
        for key in ("rgb_evidence", "sar_evidence", "matched_pairs"):
            value = evidence.get(key)
            if isinstance(value, list) and value:
                return [item for item in value if isinstance(item, dict)]
    return []


def detection_modality(api_name: str) -> Optional[str]:
    """``"rgb"`` / ``"sar"`` for detection tools, else ``None``."""
    if api_name in RGB_DETECTION_TOOLS:
        return "rgb"
    if api_name in SAR_DETECTION_TOOLS:
        return "sar"
    return None


def build_provenance(api_name: str, round_id: Any) -> Dict[str, Any]:
    """Provenance record proving the detections came from a successful tool call."""
    return {
        "source_tool": api_name,
        "source_round": round_id,
        "source_success": True,
        "source_response_id": f"round_{round_id}_{api_name.lower()}_structured",
    }


def register_detections(
    registry: Dict[str, Any],
    api_name: str,
    structured: Any,
    round_id: Any,
) -> Dict[str, Any]:
    """Record a successful detection tool's output in ``registry``.

    No-op (returns the same registry) for tools that do not produce RGB/SAR
    detections, or when the structured payload carries no usable boxes.
    """
    modality = detection_modality(api_name)
    if modality is None:
        return registry
    detections = extract_detections(structured)
    if not detections:
        return registry
    registry = dict(registry)
    provenance = build_provenance(api_name, round_id)
    if modality == "rgb":
        registry[RGB_DETECTIONS_KEY] = detections
        registry[RGB_PROVENANCE_KEY] = provenance
    else:
        registry[SAR_DETECTIONS_KEY] = detections
        registry[SAR_PROVENANCE_KEY] = provenance
    return registry


def contains_example_bbox(api_args: Dict[str, Any]) -> bool:
    text = json.dumps(api_args or {}, ensure_ascii=False)
    return any(pattern in text for pattern in EXAMPLE_BBOX_PATTERNS)


def prepare_interpret_args(
    api_args: Dict[str, Any],
    registry: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Inject registered detections + provenance into CrossModalInterpret args.

    Returns ``(args, error)``; ``error`` is ``None`` when the call may proceed,
    otherwise one of ``example_leakage_detected`` / ``invalid_dependency``.
    """
    args = dict(api_args or {})
    provenance = dict(args.get("evidence_provenance") or {})
    if not args.get("rgb_detections") and registry.get(RGB_DETECTIONS_KEY):
        args["rgb_detections"] = registry[RGB_DETECTIONS_KEY]
        provenance["rgb"] = registry.get(RGB_PROVENANCE_KEY)
    if not args.get("sar_detections") and registry.get(SAR_DETECTIONS_KEY):
        args["sar_detections"] = registry[SAR_DETECTIONS_KEY]
        provenance["sar"] = registry.get(SAR_PROVENANCE_KEY)
    if provenance:
        args["evidence_provenance"] = provenance
    if contains_example_bbox(args) and not provenance:
        return args, "example_leakage_detected"
    has_rgb = bool(args.get("rgb_detections") or args.get("rgb_boxes"))
    has_sar = bool(args.get("sar_detections") or args.get("sar_candidates") or args.get("sar_boxes"))
    if (args.get("rgb_text") or args.get("sar_text")) and not provenance:
        return args, "invalid_dependency"
    if not (has_rgb and has_sar and provenance):
        return args, "invalid_dependency"
    return args, None


def dependency_hint(error: Optional[str]) -> Optional[str]:
    """Human-readable next step for a dependency failure."""
    if error == "invalid_dependency":
        return EMPTY_DETECTION_HINT
    if error == "example_leakage_detected":
        return EXAMPLE_LEAKAGE_HINT
    return None


__all__ = [
    "EMPTY_DETECTION_HINT",
    "EXAMPLE_BBOX_PATTERNS",
    "EXAMPLE_LEAKAGE_HINT",
    "RGB_DETECTION_TOOLS",
    "SAR_DETECTION_TOOLS",
    "build_provenance",
    "contains_example_bbox",
    "dependency_hint",
    "detection_modality",
    "extract_detections",
    "prepare_interpret_args",
    "register_detections",
]
