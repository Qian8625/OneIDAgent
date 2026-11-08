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
import os
import re
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
TIME1_DETECTIONS_KEY = "latest_time1_detections"
TIME2_DETECTIONS_KEY = "latest_time2_detections"
TIME1_PROVENANCE_KEY = "latest_time1_evidence_provenance"
TIME2_PROVENANCE_KEY = "latest_time2_evidence_provenance"

#: Replay never accepts server-side switches from the request: the workbook
#: path, the inference configs and raw text detections stay out of reach.  The
#: detector *payloads* are not on this list — they are required, because replay
#: runs live detection and only the target's attributes come from the workbook.
REPLAY_FORBIDDEN = frozenset({
    "rgb_text", "sar_text", "rgb_boxes", "sar_boxes", "sar_candidates",
    "allow_unprovenanced_text", "vi_config_path", "ds_config_path",
    "workbook_path", "georef_image",
})

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
    api_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Record a successful detection tool's output in ``registry``.

    RGB/SAR stay on the online keys.  A time-pair run uses the same detectors on
    two images, so a detection tool whose image is the registry's ``time1_image``
    or ``time2_image`` is stored under the matching time key and never clobbers
    the other view.
    """
    modality = detection_modality(api_name)
    if modality is None:
        return registry
    detections = extract_detections(structured)
    if not detections:
        return registry
    registry = dict(registry)
    provenance = build_provenance(api_name, round_id)
    slot = modality
    image = str((api_args or {}).get("image") or "")
    if modality == "rgb" and image:
        time1 = str(registry.get("time1_image") or "")
        time2 = str(registry.get("time2_image") or "")
        if time1 and os.path.realpath(image) == os.path.realpath(time1):
            slot = "time1"
        elif time2 and os.path.realpath(image) == os.path.realpath(time2):
            slot = "time2"
    if slot == "time1":
        registry[TIME1_DETECTIONS_KEY] = detections
        registry[TIME1_PROVENANCE_KEY] = provenance
    elif slot == "time2":
        registry[TIME2_DETECTIONS_KEY] = detections
        registry[TIME2_PROVENANCE_KEY] = provenance
    elif modality == "rgb":
        registry[RGB_DETECTIONS_KEY] = detections
        registry[RGB_PROVENANCE_KEY] = provenance
    else:
        registry[SAR_DETECTIONS_KEY] = detections
        registry[SAR_PROVENANCE_KEY] = provenance
    return registry


#: Detection tools whose responses can be narrowed to the requested class.
_TARGET_FILTER_TOOLS = ("ObjectDetection", "SmallObjectDetection", "DenoDetV2")
_LABEL_ALIASES = {
    "airplane": "aircraft", "plane": "aircraft",
    "car": "vehicle", "small vehicle": "vehicle", "large vehicle": "vehicle",
    "storage tank": "tank", "oil tank": "tank",
    "boat": "ship", "vessel": "ship", "harbour": "harbor",
}
_LINE_RE = re.compile(r"^\(([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+)\),\s*(.*?),\s*score\s*([-+]?\d*\.?\d+)$")


def _canonical_label(value: Any) -> str:
    label = str(value or "").strip().lower()
    return _LABEL_ALIASES.get(label, label)


def _requested_targets(api_name: str, api_args: Any) -> set:
    if not isinstance(api_args, dict):
        return set()
    raw = api_args.get("classes") if api_name == "DenoDetV2" else api_args.get("text")
    if not raw:
        return set()
    parts = [part for part in re.split(r"[.,\s]+", str(raw)) if part]
    return {_canonical_label(part) for part in parts}


def parse_text_detections(text: Any) -> List[Dict[str, Any]]:
    """Parse the workers' textual detection lines into structured detections.

    Detection workers return ``"(x1, y1, x2, y2), label, score 0.91"`` lines and
    no ``structured`` payload.  The evidence pipeline needs ``bbox``/``label``/
    ``score`` dicts, so this is the one place that turns the shipped text format
    into data; without it a live detection can never be registered and
    ``CrossModalInterpret`` is always rejected as ``invalid_dependency``.
    """
    detections: List[Dict[str, Any]] = []
    for line in str(text or "").splitlines():
        match = _LINE_RE.match(line.strip())
        if not match:
            continue
        detections.append({
            "bbox": [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))],
            "label": match.group(5).strip(),
            "score": float(match.group(6)),
        })
    return detections


_DETECTION_TOOLS = ("ObjectDetection", "SmallObjectDetection", "TextToBbox", "DenoDetV2")


def enrich_detection_response(tool_resp: Any, api_name: str) -> Any:
    """Give a text-only detection response a structured, index-bearing payload.

    Workers answer with ``"(x1, y1, x2, y2), label, score s"`` lines only.  The
    evidence pipeline (and workbook replay's positional binding) needs
    ``bbox``/``label``/``score``/``index`` dicts.  Enrichment must happen
    **before** :func:`filter_detection_response`, otherwise the class filter
    renumbers the boxes and every replay binding misses.
    """
    if api_name not in _DETECTION_TOOLS or not isinstance(tool_resp, dict):
        return tool_resp
    if tool_resp.get("error_code") != 0:
        return tool_resp
    if isinstance(tool_resp.get("detections"), list) and tool_resp["detections"]:
        return tool_resp
    parsed = parse_text_detections(tool_resp.get("text"))
    if not parsed:
        return tool_resp
    enriched = dict(tool_resp)
    enriched["detections"] = parsed
    structured = dict(enriched["structured"]) if isinstance(enriched.get("structured"), dict) else {}
    structured.setdefault("detections", parsed)
    enriched["structured"] = structured
    return enriched


def filter_detection_response(tool_resp: Dict[str, Any], api_name: str, api_args: Any) -> Dict[str, Any]:
    """Keep only boxes of the class the tool was asked for.

    The workers already filter by ``text`` / ``classes``; this is the service-side
    guarantee so a response can never smuggle other classes into the observation
    or into the registry used by CrossModalInterpret.
    """
    if api_name not in _TARGET_FILTER_TOOLS or not isinstance(tool_resp, dict):
        return tool_resp
    # 文本型检测响应先补齐成结构化检测；放在过滤之前，避免过滤重排这里之后的顺序。
    tool_resp = enrich_detection_response(tool_resp, api_name)
    targets = _requested_targets(api_name, api_args)
    if not targets:
        return tool_resp

    def line_of(det: Dict[str, Any]) -> str:
        bbox = det.get("bbox") or []
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
        return f"({x1}, {y1}, {x2}, {y2}), {det.get('label', 'object')}, score {float(det.get('score', 0.0)):.2f}"

    detections = tool_resp.get("detections")
    if isinstance(detections, list):
        kept = [d for d in detections if isinstance(d, dict) and _canonical_label(d.get("label") or d.get("class_name")) in targets]
        if len(kept) == len(detections):
            return tool_resp
        out = dict(tool_resp)
        out["detections"] = kept
        out["text"] = "\n".join(line_of(d) for d in kept if isinstance(d.get("bbox"), list) and len(d["bbox"]) >= 4) or "no detections found after filtering"
        if "count" in out:
            out["count"] = len(kept)
        if "detection_count" in out:
            out["detection_count"] = len(kept)
        structured = out.get("structured")
        if isinstance(structured, dict) and isinstance(structured.get("detections"), list):
            structured = dict(structured)
            structured["detections"] = kept
            out["structured"] = structured
        return out

    text = tool_resp.get("text")
    if isinstance(text, str) and text.strip():
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        lines = []
        for line in raw_lines:
            match = _LINE_RE.match(line)
            if match and _canonical_label(match.group(5)) in targets:
                lines.append(line)
        if lines and len(lines) < len(raw_lines):
            out = dict(tool_resp)
            out["text"] = "\n".join(lines)
            return out
    return tool_resp


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
    source_mode = args.get("source_mode", "online")
    if source_mode == "workbook_replay":
        dataset = args.get("dataset_id")
        family = {"task4": "match", "task5": "time"}.get(dataset) if isinstance(dataset, str) else None
        if family is None or not isinstance(args.get("pair_id"), str) or not args["pair_id"].strip():
            return args, "invalid_replay_request"
        if args.get("family", family) != family:
            return args, "invalid_replay_request"
        # Detection is live in this mode too: the workbook only carries the
        # target's attributes, so detector payloads plus provenance are required
        # exactly as online and fall through to the shared gate below.
        if REPLAY_FORBIDDEN.intersection(args):
            return args, "invalid_replay_request"
        args["family"] = family
        image_keys = ("optical_image", "sar_image") if family == "match" else ("time1_image", "time2_image")
        argument_keys = ("rgb_image", "sar_image") if family == "match" else image_keys
        for argument, key in zip(argument_keys, image_keys):
            if registry.get(key) and not args.get(argument):
                args[argument] = registry[key]
        if family == "time":
            replay_provenance = dict(args.get("evidence_provenance") or {})
            if not args.get("rgb_detections") and registry.get(TIME1_DETECTIONS_KEY):
                args["rgb_detections"] = registry[TIME1_DETECTIONS_KEY]
                replay_provenance["time1"] = registry.get(TIME1_PROVENANCE_KEY)
            if not args.get("sar_detections") and registry.get(TIME2_DETECTIONS_KEY):
                args["sar_detections"] = registry[TIME2_DETECTIONS_KEY]
                replay_provenance["time2"] = registry.get(TIME2_PROVENANCE_KEY)
            if replay_provenance:
                args["evidence_provenance"] = replay_provenance
    elif source_mode != "online":
        return args, "invalid_replay_request"
    # A detector payload may arrive under its documented alias.  Normalise it here
    # so the gate, the registry injection below and the worker all agree on one
    # key, and the registry cannot silently shadow a caller-supplied alias.
    for argument, alias in (("rgb_detections", "rgb_detection_set"), ("sar_detections", "sar_detection_set")):
        if not args.get(argument) and args.get(alias):
            args[argument] = args.pop(alias)
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
    if error == "invalid_replay_request":
        return "Replay requires task4/task5, matching family and pair_id; pass live detector payloads, never workbook or config paths."
    if error == "invalid_dependency":
        return EMPTY_DETECTION_HINT
    if error == "example_leakage_detected":
        return EXAMPLE_LEAKAGE_HINT
    return None


__all__ = [
    "EMPTY_DETECTION_HINT",
    "EXAMPLE_BBOX_PATTERNS",
    "EXAMPLE_LEAKAGE_HINT",
    "REPLAY_FORBIDDEN",
    "RGB_DETECTION_TOOLS",
    "SAR_DETECTION_TOOLS",
    "TIME1_DETECTIONS_KEY",
    "TIME1_PROVENANCE_KEY",
    "TIME2_DETECTIONS_KEY",
    "TIME2_PROVENANCE_KEY",
    "build_provenance",
    "contains_example_bbox",
    "dependency_hint",
    "detection_modality",
    "enrich_detection_response",
    "extract_detections",
    "parse_text_detections",
    "filter_detection_response",
    "prepare_interpret_args",
    "register_detections",
]
