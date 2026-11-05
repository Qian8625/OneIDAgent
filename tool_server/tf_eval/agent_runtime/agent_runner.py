from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tool_server.tf_eval.agent_runtime.run_config import AgentRunConfig, RunMode
from tool_server.tf_eval.agent_runtime.structured_summary import (
    build_structured_observation,
    get_structured_result_for_registry,
)
from tool_server.tf_eval.agent_runtime.trace_schema import AgentTrace, create_agent_trace
from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT
from tool_server.tool_workers.skills.catalog import build_selected_skill_catalog
from tool_server.tool_workers.skills.registry import SKILL_INPUT_BINDINGS, SKILL_REGISTRY
from tool_server.tool_workers.skills.router import route_skills
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.tool_workers.cross_modal_evidence import (
    EXAMPLE_BBOX_PATTERNS,
    dependency_hint,
    extract_detections as _extract_detections_from_structured,
    prepare_interpret_args as prepare_cross_modal_interpret_args,
    register_detections,
)
from tool_server.utils.paths import PROJECT_ROOT


TOOL_ARG_BINDINGS = {
    "DrawBox": {"image": "primary_image"},
    "AddText": {"image": "primary_image"},
    "TextToBbox": {"image": "primary_image"},
    "CountGivenObject": {"image": "primary_image"},
    "ImageDescription": {"image": "primary_image"},
    "RegionAttributeDescription": {"image": "primary_image"},
    "SegmentObjectPixels": {"image": "primary_image"},
    "ObjectDetection": {"image": ["optical_image", "primary_image"]},
    "SmallObjectDetection": {"image": "primary_image"},
    "GetBboxFromGeotiff": {"geotiff": "primary_image"},
    "DisplayOnGeotiff": {"geotiff": "primary_image"},
    "ChangeDetection": {"pre_image": "time1_image", "post_image": "time2_image"},
    "DenoDetV2": {"image": ["sar_image", "primary_image"]},
    "CrossModalInterpret": {"georef_image": "optical_image"},
}

SKILL_MODALITY_BINDINGS = {
    "CrossModalMatchSkill": {"rgb_image": "optical_image", "sar_image": "sar_image"},
    "CrossModalTaskSkill": {"rgb_image": "optical_image", "sar_image": "sar_image"},
}

DETERMINISTIC_FAILURE_PATTERNS = (
    "image not found",
    "not found:",
    "missing required",
    "unsupported task_type",
    "unsupported measure type",
    "invalid_dependency",
    "unresolved references",
    "geopackage not found",
    "no crs defined",
)

NO_DETECTION_PATTERNS = (
    "no detections found",
    "no detections found after filtering",
    "no detections found after thresholding",
)

CROSS_MODAL_TASK_ALIASES = {
    "crosscheck_exists": "optical_sar_verify",
    "crosscheck": "optical_sar_verify",
    "exists": "optical_sar_verify",
    "exist": "optical_sar_verify",
    "joint_scene": "optical_sar_consistency",
    "select_largest": "cross_modal_select_largest",
    "largest": "cross_modal_select_largest",
    "select_highest_score": "cross_modal_select_highest_score",
    "highest_score": "cross_modal_select_highest_score",
    "highest_confidence": "cross_modal_select_highest_score",
    "distance_largest_smallest": "cross_modal_distance_largest_smallest",
    "largest_smallest_distance": "cross_modal_distance_largest_smallest",
    "select_smallest": "cross_modal_select_largest",
    "select_largest_smallest": "cross_modal_distance_largest_smallest",
}


TARGET_HINTS = (
    "aircraft", "airplane", "plane", "ship", "vehicle", "car", "tank", "bridge", "harbor",
    "baseball diamond", "tennis court", "basketball court", "swimming pool", "stadium",
    "ground track field", "groundtrackfield", "airport", "dam", "overpass", "train station",
    "expressway service area", "golf field", "building", "house", "excavator",
)

MEASURE_SELECTOR_WORDS = {
    "largest",
    "smallest",
    "nearest",
    "farthest",
    "leftmost",
    "rightmost",
    "topmost",
    "bottommost",
    "first",
    "second",
}

TOOL_DEFAULT_ARGUMENTS = {
    "AddText": {"color": "green"},
}

GPKG_REQUIRED_TOOLS = {
    "AddPoisLayer",
    "ComputeDistance",
    "DisplayOnGeotiff",
}

INPUT_DESCRIPTIONS = {
    "primary_image": "Single-scene image for detection, interpretation, or GeoTIFF overlay tasks.",
    "optical_image": "Optical or RGB image for cross-modal validation with SAR.",
    "sar_image": "SAR grayscale image for cross-modal validation with optical.",
    "time1_image": "Earlier timestamp image, often used as pre_image.",
    "time2_image": "Later timestamp image, often used as post_image.",
}

REGISTRY_VALUE_ALIASES = {
    "primary_image": "primary_image",
    "image": "primary_image",
    "primary": "primary_image",
    "primary_1": "primary_image",
    "img": "primary_image",
    "image_0": "primary_image",
    "rgb_image": "optical_image",
    "optical_image": "optical_image",
    "optical_1": "optical_image",
    "sar_image": "sar_image",
    "sar_1": "sar_image",
    "radar_image": "sar_image",
    "time1_image": "time1_image",
    "time2_image": "time2_image",
    "latest_image": "latest_image",
    "latest_output": "latest_output",
    "latest_structured": "latest_structured",
    "latest_vector": "latest_vector",
    "latest_gpkg": "latest_vector",
    "gpkg": "latest_vector",
    "gpkg_1": "latest_vector",
    "primary_gpkg": "latest_vector",
    "area_gpkg": "latest_vector",
    "area_boundary.gpkg": "latest_vector",
    "optical_bbox_gpkg": "latest_vector",
    "tif_1": "primary_image",
    "geotiff_1": "primary_image",
}


class LocalLLM:
    def __init__(self, pretrained_path: str, system_prompt: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path,
            use_fast=True,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        self.system_prompt = system_prompt
        self.max_new_tokens = 256

    def _build_system_prompt(self, extra_system_prompt: str = "") -> str:
        if extra_system_prompt and extra_system_prompt.strip():
            return self.system_prompt + "\n\n" + extra_system_prompt.strip()
        return self.system_prompt

    def generate(self, conversation: List[Dict[str, str]], extra_system_prompt: str = "") -> str:
        import torch

        conversation = list(conversation)
        system_prompt = self._build_system_prompt(extra_system_prompt)
        if not conversation or conversation[0].get("role") != "system":
            conversation = [{"role": "system", "content": system_prompt}] + conversation
        else:
            conversation[0] = {"role": "system", "content": system_prompt}

        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


def build_no_skill_prompt(prompt: str) -> str:
    lines = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("- SkillExecutor:"):
            continue
        if "SkillExecutor" in stripped or "Selected Skills" in stripped:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def try_parse_agent_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
    except Exception:
        pass
    return {}


def extract_actions(text: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    data = try_parse_agent_json(text)
    actions = data.get("actions")
    if isinstance(actions, dict):
        return [actions], None
    if isinstance(actions, list):
        return actions, None

    try:
        actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
        actions_match = re.search(actions_pattern, text)
        if not actions_match:
            return None, "No action found."
        parsed = json.loads(actions_match.group(1))
        return parsed if isinstance(parsed, list) else None, None
    except json.JSONDecodeError:
        return None, "Invalid JSON in actions."
    except Exception:
        return None, "Invalid action format."


def extract_thought_and_actions(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    data = try_parse_agent_json(text)
    thought = data.get("thought", "") if isinstance(data, dict) else ""
    actions = data.get("actions") if isinstance(data, dict) else None
    if actions is None:
        actions, _ = extract_actions(text)
    if isinstance(actions, dict):
        actions = [actions]
    if not isinstance(actions, list):
        actions = []
    return thought, actions


def resolve_existing_file(path: Any) -> Optional[str]:
    if not path or (isinstance(path, str) and not path.strip()):
        return None
    try:
        candidate = Path(str(path)).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, PROJECT_ROOT / candidate]
        for item in candidates:
            if item.is_file():
                return str(item.resolve())
    except Exception:
        pass
    return None


def _registry_alias_value(value: str, registry: Dict[str, Any]) -> Any:
    token = value.strip()
    if token.startswith("$"):
        token = token[1:]
    if token in {"img_1", "image_1"}:
        return registry.get("optical_image") or registry.get("primary_image") or value
    if token in {"img_2", "image_2"}:
        return registry.get("sar_image") or registry.get("time2_image") or value
    if token in registry and registry.get(token) not in [None, ""]:
        return registry[token]
    registry_key = REGISTRY_VALUE_ALIASES.get(token)
    if registry_key and registry.get(registry_key) not in [None, ""]:
        return registry[registry_key]
    return value


def _select_registry_value(registry: Dict[str, Any], registry_key: Any) -> Tuple[Any, Optional[str]]:
    keys = registry_key if isinstance(registry_key, (list, tuple)) else [registry_key]
    for key in keys:
        if registry.get(key) not in [None, ""]:
            return registry[key], str(key)
    return None, None


def _resolve_ref_value(value: Any, registry: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _registry_alias_value(value, registry)
    if isinstance(value, list):
        return [_resolve_ref_value(item, registry) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_ref_value(item, registry) for key, item in value.items()}
    return value


def resolve_argument_references(api_args: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    return _resolve_ref_value(dict(api_args or {}), registry)


def find_unresolved_references(api_args: Dict[str, Any], registry: Optional[Dict[str, Any]] = None) -> List[str]:
    unresolved: List[str] = []
    registry = registry or {}
    strict_tokens = {"gpkg_1", "latest_gpkg", "area_boundary.gpkg", "tif_1", "geotiff_1"}

    def walk(value: Any) -> None:
        if isinstance(value, str) and value.startswith("$"):
            unresolved.append(value)
        elif isinstance(value, str) and value in strict_tokens:
            registry_key = REGISTRY_VALUE_ALIASES.get(value)
            if registry_key and registry.get(registry_key) in [None, ""]:
                unresolved.append(value)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    walk(api_args)
    return unresolved


def inject_runtime_arguments(
    api_name: str,
    api_args: Dict[str, Any],
    registry: Dict[str, Any],
    current_gpkg: Optional[str],
) -> Dict[str, Any]:
    merged = dict(api_args or {})
    for tool_arg, registry_key in TOOL_ARG_BINDINGS.get(api_name, {}).items():
        selected, _ = _select_registry_value(registry, registry_key)
        if selected not in [None, ""] and (merged.get(tool_arg) in [None, ""] or api_name == "DenoDetV2" and tool_arg == "image" and registry.get("sar_image")):
            merged[tool_arg] = selected
    if api_name in GPKG_REQUIRED_TOOLS and current_gpkg and merged.get("gpkg") in [None, ""]:
        merged["gpkg"] = current_gpkg
    if api_name in GPKG_REQUIRED_TOOLS and current_gpkg and merged.get("gpkg") in {"gpkg", "gpkg_1", "latest_gpkg", "area_boundary.gpkg"}:
        merged["gpkg"] = current_gpkg
    for key, value in TOOL_DEFAULT_ARGUMENTS.get(api_name, {}).items():
        if merged.get(key) in [None, ""] or key not in merged:
            merged[key] = value
    return resolve_argument_references(merged, registry)


def inject_runtime_arguments_with_overrides(
    api_name: str,
    api_args: Dict[str, Any],
    registry: Dict[str, Any],
    current_gpkg: Optional[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    before = dict(api_args or {})
    injected = inject_runtime_arguments(api_name, api_args, registry, current_gpkg)
    overrides: List[Dict[str, Any]] = []
    for tool_arg, registry_key in TOOL_ARG_BINDINGS.get(api_name, {}).items():
        selected, selected_key = _select_registry_value(registry, registry_key)
        if selected_key and before.get(tool_arg) not in [None, ""] and injected.get(tool_arg) == selected and before.get(tool_arg) != selected:
            overrides.append({
                "api_name": api_name,
                "argument": tool_arg,
                "original_value": before.get(tool_arg),
                "corrected_value": selected,
                "registry_key": selected_key,
                "reason": "modality_binding_override",
            })
    return injected, overrides


def inject_skill_runtime_arguments(api_args: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(api_args or {})
    skill_name = merged.get("skill_name")
    if not skill_name:
        return merged
    for arg_name, registry_key in SKILL_INPUT_BINDINGS.get(skill_name, {}).items():
        if merged.get(arg_name) in [None, ""] and registry.get(registry_key):
            merged[arg_name] = registry[registry_key]
    for arg_name, registry_key in SKILL_MODALITY_BINDINGS.get(str(skill_name), {}).items():
        if registry.get(registry_key) and merged.get(arg_name) != registry[registry_key]:
            merged[arg_name] = registry[registry_key]
    if "input_registry" not in merged:
        merged["input_registry"] = registry
    return resolve_argument_references(merged, registry)


def _normalize_for_signature(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_for_signature(item) for key, item in sorted(value.items()) if key != "input_registry"}
    if isinstance(value, list):
        return [_normalize_for_signature(item) for item in value]
    return value


def call_signature(api_name: str, api_args: Dict[str, Any]) -> str:
    return json.dumps({"api_name": api_name, "arguments": _normalize_for_signature(api_args or {})}, ensure_ascii=False, sort_keys=True)


def is_deterministic_failure(tool_resp: Dict[str, Any]) -> bool:
    text = str(tool_resp.get("text") or tool_resp.get("status") or tool_resp.get("failure_reason") or "").lower()
    return any(pattern in text for pattern in DETERMINISTIC_FAILURE_PATTERNS)


def is_no_detection_response(tool_resp: Dict[str, Any]) -> bool:
    text = str(tool_resp.get("text") or "").lower()
    return any(pattern in text for pattern in NO_DETECTION_PATTERNS)


def normalize_cross_modal_task_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return text
    normalized = "_".join(text.lower().replace("-", "_").split())
    return CROSS_MODAL_TASK_ALIASES.get(normalized, normalized)


def normalize_skill_action_args(api_args: Dict[str, Any]) -> Dict[str, Any]:
    args = dict(api_args or {})
    if args.get("skill_name") != "CrossModalTaskSkill":
        return args
    raw_task = args.get("task_type") or args.get("task") or args.get("mode") or args.get("intent")
    if raw_task not in [None, ""]:
        args["task_type"] = normalize_cross_modal_task_type(raw_task)
    return args


# Detection evidence pipeline (extract / register / prepare) lives in
# tool_server/tool_workers/cross_modal_evidence.py: exactly one implementation.


def _extract_gsd(question: str) -> Optional[float]:
    match = re.search(r"\bGSD(?:\s+(?:value\s+)?(?:of|is)|\s*=)?\s*([0-9]+(?:\.[0-9]+)?)", question, re.I)
    if not match:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*m\s*/\s*p(?:ixel|x)", question, re.I)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None


def _infer_target(question: str) -> str:
    text = str(question or "").lower()
    for target in TARGET_HINTS:
        if target in text:
            return target
    return "object"


def _clean_measure_target_phrase(phrase: str) -> Tuple[str, Optional[str]]:
    text = " ".join(str(phrase or "").strip(" .?;:\n\t").split())
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    selector = None
    for word in MEASURE_SELECTOR_WORDS:
        pattern = rf"^{word}\s+"
        if re.search(pattern, text, flags=re.I):
            selector = word
            text = re.sub(pattern, "", text, count=1, flags=re.I).strip()
            break
    return text or "object", selector


def _infer_measure_request(question: str) -> Dict[str, Any]:
    text = str(question or "")
    lowered = text.lower()
    request: Dict[str, Any] = {}

    if "pixel dimension" in lowered or "pixel width" in lowered or "length-to-width" in lowered or "length to width" in lowered:
        request["measure_type"] = "pixel_dimensions"
        request["unit"] = "pixel"
    elif "distance" in lowered or re.search(r"\bbetween\b", lowered):
        request["measure_type"] = "pair_distance"
    elif "centroid" in lowered or "center" in lowered:
        request["measure_type"] = "centroid"
    elif "width" in lowered or "height" in lowered or "length" in lowered:
        request["measure_type"] = "bbox_size"
    else:
        request["measure_type"] = "area"

    ratio_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s+length\s*-?\s*to\s*-?\s*width", text, re.I)
    if not ratio_match:
        ratio_match = re.search(r"length\s*-?\s*to\s*-?\s*width\s+ratio\s+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if ratio_match:
        try:
            numerator = float(ratio_match.group(1))
            denominator = float(ratio_match.group(2))
            if denominator:
                request["length_width_ratio"] = numerator / denominator
        except Exception:
            pass

    between_match = re.search(
        r"\bbetween\s+(?:the\s+|a\s+|an\s+)?(.+?)\s+and\s+(?:the\s+|a\s+|an\s+)?(.+?)(?=,?\s*(?:considering|using|with)\b|[?.]|$)",
        text,
        re.I,
    )
    if between_match:
        target_a, selector_a = _clean_measure_target_phrase(between_match.group(1))
        target_b, selector_b = _clean_measure_target_phrase(between_match.group(2))
        request["target"] = target_a
        request["reference_target"] = target_b
        if selector_a:
            request["select_a"] = selector_a
        if selector_b:
            request["select_b"] = selector_b
    else:
        target = _infer_target(text)
        request["target"] = target
        for word in MEASURE_SELECTOR_WORDS:
            if re.search(rf"\b{word}\s+{re.escape(target)}\b", lowered):
                request["select_a"] = word
                break

    return request


def _infer_cross_modal_task_type(question: str) -> str:
    text = str(question or "").lower()
    if "highest-confidence" in text or "highest confidence" in text or "highest-score" in text:
        return "cross_modal_select_highest_score"
    if "largest" in text and "smallest" in text and "distance" in text:
        return "cross_modal_distance_largest_smallest"
    if "largest" in text:
        return "cross_modal_select_largest"
    if "jointly detect" in text or "deduplicated count" in text:
        return "optical_sar_consistency"
    if "whether" in text or "exist" in text or "final conclusion" in text:
        return "optical_sar_verify"
    return "optical_sar_verify"


def build_oracle_skill_args(gold_skill: str, question: str, registry: Dict[str, Any], eval_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    eval_meta = eval_meta or {}
    args: Dict[str, Any] = {"skill_name": gold_skill, "input_registry": registry, "question": question, "query": question, "instruction": question}
    target = str(eval_meta.get("target") or _infer_target(question))
    if gold_skill in {"ObjectMeasureSkill", "ObjectDetectSelectSkill", "CrossModalMatchSkill", "CrossModalTaskSkill"}:
        args["target"] = target
    if gold_skill == "ObjectMeasureSkill":
        measure_request = _infer_measure_request(question)
        target = str(eval_meta.get("target") or measure_request.get("target") or target)
        args["target"] = target
        args["image"] = registry.get("primary_image")
        gsd = eval_meta.get("gsd_m_per_pixel") or _extract_gsd(question)
        if gsd is not None:
            args["gsd_m_per_pixel"] = gsd
        args["measure_type"] = str(eval_meta.get("measure_type") or measure_request.get("measure_type") or "area")
        for key in ["reference_target", "select_a", "select_b", "length_width_ratio"]:
            if measure_request.get(key) not in [None, ""]:
                args[key] = measure_request[key]
        if gsd is not None and args["measure_type"] in {"area", "distance", "pair_distance"}:
            args["unit"] = "meter"
        elif measure_request.get("unit"):
            args["unit"] = measure_request["unit"]
    elif gold_skill == "ObjectDetectSelectSkill":
        args["image"] = registry.get("primary_image")
        lowered = question.lower()
        if "largest" in lowered:
            args["selection_mode"] = "largest"
        elif "smallest" in lowered:
            args["selection_mode"] = "smallest"
        else:
            args["selection_mode"] = "all"
    elif gold_skill == "CrossModalMatchSkill":
        args["rgb_image"] = registry.get("optical_image") or registry.get("primary_image")
        args["sar_image"] = registry.get("sar_image")
    elif gold_skill == "CrossModalTaskSkill":
        args["rgb_image"] = registry.get("optical_image") or registry.get("primary_image")
        args["sar_image"] = registry.get("sar_image")
        args["task_type"] = str(eval_meta.get("task_type") or _infer_cross_modal_task_type(question))
    return {key: value for key, value in args.items() if value not in [None, ""]}


def normalize_tool_response(tool_resp: Any, api_name: str) -> Dict[str, Any]:
    if isinstance(tool_resp, dict):
        return tool_resp
    if tool_resp is None:
        return {"error_code": -1, "text": f"{api_name} returned empty response."}
    if isinstance(tool_resp, str):
        return {"error_code": -1, "text": f"{api_name} returned string response: {tool_resp}"}
    return {"error_code": -1, "text": f"{api_name} returned non-dict response: {str(tool_resp)[:200]}"}


def coerce_business_negative_response(tool_resp: Dict[str, Any], api_name: str) -> Dict[str, Any]:
    if api_name not in {"ObjectDetection", "DenoDetV2", "TextToBbox", "SmallObjectDetection"}:
        return tool_resp
    if not isinstance(tool_resp, dict) or tool_resp.get("error_code") == 0 or not is_no_detection_response(tool_resp):
        return tool_resp
    coerced = dict(tool_resp)
    coerced["original_error_code"] = coerced.get("error_code")
    coerced["error_code"] = 0
    coerced.setdefault("detections", [])
    coerced.setdefault("structured", {"detections": [], "status": "no_detections"})
    coerced["status"] = "no_detections"
    return coerced


def register_artifact(
    registry: Dict[str, Any],
    artifact_id: str,
    path: str,
    artifact_type: str,
    tool_name: str,
) -> Dict[str, Any]:
    if not path:
        return registry
    registry = dict(registry)
    resolved = resolve_existing_file(path) or path
    registry[artifact_id] = resolved
    artifacts = list(registry.get("artifacts", []))
    if not any(item.get("id") == artifact_id for item in artifacts):
        artifacts.append({"id": artifact_id, "path": resolved, "type": artifact_type, "tool": tool_name})
    registry["artifacts"] = artifacts
    if artifact_type in {"image", "raster"}:
        registry["latest_image"] = resolved
        registry["latest_output"] = resolved
    elif artifact_type == "vector":
        registry["latest_vector"] = resolved
        registry["latest_output"] = resolved
    else:
        registry["latest_output"] = resolved
    return registry


def detect_output_visual(tool_resp: Dict[str, Any]) -> Optional[str]:
    if not isinstance(tool_resp, dict):
        return None
    for key in ["image", "output_path", "out_file", "path", "preview", "png"]:
        value = tool_resp.get(key)
        if isinstance(value, str):
            found = resolve_existing_file(value)
            if found and found.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                return found
    text = tool_resp.get("text", "")
    if isinstance(text, str):
        for match in reversed(re.findall(r"([A-Za-z0-9_./\\-]+\.(?:png|jpg|jpeg|tif|tiff))", text, re.I)):
            found = resolve_existing_file(match)
            if found:
                return found
    return None


def update_registry_with_tool_response(
    registry: Dict[str, Any],
    api_name: str,
    tool_resp: Dict[str, Any],
    round_id: int,
) -> Dict[str, Any]:
    if not isinstance(tool_resp, dict) or tool_resp.get("error_code") != 0:
        registry = dict(registry)
        structured = tool_resp.get("structured") if isinstance(tool_resp, dict) else None
        structured_result = structured if isinstance(structured, dict) and structured else None
        if structured_result:
            registry["latest_error_structured"] = structured_result
        return registry
    registry = dict(registry)

    structured_result = get_structured_result_for_registry(tool_resp)
    if structured_result:
        structured_key = f"round_{round_id}_{api_name.lower()}_structured"
        provenance = {
            "source_tool": api_name,
            "source_round": round_id,
            "source_success": True,
            "source_response_id": structured_key,
        }
        registry[structured_key] = structured_result
        registry["latest_structured"] = structured_result
        registry["latest_successful_structured"] = structured_result
        registry["latest_structured_provenance"] = provenance
        registry = register_detections(registry, api_name, structured_result, round_id)

    artifacts = tool_resp.get("artifacts")
    if isinstance(artifacts, list):
        for idx, item in enumerate(artifacts, start=1):
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if not path:
                continue
            artifact_id = item.get("id") or f"artifact_{idx}"
            registry = register_artifact(
                registry,
                artifact_id=f"round_{round_id}_{api_name.lower()}_{artifact_id}",
                path=path,
                artifact_type=item.get("type") or "other",
                tool_name=api_name,
            )
    if api_name == "SkillExecutor":
        return registry
    visual = detect_output_visual(tool_resp)
    if visual:
        registry = register_artifact(
            registry,
            artifact_id=f"round_{round_id}_{api_name.lower()}_image",
            path=visual,
            artifact_type="image",
            tool_name=api_name,
        )
    gpkg_path = tool_resp.get("gpkg")
    if isinstance(gpkg_path, str):
        resolved_gpkg = resolve_existing_file(gpkg_path)
        if resolved_gpkg:
            registry = register_artifact(
                registry,
                artifact_id=f"round_{round_id}_{api_name.lower()}_vector",
                path=resolved_gpkg,
                artifact_type="vector",
                tool_name=api_name,
            )
    return registry


def build_initial_user_message(user_question: str, registry: Dict[str, Any]) -> str:
    lines = [user_question.strip()]
    input_lines = [
        f"- {key}: {description}"
        for key, description in INPUT_DESCRIPTIONS.items()
        if registry.get(key)
    ]
    if input_lines:
        lines.extend([
            "",
            "Available uploaded inputs:",
            *input_lines,
            "Use the most appropriate tool and bind these uploaded inputs to the correct tool arguments.",
        ])
    lines.extend([
        "",
        "Reference rules for tool arguments:",
        "- Uploaded inputs can be referenced explicitly with $primary_image, $optical_image, $sar_image, $time1_image, $time2_image if available.",
        "- Intermediate outputs from previous tool calls will be exposed as reusable artifact ids such as $latest_image or $round_2_sartorgb_image.",
        "- If a later tool should use an intermediate result, reference it explicitly in the action arguments.",
    ])
    return "\n".join(lines)


def build_artifact_summary(registry: Dict[str, Any]) -> str:
    lines: List[str] = []
    uploaded = [
        f"- ${key} | uploaded input | path={registry[key]}"
        for key in INPUT_DESCRIPTIONS
        if registry.get(key)
    ]
    if uploaded:
        lines.append("Uploaded inputs:")
        lines.extend(uploaded)
    artifacts = registry.get("artifacts", [])
    if artifacts:
        lines.append("")
        lines.append("Reusable artifacts:")
        for item in artifacts[-10:]:
            lines.append(f"- ${item['id']} | type={item['type']} | tool={item['tool']} | path={item['path']}")
    aliases = []
    if registry.get("latest_image"):
        aliases.append(f"- $latest_image | path={registry['latest_image']}")
    if registry.get("latest_vector"):
        aliases.append(f"- $latest_vector | path={registry['latest_vector']}")
    if registry.get("latest_output"):
        aliases.append(f"- $latest_output | path={registry['latest_output']}")
    if registry.get("latest_structured"):
        aliases.append("- $latest_structured | structured result from the latest successful tool call")
    if aliases:
        lines.append("")
        lines.append("Convenience aliases:")
        lines.extend(aliases)
    return "\n".join(lines) if lines else "No uploaded inputs or reusable artifacts are currently available."


class AgentRunner:
    def __init__(
        self,
        mode: RunMode | str = RunMode.AUTO_SKILL,
        config: Optional[AgentRunConfig] = None,
        llm: Any = None,
        tool_manager: Any = None,
    ):
        self.config = config or AgentRunConfig(mode=mode)
        self.mode = RunMode.normalize(self.config.mode)
        self.llm = llm
        self.tool_manager = tool_manager

    def run_sample(
        self,
        question: str,
        input_registry: Optional[Dict[str, Any]] = None,
        oracle_skill: Optional[Any] = None,
        max_rounds: Optional[int] = None,
        sample_id: str = "",
        eval_meta: Optional[Dict[str, Any]] = None,
    ) -> AgentTrace:
        started_at = time.perf_counter()
        registry = dict(input_registry or {})
        sample_id = sample_id or str(registry.get("sample_id") or "")
        trace = create_agent_trace(sample_id=sample_id, mode=self.mode.value, question=question or "")

        def finish() -> AgentTrace:
            trace["total_tool_calls"] = trace["raw_tool_calls"] + trace["skill_tool_calls"]
            trace["wall_time"] = round(time.perf_counter() - started_at, 6)
            trace["llm_time"] = round(trace["llm_time"], 6)
            trace["tool_time"] = round(trace["tool_time"], 6)
            return trace

        if not question or not question.strip():
            self._add_error(trace, "invalid_question")
            return finish()

        max_rounds = int(max_rounds or self.config.max_rounds)
        selected_skills, router_trace = self._select_skills(question, registry, oracle_skill, trace)
        selected_skill_prompt = build_selected_skill_catalog(selected_skills)
        trace["selected_skills"] = selected_skills
        trace["router_trace"] = router_trace

        conversation = [{"role": "user", "content": build_initial_user_message(question, registry)}]
        current_gpkg = None
        failed_call_history: Dict[str, Dict[str, Any]] = {}

        if self.mode in {RunMode.ORACLE_SKILL, RunMode.ORACLE_FORCED_SKILL} and oracle_skill:
            forced_trace = self._run_forced_oracle_skill(
                question=question,
                registry=registry,
                oracle_skill=oracle_skill,
                eval_meta=eval_meta or {},
                trace=trace,
                started_at=started_at,
            )
            return forced_trace

        if self._should_force_auto_skill(selected_skills, eval_meta or {}):
            forced_trace = self._run_forced_selected_skill(
                question=question,
                registry=registry,
                selected_skill=selected_skills[0]["skill_name"],
                eval_meta=eval_meta or {},
                trace=trace,
                started_at=started_at,
            )
            return forced_trace

        for round_id in range(1, max_rounds + 1):
            trace["rounds"] = round_id
            try:
                llm_started = time.perf_counter()
                model_output = self._generate(conversation, selected_skill_prompt)
                trace["llm_time"] += time.perf_counter() - llm_started
            except Exception as exc:
                self._add_error(trace, "llm_error")
                trace["actions"].append({"round": round_id, "error": f"LLM generation failed: {exc}"})
                return finish()

            thought, actions = extract_thought_and_actions(model_output)
            if not actions:
                self._add_error(trace, "no_action")
                trace["actions"].append({
                    "round": round_id,
                    "thought": thought,
                    "model_output": model_output,
                    "actions": [],
                    "executed": False,
                    "error": "no_action",
                })
                conversation.append({"role": "assistant", "content": model_output})
                conversation.append({"role": "user", "content": question})
                continue

            action = actions[0] if isinstance(actions[0], dict) else {}
            api_name = action.get("name")
            raw_args = action.get("arguments", {})
            api_args = raw_args if isinstance(raw_args, dict) else {}
            api_args = resolve_argument_references(api_args, registry)
            raw_action_record = {
                "round": round_id,
                "thought": thought,
                "model_output": model_output,
                "raw_action": action,
            }

            if api_name == "Terminate":
                final_answer = api_args.get("ans", "") or "已完成，结果请查看输出历史或下载文件。"
                trace["final_answer"] = final_answer
                trace["terminated"] = True
                trace["actions"].append({**raw_action_record, "executed": True, "api_name": "Terminate"})
                return finish()

            if not api_name:
                self._add_error(trace, "invalid_action_format")
                trace["actions"].append({**raw_action_record, "executed": False, "error": "missing_tool_name"})
                conversation = self._append_feedback(
                    conversation,
                    model_output,
                    "The action format is invalid. Please output a valid JSON action list.",
                )
                continue

            if self.mode == RunMode.NO_SKILL and (api_name == "SkillExecutor" or api_name in SKILL_REGISTRY):
                self._add_error(trace, "disabled_skill_call")
                trace["actions"].append({**raw_action_record, "api_name": api_name, "executed": False, "error": "skill_disabled"})
                conversation = self._append_feedback(
                    conversation,
                    model_output,
                    f"{api_name} is disabled in no-skill mode. Please select a raw tool from the tool list and continue.",
                )
                continue

            if api_name in SKILL_REGISTRY:
                api_args = dict(api_args or {})
                api_args["skill_name"] = api_name
                api_args["input_registry"] = registry
                api_name = "SkillExecutor"
            if api_name == "SkillExecutor":
                api_args = normalize_skill_action_args(api_args)

            unresolved_refs = find_unresolved_references(api_args, registry)
            if unresolved_refs:
                self._add_error(trace, "unresolved_references")
                trace["actions"].append({
                    **raw_action_record,
                    "api_name": api_name,
                    "arguments": api_args,
                    "executed": False,
                    "error": "unresolved_references",
                    "unresolved_refs": unresolved_refs,
                })
                conversation = self._append_feedback(
                    conversation,
                    model_output,
                    f"The action used unresolved references: {', '.join(unresolved_refs)}. Please use available uploaded inputs or reusable artifacts.",
                )
                continue

            injected_args, modality_overrides = inject_runtime_arguments_with_overrides(api_name, api_args, registry, current_gpkg)
            if modality_overrides:
                trace.setdefault("modality_binding_overrides", []).extend(modality_overrides)

            if api_name == "SkillExecutor":
                injected_args, should_continue = self._prepare_skill_action(
                    injected_args,
                    selected_skills,
                    registry,
                    trace,
                    raw_action_record,
                    conversation,
                    model_output,
                )
                if should_continue is not None:
                    conversation = should_continue
                    continue

            if api_name == "CrossModalInterpret":
                injected_args, dependency_error = prepare_cross_modal_interpret_args(injected_args, registry)
                if dependency_error:
                    self._add_error(trace, dependency_error)
                    trace["actions"].append({
                        **raw_action_record,
                        "api_name": api_name,
                        "arguments": injected_args,
                        "executed": False,
                        "error": dependency_error,
                    })
                    conversation = self._append_feedback(
                        conversation,
                        model_output,
                        dependency_hint(dependency_error)
                        or "CrossModalInterpret requires valid RGB and SAR detection evidence from successful previous tools. Do not hand-write detection boxes.",
                    )
                    continue

            signature = call_signature(api_name, injected_args)
            prior_failure = failed_call_history.get(signature)
            if prior_failure and (prior_failure.get("count", 0) >= 2 or prior_failure.get("deterministic")):
                self._add_error(trace, "duplicate_call_blocked")
                self._add_error(trace, "duplicate_failure_terminated")
                prior_failure["count"] = int(prior_failure.get("count", 0)) + 1
                trace.setdefault("duplicate_call_blocked", []).append({
                    "round": round_id,
                    "api_name": api_name,
                    "arguments": injected_args,
                    "duplicate_count": prior_failure.get("count", 0),
                    "previous_error_code": prior_failure.get("error_code"),
                    "previous_error_text": prior_failure.get("error_text"),
                })
                trace["actions"].append({**raw_action_record, "api_name": api_name, "arguments": injected_args, "executed": False, "error": "duplicate_call_blocked"})
                trace["final_answer"] = ""
                trace["terminated"] = True
                return finish()

            trace["actions"].append({
                **raw_action_record,
                "api_name": api_name,
                "arguments": injected_args,
                "executed": True,
            })

            tool_started = time.perf_counter()
            try:
                raw_tool_resp = self._get_tool_manager().call_tool(api_name, injected_args)
            except Exception as exc:
                raw_tool_resp = {"error_code": -1, "text": f"Tool call failed: {exc}"}
            elapsed = time.perf_counter() - tool_started
            trace["tool_time"] += elapsed

            tool_resp = coerce_business_negative_response(normalize_tool_response(raw_tool_resp, api_name), api_name)
            ok = tool_resp.get("error_code") == 0
            if not ok:
                self._add_error(trace, "tool_error")
                failure = failed_call_history.get(signature, {"count": 0})
                failure.update({
                    "count": int(failure.get("count", 0)) + 1,
                    "error_code": tool_resp.get("error_code"),
                    "error_text": str(tool_resp.get("text", ""))[:500],
                    "deterministic": is_deterministic_failure(tool_resp),
                })
                failed_call_history[signature] = failure
            if api_name == "SkillExecutor":
                trace["skill_tool_calls"] += 1
                trace["skill_calls"].append({
                    "round": round_id,
                    "skill_name": injected_args.get("skill_name"),
                    "arguments": injected_args,
                    "success": ok,
                    "elapsed": round(elapsed, 6),
                    "response": tool_resp,
                })
            else:
                trace["raw_tool_calls"] += 1

            trace["tool_responses"].append({
                "round": round_id,
                "api_name": api_name,
                "elapsed": round(elapsed, 6),
                "response": tool_resp,
            })

            obs_text = tool_resp.get("text", "")
            if ok and "gpkg" in tool_resp:
                current_gpkg = tool_resp.get("gpkg")
            registry = update_registry_with_tool_response(registry, api_name, tool_resp, round_id=round_id)
            structured_observation = build_structured_observation(tool_resp)
            observation_parts = ["OBSERVATION:"]
            if ok:
                observation_parts.append(f"{api_name} outputs: {obs_text}")
            else:
                observation_parts.extend([
                    "TOOL_CALL_FAILED",
                    "Do not treat this as a task result.",
                    f"{api_name} error: {obs_text}",
                ])
            if structured_observation:
                observation_parts.extend(["", structured_observation])
            observation_parts.extend(
                [
                    "",
                    build_artifact_summary(registry),
                    "",
                    "When a later tool should use an uploaded input or an intermediate result, reference it explicitly in the action arguments using $key.",
                ]
            )
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({
                "role": "user",
                "content": "\n".join(observation_parts),
            })

        self._add_error(trace, "max_rounds")
        return finish()

    def _select_skills(
        self,
        question: str,
        registry: Dict[str, Any],
        oracle_skill: Optional[Any],
        trace: AgentTrace,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if self.mode == RunMode.NO_SKILL:
            return [], {"mode": self.mode.value, "skipped_reason": "skill_disabled"}

        if self.mode in {RunMode.ORACLE_SKILL, RunMode.ORACLE_FORCED_SKILL}:
            skill_name = oracle_skill.get("skill_name") if isinstance(oracle_skill, dict) else oracle_skill
            if not skill_name:
                self._add_error(trace, "missing_oracle_skill")
                return [], {"mode": self.mode.value, "skipped_reason": "missing_oracle_skill"}
            if skill_name not in SKILL_REGISTRY:
                self._add_error(trace, "invalid_oracle_skill")
                return [], {"mode": self.mode.value, "skipped_reason": "invalid_oracle_skill", "oracle_skill": skill_name}
            selected = [{"skill_name": skill_name, "score": 1.0, "reasons": ["oracle skill"]}]
            return selected, {"mode": self.mode.value, "skipped_reason": "oracle_skill_provided", "selected": selected}

        selected = route_skills(
            user_query=question,
            input_registry=registry,
            top_k=self.config.top_k,
            min_score=self.config.min_score,
            score_margin=self.config.score_margin,
        )
        return selected, {
            "mode": self.mode.value,
            "params": {
                "top_k": self.config.top_k,
                "min_score": self.config.min_score,
                "score_margin": self.config.score_margin,
            },
            "selected": selected,
        }

    def _run_forced_oracle_skill(
        self,
        question: str,
        registry: Dict[str, Any],
        oracle_skill: Any,
        eval_meta: Dict[str, Any],
        trace: AgentTrace,
        started_at: float,
    ) -> AgentTrace:
        skill_name = oracle_skill.get("skill_name") if isinstance(oracle_skill, dict) else str(oracle_skill)
        return self._run_forced_skill(
            question=question,
            registry=registry,
            skill_name=skill_name,
            eval_meta=eval_meta,
            trace=trace,
            started_at=started_at,
            forced_key="forced_oracle",
            forced_thought="Forced oracle skill execution.",
            final_thought="Forced oracle skill produced a final answer.",
        )

    def _run_forced_selected_skill(
        self,
        question: str,
        registry: Dict[str, Any],
        selected_skill: str,
        eval_meta: Dict[str, Any],
        trace: AgentTrace,
        started_at: float,
    ) -> AgentTrace:
        return self._run_forced_skill(
            question=question,
            registry=registry,
            skill_name=selected_skill,
            eval_meta=eval_meta,
            trace=trace,
            started_at=started_at,
            forced_key="forced_auto_skill",
            forced_thought="Forced auto selected skill execution.",
            final_thought="Forced auto selected skill produced a final answer.",
        )

    def _run_forced_skill(
        self,
        question: str,
        registry: Dict[str, Any],
        skill_name: str,
        eval_meta: Dict[str, Any],
        trace: AgentTrace,
        started_at: float,
        forced_key: str,
        forced_thought: str,
        final_thought: str,
    ) -> AgentTrace:
        if not skill_name or skill_name not in SKILL_REGISTRY:
            self._add_error(trace, "invalid_oracle_skill")
            trace["wall_time"] = round(time.perf_counter() - started_at, 6)
            return trace

        args = build_oracle_skill_args(skill_name, question, registry, eval_meta=eval_meta)
        args = normalize_skill_action_args(args)
        args = inject_skill_runtime_arguments(args, registry)
        round_id = 1
        trace["rounds"] = round_id
        trace["actions"].append({
            "round": round_id,
            "thought": forced_thought,
            "raw_action": {"name": "SkillExecutor", "arguments": args},
            "api_name": "SkillExecutor",
            "arguments": args,
            "executed": True,
            forced_key: True,
        })

        tool_started = time.perf_counter()
        try:
            raw_tool_resp = self._get_tool_manager().call_tool("SkillExecutor", args)
        except Exception as exc:
            raw_tool_resp = {"error_code": -1, "text": f"Tool call failed: {exc}"}
        elapsed = time.perf_counter() - tool_started
        trace["tool_time"] += elapsed
        tool_resp = normalize_tool_response(raw_tool_resp, "SkillExecutor")
        ok = tool_resp.get("error_code") == 0
        if not ok:
            self._add_error(trace, "tool_error")

        trace["skill_tool_calls"] += 1
        trace["skill_calls"].append({
            "round": round_id,
            "skill_name": skill_name,
            "arguments": args,
            "success": ok,
            "elapsed": round(elapsed, 6),
            "response": tool_resp,
            forced_key: True,
        })
        trace["tool_responses"].append({
            "round": round_id,
            "api_name": "SkillExecutor",
            "elapsed": round(elapsed, 6),
            "response": tool_resp,
        })
        registry = update_registry_with_tool_response(registry, "SkillExecutor", tool_resp, round_id=round_id)
        trace["input_registry_after_oracle"] = {key: registry.get(key) for key in ["primary_image", "optical_image", "sar_image", "latest_successful_structured"] if registry.get(key)}

        if ok:
            answer = ""
            task_result = tool_resp.get("task_result")
            if isinstance(task_result, dict):
                answer = str(task_result.get("answer") or "")
            if not answer:
                structured = tool_resp.get("structured") if isinstance(tool_resp.get("structured"), dict) else {}
                nested_result = structured.get("task_result") if isinstance(structured, dict) else {}
                if isinstance(nested_result, dict):
                    answer = str(nested_result.get("answer") or "")
            if not answer:
                answer = self._format_forced_skill_answer(skill_name, tool_resp)
            trace["final_answer"] = answer or str(tool_resp.get("text") or "")
            trace["terminated"] = True
            trace["actions"].append({
                "round": 2,
                "thought": final_thought,
                "raw_action": {"name": "Terminate", "arguments": {"ans": trace["final_answer"]}},
                "api_name": "Terminate",
                "arguments": {"ans": trace["final_answer"]},
                "executed": True,
                forced_key: True,
            })
            trace["rounds"] = 2

        trace["total_tool_calls"] = trace["raw_tool_calls"] + trace["skill_tool_calls"]
        trace["wall_time"] = round(time.perf_counter() - started_at, 6)
        trace["llm_time"] = round(trace["llm_time"], 6)
        trace["tool_time"] = round(trace["tool_time"], 6)
        return trace

    def _should_force_auto_skill(self, selected_skills: List[Dict[str, Any]], eval_meta: Dict[str, Any]) -> bool:
        if self.mode != RunMode.AUTO_SKILL or not selected_skills:
            return False
        if not eval_meta.get("should_use_skill"):
            return False
        selected_name = selected_skills[0].get("skill_name")
        gold_skill = eval_meta.get("gold_skill")
        if gold_skill and selected_name != gold_skill:
            return False
        return bool(selected_name) and self._is_tool_available("SkillExecutor")

    @staticmethod
    def _format_forced_skill_answer(skill_name: str, tool_resp: Dict[str, Any]) -> str:
        if skill_name != "ObjectMeasureSkill":
            return ""
        structured = tool_resp.get("structured") if isinstance(tool_resp.get("structured"), dict) else {}
        measurement = structured.get("measurement") if isinstance(structured.get("measurement"), dict) else {}
        measure_type = str(structured.get("measure_type") or "")
        target = str(structured.get("target") or "target")
        if measure_type in {"distance", "pair_distance"}:
            if measurement.get("distance_m") is not None:
                return f"The distance for {target} is {float(measurement['distance_m']):.2f} meters."
            if measurement.get("distance_px") is not None:
                return f"The distance for {target} is {float(measurement['distance_px']):.2f} pixels."
        if measure_type == "area":
            if measurement.get("area_m2") is not None:
                return f"The area of {target} is {float(measurement['area_m2']):.2f} square meters."
            if measurement.get("area_px") is not None:
                return f"The area of {target} is {float(measurement['area_px']):.2f} square pixels."
        if measure_type in {"bbox_size", "pixel_dimensions"}:
            width = measurement.get("width_px")
            height = measurement.get("height_px")
            if width is not None and height is not None:
                return f"Pixel width ≈ {float(width):.2f} px. Pixel length ≈ {float(height):.2f} px."
        if measure_type == "centroid" and measurement.get("centroid") is not None:
            return f"The centroid of {target} is {measurement['centroid']}."
        return ""

    def _prepare_skill_action(
        self,
        injected_args: Dict[str, Any],
        selected_skills: List[Dict[str, Any]],
        registry: Dict[str, Any],
        trace: AgentTrace,
        raw_action_record: Dict[str, Any],
        conversation: List[Dict[str, str]],
        model_output: str,
    ) -> Tuple[Dict[str, Any], Optional[List[Dict[str, str]]]]:
        if not selected_skills:
            self._add_error(trace, "no_selected_skills")
            trace["actions"].append({**raw_action_record, "api_name": "SkillExecutor", "arguments": injected_args, "executed": False, "error": "no_selected_skills"})
            return injected_args, self._append_feedback(
                conversation,
                model_output,
                "No suitable skill is available for this task. Do not use SkillExecutor; choose raw tools directly.",
            )

        if not self._is_tool_available("SkillExecutor"):
            self._add_error(trace, "skill_executor_unavailable")
            trace["actions"].append({**raw_action_record, "api_name": "SkillExecutor", "arguments": injected_args, "executed": False, "error": "skill_executor_unavailable"})
            return injected_args, self._append_feedback(
                conversation,
                model_output,
                "SkillExecutor is unavailable in the current runtime. Please select a raw tool from the tool list and continue.",
            )

        if not injected_args.get("skill_name"):
            injected_args["skill_name"] = selected_skills[0]["skill_name"]
        selected_names = {item["skill_name"] for item in selected_skills}
        if injected_args.get("skill_name") not in selected_names:
            self._add_error(trace, "skill_not_selected")
            trace["actions"].append({**raw_action_record, "api_name": "SkillExecutor", "arguments": injected_args, "executed": False, "error": "skill_not_selected"})
            return injected_args, self._append_feedback(
                conversation,
                model_output,
                "The requested skill_name is not in the Selected Skills list. Use one selected skill only, or choose raw tools directly.",
            )
        return inject_skill_runtime_arguments(injected_args, registry), None

    def _generate(self, conversation: List[Dict[str, str]], selected_skill_prompt: str) -> str:
        llm = self._get_llm()
        try:
            return llm.generate(conversation, extra_system_prompt=selected_skill_prompt)
        except TypeError:
            return llm.generate(conversation)

    def _get_llm(self) -> Any:
        system_prompt = build_no_skill_prompt(RS_AGENT_PROMPT) if self.mode == RunMode.NO_SKILL else RS_AGENT_PROMPT
        if self.llm is None:
            self.llm = LocalLLM(str(self.config.pretrained_path), system_prompt=system_prompt)
        elif hasattr(self.llm, "system_prompt"):
            self.llm.system_prompt = system_prompt
        return self.llm

    def _get_tool_manager(self) -> Any:
        if self.tool_manager is None:
            self.tool_manager = ToolManager(controller_url_location=self.config.controller_url_location)
        return self.tool_manager

    def _is_tool_available(self, tool_name: str) -> bool:
        manager = self._get_tool_manager()
        return (
            tool_name in getattr(manager, "available_tools", [])
            or tool_name in getattr(manager, "available_online_tools", [])
            or tool_name in getattr(manager, "available_offline_tools", [])
        )

    @staticmethod
    def _append_feedback(
        conversation: List[Dict[str, str]],
        model_output: str,
        feedback: str,
    ) -> List[Dict[str, str]]:
        conversation.append({"role": "assistant", "content": model_output})
        conversation.append({"role": "user", "content": feedback})
        return conversation

    @staticmethod
    def _add_error(trace: AgentTrace, tag: str) -> None:
        if tag not in trace["error_tags"]:
            trace["error_tags"].append(tag)
