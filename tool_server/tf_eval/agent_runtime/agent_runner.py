from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tool_server.tf_eval.agent_runtime.run_config import AgentRunConfig
from tool_server.tf_eval.agent_runtime.structured_summary import (
    build_structured_observation,
    get_structured_result_for_registry,
)
from tool_server.tf_eval.agent_runtime.trace_schema import AgentTrace, create_agent_trace
from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT
from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.tool_workers.cross_modal_evidence import (
    enrich_detection_response,
    extract_detections,
    filter_detection_response,
    register_empty_detection,
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

# 检测器约定的"业务性无目标"返回码（DenoDetV2 用 4）。判定不再依赖文本。
NO_DETECTION_ERROR_CODE = 4
# 统一文案：任何被判定为"无检测"的返回都用这一句，绝不给空字符串
NO_DETECTION_TEXT = "no detections found after thresholding"

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
    # Replay runs live detection through the shared evidence gate, but it still
    # refuses a request-injected georef image, so no binding may add one.
    bindings = {} if api_name == "CrossModalInterpret" and merged.get("source_mode") == "workbook_replay" else TOOL_ARG_BINDINGS.get(api_name, {})
    for tool_arg, registry_key in bindings.items():
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
    """按结构与返回码判断"跑完了但一个目标都没有"，不看文案。

    - status == "no_detections"（worker 显式声明）
    - error_code == 4（检测器的业务性无目标码）
    - error_code == 0 且 detections 是显式空列表
    有框、或参数/图像类错误（code 1/2/3）一律不算。
    """
    if not isinstance(tool_resp, dict):
        return False
    if str(tool_resp.get("status") or "").lower() == "no_detections":
        return True
    detections = tool_resp.get("detections")
    if isinstance(detections, list) and detections:
        return False
    code = tool_resp.get("error_code")
    if code == NO_DETECTION_ERROR_CODE:
        return True
    return code == 0 and isinstance(detections, list) and not detections


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
    if not is_no_detection_response(tool_resp):
        return tool_resp
    coerced = dict(tool_resp)
    if coerced.get("error_code") != 0:
        coerced["original_error_code"] = coerced.get("error_code")
        coerced["error_code"] = 0
    coerced.setdefault("detections", [])
    coerced.setdefault("structured", {"detections": [], "status": "no_detections"})
    coerced["status"] = "no_detections"
    if not str(coerced.get("text") or "").strip():
        coerced["text"] = NO_DETECTION_TEXT
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
    api_args: Optional[Dict[str, Any]] = None,
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
        registry = register_detections(registry, api_name, structured_result, round_id, api_args)
        if not extract_detections(structured_result):
            registry = register_empty_detection(registry, api_name, round_id, api_args)

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
        config: Optional[AgentRunConfig] = None,
        llm: Any = None,
        tool_manager: Any = None,
    ):
        self.config = config or AgentRunConfig()
        self.llm = llm
        self.tool_manager = tool_manager

    def run_sample(
        self,
        question: str,
        input_registry: Optional[Dict[str, Any]] = None,
        max_rounds: Optional[int] = None,
        sample_id: str = "",
    ) -> AgentTrace:
        started_at = time.perf_counter()
        registry = dict(input_registry or {})
        sample_id = sample_id or str(registry.get("sample_id") or "")
        trace = create_agent_trace(sample_id=sample_id, question=question or "")

        def finish() -> AgentTrace:
            trace["total_tool_calls"] = trace["raw_tool_calls"]
            trace["wall_time"] = round(time.perf_counter() - started_at, 6)
            trace["llm_time"] = round(trace["llm_time"], 6)
            trace["tool_time"] = round(trace["tool_time"], 6)
            return trace

        if not question or not question.strip():
            self._add_error(trace, "invalid_question")
            return finish()

        max_rounds = int(max_rounds or self.config.max_rounds)
        conversation = [{"role": "user", "content": build_initial_user_message(question, registry)}]
        current_gpkg = None
        failed_call_history: Dict[str, Dict[str, Any]] = {}

        for round_id in range(1, max_rounds + 1):
            trace["rounds"] = round_id
            try:
                llm_started = time.perf_counter()
                model_output = self._generate(conversation)
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
            # 先补齐 text-only 检测（带原始 index），再按类过滤，否则下标被打乱。
            tool_resp = enrich_detection_response(tool_resp, api_name)
            tool_resp = filter_detection_response(tool_resp, api_name, injected_args)
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
            registry = update_registry_with_tool_response(registry, api_name, tool_resp, round_id=round_id, api_args=injected_args)
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

    def _generate(self, conversation: List[Dict[str, str]]) -> str:
        return self._get_llm().generate(conversation)

    def _get_llm(self) -> Any:
        system_prompt = RS_AGENT_PROMPT
        if self.llm is None:
            self.llm = LocalLLM(str(self.config.pretrained_path), system_prompt=system_prompt)
        elif hasattr(self.llm, "system_prompt"):
            self.llm.system_prompt = system_prompt
        return self.llm

    def _get_tool_manager(self) -> Any:
        if self.tool_manager is None:
            self.tool_manager = ToolManager(controller_url_location=self.config.controller_url_location)
        return self.tool_manager

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
