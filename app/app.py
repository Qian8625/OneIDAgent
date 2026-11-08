import gradio as gr
import json
import re
import torch
import os
import sys
import html
import inspect
from datetime import datetime
from pathlib import Path

# Allow `python app/app.py` to import project-local packages from the repo root.
APP_FILE = Path(__file__).resolve()
APP_DIR = APP_FILE.parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tool_server.tool_workers.tool_manager.base_manager import ToolManager
from tool_server.tool_workers.cross_modal_evidence import (
    filter_detection_response,
    prepare_interpret_args,
    register_detections,
)
from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from tool_server.utils.paths import PROJECT_ROOT as REPO_ROOT, env_path
from data_preprocess import convert_tif_to_png

# 修改输入文件的路径
PROJECT_ROOT = REPO_ROOT


os.environ["GRADIO_TEMP_DIR"] = env_path("GRADIO_TEMP_DIR", "app/gradio_tmp")

PRETRAINED_PATH = env_path("PRETRAINED_PATH", "models/0_sft_train/qwen3_sft_vl_noskill_0611")
MAX_TOOL_ROUNDS = 16

INPUT_SPECS = [
    {
        "key": "primary_image",
        "label": "General Image / GeoTIFF",
        "description": "Single-scene image for detection, interpretation, or GeoTIFF overlay tasks.",
        "group": "General",
        "height": 260,
    },
    {
        "key": "optical_image",
        "label": "Optical / RGB Image",
        "description": "Optical or RGB image for cross-modal validation with SAR.",
        "group": "Cross-Modal",
        "height": 220,
    },
    {
        "key": "sar_image",
        "label": "SAR Grayscale Image",
        "description": "SAR grayscale image for cross-modal validation with optical.",
        "group": "Cross-Modal",
        "height": 220,
    },
    {
        "key": "time1_image",
        "label": "Time-1 Image",
        "description": "Earlier timestamp image, often used as pre_image.",
        "group": "Time Series",
        "height": 220,
    },
    {
        "key": "time2_image",
        "label": "Time-2 Image",
        "description": "Later timestamp image, often used as post_image.",
        "group": "Time Series",
        "height": 220,
    },
]

INPUT_GROUPS = [
    {
        "tab": "General",
        "input_keys": ["primary_image"],
    },
    {
        "tab": "Cross-Modal",
        "input_keys": ["optical_image", "sar_image"],
    },
    {
        "tab": "Time Series",
        "input_keys": ["time1_image", "time2_image"],
    },
]

TOOL_ARG_BINDINGS = {
    "DrawBox": {"image": "primary_image"},
    "AddText": {"image": "primary_image"},
    "TextToBbox": {"image": "primary_image"},
    "CountGivenObject": {"image": "primary_image"},
    "ImageDescription": {"image": "primary_image"},
    "RegionAttributeDescription": {"image": "primary_image"},
    "SegmentObjectPixels": {"image": "primary_image"},
    "ObjectDetection": {"image": "primary_image"},
    "SmallObjectDetection": {"image": "primary_image"},
    "GetBboxFromGeotiff": {"geotiff": "primary_image"},
    "DisplayOnGeotiff": {"geotiff": "primary_image"},
    "ChangeDetection": {"pre_image": "time1_image", "post_image": "time2_image"},
    "DenoDetV2": {"image": "primary_image"},
    "CrossModalInterpret": {"rgb_image": "optical_image", "sar_image": "sar_image"},
}

TOOL_DEFAULT_ARGUMENTS = {
    "AddText": {"color": "green"},
}

GPKG_REQUIRED_TOOLS = {
    "AddPoisLayer",
    "ComputeDistance",
    "DisplayOnGeotiff",
}


def validate_pretrained_path(pretrained):
    model_ref = str(pretrained).strip()
    model_path = Path(model_ref).expanduser()
    if model_path.is_absolute() and not model_path.exists():
        raise FileNotFoundError(
            "PRETRAINED_PATH points to a local path that does not exist: "
            f"{model_path}. Update PRETRAINED_PATH in .env or pass an existing model directory."
        )
    return str(model_path) if model_path.is_absolute() else model_ref

class LLM:
    def __init__(self, pretrained):
        try:
            pretrained = validate_pretrained_path(pretrained)
            # Device is configurable: default to the first visible GPU so a
            # single-GPU machine works out of the box.  ONEID_DEVICE=cpu|0|1|...
            device_pref = os.environ.get("ONEID_DEVICE", "").strip().lower()
            if device_pref == "cpu" or not torch.cuda.is_available():
                self.device = torch.device("cpu")
            else:
                index = device_pref or "0"
                try:
                    index = str(int(index))
                except ValueError:
                    index = "0"
                self.device = torch.device(f"cuda:{index}")
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained, dtype=torch.bfloat16, trust_remote_code=True
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained, use_fast=True, trust_remote_code=True
            )
            self.tokenizer.padding_side = "left"
            self.system_prompt = RS_AGENT_PROMPT
            self.max_new_tokens = 256
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")


    def _build_system_prompt(self, extra_system_prompt: str = "") -> str:
        try:
            if extra_system_prompt and extra_system_prompt.strip():
                return self.system_prompt + "\n\n" + extra_system_prompt.strip()
            return self.system_prompt
        except Exception:
            return self.system_prompt

    def prepend_system_prompt(self, conversation, extra_system_prompt: str = ""):
        try:
            final_system_prompt = self._build_system_prompt(extra_system_prompt)

            conversation = list(conversation)

            if not conversation or conversation[0]["role"] != "system":
                conversation = [{"role": "system", "content": final_system_prompt}] + conversation
            else:
                conversation[0] = {"role": "system", "content": final_system_prompt}

            return conversation
        except Exception:
            return list(conversation)

    def generate(self, conversation, extra_system_prompt: str = ""):
        try:
            conversation = self.prepend_system_prompt(
                conversation,
                extra_system_prompt=extra_system_prompt
            )

            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens
                )

            generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            return json.dumps({"thought": f"模型推理出错: {str(e)}", "actions": []})


def extract_actions(text: str):
    try:
        actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
        actions_match = re.search(actions_pattern, text)
        if not actions_match:
            return None, "No action found."
        return json.loads(actions_match.group(1)), None
    except json.JSONDecodeError:
        return None, "Invalid JSON in actions."
    except Exception:
        return None, "Invalid action format."


def try_parse_agent_json(text: str):
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


def extract_thought_and_actions(text: str):
    try:
        data = try_parse_agent_json(text)
        if isinstance(data, list):
            # A bare list of actions is accepted as well.
            return "", [item for item in data if isinstance(item, dict)]
        if not isinstance(data, dict):
            data = {}
        thought = data.get("thought", "")
        actions = data.get("actions", None)
        if actions is None and isinstance(data.get("name"), str):
            # Tolerate the un-wrapped single-action form the model sometimes
            # emits: {"name": "...", "arguments": {...}}.
            arguments = data.get("arguments")
            actions = [{
                "name": data["name"],
                "arguments": arguments if isinstance(arguments, dict) else {},
            }]
        if actions is None:
            actions, _ = extract_actions(text)
            if actions is None:
                actions = []
        if not isinstance(actions, list):
            actions = []
        return thought, actions
    except Exception:
        return "", []

def resolve_existing_file(path):
    if not path or (isinstance(path, str) and not path.strip()):
        return None

    try:
        p = Path(str(path)).expanduser()

        if p.is_absolute():
            candidates = [p]
        else:
            candidates = [
                Path.cwd() / p,
                PROJECT_ROOT / p,
                APP_DIR / p,
            ]

        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
    except Exception:
        pass

    return None

def make_preview_if_tif(path):
    if not path:
        return path
    if path.lower().endswith((".tif", ".tiff")):
        try:
            return convert_tif_to_png(path, low_pct=2.0, high_pct=98.0, overwrite=True) or path
        except Exception:
            return path
    return path


def handle_upload(image_path):
    if not image_path:
        return None, None
    try:
        resolved = resolve_existing_file(image_path) or image_path
        if resolved.lower().endswith((".tif", ".tiff")):
            return make_preview_if_tif(resolved), resolved
        return resolved, None
    except Exception:
        return None, None


def normalize_uploaded_input(image_path, original_tif, sample_aliases=None):
    raw = original_tif or image_path
    if not raw or (isinstance(raw, str) and not raw.strip()):
        return None
    resolved = resolve_existing_file(raw) or raw
    if sample_aliases:
        basename = os.path.basename(resolved)
        if basename in sample_aliases:
            alias_target = resolve_existing_file(sample_aliases[basename])
            if alias_target:
                return alias_target
    return resolved


def detect_output_visual(tool_resp):
    if not isinstance(tool_resp, dict):
        return None
    try:
        for key in ["image", "output_path", "out_file", "path", "preview", "png"]:
            value = tool_resp.get(key)
            if isinstance(value, str):
                found = resolve_existing_file(value)
                if found and found.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    return found
        text = tool_resp.get("text", "")
        if isinstance(text, str):
            for match in reversed(
                re.findall(r"([A-Za-z0-9_./\\-]+\.(?:png|jpg|jpeg|tif|tiff))", text, re.I)
            ):
                found = resolve_existing_file(match)
                if found:
                    return found
    except Exception:
        pass
    return None


def get_input_spec(key):
    try:
        for spec in INPUT_SPECS:
            if spec["key"] == key:
                return spec
    except Exception:
        pass
    raise KeyError(f"Unknown input spec key: {key}")


def build_registry_from_args(flat_args):
    expected = len(INPUT_SPECS) * 2
    flat_args = list(flat_args or [])
    if len(flat_args) < expected:
        flat_args.extend([None] * (expected - len(flat_args)))

    registry, idx = {}, 0
    for spec in INPUT_SPECS:
        try:
            registry[spec["key"]] = normalize_uploaded_input(
                flat_args[idx], flat_args[idx + 1], sample_aliases=spec.get("sample_aliases")
            )
        except Exception:
            registry[spec["key"]] = None
        idx += 2
    return registry


def describe_registry_for_prompt(registry):
    try:
        return [
            f"- {spec['key']}: {spec['description']}"
            for spec in INPUT_SPECS
            if registry.get(spec["key"])
        ]
    except Exception:
        return []


def build_initial_user_message(user_question, registry):
    try:
        lines = [user_question.strip()]
        input_lines = describe_registry_for_prompt(registry)
        if input_lines:
            lines.extend(
                [
                    "",
                    "Available uploaded inputs:",
                    *input_lines,
                    "Use the most appropriate tool and bind these uploaded inputs to the correct tool arguments.",
                ]
            )

        lines.extend(
            [
                "",
                "Reference rules for tool arguments:",
                "- Uploaded inputs can be referenced explicitly with $primary_image, $optical_image, $sar_image, $time1_image, $time2_image if available.",
                "- Intermediate outputs from previous tool calls will be exposed as reusable artifact ids such as $latest_image or $round_2_sartorgb_image.",
                "- If a later tool should use an intermediate result, reference it explicitly in the action arguments.",
            ]
        )
        return "\n".join(lines)
    except Exception:
        return user_question

def find_unresolved_references(api_args):
    unresolved = []

    try:
        def _walk(value):
            if isinstance(value, str) and value.startswith("$"):
                unresolved.append(value)
            elif isinstance(value, list):
                for item in value:
                    _walk(item)
            elif isinstance(value, dict):
                for item in value.values():
                    _walk(item)

        _walk(api_args)
    except Exception:
        pass

    return unresolved

def build_artifact_summary(registry):
    try:
        lines = []

        uploaded = []
        for spec in INPUT_SPECS:
            key = spec["key"]
            if registry.get(key):
                uploaded.append(f"- ${key} | uploaded input | path={registry[key]}")
        if uploaded:
            lines.append("Uploaded inputs:")
            lines.extend(uploaded)

        artifacts = registry.get("artifacts", [])
        if artifacts:
            lines.append("")
            lines.append("Reusable artifacts:")
            for item in artifacts[-10:]:
                lines.append(
                    f"- ${item['id']} | type={item['type']} | tool={item['tool']} | path={item['path']}"
                )

        alias_lines = []
        if registry.get("latest_image"):
            alias_lines.append(f"- $latest_image | path={registry['latest_image']}")
        if registry.get("latest_vector"):
            alias_lines.append(f"- $latest_vector | path={registry['latest_vector']}")
        if registry.get("latest_output"):
            alias_lines.append(f"- $latest_output | path={registry['latest_output']}")
        if alias_lines:
            lines.append("")
            lines.append("Convenience aliases:")
            lines.extend(alias_lines)

        if not lines:
            return "No uploaded inputs or reusable artifacts are currently available."

        return "\n".join(lines)
    except Exception:
        return "Error building artifact summary."

def _resolve_ref_value(value, registry):
    try:
        if isinstance(value, str) and value in {"img_1", "$img_1"}:
            return registry.get("optical_image") or registry.get("primary_image") or value

        if isinstance(value, str) and value in {"img_2", "$img_2"}:
            return registry.get("sar_image") or registry.get("time2_image") or value

        if isinstance(value, str) and value.startswith("$"):
            ref_key = value[1:]
            return registry.get(ref_key, value)

        if isinstance(value, list):
            return [_resolve_ref_value(v, registry) for v in value]

        if isinstance(value, dict):
            return {k: _resolve_ref_value(v, registry) for k, v in value.items()}

        return value
    except Exception:
        return value


def resolve_argument_references(api_args, registry):
    try:
        return _resolve_ref_value(dict(api_args or {}), registry)
    except Exception:
        return dict(api_args or {})

def inject_runtime_arguments(api_name, api_args, registry, current_gpkg):
    try:
        merged = dict(api_args or {})

        for tool_arg, registry_key in TOOL_ARG_BINDINGS.get(api_name, {}).items():
            current_value = merged.get(tool_arg)
            if (
                registry_key == "primary_image"
                and current_value in {"primary_image", "$primary_image"}
                and registry.get("primary_image")
            ):
                merged[tool_arg] = registry["primary_image"]
            elif current_value in [None, ""] and registry.get(registry_key):
                merged[tool_arg] = registry[registry_key]

        if api_name in GPKG_REQUIRED_TOOLS and current_gpkg and merged.get("gpkg") in [None, ""]:
            merged["gpkg"] = current_gpkg

        for k, v in TOOL_DEFAULT_ARGUMENTS.get(api_name, {}).items():
            if merged.get(k) in [None, ""] or k not in merged:
                merged[k] = v

        return merged
    except Exception:
        return dict(api_args or {})

def register_artifact(registry, artifact_id, path, artifact_type, tool_name):
    if not path:
        return registry

    try:
        resolved = resolve_existing_file(path) or path
        preview = resolved
        if artifact_type in {"image", "raster"}:
            preview = make_preview_if_tif(resolved)
            preview = resolve_existing_file(preview) or resolved

        registry[artifact_id] = resolved

        artifacts = list(registry.get("artifacts", []))
        if not any(item.get("id") == artifact_id for item in artifacts):
            artifacts.append(
                {
                    "id": artifact_id,
                    "path": resolved,
                    "preview": preview,
                    "type": artifact_type,
                    "tool": tool_name,
                }
            )
        registry["artifacts"] = artifacts

        if artifact_type in {"image", "raster"}:
            registry["latest_image"] = resolved
            registry["latest_output"] = resolved
        elif artifact_type == "vector":
            registry["latest_vector"] = resolved
            registry["latest_output"] = resolved
        else:
            registry["latest_output"] = resolved
    except Exception:
        pass

    return registry

def build_media_item(path, name=None):
    try:
        resolved = resolve_existing_file(path)
        if not resolved:
            return None

        preview = make_preview_if_tif(resolved)
        preview_resolved = resolve_existing_file(preview) or resolved

        return {
            "path": resolved,
            "preview": preview_resolved,
            "name": name or os.path.basename(resolved),
        }
    except Exception:
        return None


def collect_round_image_media(registry, round_id, api_name=None):
    media = []
    seen = set()

    try:
        for item in registry.get("artifacts", []):
            artifact_id = item.get("id", "")
            artifact_type = item.get("type")
            artifact_tool = item.get("tool")
            artifact_path = item.get("path")

            if artifact_type not in {"image", "raster"}:    
                continue
            if not artifact_id.startswith(f"round_{round_id}_"):
                continue
            if api_name is not None and artifact_tool != api_name:
                continue
            if not artifact_path or artifact_path in seen:
                continue

            media_item = {
                "path": artifact_path,
                "preview": item.get("preview") or artifact_path,
                "name": f"{artifact_tool} · {os.path.basename(artifact_path)}",
            }
            media.append(media_item)
            seen.add(artifact_path)
    except Exception:
        pass

    return media

def collect_action_input_media(api_args):
    media = []
    seen = set()

    try:
        candidate_keys = [
            "image", "geotiff", "pre_image", "post_image",
            "rgb_image", "optical_image", "sar_image", "nir_image"
        ]

        for key in candidate_keys:
            value = api_args.get(key)
            if not isinstance(value, str):
                continue

            resolved = resolve_existing_file(value)
            if not resolved or resolved in seen:
                continue

            media_item = build_media_item(resolved, name=f"input:{key}")
            if media_item:
                media.append(media_item)
                seen.add(resolved)
    except Exception:
        pass

    return media

def update_registry_with_tool_response(registry, api_name, tool_resp, round_id=None, api_args=None):
    if not isinstance(tool_resp, dict):
        return registry

    if tool_resp.get("error_code") != 0:
        return registry

    try:
        registry = dict(registry)

        # 1) Canonical artifact protocol.
        artifacts = tool_resp.get("artifacts")
        if isinstance(artifacts, list):
            for idx, item in enumerate(artifacts, start=1):
                if not isinstance(item, dict):
                    continue

                path = item.get("path")
                artifact_type = item.get("type") or "other"
                artifact_id = item.get("id") or f"artifact_{idx}"

                if not path:
                    continue

                scoped_artifact_id = f"round_{round_id}_{api_name.lower()}_{artifact_id}"

                registry = register_artifact(
                    registry,
                    artifact_id=scoped_artifact_id,
                    path=path,
                    artifact_type=artifact_type,
                    tool_name=api_name,
                )

        # RGB/SAR detection registry: single implementation shared with the eval side.
        registry = register_detections(registry, api_name, tool_resp.get("structured"), round_id, api_args)
        # 2) Legacy fallback for raw tools only.
        visual = detect_output_visual(tool_resp)
        if visual:
            artifact_id = f"round_{round_id}_{api_name.lower()}_image"
            registry = register_artifact(
                registry,
                artifact_id=artifact_id,
                path=visual,
                artifact_type="image",
                tool_name=api_name,
            )

        gpkg_path = tool_resp.get("gpkg")
        if isinstance(gpkg_path, str):
            resolved_gpkg = resolve_existing_file(gpkg_path)
            if resolved_gpkg:
                artifact_id = f"round_{round_id}_{api_name.lower()}_vector"
                registry = register_artifact(
                    registry,
                    artifact_id=artifact_id,
                    path=resolved_gpkg,
                    artifact_type="vector",
                    tool_name=api_name,
                )

    except Exception:
        pass

    return registry

def make_example(question, **paths):
    try:
        row = [question]
        for spec in INPUT_SPECS:
            row.append(paths.get(spec["key"]))
        return row
    except Exception:
        return [question] + [None] * len(INPUT_SPECS)


def render_empty_chat_html():
    try:
        return """
        <div class="chat-area">
            <div class="empty-chat-card">
                <div class="panel-kicker">Mission Console</div>
                <h3>Ready for a new remote sensing task</h3>
            </div>
        </div>
        """
    except Exception:
        return "<div class='chat-area'></div>"


def classify_chat_message(role, content):
    try:
        if role == "user":
            return "user-msg", "Request", "badge-user"
        if "✅ Final" in content or "\nFinal" in content:
            return "assistant-msg final-msg", "Final", "badge-final"
        if "🛠️ Action:" in content:
            return "assistant-msg action-msg", "Tool Call", "badge-action"
        if "Observation" in content:
            return "assistant-msg observation-msg", "Observation", "badge-observation"
        if "Thinking..." in content:
            return "assistant-msg thinking-msg", "Reasoning", "badge-thinking"
        if "达到最大轮数" in content:
            return "assistant-msg alert-msg", "Stopped", "badge-alert"
    except Exception:
        pass
    return "assistant-msg", "Agent", "badge-agent"


def split_message_header(content):
    try:
        first_line, _, remainder = content.partition("\n")
        heading = first_line.strip() or "Agent update"
        round_label = ""
        round_match = re.match(r"\[(Round \d+)\]\s*(.*)", heading)
        if round_match:
            round_label = round_match.group(1)
            heading = round_match.group(2) or "Agent update"
        return round_label, heading, remainder.strip()
    except Exception:
        return "", "Agent update", content


def render_chat_html(messages):
    try:
        if not messages:
            return render_empty_chat_html()
        blocks = ['<div class="chat-area">']
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            bubble_class, badge_label, badge_class = classify_chat_message(role, content)

            if role == "user":
                body = html.escape(content)
                blocks.append(
                    f'<div class="{bubble_class}"><div class="msg-meta">'
                    f'<span class="msg-badge {badge_class}">{badge_label}</span>'
                    f'<span class="msg-heading">Task</span></div><pre>{body}</pre></div>'
                )
                continue

            round_label, heading, body = split_message_header(content)
            meta_items = []
            if round_label:
                meta_items.append(f'<span class="round-pill">{html.escape(round_label)}</span>')
            meta_items.append(f'<span class="msg-badge {badge_class}">{badge_label}</span>')
            meta_items.append(f'<span class="msg-heading">{html.escape(heading)}</span>')

            body_html = f"<pre>{html.escape(body)}</pre>" if body else ""

            media_html = ""
            media_items = msg.get("media") or []
            if media_items:
                cards = []
                for media in media_items:
                    preview = html.escape(media.get("preview", ""))
                    path = html.escape(media.get("path", ""))
                    name = html.escape(media.get("name", "image"))

                    cards.append(
                        f"""
                        <a class="msg-media-card" href="/gradio_api/file={path}" target="_blank">
                            <img class="msg-media-img" src="/gradio_api/file={preview}" alt="{name}" />
                            <div class="msg-media-name">{name}</div>
                        </a>
                        """
                    )
                media_html = f'<div class="msg-media-grid">{"".join(cards)}</div>'

            blocks.append(
                f'<div class="{bubble_class}"><div class="msg-meta">{"".join(meta_items)}</div>{body_html}{media_html}</div>'
            )
        blocks.append("</div>")
        return "".join(blocks)
    except Exception:
        return render_empty_chat_html()


Model = None
MODEL_INIT_ERROR = None
tool_manager = ToolManager()

def is_tool_available(tool_name: str) -> bool:
    try:
        return (
            tool_name in getattr(tool_manager, "available_tools", [])
            or tool_name in getattr(tool_manager, "available_online_tools", [])
            or tool_name in getattr(tool_manager, "available_offline_tools", [])
        )
    except Exception:
        return False


def normalize_tool_response(tool_resp, api_name):
    if isinstance(tool_resp, dict):
        return tool_resp

    if tool_resp is None:
        return {"error_code": -1, "text": f"{api_name} returned empty response."}
    
    if isinstance(tool_resp, str):
        return {"error_code": -1, "text": f"{api_name} returned string response: {tool_resp}"}
    
    return {"error_code": -1, "text": f"{api_name} returned non-dict response: {str(tool_resp)[:200]}"}


def get_model():
    global Model, MODEL_INIT_ERROR

    if Model is None and MODEL_INIT_ERROR is None:
        try:
            Model = LLM(PRETRAINED_PATH)
        except Exception as exc:
            MODEL_INIT_ERROR = exc

    if MODEL_INIT_ERROR is not None:
        raise RuntimeError(f"模型初始化失败: {MODEL_INIT_ERROR}")

    return Model


def run_agent(user_question, *flat_args):
    if not user_question or not user_question.strip():
        yield render_chat_html([{"role": "user", "content": ""}, {"role": "assistant", "content": "[Round 0] ❌ Observation\n请输入有效的任务指令。"}]), None
        return

    registry = build_registry_from_args(flat_args)
    current_gpkg = None
    chat_msgs = [{"role": "user", "content": user_question}]

    initial_message = build_initial_user_message(user_question, registry)
    conversation = [{"role": "user", "content": initial_message}]

    def emit(download_path=None):
        return render_chat_html(chat_msgs), download_path

    try:
        model = get_model()
    except Exception as exc:
        chat_msgs.append(
            {"role": "assistant", "content": f"[Round 0] ❌ Observation\n{exc}"}
        )
        yield emit(None)
        return

    for round_id in range(1, MAX_TOOL_ROUNDS + 1):
        try:
            model_output = model.generate(conversation)
        except Exception as exc:
            chat_msgs.append(
                {
                    "role": "assistant",
                    "content": f"[Round {round_id}] ❌ Observation\n模型推理失败：{exc}",
                }
            )
            yield emit(None)
            return

        thought, actions = extract_thought_and_actions(model_output)

        if not actions:
            chat_msgs.append(
                {"role": "assistant", "content": f"[Round {round_id}] Thinking...\n{thought or model_output}"}
            )
            yield emit(None)
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append({"role": "user", "content": user_question})
            continue

        action = actions[0] if isinstance(actions[0], dict) else {}
        api_name = action.get("name")
        raw_args = action.get("arguments", {})
        api_args = raw_args if isinstance(raw_args, dict) else {}
        api_args = resolve_argument_references(api_args, registry)  

        unresolved_refs = find_unresolved_references(api_args)
        if unresolved_refs:
            chat_msgs.append(
                {
                    "role": "assistant",
                    "content": f"[Round {round_id}] ❌ Observation\n未解析的中间结果引用：{', '.join(unresolved_refs)}",
                }
            )
            yield emit(None)
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        f"The action used unresolved references: {', '.join(unresolved_refs)}.\n"
                        f"Please use one of the available uploaded inputs or reusable artifacts."
                    ),
                }
            )
            continue

        if not api_name:
            chat_msgs.append(
                {
                    "role": "assistant",
                    "content": f"[Round {round_id}] ❌ Observation\n无效动作格式：缺少工具名。",
                }
            )
            yield emit(None)
            conversation.append({"role": "assistant", "content": model_output})
            conversation.append(
                {
                    "role": "user",
                    "content": "The action format is invalid. Please output a valid JSON action list.",
                }
            )
            continue

        if api_name == "Terminate":
            final_answer = api_args.get("ans", "") or "已完成，结果请查看输出历史或下载文件。"
            chat_msgs.append({"role": "assistant", "content": f"[Round {round_id}] ✅ Final\n{final_answer}"})
            download = registry.get("latest_output")
            yield emit(download)
            return

        injected_args = inject_runtime_arguments(api_name, api_args, registry, current_gpkg)
        if api_name == "CrossModalInterpret":
            injected_args, _dependency_error = prepare_interpret_args(injected_args, registry)

        action_media = collect_action_input_media(injected_args)
        chat_msgs.append(
            {
                "role": "assistant",
                "content": f"[Round {round_id}] 🛠️ Action: {api_name}\nArgs:\n{json.dumps(injected_args, ensure_ascii=False, indent=2)}",
                "media": action_media,
            }
        )
        yield emit(None)

        try:
            raw_tool_resp = tool_manager.call_tool(api_name, injected_args)
        except Exception as exc:
            raw_tool_resp = {"error_code": -1, "text": f"Tool call failed: {exc}"}

        tool_resp = normalize_tool_response(raw_tool_resp, api_name)
        tool_resp = filter_detection_response(tool_resp, api_name, injected_args)
        obs_text = tool_resp.get("text", "")
        if tool_resp.get("error_code") == 0 and "gpkg" in tool_resp:
            current_gpkg = tool_resp.get("gpkg")

        registry = update_registry_with_tool_response(
            registry,
            api_name,
            tool_resp,
            round_id=round_id,
            api_args=injected_args,
        )

        round_media = collect_round_image_media(
            registry,
            round_id=round_id,
            api_name=api_name,
        )

        artifact_summary = build_artifact_summary(registry)
        status = "✅" if tool_resp.get("error_code") == 0 else "❌"
        chat_msgs.append(
            {
                "role": "assistant",
                "content": f"[Round {round_id}] {status} Observation\n{obs_text}",
                "media": round_media,
            }
        )
        download = registry.get("latest_output")
        yield emit(download)

        conversation.append({"role": "assistant", "content": model_output})
        conversation.append(
            {
                "role": "user",
                "content": (
                    f"OBSERVATION:\n"
                    f"{api_name} outputs: {obs_text}\n\n"
                    f"{artifact_summary}\n\n"
                    f"When a later tool should use an uploaded input or an intermediate result, "
                    f"reference it explicitly in the action arguments using $key."
                ),
            }
        )

    chat_msgs.append({"role": "assistant", "content": "达到最大轮数，已停止。"})
    yield emit(registry.get("latest_output"))


def render_output_history_html(history):
    try:
        history = history or []
        if not history:
            return """
            <div class="history-list empty-state">
                <strong>暂无输出历史</strong>
                <p>Agent 生成结果后，会按时间倒序显示在这里。</p>
            </div>
            """

        blocks = ['<div class="history-list">']
        for idx, item in enumerate(history, start=1):
            preview = html.escape(item.get("preview") or "")
            name = html.escape(item.get("name") or f"output_{idx}")
            created_at = html.escape(item.get("created_at") or "--")
            source = html.escape(item.get("source") or "Agent")
            status = html.escape(item.get("status") or "ready")
            path = html.escape(item.get("path") or "")
            blocks.append(
                f"""
                <div class="history-item">
                    <div class="history-thumb-wrap">
                        <img class="history-thumb" src="/gradio_api/file={preview}" alt="{name}" />
                    </div>
                    <div class="history-meta">
                        <div class="history-title-row">
                            <strong>{name}</strong>
                            <span class="history-status status-{status}">{status}</span>
                        </div>
                        <div class="history-sub">{source} · {created_at}</div>
                        <div class="history-path">{path}</div>
                    </div>
                </div>
                """
            )
        blocks.append("</div>")
        return "".join(blocks)
    except Exception:
        return """
        <div class="history-list empty-state">
            <strong>暂无输出历史</strong>
            <p>Agent 生成结果后，会按时间倒序显示在这里。</p>
        </div>
        """


def append_output_history(history, path, source="Agent"):
    if not path:
        return history or []

    try:
        resolved = resolve_existing_file(path)
        if not resolved:
            return history or []

        history = list(history or [])
        if any(item.get("path") == resolved for item in history):
            return history

        preview = make_preview_if_tif(resolved)
        history.insert(
            0,
            {
                "name": os.path.basename(resolved),
                "path": resolved,
                "preview": preview if resolve_existing_file(preview) else resolved,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": source,
                "status": "success",
            },
        )
        return history
    except Exception:
        return history or []


def build_batch_files(history):
    files = []
    try:
        for item in history or []:
            path = resolve_existing_file(item.get("path"))
            if path:
                files.append(path)
    except Exception:
        pass
    return files


def clear_output_history():
    try:
        return render_output_history_html([]), [], []
    except Exception:
        return "", [], []


def make_upload_change_handler(spec_key):
    def _handler(image_path):
        preview_path, original_tif = handle_upload(image_path)
        return preview_path, original_tif or ""
    return _handler
    
def run_agent_with_ui(user_question, *payload):
    expected_flat = len(INPUT_SPECS) * 2
    flat_args = list(payload[:expected_flat])
    output_history = payload[expected_flat] if len(payload) > expected_flat else []
    if not isinstance(output_history, list):
        output_history = []

    for chat_rendered, batch_candidate in run_agent(user_question, *flat_args):
        output_history = append_output_history(output_history, batch_candidate, source="Agent")

        yield (
            chat_rendered,
            build_batch_files(output_history),
            render_output_history_html(output_history),
            output_history,
        )


def reset_ui_v2():
    try:
        outputs = [
            "",                             # user_input
            render_chat_html([]),           # chat_html
            [],                             # batch_download_files
            render_output_history_html([]), # output_history_html
            [],                             # output_history_state
        ]
        for _ in INPUT_SPECS:
            outputs.extend([None, None])     # image component + original tif state
        return tuple(outputs)
    except Exception:
        return ("", "", [], "", []) + (None, None) * len(INPUT_SPECS)


def supports_blocks_css():
    try:
        return "css" in inspect.signature(gr.Blocks.__init__).parameters
    except Exception:
        return True


def supports_launch_css():
    try:
        return "css" in inspect.signature(gr.Blocks.launch).parameters
    except Exception:
        return False


css = """
/* ===== layout backbone: single source of truth ===== */
:root {
    --topbar-h: 72px;
    --page-pad-top: 16px;
    --page-pad-right: 16px;
    --page-pad-bottom: 16px;
    --page-pad-left: 6px;

    --rail-gap: 10px;

    --left-cluster-w: 720px;
    --right-rail-w: 360px;
    --center-min-w: 760px;

    --left-col-gap: 10px;
    --example-col-fr: 0.9fr;
    --upload-col-fr: 1.1fr;

    --panel-radius: 18px;
    --panel-pad: 16px;
    --composer-h: 148px;

    --workspace-h: 900px;

    --c-deep: #355872;
    --c-mid: #7AAACE;
    --c-light: #9CD5FF;
    --c-bg: #F7F8F0;
    --c-border: #d6deea;
    --c-text: #132a3f;
    --c-muted: #58738a;
    --c-card: #ffffff;
    --c-soft: #f8fbff;
    --c-user: #edf6ff;
    --c-action: #eef7ff;
    --c-obs: #eefaf7;
    --c-final: #f0f6fb;
    --shadow: 0 10px 28px rgba(53, 88, 114, 0.10);
}

html, body, .gradio-container {
    height: 100%;
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
    font-family: "Inter", "IBM Plex Sans", "Segoe UI", sans-serif !important;
}

body {
    overflow: auto !important;
}

.gradio-container {
    max-width: 100vw !important;
    padding:
        var(--page-pad-top)
        var(--page-pad-right)
        var(--page-pad-bottom)
        var(--page-pad-left) !important;
    overflow: visible !important;
}

/* ===== title: gradient header + glass buttons ===== */
.page-header {
    width: 100%;
    margin: 0 0 14px 0;
    padding: 20px 0 18px 0;
    background: linear-gradient(135deg, #1a3a4f 0%, #355872 50%, #2a4f6b 100%);
    border-radius: 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(26, 58, 79, 0.25), 0 1px 3px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.page-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(156, 213, 255, 0.5), rgba(122, 170, 206, 0.6), rgba(156, 213, 255, 0.5), transparent);
}

.page-header-main {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    width: 100%;
    text-align: left;
    margin-bottom: 12px;
}

.page-header-logo {
    width: 88px;
    height: 88px;
    flex: 0 0 88px;
    display: flex;
    align-items: center;
    justify-content: center;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.2));
}

.page-header-logo img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
}

.page-header-title h1 {
    margin: 0;
    font-size: 26px;
    line-height: 1.2;
    color: #ffffff;
    font-weight: 800;
    text-shadow: 0 1px 4px rgba(0,0,0,0.15);
    letter-spacing: -0.3px;
}

.page-header-title p {
    margin: 6px 0 0 0;
    font-size: 13px;
    line-height: 1.5;
    color: rgba(200, 220, 240, 0.85);
}

/* ===== workspace ===== */
.workspace-shell {
    min-height: var(--workspace-h);
    height: auto !important;
    max-height: none !important;
    gap: var(--rail-gap);
    flex-wrap: nowrap !important;
    align-items: stretch !important;
}

#left-cluster,
#center-rail,
#right-rail {
    height: var(--workspace-h) !important;
    min-height: var(--workspace-h) !important;
    max-height: var(--workspace-h) !important;
}

#left-cluster {
    order: 1;
    flex: 0 0 var(--left-cluster-w) !important;
    max-width: var(--left-cluster-w) !important;
    min-width: var(--left-cluster-w) !important;
}

#center-rail {
    order: 2;
    flex: 1 1 auto !important;
    min-width: var(--center-min-w) !important;
}

#right-rail {
    order: 3;
    flex: 0 0 var(--right-rail-w) !important;
    max-width: var(--right-rail-w) !important;
    min-width: var(--right-rail-w) !important;
}

.rail-stack,
.panel-fill,
.left-top-card,
.right-bottom-card,
#center-main-card,
.left-cluster-shell,
.left-sub-card {
    min-height: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
}

.rail-stack {
    display: flex;
    flex-direction: column;
    gap: var(--rail-gap);
}

/* ===== panels: refined shadows + hover lift ===== */
.panel-card {
    background: var(--c-card) !important;
    border: 1px solid rgba(214, 222, 234, 0.6) !important;
    border-radius: var(--panel-radius) !important;
    box-shadow:
        0 1px 2px rgba(53, 88, 114, 0.04),
        0 4px 12px rgba(53, 88, 114, 0.06),
        0 8px 24px rgba(53, 88, 114, 0.04) !important;
    padding: var(--panel-pad) !important;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-height: 0 !important;
    transition: box-shadow 0.25s ease, transform 0.25s ease;
}

.panel-fill,
#center-main-card,
.left-sub-card {
    display: flex;
    flex-direction: column;
}

/* ===== section head: gradient accent bar ===== */
.section-head {
    margin-bottom: 12px;
    position: relative;
    padding-left: 14px;
}

.section-head::before {
    content: '';
    position: absolute;
    left: 0;
    top: 2px;
    bottom: 2px;
    width: 4px;
    border-radius: 2px;
    background: linear-gradient(180deg, var(--c-deep), var(--c-mid));
}

.section-head h3 { margin: 0; color: var(--c-text); font-size: 17px; font-weight: 700; }
.section-head p { margin: 6px 0 0; color: var(--c-muted); font-size: 13px; line-height: 1.45; }

/* 左侧上下两块 */
.left-cluster-shell {
    display: grid;
    grid-template-columns: 1fr;
    grid-template-rows: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--left-col-gap);
    align-items: stretch;
    height: 100% !important;
    min-height: 0 !important;
}

.left-cluster-shell > .left-sub-card {
    min-height: 0 !important;
    height: auto !important;
    max-height: none !important;
}

/* 滚动职责 */
.scroll-y {
    min-height: 0 !important;
    max-height: none !important;
    overflow-y: auto;
}

.left-sub-scroll,
.chat-scroll,
.history-scroll {
    flex: 1 1 auto !important;
    min-height: 0 !important;
    max-height: none !important;
    overflow-y: auto !important;
}

/* WebKit 自定义滚动条 */
.left-sub-scroll::-webkit-scrollbar,
.chat-scroll::-webkit-scrollbar,
.history-scroll::-webkit-scrollbar,
.scroll-y::-webkit-scrollbar {
    width: 6px;
}

.left-sub-scroll::-webkit-scrollbar-track,
.chat-scroll::-webkit-scrollbar-track,
.history-scroll::-webkit-scrollbar-track,
.scroll-y::-webkit-scrollbar-track {
    background: transparent;
}

.left-sub-scroll::-webkit-scrollbar-thumb,
.chat-scroll::-webkit-scrollbar-thumb,
.history-scroll::-webkit-scrollbar-thumb,
.scroll-y::-webkit-scrollbar-thumb {
    background: rgba(88, 115, 138, 0.25);
    border-radius: 3px;
}

.left-sub-scroll::-webkit-scrollbar-thumb:hover,
.chat-scroll::-webkit-scrollbar-thumb:hover,
.history-scroll::-webkit-scrollbar-thumb:hover,
.scroll-y::-webkit-scrollbar-thumb:hover {
    background: rgba(88, 115, 138, 0.4);
}

.upload-panel-body {
    display: flex;
    flex-direction: column;
    min-height: 0 !important;
    height: 100% !important;
}

.upload-tabs {
    display: flex;
    flex-direction: column;
    flex: 1 1 auto !important;
    min-height: 0 !important;
}

.upload-slot {
    min-height: 0 !important;
    height: 100% !important;
}

/* 中间聊天区 */
.chat-scroll {
    border: 1px solid var(--c-border);
    border-radius: 14px;
    background: #fff;
}

.chat-area {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px;
}

#center-main-card {
    display: flex;
    flex-direction: column;
    min-height: 0 !important;
}

#mission-console {
    flex: 4 1 0 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    border: 1px solid var(--c-border);
    border-radius: 14px;
    background: #fff;
}

/* 输入区 */
#composer-card {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    margin-top: 12px;
    border: 1px solid var(--c-border);
    border-radius: 16px;
    background: #fff;
    padding: 12px;
    align-items: stretch;
    gap: 10px;
}

#composer-card .input-box textarea {
    height: 100% !important;
    min-height: 100% !important;
    max-height: 100% !important;
    border-radius: 12px !important;
    border: 1px solid #cbd8e8 !important;
}

.composer-grid {
    height: 100%;
    align-items: stretch;
    gap: 10px;
}

.input-box textarea {
    min-height: calc(var(--composer-h) - 42px) !important;
    max-height: calc(var(--composer-h) - 42px) !important;
    border-radius: 12px !important;
    border: 1px solid #cbd8e8 !important;
}

/* 右侧 */
.right-bottom-card {
    display: flex;
    flex-direction: column;
}

.right-bottom-card > .toolbar-row {
    order: 1;
    flex: 0 0 auto;
}

.right-bottom-card > .history-scroll {
    order: 2;
    flex: 1 1 auto !important;
    min-height: 0 !important;
}

.right-bottom-card > .download-slot {
    order: 3;
    flex: 0 0 220px !important;
    min-height: 220px;
    max-height: 220px;
    overflow-y: auto;
    margin-top: 12px;
    margin-bottom: 0;
}

/* ===== tabs: hover transition + active glow ===== */
.mode-tabs button {
    transition: all 0.2s ease !important;
    color: var(--c-muted) !important;
    position: relative;
}

.mode-tabs button:hover {
    background: rgba(53, 88, 114, 0.05) !important;
    color: var(--c-text) !important;
}

.mode-tabs button[aria-selected="true"] {
    background: var(--c-deep) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(53, 88, 114, 0.2);
}

.mode-note {
    background: var(--c-soft);
    border: 1px solid #e4edf8;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 10px;
}

.mode-note p {
    margin: 6px 0 0;
    color: var(--c-muted);
    font-size: 12px;
}

/* ===== upload / download slots ===== */
.upload-slot,
.download-slot {
    border-radius: 14px !important;
    border: 1px solid #dde7f3 !important;
    background: linear-gradient(135deg, #fbfdff 0%, #f6f9fc 100%) !important;
    transition: all 0.2s ease;
}

.upload-slot:hover,
.download-slot:hover {
    border-color: #b8cde0 !important;
    box-shadow: 0 2px 8px rgba(53, 88, 114, 0.06);
}

.upload-slot {
    min-height: 200px;
}

/* ===== history items ===== */
.history-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.history-item {
    display: grid;
    grid-template-columns: 76px 1fr;
    gap: 10px;
    align-items: center;
    border: 1px solid #e1e9f3;
    background: linear-gradient(135deg, #fbfdff 0%, #f8fafb 100%);
    border-radius: 14px;
    padding: 10px;
    transition: all 0.25s ease;
}

.history-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 14px rgba(53, 88, 114, 0.1);
    border-color: #c8d7e7;
}

.history-thumb-wrap {
    width: 76px;
    height: 76px;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #dbe6f2;
    background: #fff;
}

.history-thumb {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

.history-meta {
    min-width: 0;
}

.history-title-row strong {
    display: block;
    color: var(--c-text);
    font-size: 13px;
    line-height: 1.35;
}

.history-sub {
    margin-top: 4px;
    color: var(--c-muted);
    font-size: 12px;
}

.history-path {
    margin-top: 4px;
    color: #6f879b;
    font-size: 11px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.history-status {
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 800;
    text-transform: uppercase;
}

.status-success {
    background: #e7f7ef;
    color: #1d7a49;
}

/* ===== chat messages: gradient bubbles + fade-in ===== */
.user-msg, .assistant-msg {
    max-width: 88%;
    border-radius: 14px;
    padding: 12px 14px;
    border: 1px solid rgba(219, 229, 241, 0.7);
    box-shadow: 0 1px 4px rgba(53, 88, 114, 0.05);
    animation: msgFadeIn 0.3s ease-out;
}

@keyframes msgFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-msg {
    margin-left: auto;
    background: linear-gradient(135deg, #edf6ff 0%, #e4f0fc 100%);
    border-color: #cbe3fb;
}

.assistant-msg { background: linear-gradient(135deg, #f9fbff 0%, #f5f8fc 100%); }
.action-msg { background: linear-gradient(135deg, #eef7ff 0%, #e6f2fd 100%); border-color: #cfe5fb; }
.observation-msg { background: linear-gradient(135deg, #eefaf7 0%, #e5f5f0 100%); border-color: #cde9df; }
.final-msg { background: linear-gradient(135deg, #f0f6fb 0%, #e8f0f7 100%); border-color: #d4e1ef; }
.alert-msg { background: linear-gradient(135deg, #fff6f1 0%, #fef0e8 100%); border-color: #f3d6c6; }

.msg-meta {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 8px;
}

.round-pill {
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    background: linear-gradient(135deg, #e9f1fa 0%, #dde9f6 100%);
    color: var(--c-deep);
}

.msg-badge {
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2px;
}

.badge-user {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    color: #1e40af;
}

.badge-thinking {
    background: linear-gradient(135deg, #e9f1fa 0%, #dde9f6 100%);
    color: var(--c-deep);
}

.badge-action {
    background: linear-gradient(135deg, #dbeafe 0%, #c7dcf8 100%);
    color: #1d4ed8;
}

.badge-observation {
    background: linear-gradient(135deg, #d1fae5 0%, #b8f0d8 100%);
    color: #065f46;
}

.badge-final {
    background: linear-gradient(135deg, #d4e1ef 0%, #c0d4e8 100%);
    color: #1e3a5f;
}

.badge-alert {
    background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
    color: #9a3412;
}

.badge-agent {
    background: linear-gradient(135deg, #e9f1fa 0%, #dde9f6 100%);
    color: var(--c-deep);
}

.msg-heading {
    font-size: 13px;
    font-weight: 700;
}

.assistant-msg pre,
.user-msg pre {
    margin: 0;
    white-space: pre-wrap;
    font-size: 12px;
    line-height: 1.58;
}

/* ===== media cards ===== */
.msg-media-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px;
    margin-top: 10px;
}

.msg-media-card {
    display: block;
    text-decoration: none;
    background: #ffffff;
    border: 1px solid #dbe5f1;
    border-radius: 12px;
    overflow: hidden;
    transition: all 0.25s ease;
}

.msg-media-card:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 8px 24px rgba(53, 88, 114, 0.14);
    border-color: #b8cde0;
}

.msg-media-img {
    width: 100%;
    height: 180px;
    object-fit: cover;
    display: block;
    background: #f5f8fc;
}

.msg-media-name {
    padding: 8px 10px;
    font-size: 11px;
    color: var(--c-muted);
    border-top: 1px solid #e8eef5;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ===== buttons: gradient + hover glow ===== */
.send-btn button,
.clear-btn button,
.ghost-btn button {
    border-radius: 12px !important;
    min-height: 46px !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px;
}

.send-btn button {
    background: linear-gradient(135deg, #355872 0%, #4a7a9e 100%) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(53, 88, 114, 0.25);
}

.send-btn button:hover {
    background: linear-gradient(135deg, #3f6a87 0%, #5a8ab0 100%) !important;
    box-shadow: 0 4px 16px rgba(53, 88, 114, 0.35);
    transform: translateY(-1px);
}

.clear-btn button,
.ghost-btn button {
    background: #fff !important;
    border: 1px solid #c8d7e7 !important;
    color: var(--c-text) !important;
}

.clear-btn button:hover,
.ghost-btn button:hover {
    background: #f0f5fa !important;
    border-color: #a8bdd0 !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(53, 88, 114, 0.1);
}

.toolbar-row {
    display: flex;
    gap: 8px;
    margin-bottom: 10px;
}

/* ===== responsive ===== */
@media (max-width: 1600px) {
    :root {
        --left-cluster-w: 680px;
        --right-rail-w: 340px;
    }
    .page-header-title h1 {
        font-size: 24px;
    }
}

@media (max-width: 1360px) {
    :root {
        --workspace-h: auto;
    }

    .workspace-shell {
        min-height: 0;
        height: auto !important;
        max-height: none !important;
        flex-wrap: wrap !important;
    }

    #left-cluster,
    #center-rail,
    #right-rail {
        min-width: 100% !important;
        max-width: 100% !important;
        flex: 1 1 100% !important;
        height: auto !important;
        min-height: 0 !important;
        max-height: none !important;
    }

    .left-cluster-shell {
        grid-template-columns: 1fr;
        height: auto !important;
        max-height: none !important;
    }

    .rail-stack,
    .panel-fill,
    .left-top-card,
    .right-bottom-card,
    #center-main-card,
    .left-sub-card {
        height: auto !important;
        max-height: none !important;
    }

    .left-sub-scroll,
    .chat-scroll,
    .history-scroll {
        flex: 0 0 auto !important;
        max-height: 520px !important;
    }

    .page-header {
        border-radius: 12px;
    }

    .page-header-logo {
        width: 64px;
        height: 64px;
        flex: 0 0 64px;
    }

    .page-header-title h1 {
        font-size: 20px;
    }
}
"""

blocks_kwargs = {"title": "Remote Sensing Data Intelligent Interpretation System"}
BLOCKS_HAS_CSS = supports_blocks_css()
LAUNCH_HAS_CSS = supports_launch_css()
if BLOCKS_HAS_CSS:
    blocks_kwargs["css"] = css

LOGO_PATH = env_path("LOGO_PATH", "")
logo_html = f'<img src="/gradio_api/file={LOGO_PATH}" alt="logo"/>' if LOGO_PATH else ""

with gr.Blocks(**blocks_kwargs) as demo:
    state_components, image_components = {}, {}
    output_history_state = gr.State([])

    gr.HTML(
        f"""
        <div class="page-header">
            <div class="page-header-main">
                <div class="page-header-logo">
                    {logo_html}
                </div>
                <div class="page-header-title">
                    <h1>OneID : Remote Sensing Identity Interpretation</h1>
                    <p>Dual-source (RGB/SAR) shared-identity reasoning for remote sensing</p>
                </div>
            </div>
            
        </div>
        """
    )

    with gr.Row(elem_classes=["workspace-shell"]):
        # Center first in code
        with gr.Column(elem_id="center-rail", elem_classes=["rail-stack"]):
            with gr.Column(elem_id="center-main-card", elem_classes=["panel-card"]):
                gr.HTML(
                    "<div class='section-head'>"
                    "<h3>Agent 问答与流程日志</h3>"
                    "</div>"
                )

                chat_html = gr.HTML(
                    value=render_chat_html([]),
                    elem_id="mission-console",
                    elem_classes=["chat-scroll"],
                )

                with gr.Row(elem_id="composer-card", elem_classes=["composer-grid"]):
                    user_input = gr.Textbox(
                        label="",
                        placeholder="请输入任务指令，相关例子可以直接点击Example...",
                        lines=4,
                        max_lines=8,
                        elem_classes=["input-box"],
                        scale=9,
                    )
                    with gr.Column(scale=2, min_width=150):
                        submit_btn = gr.Button("发送", elem_classes=["send-btn"])
                        clear_btn = gr.Button("清空", elem_classes=["clear-btn"])

        # Left cluster
        with gr.Column(elem_id="left-cluster", elem_classes=["rail-stack"]):
            with gr.Column(elem_classes=["panel-card", "left-top-card", "panel-fill"]):
                with gr.Column(elem_classes=["left-cluster-shell"]):
                    # Upload first in code
                    with gr.Column(elem_classes=["panel-card", "left-sub-card"]):
                        gr.HTML(
                            "<div class='section-head'>"
                            "<h3>输入图片上传</h3>"
                            "</div>"
                        )

                        with gr.Column(elem_classes=["left-sub-scroll", "upload-panel-body"]):
                            with gr.Tabs(elem_classes=["mode-tabs", "upload-tabs"]):
                                for group in INPUT_GROUPS:
                                    with gr.Tab(group["tab"]):
                                        if len(group["input_keys"]) == 1:
                                            spec = get_input_spec(group["input_keys"][0])
                                            image_components[spec["key"]] = gr.Image(
                                                type="filepath",
                                                label=spec["label"],
                                                sources=["upload", "clipboard"],
                                                elem_classes=["upload-slot"],
                                            )
                                            state_components[spec["key"]] = gr.State()
                                        else:
                                            with gr.Row():
                                                for key in group["input_keys"]:
                                                    spec = get_input_spec(key)
                                                    image_components[spec["key"]] = gr.Image(
                                                        type="filepath",
                                                        label=spec["label"],
                                                        sources=["upload", "clipboard"],
                                                        elem_classes=["upload-slot"],
                                                    )
                                                    state_components[spec["key"]] = gr.State()

                    # Example second in code
                    with gr.Column(elem_classes=["panel-card", "left-sub-card"]):
                        gr.HTML(
                            "<div class='section-head'>"
                            "<h3>Example</h3>"
                            "</div>"
                        )

                        example_inputs = [user_input] + [image_components[spec["key"]] for spec in INPUT_SPECS]
                        with gr.Column(elem_classes=["left-sub-scroll"]):
                            with gr.Tabs():
                                with gr.Tab("Det/Seg"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Detect the aircraft in the picture, draw it and estimate the distance between aircrafts in the scene. assuming GSD 0.6px/meter.", 
                                                primary_image="./assets/samples/scene.jpg"
                                                ),
                                            make_example(
                                                "Use a small target detection model to detect ships in the map and label them in the image.",
                                                primary_image="./assets/samples/scene.jpg"
                                            ),
                                            make_example(
                                                "Visualize all museums and malls over the given GeoTIFF image, compute the distance between the closest pair, and finally annotate the image with this distance.", 
                                                primary_image="./assets/samples/scene.jpg"
                                                ),
                                            make_example(
                                                "Segment the road roundabout and measure its pixel area. Convert that pixel count to square meters using a ground sample distance of 0.132599419033 m/pixel. ",
                                                primary_image="./assets/samples/scene.jpg"
                                            ),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )
                                with gr.Tab("Attribute"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Detect all red house targets and calculate the area of the largest red house (in square meters), and plot it in a chart.",
                                                primary_image="./assets/samples/scene.jpg"
                                            ),
                                            make_example(
                                                "How many cars are there and are all of them traveling in the southwest direction?",
                                                primary_image="./assets/samples/scene.jpg"
                                            ),
                                            make_example(
                                                "How many aircraft are parked at the terminal apron, and what are their color attributes? Are they have same color?",
                                                primary_image="./assets/samples/scene.jpg"
                                            ),
                                            make_example(
                                                "Can you check the image if the ship is heading towards the shore?",
                                                primary_image="./assets/samples/scene.jpg"
                                            ),
                                            make_example("For the area within a 1000 m radius of Shimen Reservoir,China, generate a DEM layer, create contour lines, and visualize the create contour lines on the map.",),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )
                                with gr.Tab("SAR Trans/Det"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Please convert this SAR image to an RGB image",
                                                primary_image="./assets/samples/match_sar.jpg"
                                            ),
                                            make_example(
                                                "First, the SAR image is converted from SAR to RGB. Next, target detection is performed, and the target is then marked on the RGB image.",
                                                primary_image="./assets/samples/match_sar.jpg"
                                            ),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )
                                with gr.Tab("TimeSeries Analyze"):
                                    gr.Examples(
                                        examples=[
                                            make_example(
                                                "Visualize all museums and malls over the given GeoTIFF image, compute the distance between the closest pair, and finally annotate the image with this distance.", 
                                                primary_image="./assets/samples/scene.jpg"
                                                ),
                                            make_example(
                                                "List all the damaged buildings in the post-disaster image..",
                                                time1_image="./assets/samples/time_pre.jpg",
                                                time2_image="./assets/samples/time_post.jpg"
                                            ),
                                            make_example(
                                                "Please determine whether the airplane target in the bottom_right region left its original position between the pre-change and post-change images, and describe the scene.",
                                                time1_image="./assets/samples/time_pre.jpg",
                                                time2_image="./assets/samples/time_post.jpg"
                                            ),
                                            make_example(
                                                "Find remote sensing images from June 2024, output the number of images, and describe the last image",
                                            ),
                                        ],
                                        inputs=example_inputs,
                                        preprocess=False,
                                    )
        # Right rail
        with gr.Column(elem_id="right-rail", elem_classes=["rail-stack"]):
            with gr.Column(elem_classes=["panel-card", "right-bottom-card", "panel-fill"]):
                gr.HTML(
                    "<div class='section-head'>"
                    "<h3>输出历史下载</h3>"
                    "</div>"
                )

                batch_download_files = gr.File(
                    label="批量下载",
                    file_count="multiple",
                    elem_classes=["download-slot"],
                )

                output_history_html = gr.HTML(
                    value=render_output_history_html([]),
                    elem_classes=["scroll-y", "history-scroll"],
                )

                with gr.Row(elem_classes=["toolbar-row"]):
                    clear_history_btn = gr.Button("清空历史", elem_classes=["ghost-btn"])

    for spec in INPUT_SPECS:
        image_components[spec["key"]].change(
            fn=make_upload_change_handler(spec["key"]),
            inputs=[image_components[spec["key"]]],
            outputs=[
                image_components[spec["key"]],
                state_components[spec["key"]],
            ],
            trigger_mode="once",
        )

    submit_inputs = [user_input]
    for spec in INPUT_SPECS:
        submit_inputs.extend([image_components[spec["key"]], state_components[spec["key"]]])
    submit_inputs.append(output_history_state)

    submit_btn.click(
        fn=run_agent_with_ui,
        inputs=submit_inputs,
        outputs=[
            chat_html,
            batch_download_files,
            output_history_html,
            output_history_state,
        ],
    )

    clear_history_btn.click(
        fn=clear_output_history,
        inputs=[],
        outputs=[
            output_history_html,
            batch_download_files,
            output_history_state,
        ],
    )

    clear_outputs = [
        user_input,
        chat_html,
        batch_download_files,
        output_history_html,
        output_history_state,
    ]
    for spec in INPUT_SPECS:
        clear_outputs.extend([image_components[spec["key"]], state_components[spec["key"]]])

    clear_btn.click(
        fn=reset_ui_v2,
        inputs=[],
        outputs=clear_outputs,
    )


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 4444,
        "allowed_paths": ["./"],
    }
    if LAUNCH_HAS_CSS and not BLOCKS_HAS_CSS:
        launch_kwargs["css"] = css
    demo.queue().launch(**launch_kwargs)
