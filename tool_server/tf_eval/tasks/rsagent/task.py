
from ...utils.task_utils import *
from ...utils.utils import *
from ...utils.log_utils import get_logger
import ast
import os
import re
import json
import urllib.error
import urllib.request
from datasets import Dataset
from collections import Counter, defaultdict
from tool_server.tool_workers.offline_workers import offline_tool_workers
from tool_server.tool_workers.tool_manager.base_manager import CORE_EXPECTED_ONLINE_TOOLS
import time
import http.client

logger = get_logger(__name__)
task_config = get_task_config_from_current_dir(__file__)

judge_api_key = os.getenv("JUDGE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
if not judge_api_key:
    raise ValueError(
        "Judge API key not provided. Set JUDGE_API_KEY for evaluation scoring "
        "or DEEPSEEK_API_KEY for backward compatibility."
    )

JUDGE_API_URL = os.getenv(
    "JUDGE_API_URL",
    os.getenv(
        "DEEPSEEK_API_URL",
        task_config.get("deepseek_api_url", "https://api.deepseek.com/chat/completions"),
    ),
)
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    task_config.get("llm_judge", "deepseek-chat"),
)

LOG_DIR = "logs/scores"
os.makedirs(LOG_DIR, exist_ok=True)

TOOLS_LIST = list(dict.fromkeys(
    CORE_EXPECTED_ONLINE_TOOLS
    + list(offline_tool_workers.keys())
    + ["Terminate"]
))

def extract_action(text: str):
    def coerce_actions(obj):
        if isinstance(obj, dict):
            if "actions" in obj:
                actions = obj["actions"]
                if isinstance(actions, list):
                    return actions
                if isinstance(actions, dict):
                    return [actions]
                return None
            if "name" in obj and "arguments" in obj:
                return [obj]
        if isinstance(obj, list) and obj and all(isinstance(item, dict) for item in obj):
            return obj
        return None

    def parse_full(candidate):
        candidate = candidate.strip()
        if not candidate:
            return None
        for parser in (json.loads, ast.literal_eval):
            try:
                actions = coerce_actions(parser(candidate))
                if actions:
                    return actions
            except Exception:
                continue
        return None

    if not isinstance(text, str):
        return coerce_actions(text)

    try:
        actions = parse_full(text)
        if actions:
            return actions

        code_blocks = re.findall(r"```\s*(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        for block in code_blocks:
            actions = parse_full(block)
            if actions:
                return actions

        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\{\[]", text):
            try:
                obj, _ = decoder.raw_decode(text[match.start():])
                actions = coerce_actions(obj)
                if actions:
                    return actions
            except Exception:
                continue

        actions_pattern = r'"actions"\s*:\s*(\[(?:[^\[\]]|\[(?:[^\[\]]|\[(?:[^\[\]]|\[[^\[\]]*\])*\])*\])*\])'
        actions_match = re.search(actions_pattern, text)
        if not actions_match:
            return None
        return json.loads(actions_match.group(1))     
    except Exception as e:
        # logger.info(f"Error extracting actions list: {e}")
        return None
        
def extract_final_answer(final_response: str):
    res = ""
    actions = extract_action(final_response)
    if actions:
        action = actions[0]
        if (
            action.get("name") == "Terminate"
            and isinstance(action.get("arguments"), dict)
            and "ans" in action["arguments"]
        ):
            ans = action["arguments"]["ans"]
            return ans.strip() if isinstance(ans, str) else ans
    return res

def extract_tool_cfg(action):
    tool_cfg = None
    error = ""
    if not isinstance(action, dict):
        error = "action_is_not_valid_dict"
        return tool_cfg, error
    if "name" not in action:
        error = "missing_name_key"
        return tool_cfg, error
    if "arguments" not in action:
        error = "missing_arguments_key"
        return tool_cfg, error
    if not isinstance(action["arguments"], dict):
        error = "arguments_is_not_valid_dict"
        return tool_cfg, error
    tool_cfg = {
            "API_name": action["name"],
            "API_params": action["arguments"],
        }
    return tool_cfg, error

def is_image_param_key(key):
    key = str(key).lower()
    return any(token in key for token in ("image", "img", "path", "raster", "geotiff"))

def build_image_aliases(images, allow_numeric=False):
    aliases = {}
    for image_idx, image_path in enumerate(images or []):
        canonical = f"__image_{image_idx}__"
        basename = os.path.basename(str(image_path))
        stem, _ = os.path.splitext(basename)
        candidates = [
            str(image_path),
            basename,
            stem,
            f"img_{image_idx + 1}",
            f"img{image_idx + 1}",
            f"image_{image_idx + 1}",
            f"image{image_idx + 1}",
        ]
        if image_idx == 0:
            candidates.append("<image>")
        if allow_numeric:
            candidates.extend([str(image_idx), str(image_idx + 1)])
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate:
                aliases[candidate] = canonical
                aliases[candidate.lower()] = canonical
    return aliases

def normalize_image_ref(value, images, key=None):
    if not isinstance(value, str):
        return None
    aliases = build_image_aliases(images, allow_numeric=is_image_param_key(key))
    stripped = value.strip()
    return aliases.get(stripped) or aliases.get(stripped.lower())

def strip_wrapping_quotes(text):
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text

def normalize_param_value(value, images=None, key=None):
    if isinstance(value, dict):
        return {
            str(param_key): normalize_param_value(param_value, images, param_key)
            for param_key, param_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [normalize_param_value(item, images, key) for item in value]
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value

    text = strip_wrapping_quotes(value.strip())
    text = re.sub(r"\s+", " ", text).strip()

    image_ref = normalize_image_ref(text, images, key)
    if image_ref is not None:
        return image_ref

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None

    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        for parser in (json.loads, ast.literal_eval):
            try:
                return normalize_param_value(parser(text), images, key)
            except Exception:
                continue

    if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text):
        try:
            if any(char in text for char in (".", "e", "E")):
                return float(text)
            return int(text)
        except ValueError:
            return text

    return text

def normalized_values_equal(left, right):
    numeric_types = (int, float)
    if isinstance(left, numeric_types) and isinstance(right, numeric_types) and not isinstance(left, bool) and not isinstance(right, bool):
        return abs(float(left) - float(right)) <= 1e-6
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(normalized_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(normalized_values_equal(left_item, right_item) for left_item, right_item in zip(left, right))
    return left == right

def normalized_params_equal(gold_params, pred_params, images=None):
    gold_norm = normalize_param_value(gold_params, images)
    pred_norm = normalize_param_value(pred_params, images)
    return normalized_values_equal(gold_norm, pred_norm)

def get_all_cfg_from_conversation(conversation):
    tool_cfg = []
    for conv in conversation:
        if conv.get('from') == 'gpt':
            msg = conv.get('value')
            action = extract_action(msg)
            if action:
                cfg, _ = extract_tool_cfg(action[0])
                if cfg:
                    tool_cfg.append(cfg)
    return tool_cfg

def last_tool_in_gen(conversations):
    gen_tools = ["DrawBox", "AddText", "Plot", "DisplayOnGeotiff"]
    cfgs = get_all_cfg_from_conversation(conversations)
    tool_names = [cfg["API_name"] for cfg in cfgs if cfg["API_name"] != "Terminate"]
    if not tool_names:
        return False
    last_tool = tool_names[-1]
    return last_tool in gen_tools

EVAL_PROMPT = """
You are an evaluation assistant responsible for measuring the Answer Accuracy Score (a value between 0 and 1) for a geospatial agent’s prediction.

You are given:

- A Question  
- A Ground Truth Answer  
- A Predicted Answer produced by the agent  

Your task is to evaluate how accurately the Predicted Answer matches the Ground Truth Answer, considering both numerical precision and semantic correctness.  
When numerical values are present, allow a ±10% tolerance range for acceptable variation. For descriptive or categorical outputs, judge based on meaningful semantic equivalence.

Evaluation Guidelines

1. Numerical or Quantitative Comparisons  
- A predicted numeric value is correct if it lies within ±10% of the corresponding value in the ground truth.  
- For bounding boxes, consider the prediction correct if the IoU (Intersection over Union) ≥ 0.5.
- If multiple numeric values exist:  
    - Compute element-wise difference.
    - All values within ±10% range → Score = 0.95 – 1.00 (highly accurate)  
    - Most values within ±10% range → Score = 0.80 – 0.94 (minor acceptable deviations)  
    - Some values within ±10% range → Score = 0.30 – 0.79 (partially correct)  
    - Most values within ±20% range → Score = 0.10 – 0.29 (partially aligned)  
    - All values incorrect or nonsensical → Score = 0.00 (completely incorrect)  
- Ignore extraneous numbers that do not directly answer the question.

2. Non-Numerical or Categorical Comparisons  
    - Evaluate semantic equivalence:  
    - Perfect semantic match → Score ≈ 1.0  
    - Minor phrasing or synonym difference → 0.8–0.9  
    - Partially correct (missing non-critical details) → 0.4–0.8  
    - Vague or weakly related → 0.1–0.4  
    - Contradictory or incorrect → 0.0

3. Handling Missing or Contradictory Content  
- If the prediction omits all required answer elements → Score = 0.0
- If the prediction introduces fabricated entities, relationships, or placeholders → Score = 0.00
- If a numeric answer is required and the model responds with “no data”, “unknown”, or similar → Score = 0.00

You must provide your evaluation strictly in the following format and nothing else:
{
    "Score": <float between 0.0 and 1.0>,
    "Justification": "<1–2 concise sentences explaining the score based on the comparison between GT and Pred>"
}
"""
def extract_score(eval_text: str):
    before_just = eval_text.split("Justification")[0]
    match = re.search(r"[-+]?\d*\.\d+|\d+", before_just)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    return 0.0

def judge_chat_completion(messages, model=None, temperature=0, max_tokens=150):
    payload = {
        "model": model or JUDGE_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        JUDGE_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {judge_api_key}",
        },
        method="POST",
    )

    last_error = None

    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            break

        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")

            # 429 / 500 / 502 / 503 / 504 这类可以重试
            if error.code in {429, 500, 502, 503, 504}:
                last_error = RuntimeError(
                    f"Judge API request failed: HTTP {error.code}: {detail}"
                )
                time.sleep(5 * (attempt + 1))
                continue

            raise RuntimeError(
                f"Judge API request failed: HTTP {error.code}: {detail}"
            ) from error

        except (
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            TimeoutError,
            ConnectionResetError,
        ) as error:
            last_error = error
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"Judge API request failed after retries: {last_error}")

    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"Judge API returned unexpected response: {result}") from error

def sim_score(ans_gold: str,ans_pred: str, question: str):
    if not ans_pred:
        return 0.0
    
    prompt = f"{EVAL_PROMPT}\n\nQuestion: {question}\nGround Truth Answer: {ans_gold}\nPredicted Answer: {ans_pred}\n"

    evaluation = judge_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=JUDGE_MODEL,
        temperature=0,
        max_tokens=150
    ).strip()
    score = extract_score(evaluation)
    log_path = os.path.join(LOG_DIR, f"LLM_scores.jsonl")
    entry = {
        "question": question,
        "ground_truth": ans_gold,
        "predicted": ans_pred,
        "evaluation": evaluation,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return score
    

def gettype(name: str):
    perception = ['BaseModel', 'ImageDescription', 'RegionAttributeDescription', 'TextToBbox', 'ChangeDetection', 'ObjectDetection', 'SmallObjectDetection', 'DenoDetV2', 'CrossModalInterpret', 'SegmentObjectPixels', 'CountGivenObject']
    operation = ['DrawBox', 'AddText', 'GoogleSearch']
    logic = ['Calculator', 'Solver', 'Plot']  
    gis = ["GetAreaBoundary", "AddPoisLayer", "ComputeDistance", 'GetBboxFromGeotiff', "DisplayOnGeotiff"]
    terminate = ["Terminate"]
    if name in perception:
        return 'perception'
    elif name in operation:
        return 'operation'
    elif name in logic:
        return 'logic'
    elif name in gis:
        return 'gis'
    elif name in terminate:
        return 'terminate'
    else:
        return 'none'

def load_dataset(file_path, num_samples=None):
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["idx"] = item.get("idx", line_idx)
            dataset.append(item)
            if num_samples and len(dataset) >= num_samples:
                break
    return Dataset.from_list(dataset)

def swift_messages_to_conversation(messages):
    conversation = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            continue
        if role == "user":
            conversation.append({"from": "human", "value": content})
        elif role == "assistant":
            conversation.append({"from": "gpt", "value": content})
    return conversation

def clean_first_question(text):
    return text.replace("<image>", "").strip()

def extract_question_text(conversation):
    for entry in conversation:
        if entry.get("from") == "human":
            text = entry.get("value", "")
            if "\n\nQuestion:" in text:
                text = text.split("\n\nQuestion:", 1)[1]
            return clean_first_question(text)
    return ""

def load_data_function():
    dataset_path = task_config["dataset_path"]
    num_samples = task_config["num_sample"]
    task_type = task_config["task_type"]
    dataset = load_dataset(dataset_path, num_samples)

    meta_data = []
    step_idx = 0
    for index, item in enumerate(dataset):
        idx_val = item.get("idx", index)
        idx = f"rsagent_{index}"
        images = item.get("images", [])
        conversation = swift_messages_to_conversation(item.get("messages", []))

        if not conversation:
            logger.warning(f"Skip empty conversation for idx={idx_val}")
            continue

        if task_type == "e2e":
            text = extract_question_text(conversation)
            label = item.get("label", None)
            if label is None:
                try:
                    last_value = ""
                    for entry in reversed(conversation):
                        if entry["from"] == "gpt":
                            last_value = entry.get("value", "")
                            break
                    label = extract_final_answer(last_value)
                except Exception as e:
                    logger.warning(f"Failed to extract label for idx={idx_val}: {e}")
                    label = ""
            data_item = dict(
                idx=idx,
                sample_num=idx_val,
                text=text,
                label=label,
                images=images,
                conversation=conversation,
            )
            meta_data.append(data_item)

        elif task_type == "step":
            conv_num = 0
            for i, entry in enumerate(conversation):
                if entry["from"] != "human":
                    continue
                if i + 1 >= len(conversation) or conversation[i + 1]["from"] != "gpt":
                    logger.warning(f"Skip unmatched human turn for idx={idx_val}, turn={i}")
                    continue

                text = conversation[: i + 1]
                label = conversation[i + 1]["value"]
                conv_num += 1
                step_idx += 1

                data_item = dict(
                    idx=f"rsagent_{step_idx}",
                    sample_num=idx_val,
                    conv_num=conv_num,
                    text=text,
                    label=label,
                    images=images,
                    conversation=conversation,
                )
                meta_data.append(data_item)
        else:
            raise ValueError(f"Unsupported task type: '{task_type}'")

    ## Show statistics
    logger.info(f"Total data number: {len(meta_data)}")
    return meta_data

def compare_tool_sequences(API_names_gt, API_names_pred):
    """
    Compare predicted vs ground-truth tool call sequences.

    Returns:
      same_order: GT sequence appears as subsequence in pred (extras allowed)
      any_order:  All GT tools appear in pred with >= same counts
      unique:     All unique GT tools appear in pred (order ignored)
    """
    # --- multiset inclusion (any order, with counts) ---
    gt_counter, pred_counter = Counter(API_names_gt), Counter(API_names_pred)
    any_order = all(pred_counter[k] >= v for k, v in gt_counter.items())

    # --- order check (GT as subsequence of pred) ---
    same_order = False
    if any_order:  # Only check order if all GT tools appear
        i = 0
        for name in API_names_pred:
            if i < len(API_names_gt) and name == API_names_gt[i]:
                i += 1
        same_order = i == len(API_names_gt)

    # --- unique version (ignoring repeats) ---
    def unique(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    gt_unique, pred_unique = unique(API_names_gt), unique(API_names_pred)
    unique_ok = all(t in pred_unique for t in gt_unique)

    return same_order, any_order, unique_ok


def safe_rate(numerator, denominator, scale=1.0):
    return (numerator * scale / denominator) if denominator else 0.0


def round_float(value, digits=4):
    return round(float(value), digits)


def extract_pred_tool_steps(tool_cfg_pred):
    steps = []
    for cfg in tool_cfg_pred or []:
        if not cfg or not isinstance(cfg, list) or not isinstance(cfg[0], dict):
            continue
        item = cfg[0]
        name = item.get("API_name", item.get("api_name"))
        if not name:
            continue
        params = item.get("API_params", item.get("api_params", {}))
        steps.append({"API_name": name, "API_params": params if isinstance(params, dict) else {}})
    return steps


def non_terminal_steps(steps):
    return [step for step in steps if isinstance(step, dict) and step.get("API_name") != "Terminate"]


def classify_task_families(api_names_gt, gen_tools):
    names = set(api_names_gt)
    families = []
    if {"ObjectDetection", "DenoDetV2", "CrossModalInterpret"}.issubset(names):
        families.append("cross_modal")
    if "ChangeDetection" in names:
        families.append("change_detection")
    if names.intersection({"GetBboxFromGeotiff", "GetAreaBoundary", "AddPoisLayer", "ComputeDistance", "DisplayOnGeotiff"}):
        families.append("gis")
    if "SegmentObjectPixels" in names:
        families.append("segmentation_area")
    if "TextToBbox" in names and names.intersection({"Calculator", "Solver"}):
        families.append("measurement_math")
    if "RegionAttributeDescription" in names:
        families.append("attribute_reasoning")
    if "CountGivenObject" in names:
        families.append("counting")
    if len(api_names_gt) >= 2 and api_names_gt[-2] in gen_tools:
        families.append("visual_generation")
    return families or ["general"]


def init_bucket_stats():
    return {
        "sample_count": 0,
        "answer_score_sum": 0.0,
        "strict_tool_sequence_count": 0,
        "gt_sequence_subsequence_count": 0,
        "tool_bag_recall_count": 0,
        "unique_tool_coverage_count": 0,
        "arg_key_match_sum": 0.0,
        "arg_exact_match_sum": 0.0,
        "arg_normalized_match_sum": 0.0,
        "arg_comparable_samples": 0,
        "tool_errors": 0,
        "pred_tool_calls": 0,
        "steps_taken_sum": 0,
        "terminate_count": 0,
        "empty_answer_count": 0,
    }


def update_bucket_stats(bucket, sample_stats):
    bucket["sample_count"] += 1
    bucket["answer_score_sum"] += sample_stats.get("answer_score", 0.0)
    bucket["strict_tool_sequence_count"] += int(sample_stats.get("strict_tool_sequence", False))
    bucket["gt_sequence_subsequence_count"] += int(sample_stats.get("gt_sequence_subsequence", False))
    bucket["tool_bag_recall_count"] += int(sample_stats.get("tool_bag_recall", False))
    bucket["unique_tool_coverage_count"] += int(sample_stats.get("unique_tool_coverage", False))
    if sample_stats.get("arg_comparable"):
        bucket["arg_key_match_sum"] += sample_stats.get("arg_key_match", 0.0)
        bucket["arg_exact_match_sum"] += sample_stats.get("arg_exact_match", 0.0)
        bucket["arg_normalized_match_sum"] += sample_stats.get("arg_normalized_match", 0.0)
        bucket["arg_comparable_samples"] += 1
    bucket["tool_errors"] += sample_stats.get("tool_errors", 0)
    bucket["pred_tool_calls"] += sample_stats.get("pred_tool_calls", 0)
    bucket["steps_taken_sum"] += sample_stats.get("steps_taken", 0)
    bucket["terminate_count"] += int(sample_stats.get("terminated", False))
    bucket["empty_answer_count"] += int(sample_stats.get("empty_answer", False))


def finalize_bucket_stats(bucket):
    count = bucket["sample_count"]
    arg_count = bucket["arg_comparable_samples"]
    return {
        "sample_count": count,
        "answer_acc": round_float(safe_rate(bucket["answer_score_sum"], count, 100)),
        "strict_tool_sequence_rate": round_float(safe_rate(bucket["strict_tool_sequence_count"], count, 100)),
        "gt_sequence_subsequence_rate": round_float(safe_rate(bucket["gt_sequence_subsequence_count"], count, 100)),
        "tool_bag_recall": round_float(safe_rate(bucket["tool_bag_recall_count"], count, 100)),
        "unique_tool_coverage": round_float(safe_rate(bucket["unique_tool_coverage_count"], count, 100)),
        "arg_key_match_rate": round_float(safe_rate(bucket["arg_key_match_sum"], arg_count, 100)),
        "arg_exact_match_rate": round_float(safe_rate(bucket["arg_exact_match_sum"], arg_count, 100)),
        "arg_normalized_match_rate": round_float(safe_rate(bucket["arg_normalized_match_sum"], arg_count, 100)),
        "arg_comparable_samples": arg_count,
        "tool_error_rate": round_float(safe_rate(bucket["tool_errors"], bucket["pred_tool_calls"], 100)),
        "avg_steps_taken": round_float(safe_rate(bucket["steps_taken_sum"], count)),
        "terminate_rate": round_float(safe_rate(bucket["terminate_count"], count, 100)),
        "empty_answer_rate": round_float(safe_rate(bucket["empty_answer_count"], count, 100)),
    }


def compute_argument_match_stats(gt_steps, pred_steps, images=None):
    gt_steps = non_terminal_steps(gt_steps)
    pred_steps = non_terminal_steps(pred_steps)
    comparable = min(len(gt_steps), len(pred_steps))
    if comparable == 0:
        return {
            "arg_comparable": False,
            "arg_key_match": 0.0,
            "arg_exact_match": 0.0,
            "arg_normalized_match": 0.0,
        }
    key_matches = 0
    exact_matches = 0
    normalized_matches = 0
    for idx in range(comparable):
        gt_step = gt_steps[idx]
        pred_step = pred_steps[idx]
        if gt_step.get("API_name") != pred_step.get("API_name"):
            continue
        gt_params = gt_step.get("API_params") or {}
        pred_params = pred_step.get("API_params") or {}
        if set(gt_params.keys()) == set(pred_params.keys()):
            key_matches += 1
        if gt_params == pred_params:
            exact_matches += 1
        if normalized_params_equal(gt_params, pred_params, images or []):
            normalized_matches += 1
    return {
        "arg_comparable": True,
        "arg_key_match": key_matches / comparable,
        "arg_exact_match": exact_matches / comparable,
        "arg_normalized_match": normalized_matches / comparable,
    }


def compute_per_tool_f1(total_gt_calls_per_tool, total_pred_calls_per_tool, correct_by_tool=None):
    correct_by_tool = correct_by_tool or {}
    per_tool = {}
    for tool_name in total_gt_calls_per_tool:
        gt_count = total_gt_calls_per_tool.get(tool_name, 0)
        pred_called = total_pred_calls_per_tool.get(tool_name, {}).get("called", 0)
        correct = correct_by_tool.get(tool_name, min(gt_count, pred_called))
        precision = safe_rate(correct, pred_called, 100)
        recall = safe_rate(correct, gt_count, 100)
        f1 = safe_rate(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        per_tool[tool_name] = {
            "gt_calls": gt_count,
            "pred_calls": pred_called,
            "precision": round_float(precision),
            "recall": round_float(recall),
            "f1": round_float(f1),
        }
    return per_tool

def evaluate_function(results,meta_data):

    results_dict = {res["idx"]: res for res in results}
    meta_dict = {meta["idx"]: meta for meta in meta_data}
    task_type = task_config['task_type'] 
    metrics = {}
    error_breakdown = {}

    if task_type == "e2e":
        metrics = {
            'perception_f1': 0, 
            'operation_f1': 0, 
            'logic_f1': 0,
            'gis_f1' : 0,
            'total_tool_calls_gt' : 0,
            'total_actions_parsed': 0,
            'total_tool_calls_pred' : 0,
            'tool_call_error': 0,
            'answer_acc' : 0,
            'answer_acc_w_gen' : 0,
            'all_tools_called_any_order' : 0,  # preserves multiple instances of tool calls
            'all_tools_called_same_order' : 0,
            'all_tools_called_unique' : 0,     # check unique tool calls only 
            'avg_steps_taken': 0,
            'total_gen_samples': 0,
            'num_total_samples': len(meta_dict),
            'num_predictions': len(results_dict),
            'num_missing': max(len(meta_dict) - len(results_dict), 0),
            'num_empty_final_answer': 0,
            'num_finish_reason_terminate': 0,
            'num_finish_reason_max_rounds': 0,
            'num_action_parse_warnings': 0,
            'num_multiple_actions_recovered': 0
        }
        error_breakdown = {
            'ans_not_reached' : 0,
            'name_not_in_tools': 0,
            'no_valid_action_taken' : 0,
            'no_action_found': 0,
            'action_found_but_wrong_syntax': 0,
            'answer_reached_without_tool_calls' : 0,
            'Multiple_tool_calls_in_single_step' : 0
        }

        gen_tools = [
            'DrawBox', 
            'AddText', 
            'Plot', 
            "DisplayOnGeotiff"
            ]

        total_gt_calls_per_tool = { tool: 0 for tool in TOOLS_LIST}
        total_pred_calls_per_tool = {tool: {"called": 0, "success": 0} for tool in TOOLS_LIST}
        task_family_buckets = defaultdict(init_bucket_stats)
        image_split_buckets = defaultdict(init_bucket_stats)
        task_family_distribution = Counter()
        image_count_distribution = Counter()
        per_tool_correct = Counter()
        total_parse_errors = 0
        total_action_parse_entries = 0
        final_outcome_score_sum = 0.0
        strict_tool_sequence_count = 0
        arg_key_match_sum = 0.0
        arg_exact_match_sum = 0.0
        arg_normalized_match_sum = 0.0
        arg_comparable_samples = 0
        executed_tool_response_count = 0

        total = {'all': 0, 'all_gen': 0, 'answer': 0, 'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        total_predict = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        correct_predict = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        precision = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        recall = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}
        f1 = {'perception': 0, 'operation': 0, 'logic': 0, 'creativity': 0, 'gis': 0, 'terminate': 0}

        for idx, meta in meta_dict.items():
            result_item = results_dict.get(idx)
            result_payload = result_item.get("results", {}) if result_item else {}
            tool_cfg_pred = result_payload.get("tool_cfg") or []
            tool_response = result_payload.get("tool_response") or []
            model_response = result_payload.get("model_response") or []
            action_parse_error = result_payload.get("action_parse_error") or []
            action_parse_warning = result_payload.get("action_parse_warning") or []
            finish_reason = None

            if result_payload:
                final_ans = result_payload.get("final_answer")
                meta["prediction"] = final_ans if final_ans else None
                finish_reason = result_payload.get("finish_reason")
                if finish_reason is None:
                    if result_payload.get("current_round") == result_payload.get("max_rounds") and not meta["prediction"]:
                        finish_reason = "max_rounds"
                    elif meta["prediction"]:
                        finish_reason = "terminate"
                if finish_reason == "terminate":
                    metrics["num_finish_reason_terminate"] += 1
                elif finish_reason == "max_rounds":
                    metrics["num_finish_reason_max_rounds"] += 1
                if not meta["prediction"]:
                    metrics["num_empty_final_answer"] += 1
                    error_breakdown['ans_not_reached'] +=1
            else:
                logger.info(f"no prediction found for sample number {idx}")
                meta["prediction"] = None
                error_breakdown['ans_not_reached'] +=1

            gold = meta["label"]
            pred = meta["prediction"]
            question = extract_question_text(meta["conversation"])

            metrics["avg_steps_taken"] += result_payload.get("current_round", 0)
            tool_cfg_gt = get_all_cfg_from_conversation(meta["conversation"])
            pred_steps = extract_pred_tool_steps(tool_cfg_pred)

            API_names_gt = [cfg["API_name"] for cfg in tool_cfg_gt if cfg is not None]
            API_names_pred = [step["API_name"] for step in pred_steps]
            gt_no_terminate = [name for name in API_names_gt if name != "Terminate"]
            pred_no_terminate = [name for name in API_names_pred if name != "Terminate"]
            strict_tool_sequence = gt_no_terminate == pred_no_terminate
            if strict_tool_sequence:
                strict_tool_sequence_count += 1
            argument_stats = compute_argument_match_stats(tool_cfg_gt, pred_steps, images=meta.get("images", []))
            if argument_stats["arg_comparable"]:
                arg_key_match_sum += argument_stats["arg_key_match"]
                arg_exact_match_sum += argument_stats["arg_exact_match"]
                arg_normalized_match_sum += argument_stats["arg_normalized_match"]
                arg_comparable_samples += 1
            for gt_name, pred_name in zip(gt_no_terminate, pred_no_terminate):
                if gt_name == pred_name:
                    per_tool_correct[gt_name] += 1
            image_count = len(meta.get("images", []) or [])
            image_count_distribution[image_count] += 1

            error_codes = [res.get('error_code', 1) if isinstance(res, dict) else 1 for res in tool_response if res]
            executed_tool_response_count += len(error_codes)
            sample_answer_score = 0.0

            if len(API_names_gt) >= 2 and API_names_gt[-2] in gen_tools: #if last tool before terminate is generation tool then calc gen_acc
                total['all_gen'] += 1
                # same last tool call and tool call was successful
                last_tool_ok = bool(tool_response and isinstance(tool_response[-1], dict) and tool_response[-1].get('error_code') == 0)
                if len(API_names_pred) > 0 and API_names_gt[-2] == API_names_pred[-1] and last_tool_ok:
                    sample_answer_score = 1.0
                    metrics["answer_acc_w_gen"] += sample_answer_score
            else:
                total['all'] += 1
                sample_answer_score = sim_score(gold, pred, question)
                metrics["answer_acc"] += sample_answer_score
            final_outcome_score_sum += sample_answer_score

            metrics['total_tool_calls_gt'] += len(API_names_gt)
            metrics['total_tool_calls_pred'] += len(API_names_pred)
            same_order, any_order, unique_ok= compare_tool_sequences(API_names_gt[:-1], API_names_pred)
            if same_order:
                metrics['all_tools_called_same_order'] += 1
            if any_order:
                metrics['all_tools_called_any_order'] += 1
            if unique_ok:  
                metrics['all_tools_called_unique'] += 1

            for name in API_names_gt:
                if name in total_gt_calls_per_tool:
                    total_gt_calls_per_tool[name] += 1
            for name in API_names_pred:
                if name in total_pred_calls_per_tool:
                    total_pred_calls_per_tool[name]["called"]  += 1
            if meta["prediction"] and "Terminate" in total_pred_calls_per_tool:
                total_pred_calls_per_tool["Terminate"]["called"]  += 1

            if len(API_names_pred) == len(error_codes):
                for name, code in zip(API_names_pred, error_codes):
                    if name in total_pred_calls_per_tool and code == 0:
                        total_pred_calls_per_tool[name]["success"] += 1

            metrics['tool_call_error'] += sum(code != 0 for code in error_codes) 
            for name in API_names_pred:
                tool_type = gettype(name) 
                if tool_type != "none":
                    total_predict[tool_type] += 1
                else:
                    error_breakdown['name_not_in_tools'] += 1 
            API_names_pred_copy = API_names_pred.copy()
            for name in API_names_gt:
                total[gettype(name)] += 1
                if name in API_names_pred_copy:  # any order
                    API_names_pred_copy.remove(name)
                    correct_predict[gettype(name)] += 1
            
            # model should call tool in every response except for first response (can be thought only)
            error_breakdown['no_valid_action_taken'] += max(len(model_response)-len(tool_cfg_pred) - 1, 0)
            if meta["prediction"]: # if ans is reached
                if all(x is None for x in tool_cfg_pred): # no valid tool calls 
                    error_breakdown['answer_reached_without_tool_calls'] += 1

            for response in model_response:
                actions = extract_action(response)
                if actions and len(actions)>1:
                    error_breakdown['Multiple_tool_calls_in_single_step'] += 1

            for entry in action_parse_error:
                total_action_parse_entries += 1
                if not entry:
                    metrics["total_actions_parsed"] += 1
                else:
                    total_parse_errors += 1
                    if "Invalid action format." in entry:
                        error_breakdown['action_found_but_wrong_syntax'] += 1
                    elif "No action found." in entry:
                        error_breakdown['no_action_found'] += 1
                    elif "Multiple_tool_calls_in_single_step" in entry:
                        error_breakdown['Multiple_tool_calls_in_single_step'] += 1
                    else:
                        logger.info(f"found unknown error: {entry}")

            for entry in action_parse_warning:
                if not entry:
                    continue
                metrics["num_action_parse_warnings"] += 1
                if "Multiple_tool_calls_in_single_step" in entry:
                    metrics["num_multiple_actions_recovered"] += 1

            sample_stats = {
                "answer_score": sample_answer_score,
                "strict_tool_sequence": strict_tool_sequence,
                "gt_sequence_subsequence": same_order,
                "tool_bag_recall": any_order,
                "unique_tool_coverage": unique_ok,
                "arg_comparable": argument_stats["arg_comparable"],
                "arg_key_match": argument_stats["arg_key_match"],
                "arg_exact_match": argument_stats["arg_exact_match"],
                "arg_normalized_match": argument_stats["arg_normalized_match"],
                "tool_errors": sum(code != 0 for code in error_codes),
                "pred_tool_calls": len(error_codes),
                "steps_taken": result_payload.get("current_round", 0),
                "terminated": finish_reason == "terminate",
                "empty_answer": not bool(meta["prediction"]),
            }
            for family in classify_task_families(API_names_gt, gen_tools):
                task_family_distribution[family] += 1
                update_bucket_stats(task_family_buckets[family], sample_stats)
            image_split = "dual_image" if image_count >= 2 else "single_image"
            update_bucket_stats(image_split_buckets[image_split], sample_stats)

        for tool_type in f1.keys():
            precision[tool_type] = correct_predict[tool_type] / (total_predict[tool_type] + 1e-5)
            recall[tool_type]    = correct_predict[tool_type] / (total[tool_type] + 1e-5)
            f1[tool_type] = 2 * precision[tool_type] * recall[tool_type] / (precision[tool_type] + recall[tool_type] + 1e-5)

        metrics["answer_acc"] = round(metrics["answer_acc"] *100/ total['all'], 4) if total['all'] else 0.0
        metrics["answer_acc_w_gen"] = round(metrics["answer_acc_w_gen"] *100 / total['all_gen'], 4) if total['all_gen'] else 0.0
        metrics["perception_f1"] = f1['perception'] * 100
        metrics["operation_f1"] = f1['operation'] * 100
        metrics["logic_f1"] = f1['logic'] * 100
        metrics["gis_f1"] = f1['gis'] * 100
        metrics["avg_steps_taken"] = metrics["avg_steps_taken"] / (total['all'] + total['all_gen']) if (total['all'] + total['all_gen']) else 0.0
        metrics["total_gen_samples"] = total['all_gen']
        metrics['all_tools_called_same_order'] /= (total['all'] + total['all_gen']) if (total['all'] + total['all_gen']) else 0.0
        metrics['all_tools_called_any_order'] /= (total['all'] + total['all_gen']) if (total['all'] + total['all_gen']) else 0.0
        metrics['all_tools_called_unique'] /= (total['all'] + total['all_gen']) if (total['all'] + total['all_gen']) else 0.0
        sample_total = total['all'] + total['all_gen']
        final_outcome_score = safe_rate(final_outcome_score_sum, sample_total, 100)
        strict_tool_sequence_rate = safe_rate(strict_tool_sequence_count, sample_total, 100)
        arg_key_match_rate = safe_rate(arg_key_match_sum, arg_comparable_samples, 100)
        arg_exact_match_rate = safe_rate(arg_exact_match_sum, arg_comparable_samples, 100)
        arg_normalized_match_rate = safe_rate(arg_normalized_match_sum, arg_comparable_samples, 100)
        tool_error_rate = safe_rate(metrics['tool_call_error'], executed_tool_response_count, 100)
        tool_success_rate = 100 - tool_error_rate if executed_tool_response_count else 0.0
        terminate_rate = safe_rate(metrics['num_finish_reason_terminate'], sample_total, 100)
        empty_answer_rate = safe_rate(metrics['num_empty_final_answer'], sample_total, 100)
        max_rounds_rate = safe_rate(metrics['num_finish_reason_max_rounds'], sample_total, 100)
        parse_error_rate = safe_rate(total_parse_errors, total_action_parse_entries, 100)
        robustness_score = (
            terminate_rate +
            (100 - empty_answer_rate) +
            (100 - parse_error_rate) +
            (100 - max_rounds_rate)
        ) / 4 if sample_total else 0.0
        overall_score = (
            0.30 * final_outcome_score +
            0.25 * strict_tool_sequence_rate +
            0.20 * arg_normalized_match_rate +
            0.15 * tool_success_rate +
            0.10 * robustness_score
        )

        metrics["final_outcome_score"] = round_float(final_outcome_score)
        metrics["strict_tool_sequence_rate"] = round_float(strict_tool_sequence_rate)
        metrics["argument_normalized_match_rate"] = round_float(arg_normalized_match_rate)
        metrics["tool_error_rate"] = round_float(tool_error_rate)
        metrics["tool_success_rate"] = round_float(tool_success_rate)
        metrics["terminate_rate"] = round_float(terminate_rate)
        metrics["empty_answer_rate"] = round_float(empty_answer_rate)
        metrics["parse_error_rate"] = round_float(parse_error_rate)
        metrics["overall_score"] = round_float(overall_score)

        task_family_metrics = {
            family: finalize_bucket_stats(bucket)
            for family, bucket in sorted(task_family_buckets.items())
        }
        image_split_metrics = {
            split: finalize_bucket_stats(bucket)
            for split, bucket in sorted(image_split_buckets.items())
        }
        family_scores = {}
        for family, family_metric in task_family_metrics.items():
            family_scores[family] = round_float(
                0.35 * family_metric["answer_acc"] +
                0.25 * family_metric["strict_tool_sequence_rate"] +
                0.25 * family_metric["arg_normalized_match_rate"] +
                0.15 * (100 - family_metric["tool_error_rate"])
            )
        macro_family_score = round_float(safe_rate(sum(family_scores.values()), len(family_scores)))
        metrics["macro_family_score"] = macro_family_score

        tool_success_rate_by_tool = {}
        for tool_name, pred_stats in total_pred_calls_per_tool.items():
            if tool_name == "Terminate":
                continue
            called = pred_stats.get("called", 0)
            success = pred_stats.get("success", 0)
            tool_success_rate_by_tool[tool_name] = {
                "called": called,
                "success": success,
                "success_rate": round_float(safe_rate(success, called, 100)),
            }

        dataset_profile = {
            "num_samples": sample_total,
            "image_count_distribution": dict(sorted(image_count_distribution.items())),
            "task_family_distribution": dict(sorted(task_family_distribution.items())),
            "gt_total_tool_calls": metrics["total_tool_calls_gt"],
            "gt_tool_calls_per_tool": total_gt_calls_per_tool,
        }
        headline_metrics = {
            "overall_score": round_float(overall_score),
            "final_outcome_score": round_float(final_outcome_score),
            "answer_acc": metrics["answer_acc"],
            "artifact_success_rate": metrics["answer_acc_w_gen"],
            "strict_tool_sequence_rate": round_float(strict_tool_sequence_rate),
            "argument_normalized_match_rate": round_float(arg_normalized_match_rate),
            "tool_success_rate": round_float(tool_success_rate),
            "macro_family_score": macro_family_score,
            "terminate_rate": round_float(terminate_rate),
            "avg_steps_taken": round_float(metrics["avg_steps_taken"]),
        }
        tool_planning = {
            "strict_tool_sequence_rate": round_float(strict_tool_sequence_rate),
            "gt_sequence_subsequence_rate": round_float(metrics['all_tools_called_same_order'] * 100),
            "tool_bag_recall": round_float(metrics['all_tools_called_any_order'] * 100),
            "unique_tool_coverage": round_float(metrics['all_tools_called_unique'] * 100),
            "per_tool_f1": compute_per_tool_f1(total_gt_calls_per_tool, total_pred_calls_per_tool, per_tool_correct),
        }
        argument_grounding = {
            "arg_comparable_samples": arg_comparable_samples,
            "arg_key_match_rate": round_float(arg_key_match_rate),
            "arg_exact_match_rate": round_float(arg_exact_match_rate),
            "arg_normalized_match_rate": round_float(arg_normalized_match_rate),
        }
        tool_execution = {
            "executed_tool_response_count": executed_tool_response_count,
            "tool_error_count": metrics['tool_call_error'],
            "tool_error_rate": round_float(tool_error_rate),
            "tool_success_rate": round_float(tool_success_rate),
            "tool_success_rate_by_tool": tool_success_rate_by_tool,
        }
        robustness = {
            "terminate_rate": round_float(terminate_rate),
            "empty_answer_rate": round_float(empty_answer_rate),
            "parse_error_rate": round_float(parse_error_rate),
            "max_rounds_rate": round_float(max_rounds_rate),
            "avg_steps_taken": round_float(metrics["avg_steps_taken"]),
        }
        return {
            "metrics": metrics,
            "headline_metrics": headline_metrics,
            "dataset_profile": dataset_profile,
            "tool_planning": tool_planning,
            "argument_grounding": argument_grounding,
            "tool_execution": tool_execution,
            "task_family_metrics": task_family_metrics,
            "task_family_scores": family_scores,
            "image_split_metrics": image_split_metrics,
            "robustness": robustness,
            "error_breakdown": error_breakdown,
            "total_gt_calls_per_tool": total_gt_calls_per_tool,
            "total_pred_calls_per_tool": total_pred_calls_per_tool,
        }
    
    elif task_type == "step":
        metrics = {
                'inst_acc': 0,
                'tool_acc': 0,
                'arg_acc1' : 0,
                'arg_acc_strict': 0,
                'arg_acc2': 0,
                'answer_acc': 0,
                'count_non_gen_samples': 0,
                'count_gen_samples': 0
            }
        error_breakdown = {
            "no_valid_action" : 0,
            "action_is_not_valid_dict" : 0,
            "missing_name_key" : 0,
            "name_not_in_tools": 0,
            "incorrect_tool_called" : 0,
            "missing_arguments_key" : 0,
            "required_argument_missing" : 0,
            "arguments_is_not_valid_dict": 0,
            "argument_values_not_matching": 0,
            "argument_values_match_after_normalization": 0
        }
        total_calls = 0
        count_non_gen_samples = 0
        count_gen_samples = 0
        unique_samples = len(set(meta.get("sample_num") for meta in meta_data if "sample_num" in meta))
        logger.info(f"Total unique samples: {unique_samples}")

        for idx, meta in meta_dict.items():
            if idx in results_dict:
                model_output = results_dict[idx]["results"]["model_output"]
                meta["prediction"] = model_output[0] if isinstance(model_output, list) else model_output
                actions= extract_action(meta["prediction"])
                if actions:
                    action = actions[0] 
                    tool_cfg_pred, error = extract_tool_cfg(action)
                else:
                    error = "no_valid_action"
                    tool_cfg_pred = None
            else:
                logger.info(f"no prediction found for sample number {idx}")
                meta["prediction"] = None

            actions= extract_action(meta["label"])
            if actions:
                action = actions[0] 
                tool_cfg_gold, _ = extract_tool_cfg(action)
            else:
                tool_cfg_gold = None
  
            if tool_cfg_gold: # supposed to call tool
                total_calls += 1
                if tool_cfg_pred:  # and tool was called
                    metrics["inst_acc"] += 1
                    if tool_cfg_pred["API_name"] not in TOOLS_LIST:
                        error_breakdown['name_not_in_tools'] += 1
                    if tool_cfg_gold["API_name"] == tool_cfg_pred["API_name"]: #correct tool was called
                        metrics["tool_acc"] += 1
                        if tool_cfg_gold["API_name"] == "Terminate":
                            ans_gold = tool_cfg_gold["API_params"].get('ans')
                            ans_pred = tool_cfg_pred["API_params"].get('ans',"")
                            question = extract_question_text(meta["conversation"])
                            #only for non gen samples
                            if not last_tool_in_gen(meta["conversation"]):
                                count_non_gen_samples +=1
                                metrics['answer_acc'] += sim_score(ans_gold,ans_pred,question)
                            else: 
                                count_gen_samples += 1  
                        if tool_cfg_gold["API_params"].keys() == tool_cfg_pred["API_params"].keys(): # correct arguments were passed
                            metrics["arg_acc1"] += 1
                            strict_params_match = tool_cfg_gold["API_params"] == tool_cfg_pred["API_params"]
                            normalized_params_match = normalized_params_equal(
                                tool_cfg_gold["API_params"],
                                tool_cfg_pred["API_params"],
                                meta.get("images", []),
                            )
                            if strict_params_match:
                                metrics["arg_acc_strict"] += 1
                            if normalized_params_match: # matching argument values after normalization
                                metrics["arg_acc2"] += 1
                                if not strict_params_match:
                                    error_breakdown['argument_values_match_after_normalization'] += 1
                            else:
                                error_breakdown['argument_values_not_matching'] += 1
                        else: 
                            error_breakdown['required_argument_missing'] += 1
                    else:
                        error_breakdown['incorrect_tool_called'] += 1
                else:
                    error_breakdown[error] += 1
             
        # normalize metrics            
        if total_calls > 0:
            for key in ["inst_acc", "tool_acc", "arg_acc1", "arg_acc_strict", "arg_acc2"]:
                metrics[key] /= total_calls
                metrics[key] *= 100
        if count_non_gen_samples > 0:
            metrics["answer_acc"] = (metrics["answer_acc"] /count_non_gen_samples) *100
        metrics["count_non_gen_samples"] = count_non_gen_samples
        metrics["count_gen_samples"] = count_gen_samples
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        logger.info(f"Metrics: {metrics}")
        logger.info(f"Error breakdown: {error_breakdown}")
        return {"metrics":metrics, "error_breakdown": error_breakdown}
    else:
        raise ValueError(f"Unsupported task type: '{task_type}'")
    
