from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from tool_server.tf_eval.agent_runtime import AgentRunConfig, AgentRunner, RunMode
from tool_server.tf_eval.agent_runtime.input_paths import (
    preflight_validate_samples,
    resolve_input_path,
    resolve_registry_paths,
    resolve_sample_images,
)
from tool_server.tf_eval.agent_runtime.trace_schema import create_agent_trace
from tool_server.tf_eval.skill_eval.answer_scoring import answer_score
from tool_server.tf_eval.skill_eval.error_attribution import attribute_trace_errors
from tool_server.tf_eval.skill_eval.report import generate_report
from tool_server.tool_workers.tool_manager.base_manager import ToolManager


def load_task_config_from_current_dir(task_file_path: str) -> Dict[str, Any]:
    import yaml

    current_dir = Path(task_file_path).resolve().parent
    config_path = current_dir / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    dataset_path = config.get("dataset_path")
    if dataset_path and not os.path.isabs(str(dataset_path)):
        config["dataset_path"] = str(current_dir / str(dataset_path))
    return config


def write_json_file(data: Dict[str, Any], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)


task_config = load_task_config_from_current_dir(__file__)

RUN_TRACE_FILE = "run_traces.jsonl"
METRICS_FILE = "metrics_summary.json"
ERROR_FILE = "error_breakdown.json"


def _config_get(name: str, default: Any = None) -> Any:
    try:
        return task_config.get(name, default)
    except Exception:
        return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def resolve_dataset_asset(
    path: Any,
    dataset_file: Any = None,
    image_root: Any = None,
    dataset_root: Any = None,
) -> str:
    raw = str(path)
    if not raw.strip():
        return raw
    try:
        return resolve_input_path(raw, dataset_file=dataset_file, image_root=image_root, dataset_root=dataset_root)
    except Exception:
        pass

    asset_path = Path(raw).expanduser()
    if asset_path.is_absolute():
        return str(asset_path.resolve()) if asset_path.exists() else raw

    repo_root = Path(__file__).resolve().parents[4]
    candidate_roots = [
        os.environ.get("SKILL_EVAL_DATA_ROOT"),
        os.environ.get("RS_DATA_ROOT"),
        Path.cwd(),
        repo_root,
    ]
    seen = set()
    for root in candidate_roots:
        if not root:
            continue
        root_path = Path(root).expanduser()
        key = str(root_path)
        if key in seen:
            continue
        seen.add(key)
        candidate = root_path / asset_path
        if candidate.exists():
            return str(candidate.resolve())
    return raw


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item.get("text")))
                elif item.get("content"):
                    parts.append(str(item.get("content")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content or "")


def clean_question_text(text: str) -> str:
    text = (text or "").replace("<image>", "").strip()
    if "\n\nQuestion:" in text:
        text = text.split("\n\nQuestion:", 1)[1]
    elif "Question:" in text:
        text = text.split("Question:", 1)[1]
    return text.strip()


def extract_question(sample: Dict[str, Any]) -> str:
    for message in sample.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from")
        if role in {"user", "human"}:
            return clean_question_text(_content_to_text(message.get("content", message.get("value", ""))))
    for key in ("question", "text", "prompt"):
        if sample.get(key):
            return clean_question_text(_content_to_text(sample.get(key)))
    return ""


def _coerce_actions(obj: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(obj, dict):
        if "actions" in obj:
            actions = obj.get("actions")
            if isinstance(actions, list):
                return actions
            if isinstance(actions, dict):
                return [actions]
        if "name" in obj and "arguments" in obj:
            return [obj]
    if isinstance(obj, list) and all(isinstance(item, dict) for item in obj):
        return obj
    return None


def extract_action(text: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(text, str):
        return _coerce_actions(text)
    stripped = text.strip()
    if not stripped:
        return None
    try:
        actions = _coerce_actions(json.loads(stripped))
        if actions:
            return actions
    except Exception:
        pass
    for match in re.finditer(r"[\{\[]", stripped):
        try:
            obj, _ = json.JSONDecoder().raw_decode(stripped[match.start():])
            actions = _coerce_actions(obj)
            if actions:
                return actions
        except Exception:
            continue
    return None


def extract_final_answer(final_response: Any) -> str:
    if not isinstance(final_response, str):
        return "" if final_response is None else str(final_response)
    actions = extract_action(final_response)
    if actions:
        action = actions[0]
        if action.get("name") == "Terminate" and isinstance(action.get("arguments"), dict):
            ans = action["arguments"].get("ans", "")
            return ans.strip() if isinstance(ans, str) else str(ans)
    return final_response.strip()


def extract_label(sample: Dict[str, Any]) -> str:
    eval_meta = sample.get("eval_meta") or {}
    if isinstance(eval_meta, dict) and eval_meta.get("gold_final_answer") is not None:
        return extract_final_answer(eval_meta.get("gold_final_answer"))
    if sample.get("label") is not None:
        return extract_final_answer(sample.get("label"))
    for message in reversed(sample.get("messages") or []):
        if not isinstance(message, dict):
            continue
        role = message.get("role") or message.get("from")
        if role in {"assistant", "gpt"}:
            return extract_final_answer(_content_to_text(message.get("content", message.get("value", ""))))
    return ""


def load_jsonl_dataset(file_path: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    dataset = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["idx"] = item.get("idx", line_idx)
            dataset.append(item)
            if num_samples and len(dataset) >= int(num_samples):
                break
    return dataset


def build_input_registry(
    sample: Dict[str, Any],
    dataset_file: Any = None,
    image_root: Any = None,
    dataset_root: Any = None,
) -> Dict[str, Any]:
    explicit = sample.get("input_registry")
    if isinstance(explicit, dict):
        return resolve_registry_paths(explicit, dataset_file=dataset_file, image_root=image_root, dataset_root=dataset_root)

    registry = {
        "primary_image": None,
        "optical_image": None,
        "sar_image": None,
        "time1_image": None,
        "time2_image": None,
    }
    images = resolve_sample_images(sample, dataset_file=dataset_file, image_root=image_root, dataset_root=dataset_root)
    eval_meta = sample.get("eval_meta") or {}
    task_family = str(eval_meta.get("task_family", "")) if isinstance(eval_meta, dict) else ""

    if images:
        registry["primary_image"] = images[0]
    if len(images) >= 2:
        registry["optical_image"] = images[0]
        registry["sar_image"] = images[1]
        registry["time1_image"] = images[0]
        registry["time2_image"] = images[1]

    if task_family == "sar_evidence" and images:
        registry["sar_image"] = images[0]
    if task_family in {"cross_modal_match", "cross_modal_task"} and len(images) >= 2:
        registry["optical_image"] = images[0]
        registry["sar_image"] = images[1]
    if task_family.startswith("single_image") and images:
        registry["primary_image"] = images[0]

    return registry


def normalize_sample(
    sample: Dict[str, Any],
    index: int,
    dataset_file: Any = None,
    image_root: Any = None,
    dataset_root: Any = None,
) -> Dict[str, Any]:
    eval_meta = sample.get("eval_meta") if isinstance(sample.get("eval_meta"), dict) else {}
    sample_id = str(sample.get("idx", sample.get("id", index)))
    return {
        "idx": sample_id,
        "sample_id": sample_id,
        "question": extract_question(sample),
        "label": extract_label(sample),
        "images": resolve_sample_images(sample, dataset_file=dataset_file, image_root=image_root, dataset_root=dataset_root),
        "input_registry": build_input_registry(sample, dataset_file=dataset_file, image_root=image_root, dataset_root=dataset_root),
        "eval_meta": dict(eval_meta),
        "raw_sample": sample,
    }


def load_data_function() -> List[Dict[str, Any]]:
    dataset_path = _config_get("dataset_path")
    if not dataset_path:
        return []
    num_samples = _config_get("num_sample")
    return [
        normalize_sample(item, index)
        for index, item in enumerate(load_jsonl_dataset(str(dataset_path), num_samples=num_samples))
    ]


def append_jsonl_line(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def selected_skill_names(trace: Dict[str, Any]) -> List[str]:
    return [item.get("skill_name") for item in trace.get("selected_skills", []) if item.get("skill_name")]


def first_selected_skill(trace: Dict[str, Any]) -> Optional[str]:
    names = selected_skill_names(trace)
    return names[0] if names else None


def called_skill_names(trace: Dict[str, Any]) -> List[str]:
    return [item.get("skill_name") for item in trace.get("skill_calls", []) if item.get("skill_name")]


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else 0.0


def _rate(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def summarize_metrics(traces: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for trace in traces:
        mode = trace.get("mode", "")
        by_mode[mode].append(trace)
        by_sample[str(trace.get("sample_id", ""))][mode] = trace

    mode_metrics = {}
    for mode, mode_traces in sorted(by_mode.items()):
        scores = [trace.get("answer_score") for trace in mode_traces if trace.get("answer_score") is not None]
        skill_success = [
            call.get("success")
            for trace in mode_traces
            for call in trace.get("skill_calls", [])
        ]
        mode_metrics[mode] = {
            "runs": len(mode_traces),
            "terminated_rate": _rate(sum(1 for trace in mode_traces if trace.get("terminated")), len(mode_traces)),
            "answer_accuracy": _mean(scores),
            "avg_rounds": _mean(trace.get("rounds", 0) for trace in mode_traces),
            "avg_raw_tool_calls": _mean(trace.get("raw_tool_calls", 0) for trace in mode_traces),
            "avg_skill_tool_calls": _mean(trace.get("skill_tool_calls", 0) for trace in mode_traces),
            "avg_total_tool_calls": _mean(trace.get("total_tool_calls", 0) for trace in mode_traces),
            "avg_wall_time": _mean(trace.get("wall_time", 0) for trace in mode_traces),
            "avg_llm_time": _mean(trace.get("llm_time", 0) for trace in mode_traces),
            "avg_tool_time": _mean(trace.get("tool_time", 0) for trace in mode_traces),
            "skill_call_success_rate": _rate(sum(1 for ok in skill_success if ok), len(skill_success)),
        }

    auto_traces = by_mode.get(RunMode.AUTO_SKILL.value, [])
    gold_auto = [trace for trace in auto_traces if trace.get("eval_meta", {}).get("gold_skill")]
    should_use = [trace for trace in auto_traces if trace.get("eval_meta", {}).get("should_use_skill") is True]
    should_not_use = [trace for trace in auto_traces if trace.get("eval_meta", {}).get("should_use_skill") is False]
    router_top1 = 0
    router_topk = 0
    skill_gold_called = 0
    for trace in gold_auto:
        gold_skill = trace.get("eval_meta", {}).get("gold_skill")
        selected = selected_skill_names(trace)
        called = called_skill_names(trace)
        if selected and selected[0] == gold_skill:
            router_top1 += 1
        if gold_skill in selected:
            router_topk += 1
        if gold_skill in called:
            skill_gold_called += 1

    router_metrics = {
        "gold_skill_samples": len(gold_auto),
        "top1_accuracy": _rate(router_top1, len(gold_auto)),
        "topk_accuracy": _rate(router_topk, len(gold_auto)),
        "skill_gold_call_rate": _rate(skill_gold_called, len(gold_auto)),
        "should_use_skill_samples": len(should_use),
        "should_use_skill_selected_rate": _rate(sum(1 for trace in should_use if selected_skill_names(trace)), len(should_use)),
        "should_not_use_skill_samples": len(should_not_use),
        "false_positive_selected_rate": _rate(sum(1 for trace in should_not_use if selected_skill_names(trace)), len(should_not_use)),
    }

    efficiency = {}
    for target_mode in (RunMode.AUTO_SKILL.value, RunMode.ORACLE_SKILL.value, RunMode.ORACLE_FORCED_SKILL.value):
        deltas = {"rounds": [], "total_tool_calls": [], "wall_time": [], "llm_time": [], "tool_time": []}
        for sample_modes in by_sample.values():
            base = sample_modes.get(RunMode.NO_SKILL.value)
            target = sample_modes.get(target_mode)
            if not base or not target:
                continue
            for key in deltas:
                deltas[key].append(float(target.get(key, 0)) - float(base.get(key, 0)))
        efficiency[target_mode + "_minus_no_skill"] = {key: _mean(value) for key, value in deltas.items()}

    return {
        "num_samples": len(by_sample),
        "num_traces": len(traces),
        "mode_metrics": mode_metrics,
        "router_metrics": router_metrics,
        "efficiency_delta": efficiency,
        "oracle_skipped": {
            "count": len(skipped),
            "by_reason": dict(Counter(item.get("reason", "unknown") for item in skipped)),
        },
    }


def summarize_errors(traces: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode = defaultdict(Counter)
    tool_errors = defaultdict(int)
    for trace in traces:
        mode = trace.get("mode", "")
        by_mode[mode].update(trace.get("error_tags", []))
        tool_errors[mode] += sum(
            1
            for item in trace.get("tool_responses", [])
            if item.get("response", {}).get("error_code") not in (None, 0)
        )
    return {
        "error_tags_by_mode": {mode: dict(counter) for mode, counter in sorted(by_mode.items())},
        "tool_errors_by_mode": dict(tool_errors),
        "oracle_skipped": skipped,
    }


RunnerFactory = Callable[[str, AgentRunConfig, Any, Any], AgentRunner]


def _skill_modes_enabled(selected_modes: List[str]) -> bool:
    return any(mode in {RunMode.AUTO_SKILL.value, RunMode.ORACLE_SKILL.value, RunMode.ORACLE_FORCED_SKILL.value} for mode in selected_modes)


def _validate_required_online_tools(tool_manager: Any, selected_modes: List[str], records: List[Dict[str, Any]]) -> None:
    if os.environ.get("SKILL_EVAL_ALLOW_MISSING_ONLINE_TOOLS") in {"1", "true", "TRUE", "yes", "YES"}:
        return
    if not _skill_modes_enabled(selected_modes):
        return

    available_online_tools = getattr(tool_manager, "available_online_tools", None)
    if available_online_tools is None:
        return

    required = {"SkillExecutor"}
    for record in records:
        gold_skill = (record.get("eval_meta") or {}).get("gold_skill")
        if gold_skill:
            required.add(str(gold_skill))

    missing = sorted(required - set(available_online_tools or []))
    if missing:
        raise RuntimeError(
            "Skill eval requires online workers for skill modes, but these tools are not registered: "
            + ", ".join(missing)
            + ". Start or restart the online tool stack, for example: "
            + "python tool_server/tool_workers/scripts/launch_scripts/start_server_local.py "
            + "--config tool_server/tool_workers/scripts/launch_scripts/config/all_service_example_local.yaml. "
            + "Set SKILL_EVAL_ALLOW_MISSING_ONLINE_TOOLS=1 only for offline/unit tests."
        )


def default_runner_factory(
    mode: str,
    config: AgentRunConfig,
    shared_llm: Any,
    shared_tool_manager: Any,
) -> AgentRunner:
    return AgentRunner(mode=mode, config=config, llm=shared_llm, tool_manager=shared_tool_manager)


def run_skill_ablation(
    dataset_path: str,
    output_dir: str = "outputs/skill_eval",
    num_samples: Optional[int] = None,
    max_rounds: Optional[int] = None,
    modes: Optional[List[str]] = None,
    image_root: Optional[str] = None,
    dataset_root: Optional[str] = None,
    runner_factory: Optional[RunnerFactory] = None,
    llm: Any = None,
    tool_manager: Any = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    trace_path = output_path / RUN_TRACE_FILE
    metrics_path = output_path / METRICS_FILE
    errors_path = output_path / ERROR_FILE
    trace_path.write_text("", encoding="utf-8")

    selected_modes = modes or list(_config_get("modes", [mode.value for mode in RunMode]))
    selected_modes = [RunMode.normalize(mode).value for mode in selected_modes]
    runner_factory = runner_factory or default_runner_factory
    raw_records = load_jsonl_dataset(dataset_path, num_samples=num_samples)
    preflight_validate_samples(raw_records, dataset_file=dataset_path, image_root=image_root, dataset_root=dataset_root)
    records = [
        normalize_sample(item, index, dataset_file=dataset_path, image_root=image_root, dataset_root=dataset_root)
        for index, item in enumerate(raw_records)
    ]

    traces: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    shared_llm = llm
    shared_tool_manager = tool_manager or ToolManager()
    _validate_required_online_tools(shared_tool_manager, selected_modes, records)
    run_config = AgentRunConfig(
        top_k=int(_config_get("router_top_k", 3)),
        min_score=float(_config_get("router_min_score", 2.5)),
        score_margin=float(_config_get("router_score_margin", 0.5)),
        max_rounds=int(max_rounds or _config_get("max_rounds", 16)),
    )

    for sample in records:
        eval_meta = sample.get("eval_meta", {})
        gold_skill = eval_meta.get("gold_skill")
        for mode in selected_modes:
            if mode in {RunMode.ORACLE_SKILL.value, RunMode.ORACLE_FORCED_SKILL.value} and not gold_skill:
                skipped.append({
                    "sample_id": sample["sample_id"],
                    "mode": mode,
                    "reason": "missing_gold_skill",
                })
                continue

            mode_config = deepcopy(run_config)
            mode_config.mode = RunMode.normalize(mode)
            runner = runner_factory(mode, mode_config, shared_llm, shared_tool_manager)
            started_at = time.perf_counter()
            try:
                trace = runner.run_sample(
                    question=sample["question"],
                    input_registry=sample["input_registry"],
                    oracle_skill=gold_skill if mode in {RunMode.ORACLE_SKILL.value, RunMode.ORACLE_FORCED_SKILL.value} else None,
                    max_rounds=mode_config.max_rounds,
                    sample_id=sample["sample_id"],
                    eval_meta=eval_meta,
                )
            except Exception as exc:
                trace = create_agent_trace(sample_id=sample["sample_id"], mode=mode, question=sample["question"])
                trace["wall_time"] = round(time.perf_counter() - started_at, 6)
                trace["error_tags"].append("runner_exception")
                trace["final_answer"] = ""
                trace["exception"] = str(exc)

            shared_llm = getattr(runner, "llm", shared_llm)
            shared_tool_manager = getattr(runner, "tool_manager", shared_tool_manager)
            trace["sample_id"] = sample["sample_id"]
            trace["eval_meta"] = eval_meta
            trace["label"] = sample["label"]
            trace["images"] = sample["images"]
            trace["answer_score"] = answer_score(
                sample["label"],
                trace.get("final_answer", ""),
                eval_meta=eval_meta,
                trace=trace,
            )
            trace = attribute_trace_errors(trace)
            traces.append(trace)
            append_jsonl_line(str(trace_path), trace)

    return generate_report(trace_path, output_dir=output_path, skipped=skipped, rewrite_traces=True)


def evaluate_function(results: List[Dict[str, Any]], meta_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    traces = []
    for item in results or []:
        if isinstance(item, dict) and isinstance(item.get("results"), dict):
            traces.append(item["results"])
        elif isinstance(item, dict):
            traces.append(item)
    return {
        "metrics": summarize_metrics(traces, skipped=[]),
        "error_breakdown": summarize_errors(traces, skipped=[]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RSAgent skill ablation evaluation.")
    parser.add_argument("--dataset-path", default=_config_get("dataset_path"), help="Path to jsonl evaluation set.")
    parser.add_argument("--output-dir", default=_config_get("output_dir", "outputs/skill_eval"), help="Directory for ablation outputs.")
    parser.add_argument("--num-sample", type=int, default=_config_get("num_sample"), help="Optional sample limit.")
    parser.add_argument("--max-rounds", type=int, default=_config_get("max_rounds", 16), help="Max agent rounds per run.")
    parser.add_argument("--modes", default=",".join(_config_get("modes", [mode.value for mode in RunMode])), help="Comma-separated modes to run.")
    parser.add_argument("--image-root", default=os.environ.get("ONEID_IMAGE_ROOT"), help="Root containing image assets such as images_test/.")
    parser.add_argument("--dataset-root", default=os.environ.get("SKILL_EVAL_DATA_ROOT") or os.environ.get("RS_DATA_ROOT"), help="Dataset asset root used for relative paths.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_path:
        raise SystemExit("--dataset-path is required or must be set in config.yaml")
    modes = [item.strip() for item in str(args.modes).split(",") if item.strip()]
    result = run_skill_ablation(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_samples=args.num_sample,
        max_rounds=args.max_rounds,
        modes=modes,
        image_root=args.image_root,
        dataset_root=args.dataset_root,
    )
    print(json.dumps(result["output_files"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
