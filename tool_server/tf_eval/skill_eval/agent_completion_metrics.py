from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


VISUAL_FAMILIES = {"visual_output"}


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else 0.0


def _rate(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _mode_traces(traces: Iterable[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    return [trace for trace in traces if trace.get("mode") == mode]


def _is_skill_applicable(trace: Dict[str, Any]) -> bool:
    eval_meta = trace.get("eval_meta", {}) if isinstance(trace.get("eval_meta"), dict) else {}
    return eval_meta.get("should_use_skill") is True or bool(eval_meta.get("gold_skill"))


def _executed_tool_sequence(trace: Dict[str, Any]) -> List[str]:
    return [
        action.get("api_name")
        for action in trace.get("actions", [])
        if action.get("executed") and action.get("api_name")
    ]


def _sequence_contains(gold: List[str], pred: List[str]) -> bool:
    if not gold:
        return False
    cursor = 0
    for name in pred:
        if cursor < len(gold) and name == gold[cursor]:
            cursor += 1
    return cursor == len(gold)


def _requires_visual(trace: Dict[str, Any]) -> bool:
    eval_meta = trace.get("eval_meta", {}) if isinstance(trace.get("eval_meta"), dict) else {}
    task_family = str(eval_meta.get("task_family", ""))
    return bool(eval_meta.get("requires_visual_output")) or task_family in VISUAL_FAMILIES or "visual" in task_family


def _has_artifact(trace: Dict[str, Any]) -> bool:
    for response_item in trace.get("tool_responses", []):
        response = response_item.get("response") if isinstance(response_item, dict) else {}
        if not isinstance(response, dict):
            continue
        artifacts = response.get("artifacts")
        if isinstance(artifacts, list) and artifacts:
            return True
        if response.get("primary_image") or response.get("image") or response.get("download_path"):
            return True
    for call in trace.get("skill_calls", []):
        response = call.get("response") if isinstance(call, dict) else {}
        if isinstance(response, dict) and isinstance(response.get("artifacts"), list) and response.get("artifacts"):
            return True
    return False


def _tool_chain_success(trace: Dict[str, Any]) -> Optional[bool]:
    eval_meta = trace.get("eval_meta", {}) if isinstance(trace.get("eval_meta"), dict) else {}
    gold = eval_meta.get("gold_tool_sequence") or eval_meta.get("tool_sequence")
    if not isinstance(gold, list) or not gold:
        return None
    return _sequence_contains([str(item) for item in gold], _executed_tool_sequence(trace))


def _has_tool_failure(trace: Dict[str, Any]) -> bool:
    for item in trace.get("tool_responses", []):
        response = item.get("response") if isinstance(item, dict) else {}
        if isinstance(response, dict) and response.get("error_code") not in (None, 0):
            return True
    return False


def _required_dependencies_succeeded(trace: Dict[str, Any]) -> bool:
    eval_meta = trace.get("eval_meta", {}) if isinstance(trace.get("eval_meta"), dict) else {}
    if eval_meta.get("should_use_skill") is True or eval_meta.get("gold_skill"):
        calls = [call for call in trace.get("skill_calls", []) if isinstance(call, dict)]
        return any(call.get("success") is True for call in calls) and not _has_tool_failure(trace)
    return not _has_tool_failure(trace)


def _evidence_grounded(trace: Dict[str, Any]) -> bool:
    if "invalid_dependency" in trace.get("error_tags", []) or "example_leakage_detected" in trace.get("error_tags", []):
        return False
    if _has_tool_failure(trace):
        return False
    eval_meta = trace.get("eval_meta", {}) if isinstance(trace.get("eval_meta"), dict) else {}
    if str(eval_meta.get("task_family", "")).startswith("cross_modal"):
        for call in trace.get("skill_calls", []):
            response = call.get("response") if isinstance(call, dict) else {}
            if isinstance(response, dict) and response.get("error_code") == 0:
                internal = response.get("internal_tool_calls") or []
                if not internal:
                    return True
                return all(isinstance(item, dict) and item.get("success") is True for item in internal)
        return False
    return True


def _compute_for_mode(traces: List[Dict[str, Any]], answer_threshold: float) -> Dict[str, Any]:
    scored = [trace for trace in traces if trace.get("answer_score") is not None]
    semantic_success = [trace for trace in scored if float(trace.get("answer_score", 0)) >= answer_threshold]
    final_success = [
        trace for trace in scored
        if float(trace.get("answer_score", 0)) >= answer_threshold
        and _required_dependencies_succeeded(trace)
        and _evidence_grounded(trace)
    ]
    max_round_failures = [
        trace for trace in traces
        if "max_rounds" in trace.get("error_tags", []) or (not trace.get("terminated") and trace.get("rounds", 0) > 0)
    ]
    chain_evals = [value for value in (_tool_chain_success(trace) for trace in traces) if value is not None]
    visual_evals = [trace for trace in traces if _requires_visual(trace)]
    visual_success = [trace for trace in visual_evals if _has_artifact(trace)]

    return {
        "samples": len(traces),
        "completion_rate": _rate(sum(1 for trace in traces if trace.get("terminated")), len(traces)),
        "semantic_success_rate": _rate(len(semantic_success), len(scored)),
        "final_success_rate": _rate(len(final_success), len(scored)),
        "semantic_answer_score": _mean(trace.get("answer_score") for trace in scored),
        "answer_acc": _mean(trace.get("answer_score") for trace in scored),
        "max_round_failure_rate": _rate(len(max_round_failures), len(traces)),
        "tool_chain_success": _rate(sum(1 for value in chain_evals if value), len(chain_evals)),
        "required_dependencies_succeeded": _rate(sum(1 for trace in traces if _required_dependencies_succeeded(trace)), len(traces)),
        "evidence_grounded": _rate(sum(1 for trace in traces if _evidence_grounded(trace)), len(traces)),
        "visual_success_rate": _rate(len(visual_success), len(visual_evals)),
    }


def compute_agent_completion_metrics(
    traces: Iterable[Dict[str, Any]],
    answer_threshold: float = 0.8,
) -> Dict[str, Any]:
    trace_list = list(traces)
    modes = sorted({trace.get("mode") for trace in trace_list if trace.get("mode")})
    metrics = {
        mode: _compute_for_mode(_mode_traces(trace_list, mode), answer_threshold)
        for mode in modes
    }
    metrics["skill_applicable_subset"] = {
        mode: _compute_for_mode([trace for trace in _mode_traces(trace_list, mode) if _is_skill_applicable(trace)], answer_threshold)
        for mode in modes
    }
    return metrics
