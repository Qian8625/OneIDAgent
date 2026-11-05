from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from tool_server.tf_eval.skill_eval.router_metrics import selected_skill_names


VISUAL_FAMILIES = {"visual_output"}


def _eval_meta(trace: Dict[str, Any]) -> Dict[str, Any]:
    meta = trace.get("eval_meta")
    return meta if isinstance(meta, dict) else {}


def _requires_visual(trace: Dict[str, Any]) -> bool:
    meta = _eval_meta(trace)
    task_family = str(meta.get("task_family", ""))
    return bool(meta.get("requires_visual_output")) or task_family in VISUAL_FAMILIES or "visual" in task_family


def _skill_calls(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in trace.get("skill_calls", []) if isinstance(item, dict)]


def _skill_call_failed(call: Dict[str, Any]) -> bool:
    response = call.get("response") if isinstance(call.get("response"), dict) else {}
    if response:
        return response.get("error_code") != 0
    return call.get("success") is False


def _skill_call_success(call: Dict[str, Any]) -> bool:
    return not _skill_call_failed(call) and (call.get("success") is True or bool(call.get("response")))


def _skill_has_required_output(trace: Dict[str, Any], call: Dict[str, Any]) -> bool:
    response = call.get("response") if isinstance(call.get("response"), dict) else {}
    if not _requires_visual(trace):
        return True
    artifacts = response.get("artifacts")
    return isinstance(artifacts, list) and bool(artifacts)


def _handoff_failed(trace: Dict[str, Any], call: Dict[str, Any]) -> bool:
    if not _skill_call_success(call):
        return False
    call_round = call.get("round", 0)
    for action in trace.get("actions", []):
        if action.get("round", 0) <= call_round:
            continue
        if action.get("executed") and action.get("api_name"):
            return False
    return not trace.get("terminated")


def _raw_tool_failed(trace: Dict[str, Any]) -> bool:
    for item in trace.get("tool_responses", []):
        if item.get("api_name") == "SkillExecutor":
            continue
        response = item.get("response") if isinstance(item.get("response"), dict) else {}
        if response and response.get("error_code") != 0:
            return True
    return False


def _append_unique(values: List[str], tag: str) -> None:
    if tag not in values:
        values.append(tag)


def attributed_error_tags(trace: Dict[str, Any], answer_threshold: float = 0.8) -> List[str]:
    tags = list(trace.get("error_tags", []))
    mode = trace.get("mode")
    meta = _eval_meta(trace)
    gold_skill = meta.get("gold_skill")
    should_use = meta.get("should_use_skill") is True or bool(gold_skill)
    selected = selected_skill_names(trace)
    calls = _skill_calls(trace)

    if mode == "auto_skill" and gold_skill:
        if not selected or gold_skill not in selected:
            _append_unique(tags, "router_miss")
        elif selected[0] != gold_skill:
            _append_unique(tags, "router_wrong")

    if mode in {"auto_skill", "oracle_skill"} and should_use and not calls:
        _append_unique(tags, "skill_not_called")

    if any(_skill_call_failed(call) for call in calls):
        _append_unique(tags, "skill_failed")

    if any(_skill_call_success(call) and not _skill_has_required_output(trace, call) for call in calls):
        _append_unique(tags, "skill_output_wrong")

    if any(_handoff_failed(trace, call) for call in calls):
        _append_unique(tags, "handoff_failed")

    if trace.get("answer_score") is not None and float(trace.get("answer_score", 0)) < answer_threshold:
        _append_unique(tags, "agent_final_wrong")

    if _raw_tool_failed(trace):
        _append_unique(tags, "raw_tool_failed")

    for runtime_tag in ["invalid_dependency", "example_leakage_detected", "duplicate_call_blocked"]:
        if runtime_tag in trace.get("error_tags", []) or trace.get(runtime_tag):
            _append_unique(tags, runtime_tag)

    if trace.get("mode") in {"oracle_skill", "oracle_forced_skill"} and gold_skill:
        calls = _skill_calls(trace)
        if not calls or calls[0].get("skill_name") != gold_skill:
            _append_unique(tags, "oracle_not_called")

    if trace.get("answer_score") is not None and float(trace.get("answer_score", 0)) >= answer_threshold:
        if any(tag in tags for tag in ["tool_error", "raw_tool_failed", "skill_failed", "invalid_dependency", "example_leakage_detected"]):
            _append_unique(tags, "ungrounded_final_answer")

    if "max_rounds" in tags or (not trace.get("terminated") and trace.get("rounds", 0) > 0):
        _append_unique(tags, "timeout_or_max_round")

    if any(tag in tags for tag in ["invalid_action_format", "no_action"]):
        _append_unique(tags, "invalid_action_format")

    if not trace.get("terminated") or not str(trace.get("final_answer", "")).strip():
        _append_unique(tags, "no_final_answer")

    return tags


def attribute_trace_errors(trace: Dict[str, Any], answer_threshold: float = 0.8) -> Dict[str, Any]:
    trace = dict(trace)
    trace["error_tags"] = attributed_error_tags(trace, answer_threshold=answer_threshold)
    return trace


def summarize_error_attribution(
    traces: Iterable[Dict[str, Any]],
    skipped: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    skipped = skipped or []
    by_mode = defaultdict(Counter)
    sample_ids_by_error = defaultdict(lambda: defaultdict(list))
    runtime_tags_by_mode = defaultdict(Counter)

    for trace in traces:
        mode = trace.get("mode", "unknown")
        tags = trace.get("error_tags", [])
        by_mode[mode].update(tags)
        for tag in tags:
            sample_ids_by_error[mode][tag].append(trace.get("sample_id", ""))
        for tag in tags:
            if tag not in {
                "router_wrong",
                "router_miss",
                "skill_not_called",
                "skill_failed",
                "skill_output_wrong",
                "handoff_failed",
                "agent_final_wrong",
                "raw_tool_failed",
                "timeout_or_max_round",
                "invalid_action_format",
                "no_final_answer",
            }:
                runtime_tags_by_mode[mode][tag] += 1

    return {
        "error_tags_by_mode": {mode: dict(counter) for mode, counter in sorted(by_mode.items())},
        "runtime_tags_by_mode": {mode: dict(counter) for mode, counter in sorted(runtime_tags_by_mode.items())},
        "sample_ids_by_error": {
            mode: {tag: ids for tag, ids in sorted(errors.items())}
            for mode, errors in sorted(sample_ids_by_error.items())
        },
        "oracle_skipped": skipped,
    }
