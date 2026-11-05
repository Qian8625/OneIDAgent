from __future__ import annotations

from typing import Any, Dict, Iterable, List


VISUAL_FAMILIES = {"visual_output"}


def _rate(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _mode_traces(traces: Iterable[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    return [trace for trace in traces if trace.get("mode") == mode]


def _should_use_skill(trace: Dict[str, Any]) -> bool:
    eval_meta = trace.get("eval_meta", {}) if isinstance(trace.get("eval_meta"), dict) else {}
    if trace.get("mode") in {"oracle_skill", "oracle_forced_skill"}:
        return bool(eval_meta.get("gold_skill") or trace.get("selected_skills"))
    return eval_meta.get("should_use_skill") is True or bool(eval_meta.get("gold_skill"))


def _requires_visual(trace: Dict[str, Any]) -> bool:
    eval_meta = trace.get("eval_meta", {}) if isinstance(trace.get("eval_meta"), dict) else {}
    task_family = str(eval_meta.get("task_family", ""))
    return bool(eval_meta.get("requires_visual_output")) or task_family in VISUAL_FAMILIES or "visual" in task_family


def _skill_calls(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [item for item in trace.get("skill_calls", []) if isinstance(item, dict)]


def _first_skill_name(trace: Dict[str, Any]) -> str:
    calls = _skill_calls(trace)
    if calls:
        return str(calls[0].get("skill_name") or "")
    for action in trace.get("actions", []):
        if action.get("api_name") == "SkillExecutor" and isinstance(action.get("arguments"), dict):
            return str(action["arguments"].get("skill_name") or "")
    return ""


def _response(call: Dict[str, Any]) -> Dict[str, Any]:
    response = call.get("response")
    return response if isinstance(response, dict) else {}


def _call_success(call: Dict[str, Any]) -> bool:
    response = _response(call)
    if response:
        return response.get("error_code") == 0
    return call.get("success") is True


def _has_structured(call: Dict[str, Any]) -> bool:
    structured = _response(call).get("structured")
    return bool(structured)


def _has_artifacts(call: Dict[str, Any]) -> bool:
    response = _response(call)
    artifacts = response.get("artifacts")
    return isinstance(artifacts, list) and bool(artifacts)


def _handoff_success(trace: Dict[str, Any], call: Dict[str, Any]) -> bool:
    if not _call_success(call):
        return False
    call_round = call.get("round", 0)
    for action in trace.get("actions", []):
        if action.get("round", 0) <= call_round:
            continue
        if action.get("api_name") == "Terminate" and action.get("executed"):
            return True
        if action.get("executed") and action.get("api_name") not in {None, "SkillExecutor"}:
            return True
    return bool(trace.get("terminated"))


def _compute_for_mode(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    expected = [trace for trace in traces if _should_use_skill(trace)]
    called = [trace for trace in expected if _skill_calls(trace)]
    calls = [call for trace in traces for call in _skill_calls(trace)]
    success_calls = [call for call in calls if _call_success(call)]
    structured_calls = [call for call in calls if _has_structured(call)]
    successful_structured_calls = [call for call in calls if _call_success(call) and _has_structured(call)]
    failed_structured_calls = [call for call in calls if not _call_success(call) and _has_structured(call)]
    visual_expected = [trace for trace in expected if _requires_visual(trace)]
    visual_with_artifact = [trace for trace in visual_expected if any(_has_artifacts(call) for call in _skill_calls(trace))]

    name_denominator = [trace for trace in called if trace.get("eval_meta", {}).get("gold_skill")]
    name_correct = sum(
        1
        for trace in name_denominator
        if _first_skill_name(trace) == trace.get("eval_meta", {}).get("gold_skill")
    )
    handoff_success = sum(
        1
        for trace in traces
        for call in _skill_calls(trace)
        if _handoff_success(trace, call)
    )

    return {
        "samples": len(traces),
        "expected_skill_samples": len(expected),
        "skill_call_rate": _rate(len(called), len(expected)),
        "skill_name_acc": _rate(name_correct, len(name_denominator)),
        "skill_success_rate": _rate(len(success_calls), len(calls)),
        "skill_structured_rate": _rate(len(structured_calls), len(calls)),
        "response_structured_rate": _rate(len(structured_calls), len(calls)),
        "successful_structured_rate": _rate(len(successful_structured_calls), len(success_calls)),
        "structured_error_rate": _rate(len(failed_structured_calls), len(calls)),
        "skill_artifact_rate": _rate(len(visual_with_artifact), len(visual_expected)),
        "skill_handoff_success": _rate(handoff_success, len(calls)),
    }


def compute_skill_execution_metrics(traces: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    trace_list = list(traces)
    return {
        "auto_skill": _compute_for_mode(_mode_traces(trace_list, "auto_skill")),
        "oracle_skill": _compute_for_mode(_mode_traces(trace_list, "oracle_skill")),
        "oracle_forced_skill": _compute_for_mode(_mode_traces(trace_list, "oracle_forced_skill")),
    }
