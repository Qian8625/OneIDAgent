from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


TOOL_WEIGHTS = {
    "Calculator": 1,
    "Solver": 1,
    "Plot": 1,
    "DrawBox": 1,
    "TextToBbox": 3,
    "ObjectDetection": 3,
    "SmallObjectDetection": 3,
    "SARToRGB": 4,
    "CrossModalInterpret": 4,
    "SkillExecutor": 3,
    "Terminate": 0,
}

SKILL_EXECUTOR_WEIGHTS = {
    "ObjectDetectSelectSkill": 3,
    "ObjectMeasureSkill": 3,
    "ObjectVisualOutputSkill": 1,
    "SAREvidenceExtractSkill": 4,
    "CrossModalMatchSkill": 4,
    "CrossModalTaskSkill": 4,
}


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else 0.0


def _rate(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def _tool_weight(name: Optional[str]) -> int:
    if not name:
        return 1
    return SKILL_EXECUTOR_WEIGHTS.get(name, TOOL_WEIGHTS.get(name, 2))


def _executor_from_skill_call(call: Dict[str, Any]) -> Optional[str]:
    response = call.get("response") if isinstance(call.get("response"), dict) else {}
    if response.get("executor_model"):
        return response.get("executor_model")
    for item in response.get("skill_trace", []) if isinstance(response.get("skill_trace"), list) else []:
        if isinstance(item, dict) and item.get("stage") == "dispatch_executor" and item.get("executor_model"):
            return item.get("executor_model")
    return None


def internal_tool_calls_from_response(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls = response.get("internal_tool_calls") if isinstance(response, dict) else None
    return [item for item in calls if isinstance(item, dict)] if isinstance(calls, list) else []


def internal_cost_from_response(response: Dict[str, Any]) -> Optional[float]:
    if not isinstance(response, dict) or "internal_cost" not in response:
        return None
    try:
        return float(response.get("internal_cost"))
    except Exception:
        return None


def internal_tool_call_count(call: Dict[str, Any]) -> int:
    response = call.get("response") if isinstance(call.get("response"), dict) else {}
    return len(internal_tool_calls_from_response(response))


def internal_cost_from_calls(calls: List[Dict[str, Any]]) -> float:
    return float(sum(_tool_weight(item.get("tool")) for item in calls))


def _skill_call_base_cost(call: Dict[str, Any]) -> int:
    executor_model = _executor_from_skill_call(call)
    if executor_model:
        return _tool_weight(executor_model)
    skill_name = call.get("skill_name")
    return SKILL_EXECUTOR_WEIGHTS.get(skill_name, TOOL_WEIGHTS["SkillExecutor"])


def _skill_call_cost(call: Dict[str, Any]) -> float:
    response = call.get("response") if isinstance(call.get("response"), dict) else {}
    internal_calls = internal_tool_calls_from_response(response)
    internal_cost = internal_cost_from_response(response)
    if internal_cost is None:
        internal_cost = internal_cost_from_calls(internal_calls)
    return float(_skill_call_base_cost(call)) + float(internal_cost or 0.0)


def weighted_tool_cost(trace: Dict[str, Any]) -> float:
    cost = 0.0
    skill_calls = iter([call for call in trace.get("skill_calls", []) if isinstance(call, dict)])
    for action in trace.get("actions", []):
        if not action.get("executed"):
            continue
        api_name = action.get("api_name")
        if api_name == "SkillExecutor":
            call = next(skill_calls, None)
            cost += _skill_call_cost(call or {})
        else:
            cost += _tool_weight(api_name)
    return round(cost, 6)


def real_tool_calls(trace: Dict[str, Any]) -> int:
    raw_calls = int(trace.get("raw_tool_calls", 0) or 0)
    internal_calls = 0
    for call in trace.get("skill_calls", []):
        response = call.get("response") if isinstance(call, dict) else {}
        explicit_internal_calls = internal_tool_calls_from_response(response)
        if explicit_internal_calls:
            internal_calls += 1 + len(explicit_internal_calls)
            continue
        skill_trace = response.get("skill_trace", []) if isinstance(response, dict) else []
        dispatch_count = sum(
            1
            for item in skill_trace
            if isinstance(item, dict) and item.get("stage") == "dispatch_executor"
        )
        internal_calls += dispatch_count or 1
    return raw_calls + internal_calls


def internal_tool_calls(trace: Dict[str, Any]) -> int:
    return sum(
        internal_tool_call_count(call)
        for call in trace.get("skill_calls", [])
        if isinstance(call, dict)
    )


def internal_cost(trace: Dict[str, Any]) -> float:
    total = 0.0
    for call in trace.get("skill_calls", []):
        if not isinstance(call, dict):
            continue
        response = call.get("response") if isinstance(call.get("response"), dict) else {}
        explicit_cost = internal_cost_from_response(response)
        if explicit_cost is not None:
            total += explicit_cost
        else:
            total += internal_cost_from_calls(internal_tool_calls_from_response(response))
    return round(total, 6)


def _mode_summary(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "samples": len(traces),
        "avg_rounds": _mean(trace.get("rounds", 0) for trace in traces),
        "avg_total_tool_calls": _mean(trace.get("total_tool_calls", 0) for trace in traces),
        "avg_raw_tool_calls": _mean(trace.get("raw_tool_calls", 0) for trace in traces),
        "avg_skill_calls": _mean(trace.get("skill_tool_calls", 0) for trace in traces),
        "avg_surface_tool_calls": _mean(trace.get("total_tool_calls", 0) for trace in traces),
        "avg_real_tool_calls": _mean(real_tool_calls(trace) for trace in traces),
        "avg_internal_tool_calls": _mean(internal_tool_calls(trace) for trace in traces),
        "avg_internal_cost": _mean(internal_cost(trace) for trace in traces),
        "avg_wall_time": _mean(trace.get("wall_time", 0) for trace in traces),
        "avg_llm_time": _mean(trace.get("llm_time", 0) for trace in traces),
        "avg_tool_time": _mean(trace.get("tool_time", 0) for trace in traces),
        "weighted_tool_cost": _mean(weighted_tool_cost(trace) for trace in traces),
    }


def _comparison(by_sample: Dict[str, Dict[str, Dict[str, Any]]], target_mode: str) -> Dict[str, Any]:
    cost_reductions = []
    speedups = []
    for modes in by_sample.values():
        base = modes.get("no_skill")
        target = modes.get(target_mode)
        if not base or not target:
            continue
        base_cost = weighted_tool_cost(base)
        target_cost = weighted_tool_cost(target)
        base_time = float(base.get("wall_time", 0) or 0)
        target_time = float(target.get("wall_time", 0) or 0)
        if base_cost > 0:
            cost_reductions.append((base_cost - target_cost) / base_cost)
        if base_time > 0:
            speedups.append((base_time - target_time) / base_time)
    return {
        "cost_reduction_rate": _mean(cost_reductions),
        "speedup_rate": _mean(speedups),
    }


def compute_efficiency_metrics(traces: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    trace_list = list(traces)
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_sample: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for trace in trace_list:
        mode = trace.get("mode")
        if not mode:
            continue
        by_mode[mode].append(trace)
        by_sample[str(trace.get("sample_id", ""))][mode] = trace

    return {
        "by_mode": {mode: _mode_summary(mode_traces) for mode, mode_traces in sorted(by_mode.items())},
        "comparisons": {
            "auto_skill_vs_no_skill": _comparison(by_sample, "auto_skill"),
            "oracle_skill_vs_no_skill": _comparison(by_sample, "oracle_skill"),
        },
    }
