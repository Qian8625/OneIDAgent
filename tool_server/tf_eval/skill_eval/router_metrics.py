from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


SKILL_TO_FAMILY = {
    "ObjectDetectSelectSkill": "single_image_detect_select",
    "ObjectMeasureSkill": "single_image_measure",
    "ObjectVisualOutputSkill": "visual_output",
    "SAREvidenceExtractSkill": "sar_evidence",
    "CrossModalMatchSkill": "cross_modal_match",
    "CrossModalTaskSkill": "cross_modal_task",
}


def _rate(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def selected_skill_names(trace: Dict[str, Any]) -> List[str]:
    return [
        item.get("skill_name")
        for item in trace.get("selected_skills", [])
        if isinstance(item, dict) and item.get("skill_name")
    ]


def skill_family(skill_name: Optional[str]) -> Optional[str]:
    if not skill_name:
        return None
    return SKILL_TO_FAMILY.get(skill_name)


def _gold_family(trace: Dict[str, Any]) -> Optional[str]:
    eval_meta = trace.get("eval_meta", {}) if isinstance(trace.get("eval_meta"), dict) else {}
    gold_skill = eval_meta.get("gold_skill")
    return skill_family(gold_skill) or eval_meta.get("task_family")


def compute_router_metrics(traces: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    auto_traces = [trace for trace in traces if trace.get("mode") == "auto_skill"]
    gold_traces = [trace for trace in auto_traces if trace.get("eval_meta", {}).get("gold_skill")]
    should_use = [trace for trace in auto_traces if trace.get("eval_meta", {}).get("should_use_skill") is True]
    should_not_use = [trace for trace in auto_traces if trace.get("eval_meta", {}).get("should_use_skill") is False]

    top1 = 0
    topk = 0
    miss = 0
    wrong_family = 0

    for trace in gold_traces:
        gold_skill = trace.get("eval_meta", {}).get("gold_skill")
        selected = selected_skill_names(trace)
        if selected and selected[0] == gold_skill:
            top1 += 1
        if gold_skill in selected:
            topk += 1
        selected_family = skill_family(selected[0]) if selected else None
        gold_family = _gold_family(trace)
        if selected_family and gold_family and selected_family != gold_family:
            wrong_family += 1

    for trace in should_use:
        gold_skill = trace.get("eval_meta", {}).get("gold_skill")
        selected = selected_skill_names(trace)
        if gold_skill:
            miss += int(gold_skill not in selected)
        else:
            miss += int(not selected)

    false_positive = sum(1 for trace in should_not_use if selected_skill_names(trace))

    return {
        "samples": len(auto_traces),
        "gold_skill_samples": len(gold_traces),
        "should_use_skill_samples": len(should_use),
        "should_not_use_skill_samples": len(should_not_use),
        "router_top1_acc": _rate(top1, len(gold_traces)),
        "router_topk_acc": _rate(topk, len(gold_traces)),
        "router_miss_rate": _rate(miss, len(should_use)),
        "router_false_positive_rate": _rate(false_positive, len(should_not_use)),
        "router_wrong_family_rate": _rate(wrong_family, len(gold_traces)),
    }
