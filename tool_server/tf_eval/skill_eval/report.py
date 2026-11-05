from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tool_server.tf_eval.skill_eval.agent_completion_metrics import compute_agent_completion_metrics
from tool_server.tf_eval.skill_eval.answer_scoring import answer_score
from tool_server.tf_eval.skill_eval.efficiency_metrics import compute_efficiency_metrics
from tool_server.tf_eval.skill_eval.error_attribution import attribute_trace_errors, summarize_error_attribution
from tool_server.tf_eval.skill_eval.router_metrics import compute_router_metrics
from tool_server.tf_eval.skill_eval.skill_execution_metrics import compute_skill_execution_metrics


def load_traces(path: str | Path) -> List[Dict[str, Any]]:
    traces = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)


def write_traces(path: str | Path, traces: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")


def _mode_value(metrics: Dict[str, Dict[str, Any]], mode: str, key: str) -> float:
    value = metrics.get(mode, {}).get(key, 0.0)
    return float(value or 0.0)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _gain(no_skill: float, target: float, higher_is_better: bool) -> float:
    return target - no_skill if higher_is_better else no_skill - target


def _table(headers: List[str], rows: List[List[Any]]) -> str:
    align = [":---"] + ["---:" for _ in headers[1:]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(item) for item in row) + " |")
    return "\n".join(lines)


def build_main_comparison(
    completion: Dict[str, Dict[str, Any]],
    efficiency: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    eff_by_mode = efficiency.get("by_mode", {})
    rows = {
        "final_success_rate": {
            "no_skill": _mode_value(completion, "no_skill", "final_success_rate"),
            "auto_skill": _mode_value(completion, "auto_skill", "final_success_rate"),
            "oracle_skill": _mode_value(completion, "oracle_skill", "final_success_rate"),
            "higher_is_better": True,
        },
        "completion_rate": {
            "no_skill": _mode_value(completion, "no_skill", "completion_rate"),
            "auto_skill": _mode_value(completion, "auto_skill", "completion_rate"),
            "oracle_skill": _mode_value(completion, "oracle_skill", "completion_rate"),
            "higher_is_better": True,
        },
        "avg_rounds": {
            "no_skill": _mode_value(eff_by_mode, "no_skill", "avg_rounds"),
            "auto_skill": _mode_value(eff_by_mode, "auto_skill", "avg_rounds"),
            "oracle_skill": _mode_value(eff_by_mode, "oracle_skill", "avg_rounds"),
            "higher_is_better": False,
        },
        "avg_total_tool_calls": {
            "no_skill": _mode_value(eff_by_mode, "no_skill", "avg_total_tool_calls"),
            "auto_skill": _mode_value(eff_by_mode, "auto_skill", "avg_total_tool_calls"),
            "oracle_skill": _mode_value(eff_by_mode, "oracle_skill", "avg_total_tool_calls"),
            "higher_is_better": False,
        },
        "avg_raw_tool_calls": {
            "no_skill": _mode_value(eff_by_mode, "no_skill", "avg_raw_tool_calls"),
            "auto_skill": _mode_value(eff_by_mode, "auto_skill", "avg_raw_tool_calls"),
            "oracle_skill": _mode_value(eff_by_mode, "oracle_skill", "avg_raw_tool_calls"),
            "higher_is_better": False,
        },
        "avg_wall_time": {
            "no_skill": _mode_value(eff_by_mode, "no_skill", "avg_wall_time"),
            "auto_skill": _mode_value(eff_by_mode, "auto_skill", "avg_wall_time"),
            "oracle_skill": _mode_value(eff_by_mode, "oracle_skill", "avg_wall_time"),
            "higher_is_better": False,
        },
        "weighted_tool_cost": {
            "no_skill": _mode_value(eff_by_mode, "no_skill", "weighted_tool_cost"),
            "auto_skill": _mode_value(eff_by_mode, "auto_skill", "weighted_tool_cost"),
            "oracle_skill": _mode_value(eff_by_mode, "oracle_skill", "weighted_tool_cost"),
            "higher_is_better": False,
        },
    }
    for item in rows.values():
        higher = bool(item.pop("higher_is_better"))
        item["auto_gain"] = _gain(item["no_skill"], item["auto_skill"], higher)
        item["oracle_gain"] = _gain(item["no_skill"], item["oracle_skill"], higher)
    return rows


def build_summary_md(metrics: Dict[str, Any], errors: Dict[str, Any]) -> str:
    main = metrics.get("main_comparison", {})
    router = metrics.get("router_metrics", {})
    skill = metrics.get("skill_execution_metrics", {})
    skill_applicable = metrics.get("agent_completion_metrics", {}).get("skill_applicable_subset", {})

    main_rows = [
        ["Final Success Rate ↑", main.get("final_success_rate", {}).get("no_skill"), main.get("final_success_rate", {}).get("auto_skill"), main.get("final_success_rate", {}).get("oracle_skill"), main.get("final_success_rate", {}).get("auto_gain"), main.get("final_success_rate", {}).get("oracle_gain")],
        ["Completion Rate ↑", main.get("completion_rate", {}).get("no_skill"), main.get("completion_rate", {}).get("auto_skill"), main.get("completion_rate", {}).get("oracle_skill"), main.get("completion_rate", {}).get("auto_gain"), main.get("completion_rate", {}).get("oracle_gain")],
        ["Avg Rounds ↓", main.get("avg_rounds", {}).get("no_skill"), main.get("avg_rounds", {}).get("auto_skill"), main.get("avg_rounds", {}).get("oracle_skill"), main.get("avg_rounds", {}).get("auto_gain"), main.get("avg_rounds", {}).get("oracle_gain")],
        ["Avg Tool Calls ↓", main.get("avg_total_tool_calls", {}).get("no_skill"), main.get("avg_total_tool_calls", {}).get("auto_skill"), main.get("avg_total_tool_calls", {}).get("oracle_skill"), main.get("avg_total_tool_calls", {}).get("auto_gain"), main.get("avg_total_tool_calls", {}).get("oracle_gain")],
        ["Avg Raw Tool Calls ↓", main.get("avg_raw_tool_calls", {}).get("no_skill"), main.get("avg_raw_tool_calls", {}).get("auto_skill"), main.get("avg_raw_tool_calls", {}).get("oracle_skill"), main.get("avg_raw_tool_calls", {}).get("auto_gain"), main.get("avg_raw_tool_calls", {}).get("oracle_gain")],
        ["Avg Wall Time ↓", main.get("avg_wall_time", {}).get("no_skill"), main.get("avg_wall_time", {}).get("auto_skill"), main.get("avg_wall_time", {}).get("oracle_skill"), main.get("avg_wall_time", {}).get("auto_gain"), main.get("avg_wall_time", {}).get("oracle_gain")],
        ["Weighted Tool Cost ↓", main.get("weighted_tool_cost", {}).get("no_skill"), main.get("weighted_tool_cost", {}).get("auto_skill"), main.get("weighted_tool_cost", {}).get("oracle_skill"), main.get("weighted_tool_cost", {}).get("auto_gain"), main.get("weighted_tool_cost", {}).get("oracle_gain")],
    ]
    router_rows = [
        ["Top-1 Acc ↑", router.get("router_top1_acc")],
        ["Top-k Acc ↑", router.get("router_topk_acc")],
        ["Miss Rate ↓", router.get("router_miss_rate")],
        ["False Positive Rate ↓", router.get("router_false_positive_rate")],
    ]
    skill_rows = [
        ["Skill Call Rate ↑", skill.get("auto_skill", {}).get("skill_call_rate"), skill.get("oracle_skill", {}).get("skill_call_rate")],
        ["Skill Name Acc ↑", skill.get("auto_skill", {}).get("skill_name_acc"), skill.get("oracle_skill", {}).get("skill_name_acc")],
        ["Skill Success Rate ↑", skill.get("auto_skill", {}).get("skill_success_rate"), skill.get("oracle_skill", {}).get("skill_success_rate")],
        ["Successful Structured Rate ↑", skill.get("auto_skill", {}).get("successful_structured_rate"), skill.get("oracle_skill", {}).get("successful_structured_rate")],
        ["Structured Error Rate ↓", skill.get("auto_skill", {}).get("structured_error_rate"), skill.get("oracle_skill", {}).get("structured_error_rate")],
        ["Handoff Success ↑", skill.get("auto_skill", {}).get("skill_handoff_success"), skill.get("oracle_skill", {}).get("skill_handoff_success")],
    ]
    skill_applicable_rows = [
        ["Final Success Rate ↑", skill_applicable.get("no_skill", {}).get("final_success_rate"), skill_applicable.get("auto_skill", {}).get("final_success_rate"), skill_applicable.get("oracle_skill", {}).get("final_success_rate")],
        ["Semantic Success Rate ↑", skill_applicable.get("no_skill", {}).get("semantic_success_rate"), skill_applicable.get("auto_skill", {}).get("semantic_success_rate"), skill_applicable.get("oracle_skill", {}).get("semantic_success_rate")],
        ["Evidence Grounded ↑", skill_applicable.get("no_skill", {}).get("evidence_grounded"), skill_applicable.get("auto_skill", {}).get("evidence_grounded"), skill_applicable.get("oracle_skill", {}).get("evidence_grounded")],
    ]

    lines = [
        "# Skill Ablation Summary",
        "",
        f"- Samples: {metrics.get('num_samples', 0)}",
        f"- Traces: {metrics.get('num_traces', 0)}",
        f"- Oracle skipped: {metrics.get('oracle_skipped', {}).get('count', 0)}",
        "",
        "## Main Comparison",
        _table(["Metric", "No Skill", "Auto Skill", "Oracle Skill", "Auto Gain", "Oracle Gain"], main_rows),
        "",
        "## Router",
        _table(["Metric", "Auto Skill"], router_rows),
        "",
        "## Skill Execution",
        _table(["Metric", "Auto Skill", "Oracle Skill"], skill_rows),
        "",
        "## Skill-Applicable Subset",
        _table(["Metric", "No Skill", "Auto Skill", "Oracle Skill"], skill_applicable_rows),
        "",
        "## Error Breakdown",
        "```json",
        json.dumps(errors.get("error_tags_by_mode", {}), ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def generate_report(
    run_traces_path: str | Path,
    output_dir: str | Path | None = None,
    skipped: List[Dict[str, Any]] | None = None,
    rewrite_traces: bool = True,
) -> Dict[str, Any]:
    run_traces_path = Path(run_traces_path)
    output_dir = Path(output_dir) if output_dir else run_traces_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    skipped = skipped or []

    traces = []
    for trace in load_traces(run_traces_path):
        if trace.get("label") is not None:
            trace["answer_score"] = answer_score(
                trace.get("label"),
                trace.get("final_answer", ""),
                eval_meta=trace.get("eval_meta"),
                trace=trace,
            )
        traces.append(attribute_trace_errors(trace))
    if rewrite_traces:
        write_traces(run_traces_path, traces)

    completion = compute_agent_completion_metrics(traces)
    efficiency = compute_efficiency_metrics(traces)
    metrics = {
        "num_samples": len({trace.get("sample_id") for trace in traces}),
        "num_traces": len(traces),
        "router_metrics": compute_router_metrics(traces),
        "skill_execution_metrics": compute_skill_execution_metrics(traces),
        "agent_completion_metrics": completion,
        "efficiency_metrics": efficiency,
        "main_comparison": build_main_comparison(completion, efficiency),
        "oracle_skipped": {
            "count": len(skipped),
            "by_reason": {},
        },
    }
    for item in skipped:
        reason = item.get("reason", "unknown")
        metrics["oracle_skipped"]["by_reason"][reason] = metrics["oracle_skipped"]["by_reason"].get(reason, 0) + 1

    errors = summarize_error_attribution(traces, skipped=skipped)

    metrics_path = output_dir / "metrics_summary.json"
    errors_path = output_dir / "error_breakdown.json"
    summary_path = output_dir / "summary.md"
    write_json(metrics, metrics_path)
    write_json(errors, errors_path)
    summary_path.write_text(build_summary_md(metrics, errors), encoding="utf-8")

    return {
        "metrics": metrics,
        "error_breakdown": errors,
        "output_files": {
            "run_traces": str(run_traces_path),
            "metrics_summary": str(metrics_path),
            "error_breakdown": str(errors_path),
            "summary_md": str(summary_path),
        },
    }
