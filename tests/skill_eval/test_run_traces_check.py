import json
from pathlib import Path

from scripts.eval.skill_eval.run_traces_check import analyze


def test_run_traces_check_counts(tmp_path: Path):
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text(
        json.dumps({
            "sample_id": "1",
            "mode": "oracle_skill",
            "images": ["images_test/sample.png"],
            "skill_calls": [{"skill_name": "CrossModalTaskSkill", "success": True, "response": {"error_code": 0, "internal_tool_calls": [{"success": True}]}}],
            "tool_responses": [{"api_name": "SkillExecutor", "arguments": {}, "response": {"error_code": 0, "text": "ok"}}],
            "selected_skills": [{"skill_name": "CrossModalTaskSkill"}],
            "eval_meta": {"gold_skill": "CrossModalTaskSkill"},
            "error_tags": [],
            "actions": [],
        }) + "\n",
        encoding="utf-8",
    )
    summary = analyze(trace_path)
    assert summary["counters"]["relative_path_count"] == 1
