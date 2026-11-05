from tool_server.tf_eval.skill_eval.agent_completion_metrics import compute_agent_completion_metrics


def test_final_success_requires_grounding():
    traces = [
        {
            "mode": "auto_skill",
            "terminated": True,
            "answer_score": 1.0,
            "eval_meta": {"should_use_skill": True, "gold_skill": "CrossModalTaskSkill", "task_family": "cross_modal_task"},
            "skill_calls": [{"success": False, "response": {"error_code": 3, "text": "image not found"}}],
            "tool_responses": [{"api_name": "SkillExecutor", "response": {"error_code": 3, "text": "image not found"}}],
            "error_tags": ["raw_tool_failed"],
        }
    ]
    metrics = compute_agent_completion_metrics(traces)
    assert metrics["auto_skill"]["semantic_success_rate"] == 1.0
    assert metrics["auto_skill"]["final_success_rate"] == 0.0
