"""CrossModalTaskSkill 对 DS+VI 裁决词的归一化。

任务分支只认"配上了/没配上"，而共享身份会产出论文四值证据状态
（Confirmed / Weak Support / Conflict / Uncertain）以及降级原因；
这里验证它们统一落到任务级 insufficient_evidence，且绝不把弱证据升级成 success。
"""

from tool_server.tool_workers.skills.plugins.cross_modal_task.worker import (
    CrossModalTaskSkillWorker,
)


def _response(evidence, status="success"):
    return CrossModalTaskSkillWorker._response(
        "text", 0, "ship", "optical_sar_verify", {"label": "yes"}, evidence, None, status,
    )


def test_uncertain_verdict_downgrades_to_insufficient_evidence():
    response = _response({"matched_pairs": [{"vi_status": "Uncertain", "match_score": 0.5}]})
    assert response["structured"]["status"] == "insufficient_evidence"
    assert response["structured"]["failure_reason"] == "vi_status_unknown"


def test_weak_support_verdict_downgrades():
    response = _response({"rgb_only": [{"vi_status": "Weak Support"}]})
    assert response["structured"]["status"] == "insufficient_evidence"


def test_confirmed_verdict_stays_success():
    response = _response({"matched_pairs": [{"vi_status": "Confirmed", "match_score": 0.5}]})
    assert response["structured"]["status"] == "success"
    assert response["structured"]["vi_evidence"] == ["Confirmed"]


def test_degraded_pairing_downgrades():
    response = _response({
        "matched_pairs": [{"vi_status": None}],
        "pairing": {"degraded": True, "degrade_reason": "interpret_unavailable"},
    })
    assert response["structured"]["status"] == "insufficient_evidence"


def test_mixed_states_keep_success_when_any_strong():
    response = _response({
        "matched_pairs": [{"vi_status": "Confirmed"}],
        "rgb_only": [{"vi_status": "Uncertain"}],
    })
    assert response["structured"]["status"] == "success"


def test_non_success_status_is_never_upgraded():
    response = _response({"matched_pairs": [{"vi_status": "Confirmed"}]}, status="insufficient_evidence")
    assert response["structured"]["status"] == "insufficient_evidence"
