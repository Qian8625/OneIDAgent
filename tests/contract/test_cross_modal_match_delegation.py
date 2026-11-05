"""CrossModalMatchSkill 委托 CrossModalInterpret 的契约测试（M5 / G2 / G3）。

不依赖服务端：用 __new__ 造实例并替换 _call_tool，离线验证映射逻辑。
"""

from tool_server.tool_workers.skills.plugins.cross_modal_match.worker import (
    CrossModalMatchSkillWorker,
)

RGB = [
    {"bbox": [100, 100, 200, 200], "label": "ship", "score": 0.9},
    {"bbox": [520, 520, 600, 600], "label": "ship", "score": 0.6},
]
SAR = [
    {"bbox": [102, 101, 201, 199], "label": "ship", "score": 0.85},
]


def _worker(response):
    worker = CrossModalMatchSkillWorker.__new__(CrossModalMatchSkillWorker)
    worker.model_name = "CrossModalMatchSkill"

    def fake_call_tool(tool_name, payload, timeout=180):
        assert tool_name == "CrossModalInterpret"
        assert payload["mode"] == "shared_identity"
        return response

    worker._call_tool = fake_call_tool
    return worker


def _interpret_response():
    return {
        "error_code": 0,
        "mode": "shared_identity",
        "degraded": False,
        "degrade_reason": None,
        "summary": {"dsvi": {"blend_weight": 0.5, "accept_betp": 0.5}},
        "dedup_detections": [
            {
                "source": "matched", "label": "ship", "bbox": [101.0, 100.5, 200.5, 199.5],
                "score": 0.93, "match_score": 0.93,
                "rgb_bbox": [100, 100, 200, 200], "sar_bbox": [102, 101, 201, 199],
                "rgb_label": "ship", "sar_label": "ship",
                "vi_status": "Confirmed", "vi_p_class": 0.95, "vi_association": 0.91,
            },
            {
                "source": "rgb_only", "label": "ship", "bbox": [520, 520, 600, 600],
                "score": 0.6, "rgb_bbox": [520, 520, 600, 600], "sar_bbox": None,
            },
        ],
    }


def test_shares_geometry_and_blend_score():
    """G3：match_score 换成 VI blend 分，几何分另存。"""
    worker = _worker(_interpret_response())
    trace, calls = [], []
    result = worker._shared_identity_match(
        {"rgb_image": "a.png", "sar_image": "b.png"}, "ship", RGB, SAR,
        0.2, "auto", "hybrid", trace, calls,
    )
    assert result is not None
    matched, rgb_only, sar_only, meta = result
    assert len(matched) == 1 and len(rgb_only) == 1 and len(sar_only) == 0
    pair = matched[0]
    assert pair["match_score"] == 0.93                      # VI blend 分
    assert pair["geometry_match_score"] > 0.0               # 旧几何分保留
    assert pair["vi_status"] == "Confirmed"
    assert pair["vi_p_class"] == 0.95
    assert meta["mode"] == "shared_identity"
    assert meta["degraded"] is False
    assert meta["blend_weight"] == 0.5
    assert calls and calls[0]["tool"] == "CrossModalInterpret"


def test_unsupported_target_falls_back_with_reason():
    """G2：非 6 类 canonical target 直接走本地贪心并标注原因。"""
    worker = _worker(_interpret_response())
    trace, calls = [], []
    result = worker._shared_identity_match(
        {}, "bus", RGB, SAR, 0.2, "auto", "hybrid", trace, calls,
    )
    assert result is None
    assert worker._shared_identity_reason == "unsupported_target"
    assert calls == []


def test_target_alias_is_canonicalised():
    worker = _worker(_interpret_response())
    trace, calls = [], []
    result = worker._shared_identity_match(
        {}, "airplane", RGB, SAR, 0.2, "auto", "hybrid", trace, calls,
    )
    # airplane 映射到 aircraft，属于被支持的 target，因此会真的发起调用
    assert result is not None
    assert calls and calls[0]["success"] is True


def test_interpret_failure_falls_back():
    worker = _worker({"error_code": 2, "text": "invalid_dependency"})
    trace, calls = [], []
    result = worker._shared_identity_match(
        {}, "ship", RGB, SAR, 0.2, "auto", "hybrid", trace, calls,
    )
    assert result is None
    assert worker._shared_identity_reason == "interpret_unavailable"


def test_empty_dedup_falls_back():
    worker = _worker({"error_code": 0, "mode": "shared_identity", "dedup_detections": []})
    trace, calls = [], []
    result = worker._shared_identity_match(
        {}, "ship", RGB, SAR, 0.2, "auto", "hybrid", trace, calls,
    )
    assert result is None
    assert worker._shared_identity_reason == "interpret_empty"
