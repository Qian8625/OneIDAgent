"""CrossModalInterpret 对外契约测试（M2 验收 ①，M3 验收 ③，G1）。

不依赖 pytest：函数式断言，由 scripts/ci/run_contract_tests.py 调用。
"""

import copy
import json

from tool_server.tool_workers.online_workers.CrossModalInterpret_worker import (
    CrossModalInterpreter,
    OneIDCrossModalEngine,
)

PAIR = {
    "target": "ship",
    "task": "crosscheck_exists",
    "rgb_detections": [
        {"bbox": [100, 100, 200, 200], "label": "ship", "score": 0.9},
        {"bbox": [300, 300, 400, 400], "label": "ship", "score": 0.8},
        {"bbox": [520, 520, 600, 600], "label": "ship", "score": 0.6},
    ],
    "sar_detections": [
        {"bbox": [102, 101, 201, 199], "label": "ship", "score": 0.85},
        {"bbox": [305, 302, 402, 401], "label": "ship", "score": 0.7},
    ],
}

LEGACY_FIELDS = {
    "text", "error_code", "task", "target", "output_mode", "truncated",
    "mode", "summary", "conclusion_type", "final_bbox", "selected_targets", "structured",
}

LEGACY_SUMMARY_KEYS = {
    "target", "raw_target", "rgb_input_count", "sar_input_count",
    "rgb_filtered_count", "sar_filtered_count", "matched_count",
    "rgb_only_count", "sar_only_count", "dedup_count", "source_counts",
    "match_method", "match_threshold", "weights",
}


def test_legacy_hard_field_envelope_is_unchanged():
    engine = OneIDCrossModalEngine()
    response = engine.generate({**copy.deepcopy(PAIR), "mode": "legacy_hard"})
    assert response["error_code"] == 0
    assert LEGACY_FIELDS.issubset(set(response)), sorted(LEGACY_FIELDS - set(response))
    assert LEGACY_SUMMARY_KEYS.issubset(set(response["summary"]))
    assert response["summary"]["match_method"] == "global_sorted_greedy_hybrid_score"


def test_shared_identity_carries_evidence_layer():
    engine = OneIDCrossModalEngine()
    response = engine.generate(copy.deepcopy(PAIR))
    assert response["error_code"] == 0
    assert response["mode"] == "shared_identity"
    assert response["degraded"] is False
    for key in ("evidence_summary", "inference", "object_hypotheses"):
        assert key in response, key
    assert response["summary"]["match_method"] == "shared_identity_cascade"


def test_shared_identity_is_deterministic():
    engine = OneIDCrossModalEngine()
    first = engine.generate(copy.deepcopy(PAIR))
    second = engine.generate(copy.deepcopy(PAIR))
    for response in (first, second):
        response.pop("structured", None)
    assert json.dumps(first, sort_keys=True, ensure_ascii=False) == json.dumps(second, sort_keys=True, ensure_ascii=False)


#: 在线适配器在 full 模式必须给出的证据层细节。
FULL_ONLY_KEYS = ("primitive_observations", "claims", "dedup_detections", "objects", "vi_diagnostics")


def test_compact_strips_big_payload_fields():
    engine = OneIDCrossModalEngine()
    compact = engine.generate(copy.deepcopy(PAIR))
    full = engine.generate({**copy.deepcopy(PAIR), "output_mode": "full"})
    for key in FULL_ONLY_KEYS:
        assert key not in compact, key
        assert key in full, key
    # compact 的证据层只保留摘要，不得把推断细节带进上下文
    for key in ("primitive_observations", "claims", "candidate_edges"):
        assert key not in compact, key
    assert compact["structured"]["payload_bytes"] < 20000


def test_unknown_task_is_rejected():
    engine = OneIDCrossModalEngine()
    response = engine.generate({**copy.deepcopy(PAIR), "task": "teleport_the_ship"})
    assert response["error_code"] == 2
    assert response["structured"]["failure_reason"] == "unsupported task"


def test_unsupported_target_is_rejected():
    engine = OneIDCrossModalEngine()
    response = engine.generate({**copy.deepcopy(PAIR), "target": "submarine"})
    assert response["error_code"] == 2
    assert response["structured"]["failure_reason"] == "unsupported target"


def test_text_without_provenance_is_rejected():
    engine = OneIDCrossModalEngine()
    response = engine.generate({"target": "ship", "task": "crosscheck_exists",
                                "rgb_text": "(10, 20, 50, 80), ship, score 0.73"})
    assert response["error_code"] == 2
    assert response["structured"]["failure_reason"] == "missing evidence_provenance"


def test_provenance_lets_text_through():
    engine = OneIDCrossModalEngine()
    response = engine.generate({
        "target": "ship", "task": "crosscheck_exists",
        "rgb_text": "(100, 100, 200, 200), ship, score 0.9",
        "sar_text": "(102, 101, 201, 199), ship, score 0.85",
        "evidence_provenance": {"rgb": {"source_tool": "ObjectDetection"},
                                "sar": {"source_tool": "DenoDetV2"}},
    })
    assert response["error_code"] == 0


def test_empty_detections_return_empty_success():
    engine = OneIDCrossModalEngine()
    response = engine.generate({"target": "ship", "task": "crosscheck_exists",
                                "rgb_detections": [], "sar_detections": []})
    assert response["error_code"] == 0
    assert response["dedup_count"] == 0


def test_public_symbols_are_still_exported():
    """评测脚本依赖这些符号，不能因为重构而丢失。"""
    for name in ("FINAL_TARGETS", "TARGET_ALIASES", "TASK_ALIASES",
                 "MATCH_THRESHOLD", "DEFAULT_DETECTION_CONFIDENCE"):
        assert hasattr(CrossModalInterpreter, name), name
    for name in ("normalize_label", "canonical_target", "canonical_task",
                 "compute_pair_score", "match_detections", "generate"):
        assert callable(getattr(CrossModalInterpreter(), name, None)), name
