"""RGB/SAR 检测证据管道的契约测试（M3.5 验收 ②，G5 回归）。

覆盖两条真实链路：
- ObjectDetection（RGB）→ DenoDetV2（SAR）→ CrossModalInterpret
- 任一侧缺失 / 空检测 / 示例框泄漏 / 非检测工具
"""

from tool_server.tool_workers.cross_modal_evidence import (
    EMPTY_DETECTION_HINT,
    RGB_DETECTIONS_KEY,
    SAR_DETECTIONS_KEY,
    build_provenance,
    dependency_hint,
    extract_detections,
    prepare_interpret_args,
    register_detections,
)


def _rgb_structured():
    return {
        "status": "ok",
        "detections": [
            {"bbox": [100, 100, 200, 200], "label": "ship", "score": 0.91},
            {"bbox": [300, 300, 400, 400], "label": "ship", "score": 0.72},
        ],
    }


def _sar_structured():
    return {
        "status": "ok",
        "detections": [
            {"bbox": [102, 101, 201, 199], "label": "ship", "score": 0.83},
        ],
    }


def test_object_detection_then_denodet_then_interpret():
    registry = {}
    registry = register_detections(registry, "ObjectDetection", _rgb_structured(), 1)
    registry = register_detections(registry, "DenoDetV2", _sar_structured(), 2)

    assert len(registry[RGB_DETECTIONS_KEY]) == 2
    assert len(registry[SAR_DETECTIONS_KEY]) == 1
    assert registry["latest_rgb_evidence_provenance"]["source_tool"] == "ObjectDetection"
    assert registry["latest_sar_evidence_provenance"]["source_round"] == 2

    args, error = prepare_interpret_args({"target": "ship", "task": "crosscheck_exists"}, registry)
    assert error is None
    assert len(args["rgb_detections"]) == 2
    assert len(args["sar_detections"]) == 1
    assert args["evidence_provenance"]["rgb"]["source_tool"] == "ObjectDetection"
    assert args["evidence_provenance"]["sar"]["source_tool"] == "DenoDetV2"


def test_small_object_detection_counts_as_rgb():
    registry = register_detections({}, "SmallObjectDetection", _rgb_structured(), 3)
    assert RGB_DETECTIONS_KEY in registry


def test_missing_sar_returns_invalid_dependency():
    registry = register_detections({}, "ObjectDetection", _rgb_structured(), 1)
    args, error = prepare_interpret_args({"target": "ship"}, registry)
    assert error == "invalid_dependency"
    assert args.get("rgb_detections")


def test_register_ignores_non_detection_tools():
    registry = {"keep": 1}
    assert register_detections(registry, "AddText", _rgb_structured(), 4) == registry
    assert register_detections(registry, "ObjectDetection", {"status": "no_detections", "detections": []}, 5) == registry


def test_text_without_provenance_is_rejected():
    args, error = prepare_interpret_args({"rgb_text": "(1, 2, 3, 4), ship, score 0.9"}, {})
    assert error == "invalid_dependency"


def test_example_bbox_leakage_is_detected():
    args, error = prepare_interpret_args(
        {"rgb_text": "(10, 20, 50, 80), ship, score 0.73", "sar_text": "(12, 22, 52, 82), ship, score 0.86"},
        {},
    )
    assert error in {"example_leakage_detected", "invalid_dependency"}


def test_empty_chain_gets_explicit_next_step():
    """G5：全链路检测失败时不能只说"依赖不满足"，要给出下一步。"""
    hint = dependency_hint("invalid_dependency")
    assert hint == EMPTY_DETECTION_HINT
    assert "ObjectDetection" in hint and "DenoDetV2" in hint
    assert dependency_hint(None) is None


def test_extract_detections_handles_nested_evidence():
    structured = {"evidence": {"rgb_evidence": [{"bbox": [1, 2, 3, 4], "label": "tank", "score": 0.5}]}}
    assert extract_detections(structured)[0]["label"] == "tank"
    assert extract_detections({"detections": []}) == []
    assert extract_detections(None) == []


def test_provenance_records_round_and_tool():
    record = build_provenance("DenoDetV2", 7)
    assert record["source_tool"] == "DenoDetV2"
    assert record["source_round"] == 7
    assert record["source_response_id"] == "round_7_denodetv2_structured"
