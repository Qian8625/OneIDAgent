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


def _replay_registry():
    return {
        RGB_DETECTIONS_KEY: [{"bbox": [1, 2, 3, 4], "label": "ship", "score": 0.9}],
        SAR_DETECTIONS_KEY: [{"bbox": [5, 6, 7, 8], "label": "ship", "score": 0.8}],
        "latest_rgb_evidence_provenance": {"source_tool": "ObjectDetection", "source_round": 1},
        "latest_sar_evidence_provenance": {"source_tool": "DenoDetV2", "source_round": 2},
    }


def test_replay_requires_live_detections_and_inherits_registry():
    # Replay runs live detection; the workbook only carries attributes.
    for dataset, family in (("task4", "match"), ("task5", "time")):
        base = {"source_mode": "workbook_replay", "dataset_id": dataset, "pair_id": dataset + "_000001"}
        _, error = prepare_interpret_args(base, {RGB_DETECTIONS_KEY: [{"bbox": [1, 2, 3, 4]}]})
        assert error == "invalid_dependency"
        args, error = prepare_interpret_args(base, _replay_registry())
        assert error is None
        assert args["family"] == family
        assert args["rgb_detections"] == _replay_registry()[RGB_DETECTIONS_KEY]
        assert args["sar_detections"] == _replay_registry()[SAR_DETECTIONS_KEY]
        assert args["evidence_provenance"]["rgb"]["source_tool"] == "ObjectDetection"
        assert args["evidence_provenance"]["sar"]["source_tool"] == "DenoDetV2"


def test_replay_rejects_mixed_and_malformed_requests():
    base = {"source_mode": "workbook_replay", "dataset_id": "task4", "pair_id": "task4_000001"}
    for extra in ({"family": "time"}, {"workbook_path": "/tmp/truth.xlsx"}, {"vi_config_path": "custom.yaml"},
                  {"dataset_id": []}, {"pair_id": ""}, {"rgb_text": "(1, 2, 3, 4), ship"},
                  {"georef_image": "/tmp/g.tif"}, {"allow_unprovenanced_text": True}):
        _, error = prepare_interpret_args(dict(base, **extra), {})
        assert error == "invalid_replay_request"
    assert dependency_hint("invalid_replay_request")


def test_detection_set_alias_is_normalised_for_both_modes():
    payload = {
        "rgb_detection_set": [{"bbox": [1, 2, 3, 4], "label": "ship", "score": 0.9}],
        "sar_detection_set": [{"bbox": [5, 6, 7, 8], "label": "ship", "score": 0.8}],
        "evidence_provenance": {"rgb": {"source_tool": "ObjectDetection", "source_round": 1}},
    }
    for request in ({"target": "ship", "task": "joint_scene"},
                    {"source_mode": "workbook_replay", "dataset_id": "task4", "pair_id": "task4_000001"}):
        args, error = prepare_interpret_args(dict(request, **payload), _replay_registry())
        assert error is None, error
        assert args["rgb_detections"] == payload["rgb_detection_set"]
        assert args["sar_detections"] == payload["sar_detection_set"]
        assert "rgb_detection_set" not in args and "sar_detection_set" not in args


def test_replay_binds_temporal_images_not_sar_alias():
    registry = dict(_replay_registry(), time1_image="/data/pre.png", time2_image="/data/post.png", sar_image="/wrong.png")
    args, error = prepare_interpret_args({"source_mode": "workbook_replay", "dataset_id": "task5", "pair_id": "task5_000101"}, registry)
    assert error is None
    assert args["time1_image"] == "/data/pre.png"
    assert args["time2_image"] == "/data/post.png"
    assert "sar_image" not in args


def test_registry_keeps_full_evidence_but_observation_is_bounded():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "tool_server/tf_eval/agent_runtime/structured_summary.py"
    spec = importlib.util.spec_from_file_location("replay_structured_summary", path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    get_structured_result_for_registry = summary.get_structured_result_for_registry
    build_structured_observation = summary.build_structured_observation
    detections = [{"bbox": [i, 0, i + 1, 1], "label": "ship", "score": 0.9, "attributes": {"color": "gray"}} for i in range(25)]
    response = {"error_code": 0, "structured": {"detections": detections}}
    result = get_structured_result_for_registry(response)
    assert len(result["detections"]) == 25
    assert result["detections"][24]["attributes"]["color"] == "gray"
    assert "truncated" in build_structured_observation(response)
    assert get_structured_result_for_registry({"error_code": 1}) is None
