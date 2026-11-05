from tool_server.tf_eval.agent_runtime.agent_runner import prepare_cross_modal_interpret_args


def test_cross_modal_requires_provenance():
    args, error = prepare_cross_modal_interpret_args({"rgb_text": "bbox", "sar_text": "bbox"}, {})
    assert error == "invalid_dependency"
    assert args["rgb_text"] == "bbox"


def test_cross_modal_accepts_provenance():
    args, error = prepare_cross_modal_interpret_args(
        {
            "rgb_detections": [{"bbox": [0, 0, 10, 10], "label": "ship", "score": 0.9}],
            "sar_detections": [{"bbox": [0, 0, 10, 10], "label": "ship", "score": 0.8}],
            "evidence_provenance": {"rgb": {"source_tool": "ObjectDetection"}, "sar": {"source_tool": "DenoDetV2"}},
        },
        {},
    )
    assert error is None
    assert args["evidence_provenance"]
