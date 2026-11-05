"""Unit tests for the dual-view (match/time) multi-class D-S pipeline."""

from __future__ import annotations




import DS_VI.models.ds_fusion.pipeline as dual_pipeline
run_pair = dual_pipeline.run_pair
from DS_VI.models.ds_fusion.config import load_config as load_config


def _pair(family):
    return {
        "base_id": "synthetic",
        "family": family,
        "images": {"a": "/nonexistent/a.png", "b": "/nonexistent/b.png"},
    }


def _det(bbox, label, score=0.9, index=0):
    return {"bbox": list(bbox), "label": label, "score": score, "index": index, "view": "a", "source": "tool_json"}


def test_match_pairs_and_single_view_objects():
    config = load_config({"targets": ["ship", "tank"]})
    detections = {
        "a": [_det([0, 0, 100, 100], "ship", 0.9, 0), _det([500, 500, 560, 560], "tank", 0.9, 1)],
        "b": [_det([5, 5, 105, 105], "ship", 0.9, 0)],
    }
    result = run_pair(_pair("match"), detections, config)
    sources = {(obj["label"], obj["source"]) for obj in result["objects"]}
    assert ("ship", "matched") in sources
    assert ("tank", "view_a_only") in sources
    ship = next(obj for obj in result["objects"] if obj["label"] == "ship")
    assert ship["bbox"] is not None
    assert ship["view_a_bbox"] == [0, 0, 100, 100]
    assert ship["view_b_bbox"] == [5, 5, 105, 105]


def test_time_change_labels():
    config = load_config({"targets": ["ship"]})
    detections = {
        "a": [
            _det([0, 0, 100, 100], "ship", 0.9, 0),
            _det([400, 0, 450, 50], "ship", 0.9, 1),
        ],
        "b": [
            _det([5, 5, 105, 105], "ship", 0.9, 0),
            _det([445, 0, 495, 50], "ship", 0.9, 1),
        ],
    }
    result = run_pair(_pair("time"), detections, config)
    by_t0 = {tuple(obj["view_a_bbox"]): obj for obj in result["objects"] if obj["view_a_bbox"]}
    stayed = by_t0[(0, 0, 100, 100)]
    assert stayed["change"] == "stayed"
    assert stayed["source"] == "matched"


def test_time_left_and_appeared():
    config = load_config({"targets": ["ship"]})
    detections = {
        "a": [_det([0, 0, 80, 80], "ship", 0.9, 0)],
        "b": [_det([300, 300, 380, 380], "ship", 0.9, 0)],
    }
    result = run_pair(_pair("time"), detections, config)
    change_by_source = {obj["source"]: obj["change"] for obj in result["objects"]}
    assert change_by_source.get("view_a_only") == "left"
    assert change_by_source.get("view_b_only") == "appeared"


def test_time_moved_label():
    config = load_config({"targets": ["ship"]})
    detections = {
        "a": [_det([0, 0, 50, 50], "ship", 0.9, 0)],
        "b": [_det([40, 0, 90, 50], "ship", 0.9, 0)],
    }
    result = run_pair(_pair("time"), detections, config)
    matched = [obj for obj in result["objects"] if obj["source"] == "matched"]
    assert len(matched) == 1
    assert matched[0]["change"] == "moved"


def test_class_filtering_and_empty_inputs():
    config = load_config({"targets": ["ship", "tank"]})
    result = run_pair(_pair("match"), {"a": [], "b": []}, config)
    assert result["objects"] == []
    assert result["counts"]["output"] == 0


def test_ship_only_override_matches_legacy_target_list():
    config = load_config({"targets": ["ship"]})
    detections = {
        "a": [_det([0, 0, 100, 100], "tank", 0.9, 0)],
        "b": [_det([0, 0, 100, 100], "tank", 0.9, 0)],
    }
    result = run_pair(_pair("match"), detections, config)
    assert result["objects"] == []
