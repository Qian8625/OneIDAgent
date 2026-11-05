"""Unit test: cascade VI records keep the class hint and posterior."""

from __future__ import annotations




import DS_VI.models.cascade.vi as dual_vi
import DS_VI.models.ds_fusion.config as pipeline


def _pair():
    return {
        "base_id": "synthetic",
        "family": "match",
        "images": {"a": "/nonexistent/a.png", "b": "/nonexistent/b.png"},
    }


def _det(bbox, label, score=0.9, index=0):
    return {"bbox": list(bbox), "label": label, "score": score, "index": index, "view": "a", "source": "tool_json"}


def test_matched_records_keep_class_and_posterior():
    config = pipeline.load_config({"targets": ["ship"]})
    detections = {
        "a": [_det([0, 0, 100, 100], "ship", 0.9, 0)],
        "b": [_det([5, 5, 105, 105], "ship", 0.9, 0)],
    }
    matched = [{
        "label": "ship",
        "source": "matched",
        "bbox": [2.5, 2.5, 102.5, 102.5],
        "score": 0.8,
        "view_a_index": 0,
        "view_b_index": 0,
        "view_a_bbox": [0, 0, 100, 100],
        "view_b_bbox": [5, 5, 105, 105],
    }]
    result = dual_vi.run_vi_pair(_pair(), detections, config, "match", matched=matched)
    assert len(result["objects"]) == 1
    record = result["objects"][0]
    assert record["class"] == "ship"
    assert record["p_class"] > 0.0
    assert record["association"] >= 0.0
    assert record["bbox"] is not None


def test_alone_mode_registers_pruned_candidates():
    config = pipeline.load_config({"targets": ["ship"]})
    detections = {
        "a": [_det([0, 0, 100, 100], "ship", 0.9, 0)],
        "b": [_det([5, 5, 105, 105], "ship", 0.9, 0), _det([500, 500, 600, 600], "ship", 0.9, 1)],
    }
    result = dual_vi.run_vi_pair(_pair(), detections, config, "match", matched=None)
    assert len(result["objects"]) >= 1
    assert all(record.get("class") == "ship" for record in result["objects"])
    assert result["diagnostics"]["observations"] == 3
