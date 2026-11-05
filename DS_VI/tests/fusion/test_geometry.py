"""Unit tests for NMS, registration and Hungarian association."""

from __future__ import annotations


import numpy as np
import pytest



from DS_VI.models.ds_fusion.geometry import (
    affine_registration,
    hungarian_association,
    pair_geometry,
    rotated_nms,
    transform_boxes,
)


def _box(bbox, score=0.5, label="ship", index=0):
    return {"bbox": list(bbox), "score": score, "label": label, "index": index}


def test_rotated_nms_suppresses_overlap_keeps_distant():
    boxes = [
        _box([0, 0, 100, 100], score=0.9, index=0),
        _box([5, 5, 105, 105], score=0.8, index=1),
        _box([500, 500, 600, 600], score=0.7, index=2),
    ]
    kept = rotated_nms(boxes, iou_threshold=0.5)
    kept_indexes = sorted(item["index"] for item in kept)
    assert kept_indexes == [0, 2]


def test_rotated_nms_keeps_other_class():
    boxes = [
        _box([0, 0, 100, 100], score=0.9, label="ship", index=0),
        _box([0, 0, 100, 100], score=0.8, label="harbor", index=1),
    ]
    kept = rotated_nms(boxes, iou_threshold=0.5)
    assert len(kept) == 2


def test_transform_boxes_translation():
    boxes = [_box([10, 20, 30, 40])]
    moved = transform_boxes(boxes, [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0]])
    assert moved[0]["bbox"] == pytest.approx([15.0, 17.0, 35.0, 37.0])


def test_affine_registration_identity_keeps_sar_boxes():
    rgb = [_box([0, 0, 50, 50], index=0)]
    sar = [_box([10, 10, 60, 60], index=1)]
    rgb_out, sar_out, info = affine_registration(rgb, sar, None, None, mode="identity")
    assert info["applied"] == "identity"
    assert rgb_out[0]["bbox"] == [0, 0, 50, 50]
    assert sar_out[0]["bbox"] == [10, 10, 60, 60]


def test_pair_geometry_identical_boxes():
    box = _box([0, 0, 100, 100])
    record = pair_geometry(box, box, {"w_iou": 0.5, "w_center": 0.3, "w_area": 0.2}, center_scale=0.75)
    assert record["iou"] == pytest.approx(1.0)
    assert record["center_score"] == pytest.approx(1.0)
    assert record["area_ratio"] == pytest.approx(1.0)
    assert record["match_score"] == pytest.approx(1.0)


def test_hungarian_pairs_and_gates():
    rgb = [_box([0, 0, 100, 100], score=0.9, index=0), _box([500, 500, 560, 560], score=0.8, index=1)]
    sar = [_box([5, 5, 105, 105], score=0.85, index=0)]
    matched, rgb_only, sar_only = hungarian_association(rgb, sar, gate_iou=0.05, gate_score=0.35)
    assert len(matched) == 1
    assert matched[0]["rgb_index"] == 0
    assert matched[0]["sar_index"] == 0
    assert [item["rgb_index"] for item in rgb_only] == [1]
    assert sar_only == []


def test_hungarian_no_pair_when_gate_fails():
    rgb = [_box([0, 0, 50, 50], index=0)]
    sar = [_box([900, 900, 950, 950], index=0)]
    matched, rgb_only, sar_only = hungarian_association(rgb, sar, gate_iou=0.05, gate_score=0.35)
    assert matched == []
    assert len(rgb_only) == 1
    assert len(sar_only) == 1


def test_hungarian_empty_inputs():
    matched, rgb_only, sar_only = hungarian_association([], [])
    assert matched == [] and rgb_only == [] and sar_only == []
    matched, rgb_only, sar_only = hungarian_association([_box([0, 0, 10, 10])], [])
    assert matched == [] and len(rgb_only) == 1 and sar_only == []


def test_hungarian_returns_finite_scores():
    rng = np.random.default_rng(0)
    rgb = [_box([float(x), float(y), float(x) + 60, float(y) + 40], score=0.6, index=i) for i, (x, y) in enumerate(rng.uniform(0, 900, size=(6, 2)))]
    sar = [_box([float(x), float(y), float(x) + 60, float(y) + 40], score=0.6, index=i) for i, (x, y) in enumerate(rng.uniform(0, 900, size=(6, 2)))]
    matched, _rgb_only, _sar_only = hungarian_association(rgb, sar)
    for item in matched:
        assert 0.0 <= item["match_score"] <= 1.0
