"""Unit tests for the time-arm evaluation (identity + change metrics)."""

from __future__ import annotations


import pytest


import DS_VI.experiments.metrics.evaluate as eval_dual_arm
time_metrics = eval_dual_arm.time_metrics
_greedy_pairs = eval_dual_arm._greedy_pairs


def _track(label, change, bbox, bbox_t0=None, bbox_t1=None, source="matched"):
    return {
        "label": label,
        "change": change,
        "bbox": list(bbox),
        "bbox_t0": list(bbox_t0) if bbox_t0 else None,
        "bbox_t1": list(bbox_t1) if bbox_t1 else None,
        "source": source,
    }


def _pred(label, change, bbox, bbox_t0=None, bbox_t1=None, source="matched", score=0.9):
    return {
        "label": label,
        "change": change,
        "bbox": list(bbox),
        "bbox_t0": list(bbox_t0) if bbox_t0 else None,
        "bbox_t1": list(bbox_t1) if bbox_t1 else None,
        "source": source,
        "score": score,
    }


def test_greedy_pairs_one_to_one_and_class_aware():
    gt = [{"label": "ship", "bbox": [0, 0, 10, 10]}, {"label": "ship", "bbox": [20, 20, 30, 30]}]
    pred = [{"label": "ship", "bbox": [0, 0, 10, 10]}, {"label": "tank", "bbox": [20, 20, 30, 30]}]
    pairs = _greedy_pairs(gt, pred)
    assert len(pairs) == 1
    assert pairs[0][0] == 0 and pairs[0][1] == 0


def test_perfect_time_metrics():
    gt = {
        "p1": [
            _track("ship", "stayed", [0, 0, 10, 10], [0, 0, 10, 10], [1, 0, 11, 10]),
            _track("ship", "moved", [20, 20, 30, 30], [20, 20, 30, 30], [25, 20, 35, 30]),
            _track("tank", "left", [40, 40, 50, 50], [40, 40, 50, 50], None, source="view_a_only"),
            _track("tank", "appeared", [60, 60, 70, 70], None, [60, 60, 70, 70], source="view_b_only"),
        ]
    }
    pred = {
        "p1": [
            _pred("ship", "stayed", [0, 0, 10, 10], [0, 0, 10, 10], [1, 0, 11, 10]),
            _pred("ship", "moved", [20, 20, 30, 30], [20, 20, 30, 30], [25, 20, 35, 30]),
            _pred("tank", "left", [40, 40, 50, 50], [40, 40, 50, 50], None, source="view_a_only"),
            _pred("tank", "appeared", [60, 60, 70, 70], None, [60, 60, 70, 70], source="view_b_only"),
        ]
    }
    metrics = time_metrics(gt, pred, ["ship", "tank"])
    assert metrics["identity"]["micro"]["f1"] == pytest.approx(1.0)
    assert metrics["change"]["accuracy"] == pytest.approx(1.0)
    assert metrics["presence"]["t0_accuracy"] == pytest.approx(1.0)
    assert metrics["presence"]["t1_accuracy"] == pytest.approx(1.0)
    assert metrics["motion"]["accuracy"] == pytest.approx(1.0)


def test_false_positive_and_missed_identity():
    gt = {"p1": [_track("ship", "stayed", [0, 0, 10, 10], [0, 0, 10, 10], [1, 0, 11, 10])]}
    pred = {
        "p1": [
            _pred("ship", "stayed", [100, 100, 110, 110], [100, 100, 110, 110], [101, 100, 111, 110]),
            _pred("ship", "stayed", [0, 0, 10, 10], [0, 0, 10, 10], [1, 0, 11, 10]),
        ]
    }
    metrics = time_metrics(gt, pred, ["ship"])
    assert metrics["identity"]["per_class"]["ship"]["tp"] == 1
    assert metrics["identity"]["per_class"]["ship"]["fp"] == 1
    assert metrics["identity"]["per_class"]["ship"]["fn"] == 0


def test_wrong_change_label_lowers_accuracy():
    gt = {"p1": [_track("ship", "stayed", [0, 0, 10, 10], [0, 0, 10, 10], [1, 0, 11, 10])]}
    pred = {"p1": [_pred("ship", "moved", [0, 0, 10, 10], [0, 0, 10, 10], [1, 0, 11, 10])]}
    metrics = time_metrics(gt, pred, ["ship"])
    assert metrics["change"]["accuracy"] == pytest.approx(0.0)
    assert metrics["motion"]["accuracy"] == pytest.approx(0.0)


def test_empty_inputs():
    metrics = time_metrics({}, {}, ["ship"])
    assert metrics["identity"]["micro"]["f1"] == 0.0
    assert metrics["change"]["support"] == 0


def test_same_average_box_does_not_prove_identity():
    a, b = [0, 0, 10, 10], [20, 0, 30, 10]
    gt = {"p1": [_track("ship", "moved", [10, 0, 20, 10], a, b)]}
    pred = {"p1": [_pred("ship", "moved", [10, 0, 20, 10], b, a)]}
    metrics = time_metrics(gt, pred, ["ship"])
    assert metrics["identity"]["micro"]["tp"] == 0
    assert metrics["identity"]["micro"]["fp"] == 1
    assert metrics["identity"]["micro"]["fn"] == 1
    assert metrics["change"]["missed"] == metrics["change"]["spurious"] == 1


def test_e2e_change_penalizes_misses_duplicates_and_extra_images():
    box = [0, 0, 10, 10]
    event = _track("ship", "stayed", box, box, box)
    prediction = _pred("ship", "stayed", box, box, box)
    gt = {"p1": [event], "p2": [event]}
    pred = {"p1": [prediction, prediction], "extra": [prediction]}
    change = time_metrics(gt, pred, ["ship"])["change"]
    assert change["conditional_accuracy"] == 1
    assert change["missed"] == 1 and change["spurious"] == 2
    assert change["accuracy"] == 0.25
    assert change["micro_f1"] == 0.4
    assert change["per_class"]["stayed"]["fn"] == 1
    assert change["per_class"]["stayed"]["fp"] == 2


def test_wrong_presence_is_missed_event_and_spurious_prediction():
    box = [0, 0, 10, 10]
    gt = {"p1": [_track("ship", "stayed", box, box, box)]}
    pred = {"p1": [_pred("ship", "left", box, box, None, source="view_a_only")]}
    change = time_metrics(gt, pred, ["ship"])["change"]
    assert change["per_class"]["stayed"]["fn"] == 1
    assert change["per_class"]["left"]["fp"] == 1
    assert change["micro_f1"] == 0
