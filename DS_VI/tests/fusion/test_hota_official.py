"""Official HOTA implementation: hand-checkable cases.

Two views = two frames, one object = one track.  The cases below have exact
closed forms, so a regression in the matching or the A(c) accounting shows up as
a wrong number rather than as a plausible-looking one.
"""
from __future__ import annotations

import math

from DS_VI.experiments.metrics.hota import THRESHOLDS, aggregate, pair_hota


def _instance(instance_id, view, box, category="ship"):
    return {
        "instance_id": instance_id,
        "pair_id": instance_id.split(":")[0],
        "view": view,
        "category": category,
        "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
    }


def _object(object_id, members, category="ship"):
    row = {"object_id": object_id, "pair_id": object_id.split(":")[0], "category": category}
    for key, instance_id in members.items():
        row[f"instance_{key}"] = instance_id
    return row


BOX_A = [0.0, 0.0, 10.0, 10.0]
BOX_B = [100.0, 100.0, 110.0, 110.0]

INSTANCES = {
    "p:a1": _instance("p:a1", "rgb", BOX_A),
    "p:a2": _instance("p:a2", "rgb", BOX_B),
    "p:b1": _instance("p:b1", "sar", BOX_A),
    "p:b2": _instance("p:b2", "sar", BOX_B),
}

TRUTH = [
    _object("gt1", {"a": "p:a1", "b": "p:b1"}),
    _object("gt2", {"a": "p:a2", "b": "p:b2"}),
]


def test_perfect_prediction_scores_one():
    counts = pair_hota(list(TRUTH), TRUTH, INSTANCES, INSTANCES)
    result = aggregate([counts])
    assert result["hota"] == 1.0
    assert result["deta"] == 1.0
    assert result["assa"] == 1.0
    assert len(result["per_threshold"]) == len(THRESHOLDS)


def test_crossed_ids_keep_detection_but_lose_association():
    # The boxes are right, the identities are swapped: DetA stays 1, AssA drops.
    crossed = [
        _object("pr1", {"a": "p:a1", "b": "p:b2"}),
        _object("pr2", {"a": "p:a2", "b": "p:b1"}),
    ]
    result = aggregate([pair_hota(crossed, TRUTH, INSTANCES, INSTANCES)])
    assert result["deta"] == 1.0
    # Each matched detection sees one sibling on its GT track with another ID and
    # one sibling on its predicted track with another GT -> A(c) = 1/3.
    assert abs(result["assa"] - 1 / 3) < 1e-6
    assert abs(result["hota"] - math.sqrt(1 / 3)) < 1e-6


def test_missing_object_costs_detection_only():
    partial = list(TRUTH[:1])          # one of two GT objects is predicted
    result = aggregate([pair_hota(partial, TRUTH, INSTANCES, INSTANCES)])
    assert result["deta"] == 0.5       # 2 matched detections, 2 missed, 0 extra
    assert result["assa"] == 1.0       # the surviving track is perfectly aligned
    assert abs(result["hota"] - math.sqrt(0.5)) < 1e-6


def test_threshold_controls_localisation():
    # A 1-pixel shift on a 10-pixel box keeps IoU high but not 1.0.
    shifted = dict(INSTANCES)
    shifted["p:a1"] = _instance("p:a1", "rgb", [1.0, 0.0, 11.0, 10.0])
    shifted["p:b1"] = _instance("p:b1", "sar", [1.0, 0.0, 11.0, 10.0])
    # Only the shifted object is predicted, so every GT detection it covers is off.
    result = aggregate([pair_hota(list(TRUTH[:1]), TRUTH, shifted, INSTANCES)])
    assert result["per_threshold"][0]["alpha"] == THRESHOLDS[0]
    assert result["per_threshold"][-1]["alpha"] == THRESHOLDS[-1]
    # IoU = 9/11 = 0.818: matched up to alpha 0.80, unmatched from 0.85 on.
    matched_high = [row for row in result["per_threshold"] if row["deta"] > 0.0]
    assert max(row["alpha"] for row in matched_high) == 0.8
    assert 0.0 < result["hota"] < 1.0


def test_counts_aggregate_across_pairs():
    one = pair_hota(list(TRUTH), TRUTH, INSTANCES, INSTANCES)
    two = pair_hota(list(TRUTH[:1]), TRUTH, INSTANCES, INSTANCES)
    result = aggregate([one, two])
    # 6 matched of 8 detections, no FPs, association sum 2 + 2 over 4 + 2 matches.
    assert 0.0 < result["deta"] < 1.0
    assert result["hota"] < aggregate([one])["hota"]
