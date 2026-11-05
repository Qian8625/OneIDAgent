"""Reported per-object status vocabulary: labels, assembly and scoring.

The status vocabulary is part of the output contract (match: ``match`` /
``rgb-only`` / ``sar-only`` / ``unknown``; time: ``static`` / ``moved`` /
``time1-only`` / ``time2-only``), so it gets its own tests rather than riding on
the metric fixtures.
"""
from __future__ import annotations

import pytest

from DS_VI.models.cascade import MATCH_STATUSES, STATUS_TO_CHANGE, TIME_STATUSES, object_status
from DS_VI.models.cascade.attributes import cascade_objects
from DS_VI.experiments.metrics.property_metrics import reported_change_label


def test_match_vocabulary():
    assert object_status("match", True, True) == "match"
    assert object_status("match", True, False) == "rgb-only"
    assert object_status("match", False, True) == "sar-only"
    # A candidate that lost both member indices is still reportable.
    assert object_status("match", False, False) == "unknown"
    assert set(object_status("match", a, b) for a in (0, 1) for b in (0, 1)) == set(MATCH_STATUSES)


def test_time_vocabulary():
    assert object_status("time", True, True, "stayed") == "static"
    assert object_status("time", True, True, "moved") == "moved"
    assert object_status("time", True, False) == "time1-only"
    assert object_status("time", False, True) == "time2-only"
    # An explicit change label is authoritative: a two-view object is never
    # reported as a single-side status.
    assert object_status("time", True, True, "appeared") == "static"
    assert {
        object_status("time", a, b, "moved")
        for a in (0, 1) for b in (0, 1) if a or b
    } <= set(TIME_STATUSES)


def test_time_rejects_a_member_less_object():
    with pytest.raises(ValueError):
        object_status("time", False, False)


def test_unknown_family_is_an_error():
    with pytest.raises(ValueError):
        object_status("multispectral", True, True)


def test_status_to_change_covers_the_time_vocabulary():
    assert set(STATUS_TO_CHANGE) == set(TIME_STATUSES)


def test_cascade_objects_labels_every_object():
    matched = [
        {"label": "ship", "bbox": [0, 0, 10, 10], "score": 0.9, "betp": 0.9,
         "view_a_index": 0, "view_b_index": 0, "family": "match", "source": "matched"},
    ]
    singles = [
        {"label": "ship", "bbox": [0, 0, 10, 10], "score": 0.8, "betp": 0.8,
         "view_a_index": 1, "view_b_index": None, "family": "match", "source": "view_a_only"},
        {"label": "ship", "bbox": [0, 0, 10, 10], "score": 0.8, "betp": 0.8,
         "view_a_index": None, "view_b_index": 1, "family": "match", "source": "view_b_only"},
    ]
    vi_records = [{"status": "supported", "value": "ship", "p_class": 0.9,
                   "association": 0.9, "confidence": 0.9}]
    objects = cascade_objects(matched, singles, vi_records, policy="blend", threshold=0.5)
    # Assembly order is singles first, then the D-S pairs, then extras.
    assert [item["status"] for item in objects] == ["rgb-only", "sar-only", "match"]


def test_cascade_objects_time_status_uses_the_change_label():
    matched = [
        {"label": "ship", "bbox": [0, 0, 10, 10], "score": 0.9, "betp": 0.9,
         "view_a_index": 0, "view_b_index": 0, "family": "time", "source": "matched",
         "change": "moved"},
        {"label": "ship", "bbox": [0, 0, 10, 10], "score": 0.9, "betp": 0.9,
         "view_a_index": 1, "view_b_index": 1, "family": "time", "source": "matched",
         "change": "stayed"},
    ]
    singles = [
        {"label": "ship", "bbox": [0, 0, 10, 10], "score": 0.8, "betp": 0.8,
         "view_a_index": 2, "view_b_index": None, "family": "time", "source": "view_a_only",
         "change": "left"},
    ]
    vi_records = [{"status": "supported", "value": "ship", "p_class": 0.9, "association": 0.9,
                   "confidence": 0.9},
                  {"status": "supported", "value": "ship", "p_class": 0.9, "association": 0.9,
                   "confidence": 0.9}]
    objects = cascade_objects(matched, singles, vi_records, policy="blend", threshold=0.5,
                              family="time")
    assert [item["status"] for item in objects] == ["time1-only", "moved", "static"]


def test_evaluator_reads_the_reported_status():
    assert reported_change_label({"status": "static"}) == "stayed"
    assert reported_change_label({"status": "moved"}) == "moved"
    assert reported_change_label({"status": "time1-only"}) == "disappeared"
    assert reported_change_label({"status": "time2-only"}) == "appeared"
    # match statuses and absent columns fall back to the geometric derivation.
    assert reported_change_label({"status": "match"}) is None
    assert reported_change_label({}) is None
