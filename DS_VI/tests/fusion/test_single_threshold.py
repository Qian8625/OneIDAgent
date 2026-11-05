"""Single-side detections must not be governed by the pair accept threshold.

``decision.accept_betp`` decides whether an *association* is accepted. Reusing it
for single-view detections makes the threshold do double duty: raising it to reject
bad pairs silently deletes most valid single-view detections (measured on
task4_000001: 13 -> 1 objects). ``decision.single_accept_betp`` decouples the two
and defaults to the historic behaviour when absent.
"""
from __future__ import annotations

from DS_VI.models.ds_fusion.config import load_config
from DS_VI.models.ds_fusion.pipeline import run_pair


def _pair():
    return {"base_id": "p0001", "family": "match", "images": {"a": "a.png", "b": "b.png"}}


def _detections(label="ship"):
    return {
        "a": [{"bbox": [0, 0, 10, 10], "label": label, "score": 0.9, "index": 0},
              {"bbox": [50, 50, 60, 60], "label": label, "score": 0.8, "index": 1}],
        "b": [],
    }


def _sources(config):
    result = run_pair(_pair(), _detections(), config)
    return [obj["source"] for obj in result["objects"]]


def test_high_pair_threshold_deletes_singles_without_the_new_key():
    config = load_config({"decision": {"accept_betp": 0.99, "matched_fallback": True}})
    assert _sources(config) == []


def test_single_threshold_keeps_singles_alive():
    config = load_config({"decision": {
        "accept_betp": 0.99, "matched_fallback": True, "single_accept_betp": 0.0,
    }})
    assert _sources(config) == ["view_a_only", "view_a_only"]


def test_absent_single_threshold_is_backward_compatible():
    low = load_config({"decision": {"accept_betp": 0.0, "matched_fallback": True}})
    low_with_key = load_config({"decision": {
        "accept_betp": 0.0, "matched_fallback": True, "single_accept_betp": 0.0,
    }})
    assert _sources(low) == _sources(low_with_key) == ["view_a_only", "view_a_only"]
