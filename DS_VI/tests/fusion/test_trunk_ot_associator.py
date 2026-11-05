"""Trunk wire-up of the OT associator: same contract, different pairing rule.

``association.associator: hungarian | ot`` selects the pairing rule inside
``run_pair``; ``association.ot`` carries the OT knobs. The default stays
``hungarian``, so a config without these keys behaves exactly as before.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from DS_VI.models.ds_fusion.config import load_config
from DS_VI.models.ds_fusion.pipeline import run_pair


def _pair():
    return {"base_id": "p0001", "family": "match", "images": {"a": "a.png", "b": "b.png"}}


def _detections():
    return {
        "a": [
            {"bbox": [0, 0, 10, 10], "label": "ship", "score": 0.9, "index": 0},
            {"bbox": [100, 100, 120, 120], "label": "ship", "score": 0.8, "index": 1},
            {"bbox": [300, 300, 320, 320], "label": "ship", "score": 0.5, "index": 2},
        ],
        "b": [
            {"bbox": [1, 1, 11, 11], "label": "ship", "score": 0.9, "index": 0},
            {"bbox": [102, 101, 122, 121], "label": "ship", "score": 0.8, "index": 1},
        ],
    }


def _sources(config):
    result = run_pair(_pair(), _detections(), config)
    return [(obj["source"], obj.get("view_a_index"), obj.get("view_b_index"))
            for obj in result["objects"]]


def _objects(config):
    result = run_pair(_pair(), _detections(), config)
    return json.dumps(result["objects"], sort_keys=True, ensure_ascii=False)


def test_default_stays_hungarian():
    config = load_config({"decision": {"accept_betp": 0.0, "matched_fallback": True}})
    assert "associator" not in (config.get("association") or {})
    expected = _sources(config)
    explicit = load_config({"decision": {"accept_betp": 0.0, "matched_fallback": True}})
    explicit["association"] = {**explicit["association"], "associator": "hungarian"}
    assert _sources(explicit) == expected
    # Whole objects, not only their sources: the default must be byte-identical.
    assert _objects(explicit) == _objects(config)


def test_ot_associator_matches_hungarian_on_an_unambiguous_pair():
    base = load_config({"decision": {"accept_betp": 0.0, "matched_fallback": True}})
    ot = load_config({"decision": {"accept_betp": 0.0, "matched_fallback": True}})
    ot["association"] = {**ot["association"], "associator": "ot", "ot": {"reg": 0.1}}
    hungarian_rows = _sources(base)
    ot_rows = _sources(ot)
    assert hungarian_rows == ot_rows
    assert [row[0] for row in ot_rows].count("matched") == 2


def test_ot_associator_is_deterministic():
    config = load_config({"decision": {"accept_betp": 0.0, "matched_fallback": True}})
    config["association"] = {**config["association"], "associator": "ot", "ot": {"reg": 0.1}}
    assert _sources(config) == _sources(config)
