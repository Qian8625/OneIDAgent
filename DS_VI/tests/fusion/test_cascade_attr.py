"""Unit tests for the cascade-v2 attribute adapter and metrics."""

from __future__ import annotations


import pytest


import DS_VI.models.cascade as cascade_attr
import DS_VI.experiments.metrics.attribute_metrics as eval_attr
import DS_VI.models.cascade as dual_cascade
import DS_VI.models.ds_fusion.config as pipeline


def _pair(family="match"):
    return {
        "base_id": "s1",
        "family": family,
        "images": {"a": "/nonexistent/a.png", "b": "/nonexistent/b.png"},
    }


def _det(bbox, label, score=0.9, index=0):
    return {"bbox": list(bbox), "label": label, "score": score, "index": index, "view": "a", "source": "tool_json"}


def _ctx_payload():
    return {
        "samples": [
            {
                "sample_id": "s1",
                "claims": [
                    {"observation_id": "s1-RGB-000", "attribute": "subtype", "value": "cargo_ship", "raw_confidence": None},
                    {"observation_id": "s1-RGB-000", "attribute": "size_bin", "value": "large", "raw_confidence": None},
                    {"observation_id": "s1-RGB-000", "attribute": "category", "value": "ship", "raw_confidence": 0.9},
                    {"observation_id": "s1-SAR-000", "attribute": "subtype", "value": "cargo_ship", "raw_confidence": None},
                ],
                "gt_assignment": {"s1-RGB-000": "OBJ1", "s1-SAR-000": "OBJ1"},
                "gt_truth": {
                    "OBJ1|subtype|t0": "cargo_ship",
                    "OBJ1|size_bin|t0": "large",
                },
            }
        ]
    }


def test_observation_id_format():
    assert cascade_attr.observation_id_for("task4_000001", "match", "a", 0) == "task4_000001-RGB-000"
    assert cascade_attr.observation_id_for("task5_000101", "time", "b", 7) == "task5_000101-time2-007"


def test_ctx_annotation_claim_lookup():
    ctx = cascade_attr.CtxIndex(payload=_ctx_payload())
    claims = ctx.annotation_claims("s1", "match", "a", 0)
    attributes = {claim["attribute"] for claim in claims}
    assert attributes == {"subtype", "size_bin"}
    assert all(claim["attribute"] != "category" for claim in claims)
    assert ctx.annotation_claims("s1", "match", "a", 3) == []


def test_cascade_attr_pair_attaches_claims_and_beliefs():
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
    ctx = cascade_attr.CtxIndex(payload=_ctx_payload())
    result = cascade_attr.run_cascade_attr_pair(_pair(), detections, matched, config, "match", ctx)
    assert result["claim_coverage"]["members_with_claims"] == 2
    record = result["objects"][0]
    # The shipped config pools time-invariant attributes, so subtype is keyed by
    # the pooled bucket ("*") instead of "t0". Accept either spelling: the belief
    # is addressed by (attribute, time bucket), and only the bucket changed.
    belief_keys = [key for key in record["attribute_beliefs"] if key.startswith("subtype@")]
    assert belief_keys, sorted(record["attribute_beliefs"])
    belief = record["attribute_beliefs"][belief_keys[0]]
    assert belief["value"] == "cargo_ship"
    assert belief["probability"] > 0.0


def test_run_vi_pair_without_extra_claims_has_empty_attribute_beliefs():
    import DS_VI.models.cascade.vi as dual_vi
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
    assert result["objects"][0]["attribute_beliefs"] == {}


def test_all_policy_keeps_every_matched_object():
    matched = [{"label": "ship", "bbox": [0, 0, 10, 10], "source": "matched"}]
    single = [{"label": "ship", "bbox": [50, 50, 60, 60], "source": "view_a_only"}]
    vi = [{"status": "insufficient_evidence", "value": None, "p_class": 0.3, "association": 0.4}]
    final = dual_cascade.cascade_objects(matched, single, vi, "all", 0.0)
    assert len(final) == 2


def test_attribute_metrics_conditional_and_end_to_end():
    ctx = cascade_attr.CtxIndex(payload=_ctx_payload())
    prediction = {
        "label": "ship",
        "bbox": [2.5, 2.5, 102.5, 102.5],
        "source": "matched",
        "view_a_index": 0,
        "view_b_index": 0,
        "attribute_beliefs": {
            "subtype@t0": {"attribute": "subtype", "time_key": "t0", "value": "cargo_ship", "probability": 0.9},
            "size_bin@t0": {"attribute": "size_bin", "time_key": "t0", "value": "small", "probability": 0.6},
        },
    }
    metrics = eval_attr.attribute_metrics(ctx, "match", {"s1": [prediction]}, ("subtype", "size_bin"))
    assert metrics["attributes"]["subtype"]["conditional_accuracy"] == pytest.approx(1.0)
    assert metrics["attributes"]["subtype"]["end_to_end_accuracy"] == pytest.approx(1.0)
    assert metrics["attributes"]["size_bin"]["conditional_accuracy"] == pytest.approx(0.0)
    assert metrics["attributes"]["size_bin"]["end_to_end_accuracy"] == pytest.approx(0.0)
    assert metrics["coverage"]["identity_ok"] == 1


def test_attribute_metrics_miss_counts_as_end_to_end_wrong():
    ctx = cascade_attr.CtxIndex(payload=_ctx_payload())
    metrics = eval_attr.attribute_metrics(ctx, "match", {"s1": []}, ("subtype",))
    assert metrics["attributes"]["subtype"]["conditional_support"] == 0
    assert metrics["attributes"]["subtype"]["end_to_end_accuracy"] == pytest.approx(0.0)
    assert metrics["attributes"]["subtype"]["end_to_end_support"] == 1


def test_attribute_metrics_rejects_null_partner():
    payload = _ctx_payload()
    payload['samples'][0]['gt_assignment']['s1-SAR-000'] = '__NULL__'
    ctx = cascade_attr.CtxIndex(payload=payload)
    prediction = {'view_a_index': 0, 'view_b_index': 0, 'attribute_beliefs': {
        'subtype@t0': {'attribute': 'subtype', 'time_key': 't0', 'value': 'cargo_ship', 'probability': 0.9},
    }}
    result = eval_attr.attribute_metrics(ctx, 'match', {'s1': [prediction]}, ('subtype',))
    assert result['coverage']['identity_ok'] == 0
    assert result['attributes']['subtype']['conditional_support'] == 0
    assert result['attributes']['subtype']['end_to_end_accuracy'] == 0


def test_attribute_metrics_scores_both_times():
    payload = {'samples': [{'sample_id': 's1', 'claims': [],
        'gt_assignment': {'s1-time1-000': 'OBJ1', 's1-time2-000': 'OBJ1'},
        'gt_truth': {'OBJ1|state|t0': 'moving', 'OBJ1|state|t1': 'moving'}}]}
    ctx = cascade_attr.CtxIndex(payload=payload)
    prediction = {'view_a_index': 0, 'view_b_index': 0, 'attribute_beliefs': {
        'state@t0': {'attribute': 'state', 'time_key': 't0', 'value': 'moving', 'probability': 0.9},
        'state@t1': {'attribute': 'state', 'time_key': 't1', 'value': 'stopped', 'probability': 0.6},
    }}
    result = eval_attr.attribute_metrics(ctx, 'time', {'s1': [prediction]}, ('state',))['attributes']['state']
    assert result['conditional_support'] == 2
    assert result['end_to_end_support'] == 2
    assert result['conditional_accuracy'] == 0.5
    assert result['end_to_end_accuracy'] == 0.5
