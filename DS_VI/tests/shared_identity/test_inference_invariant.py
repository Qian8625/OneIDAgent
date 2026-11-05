"""Time-invariant attribute pooling.

A cross-view pair asserts the same object twice. If the two views carry different
``time_key`` values and an attribute that cannot change between views is stored
per-time, the evidence splits over two truth cells and the posterior halves --
which is exactly what kept the independent-data candidates just below the
identity threshold in the time family.
"""
from __future__ import annotations

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.inference import _run_joint_inference_single
from DS_VI.models.shared_identity.observation_builder import build_claim, build_observation
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    CandidateEdge,
    InferenceProblem,
    ObjectHypothesis,
    SpatialSupport,
)


def _problem(time_keys=("t0", "t1"), attribute: str = "category"):
    observations = [
        build_observation("tool", "optical", bbox=[0, 0, 10, 10], acquisition_id=f"img{i}",
                          time_key=time_keys[i],
                          context={"image_group_id": f"img{i}", "channel": "ab"[i]})
        for i in range(2)
    ]
    claims = [build_claim(obs, attribute, "ship", source_channel="optical") for obs in observations]
    hypothesis = ObjectHypothesis(
        object_id="pair",
        spatial_support=SpatialSupport(bbox=[0, 0, 10, 10]),
        capacity=2,
        attribute_hints={"category": "ship"},
        context={
            "candidate_type": "cross_image_pair",
            "relation_group_id": "g",
            "member_observation_ids": [obs.observation_id for obs in observations],
            "member_image_group_ids": ["img0", "img1"],
            "pair_features": {"iou": 0.0, "center_distance_px": 5.0, "center_score": 0.9,
                              "area_ratio": 1.0, "confidence_score": 0.9, "diagnostic_score": 0.9},
            "prior_floor": 0.5,
        },
    )
    edges = []
    for obs in observations:
        edges.append(CandidateEdge(observation_id=obs.observation_id, object_id="pair",
                                   features={"diagnostic_score": 0.9}, admissible=True))
        edges.append(CandidateEdge(observation_id=obs.observation_id, object_id=NULL_OBJECT_ID,
                                   features={}, admissible=True, gate_reason="null"))
    problem = InferenceProblem(observations=observations, claims=claims, objects=[hypothesis],
                               candidate_edges=edges, capacities={"pair": 2})
    return problem, observations


def _run(invariant):
    config = load_config(overrides={"inference": {
        "evidence_cap": None,
        "time_invariant_attributes": invariant,
    }})
    problem, observations = _problem()
    state = _run_joint_inference_single(problem, config)
    first = observations[0].observation_id
    return state, state.association[first]["pair"]


def test_per_time_axes_split_the_evidence():
    state, association = _run([])
    assert ("pair", "category", "t0") in state.truth
    assert ("pair", "category", "t1") in state.truth
    split_posterior = state.truth[("pair", "category", "t0")]["ship"]
    # One claim per cell cannot reach the pooled posterior below.
    assert split_posterior < 0.9
    _, pooled_association = _run(["category"])
    assert association < pooled_association


def test_invariant_attributes_pool_into_one_cell():
    state, association = _run(["category"])
    assert ("pair", "category", "*") in state.truth
    assert ("pair", "category", "t0") not in state.truth
    assert state.truth[("pair", "category", "*")]["ship"] > 0.95
    assert association > 0.6


def test_state_stays_per_time_even_when_listed_as_invariant_only_for_others():
    state, _ = _run(["category"])
    # ``state`` was not pooled, so its axes keep their own time keys.
    assert not any(key[1] == "state" for key in state.truth)
