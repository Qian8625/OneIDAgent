from __future__ import annotations

from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.inference import run_joint_inference
from DS_VI.models.shared_identity.observation_builder import (
    build_claim,
    build_observation,
    cross_image_object_hypotheses,
)
from DS_VI.models.shared_identity.schema import InferenceProblem, Modality


def _observation(observation_id: str, image: str, relation: str, bbox):
    observation = build_observation(
        source_tool="test",
        modality=Modality.OPTICAL,
        bbox=bbox,
        acquisition_id=image,
        context={
            "relation_group_id": relation,
            "image_group_id": image,
            "related_image_group_ids": [
                "image_b" if image == "image_a" else "image_a"
            ],
        },
    )
    observation.observation_id = observation_id
    return observation


def test_cross_image_object_candidates_never_join_same_image():
    observations = [
        _observation("a1", "image_a", "pair_1", [0, 0, 10, 10]),
        _observation("a2", "image_a", "pair_1", [20, 0, 30, 10]),
        _observation("b1", "image_b", "pair_1", [0, 0, 10, 10]),
    ]

    objects = cross_image_object_hypotheses(observations)

    assert objects
    for obj in objects:
        assert len(set(obj.context["member_image_group_ids"])) == 2
        assert set(obj.context["member_observation_ids"]) & {"a1", "a2"}


def test_candidate_graph_rejects_unrelated_observations():
    observations = [
        _observation("a1", "image_a", "pair_1", [0, 0, 10, 10]),
        _observation("c1", "image_c", "pair_2", [0, 0, 10, 10]),
    ]
    objects = cross_image_object_hypotheses(observations)
    assert objects == []


def test_independent_inference_keeps_relation_groups_separate():
    left = _observation("a1", "image_a", "pair_1", [0, 0, 10, 10])
    right = _observation("b1", "image_b", "pair_1", [0, 0, 10, 10])
    other_left = _observation("c1", "image_c", "pair_2", [100, 100, 110, 110])
    other_right = _observation("d1", "image_d", "pair_2", [100, 100, 110, 110])
    observations = [left, right, other_left, other_right]
    objects = cross_image_object_hypotheses(observations)
    claims = [
        build_claim(left, "category", "ship"),
        build_claim(right, "category", "ship"),
        build_claim(other_left, "category", "vehicle"),
        build_claim(other_right, "category", "vehicle"),
    ]
    edges = build_candidate_graph(observations, objects, load_config())
    state = run_joint_inference(
        InferenceProblem(
            observations=observations,
            claims=claims,
            objects=objects,
            candidate_edges=edges,
        ),
        load_config(),
    )

    assert state.diagnostics["mode"] == "cross_image_independent"
    assert state.diagnostics["group_count"] == 2
    assert set(state.association) == {"a1", "b1", "c1", "d1"}
