"""Independent-data channel: prior floor, gate bypass, and the cascade merge."""
from __future__ import annotations

from DS_VI.models.cascade import cascade_objects
from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.physical_prior import compute_physical_prior
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    CandidateEdge,
    ObjectHypothesis,
    SpatialSupport,
)
from DS_VI.models.shared_identity.observation_builder import build_observation


def _observation(observation_id: str, bbox, image_group: str):
    observation = build_observation(
        source_tool="tool",
        modality="optical",
        bbox=bbox,
        acquisition_id=image_group,
        context={"image_group_id": image_group, "channel": "a"},
    )
    observation.observation_id = observation_id
    return observation


def _hypothesis(object_id: str, bbox, members, **context):
    return ObjectHypothesis(
        object_id=object_id,
        spatial_support=SpatialSupport(bbox=list(bbox)),
        capacity=2,
        attribute_hints={"category": "ship"},
        context={
            "candidate_type": "cross_image_pair",
            "relation_group_id": "g",
            "member_observation_ids": list(members),
            "member_image_group_ids": ["img_a", "img_b"],
            **context,
        },
    )


def _edges(observation_id: str, object_id: str, score: float, is_null: bool = False) -> CandidateEdge:
    return CandidateEdge(
        observation_id=observation_id,
        object_id=object_id,
        features={} if is_null else {"diagnostic_score": score},
        admissible=True,
        gate_reason="test",
    )


def test_prior_floor_neutralises_quality_scaling_for_flagged_candidates():
    edges = [
        _edges("obs", "extra", score=0.02),
        _edges("obs", NULL_OBJECT_ID, score=0.0, is_null=True),
    ]
    plain = _hypothesis("extra", [0, 0, 10, 10], ["obs"])
    floored = _hypothesis("extra", [0, 0, 10, 10], ["obs"], prior_floor=0.5)
    config = load_config()

    # Without the flag the weak candidate keeps almost all mass on NULL.
    prior_plain = compute_physical_prior(edges, config, [plain])["obs"]
    assert prior_plain["extra"] < 0.05

    # With the flag the real budget is raised to the floor.
    prior_floored = compute_physical_prior(edges, config, [floored])["obs"]
    assert prior_floored["extra"] == 0.5
    assert prior_floored[NULL_OBJECT_ID] == 0.5


def test_gate_bypass_admits_geometrically_hopeless_candidates():
    observation = _observation("obs_a", [0, 0, 10, 10], "img_a")
    config = load_config()

    hopeless = {"iou": 0.0, "center_distance_px": 1272.0, "center_score": 0.0,
                "area_ratio": 1.0, "confidence_score": 0.9, "diagnostic_score": 0.09}
    gated = _hypothesis("pair", [0, 0, 10, 10], ["obs_a", "obs_b"], pair_features=hopeless)
    edges = build_candidate_graph([observation], [gated], config)
    assert all(edge.object_id != "pair" for edge in edges)

    bypass = _hypothesis("pair", [0, 0, 10, 10], ["obs_a", "obs_b"], bypass_gates=True,
                         pair_features=hopeless)
    edges = build_candidate_graph([observation], [bypass], config)
    assert any(edge.object_id == "pair" for edge in edges)


def test_confirmed_extra_takes_over_its_member_singles():
    single_a = {"label": "ship", "bbox": [0, 0, 10, 10], "score": 0.4, "source": "view_a_only",
                "view_a_index": 0, "view_b_index": None}
    single_b = {"label": "ship", "bbox": [50, 50, 60, 60], "score": 0.3, "source": "view_b_only",
                "view_a_index": None, "view_b_index": 1}
    untouched = {"label": "ship", "bbox": [80, 80, 90, 90], "score": 0.2, "source": "view_b_only",
                 "view_a_index": None, "view_b_index": 2}
    extra = {"class": "ship", "status": "Confirmed", "value": "ship",
             "confidence": 0.9, "p_class": 0.95, "association": 0.8,
             "bbox": [0, 0, 10, 10], "view_a_index": 0, "view_b_index": 1, "members": ["o1", "o2"]}

    merged = cascade_objects([], [single_a, single_b, untouched], [], policy="status",
                             extra_records=[extra])
    assert len(merged) == 2
    assert [item.get("source") for item in merged] == ["view_b_only", "vi_extra"]
    assert merged[1]["view_a_index"] == 0 and merged[1]["view_b_index"] == 1


def test_unconfirmed_extra_is_dropped_but_policy_all_keeps_it():
    single = {"label": "ship", "bbox": [0, 0, 10, 10], "score": 0.4, "source": "view_a_only",
              "view_a_index": 0, "view_b_index": None}
    extra = {"class": "ship", "status": "Uncertain", "value": "ship",
             "confidence": 0.1, "p_class": 0.2, "association": 0.1,
             "bbox": [0, 0, 10, 10], "view_a_index": 0, "view_b_index": 1, "members": []}

    merged = cascade_objects([], [single], [], policy="status", extra_records=[extra])
    assert len(merged) == 1
    assert merged[0]["source"] == "view_a_only"

    # policy "all" keeps the extra too, and it still takes over the covered single,
    # so the pair is not reported twice.
    merged_all = cascade_objects([], [single], [], policy="status", extra_records=[extra],
                                 extra_policy="all")
    assert len(merged_all) == 1
    assert merged_all[0]["source"] == "vi_extra"
