"""Candidate graph construction: high-recall enumeration of possible identities.

Two properties matter most here.  First, every observation always gets exactly
one ``__NULL__`` edge, appended last and always admissible, so its association
row can be normalized even when nothing real is plausible.  Second, gating is an
OR over IoU / center / geo, because this stage must not decide identity.
"""
from __future__ import annotations

import pytest

from DS_VI.models.shared_identity.candidate_graph import (
    build_candidate_graph,
    candidate_mask,
    candidate_recall,
    evaluate_admissibility,
)
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.physical_features import PairFeatures, compute_pair_features
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    Modality,
    ObjectHypothesis,
    PrimitiveObservation,
    SpatialSupport,
    TimeScope,
)

GATES = dict(load_config()["physical"]["gates"])
LOW_IOU_GATE = GATES["low_iou_gate"]      # 0.05
CENTER_GATE = GATES["center_gate"]        # 0.25
GEO_GATE_M = GATES["geo_gate_m"]          # 250.0


def _observation(
    observation_id: str = "m1",
    bbox=(0.0, 0.0, 10.0, 10.0),
    detection_confidence=0.9,
    geo_center=None,
) -> PrimitiveObservation:
    return PrimitiveObservation(
        observation_id=observation_id,
        modality=Modality.coerce("optical"),
        source_tool="SyntheticSensor",
        time_scope=TimeScope(key="t0"),
        spatial_support=SpatialSupport(
            bbox=[float(v) for v in bbox],
            geo_center=list(geo_center) if geo_center else None,
        ),
        detection_confidence=detection_confidence,
        input_roots=["root_a"],
    )


def _object(object_id: str, bbox=(0.0, 0.0, 10.0, 10.0), geo_center=None) -> ObjectHypothesis:
    return ObjectHypothesis(
        object_id=object_id,
        spatial_support=SpatialSupport(
            bbox=[float(v) for v in bbox],
            geo_center=list(geo_center) if geo_center else None,
        ),
        attribute_hints={"category": "ship"},
    )


def _real(edges):
    return [e for e in edges if not e.is_null]


def _nulls(edges):
    return [e for e in edges if e.is_null]


# ----------------------------------------------------------------------
# the NULL edge invariant
# ----------------------------------------------------------------------
def test_every_observation_gets_exactly_one_null_edge():
    observations = [
        _observation("m1", (0.0, 0.0, 10.0, 10.0)),
        _observation("m2", (100.0, 100.0, 110.0, 110.0)),
        _observation("m3", (900.0, 900.0, 940.0, 940.0)),
    ]
    edges = build_candidate_graph(observations, [_object("k1", (0.0, 0.0, 10.0, 10.0))])

    for observation in observations:
        row_nulls = [
            e for e in edges if e.observation_id == observation.observation_id and e.is_null
        ]
        assert len(row_nulls) == 1


def test_null_edge_is_appended_last_within_each_observation_row():
    # Index order matters downstream: the prior and the association matrix are
    # built from this ordering, and NULL is expected in the final column.
    observations = [_observation("m1"), _observation("m2", (2.0, 2.0, 12.0, 12.0))]
    objects = [_object("k1", (0.0, 0.0, 10.0, 10.0)), _object("k2", (1.0, 1.0, 11.0, 11.0))]
    edges = build_candidate_graph(observations, objects)

    for observation in observations:
        row = [e for e in edges if e.observation_id == observation.observation_id]
        assert len(row) > 1
        assert row[-1].is_null
        assert not any(e.is_null for e in row[:-1])
        # The observation's own candidate list mirrors that ordering.
        assert observation.candidate_object_ids[-1] == NULL_OBJECT_ID
        assert observation.candidate_object_ids == [e.object_id for e in row]


def test_null_edge_is_admissible_even_when_no_real_candidate_is():
    # Without this the association row for a fully unmatched observation would
    # be empty and impossible to normalize.
    observation = _observation("m_lonely", (0.0, 0.0, 10.0, 10.0))
    edges = build_candidate_graph([observation], [_object("k_far", (900.0, 900.0, 940.0, 940.0))])

    assert _real(edges) == []
    assert len(edges) == 1
    assert edges[0].is_null
    assert edges[0].admissible is True
    assert edges[0].gate_reason == "null_always_admissible"


def test_null_edge_exists_with_no_objects_at_all():
    observation = _observation("m_alone")
    edges = build_candidate_graph([observation], [])
    assert [e.object_id for e in edges] == [NULL_OBJECT_ID]
    assert edges[0].admissible is True
    assert observation.candidate_object_ids == [NULL_OBJECT_ID]


def test_null_edge_exists_when_the_observation_has_no_usable_bbox():
    # Degenerate geometry yields zero features, so nothing real gates in, but
    # the row must still be normalizable.
    observation = PrimitiveObservation(
        observation_id="m_nobox",
        modality=Modality.coerce("optical"),
        time_scope=TimeScope(key="t0"),
        spatial_support=SpatialSupport(bbox=None),
        input_roots=["root_a"],
    )
    edges = build_candidate_graph([observation], [_object("k1")])
    assert [e.object_id for e in edges] == [NULL_OBJECT_ID]


def test_null_edge_carries_no_geometric_features():
    edges = build_candidate_graph([_observation("m1")], [_object("k1")])
    assert _nulls(edges)[0].features == {}


def test_inactive_objects_are_excluded_but_null_remains():
    inactive = _object("k_off", (0.0, 0.0, 10.0, 10.0))
    inactive.active = False
    edges = build_candidate_graph([_observation("m1")], [inactive])
    assert [e.object_id for e in edges] == [NULL_OBJECT_ID]


# ----------------------------------------------------------------------
# OR-gating
# ----------------------------------------------------------------------
def test_edge_is_admissible_via_iou_alone_when_center_score_fails():
    # a=[0,0,10,10] vs b=[8,0,18,10]: iou = 20/180 = 0.111 >= 0.05 passes,
    # while center_score = 1 - 8/(14.142*0.75) = 0.2457 < 0.25 fails.
    observation_bbox = (0.0, 0.0, 10.0, 10.0)
    object_bbox = (8.0, 0.0, 18.0, 10.0)
    features = compute_pair_features(list(observation_bbox), list(object_bbox))
    assert features.iou >= LOW_IOU_GATE
    assert features.center_score < CENTER_GATE

    edges = build_candidate_graph(
        [_observation("m_iou", observation_bbox, detection_confidence=None)],
        [_object("k_iou", object_bbox)],
    )
    real = _real(edges)
    assert [e.object_id for e in real] == ["k_iou"]
    assert real[0].admissible is True
    assert real[0].gate_reason == f"iou>={LOW_IOU_GATE}"


def test_edge_is_admissible_via_center_distance_alone_when_iou_is_zero():
    # Disjoint boxes (iou exactly 0) whose centers are close relative to the
    # larger diagonal: center_score = 0.3299 >= 0.25 carries the edge alone.
    observation_bbox = (0.0, 0.0, 10.0, 10.0)
    object_bbox = (11.0, 0.0, 200.0, 200.0)
    features = compute_pair_features(list(observation_bbox), list(object_bbox))
    assert features.iou == 0.0
    assert features.center_score >= CENTER_GATE

    edges = build_candidate_graph(
        [_observation("m_center", observation_bbox, detection_confidence=None)],
        [_object("k_center", object_bbox)],
    )
    real = _real(edges)
    assert [e.object_id for e in real] == ["k_center"]
    assert real[0].gate_reason == f"center_score>={CENTER_GATE}"


def test_edge_is_admissible_via_geo_distance_alone_when_pixel_gates_fail():
    # Pixel geometry disagrees completely, but the geo centers are ~10 m apart,
    # well inside the 250 m gate. OR-gating must still admit the edge.
    observation = _observation(
        "m_geo", (0.0, 0.0, 10.0, 10.0), detection_confidence=None, geo_center=(10.0, 50.0)
    )
    obj = _object("k_geo", (900.0, 900.0, 940.0, 940.0), geo_center=(10.00005, 50.00005))
    edges = build_candidate_graph([observation], [obj])

    real = _real(edges)
    assert [e.object_id for e in real] == ["k_geo"]
    assert real[0].gate_reason == f"geo_distance<={GEO_GATE_M}m"
    # The geo distance is recorded as an extra feature for the prior.
    assert real[0].features["geo_distance_m"] < GEO_GATE_M


def test_evaluate_admissibility_passes_on_iou_alone():
    features = PairFeatures(
        iou=0.06,
        center_distance_px=10_000.0,
        center_score=0.0,
        area_ratio=0.0,
        confidence_score=0.5,
        diagnostic_score=0.027,
    )
    admissible, reason = evaluate_admissibility(features, None, GATES)
    assert admissible is True
    assert reason == f"iou>={LOW_IOU_GATE}"


def test_evaluate_admissibility_passes_on_center_score_alone():
    features = PairFeatures(
        iou=0.0,
        center_distance_px=5.0,
        center_score=0.30,
        area_ratio=0.0,
        confidence_score=0.5,
        diagnostic_score=0.09,
    )
    admissible, reason = evaluate_admissibility(features, None, GATES)
    assert admissible is True
    assert reason == f"center_score>={CENTER_GATE}"


def test_evaluate_admissibility_rejects_when_no_gate_passes():
    features = PairFeatures(
        iou=0.0,
        center_distance_px=10_000.0,
        center_score=0.0,
        area_ratio=0.0,
        confidence_score=0.5,
        diagnostic_score=0.05,
    )
    admissible, reason = evaluate_admissibility(features, 10_000.0, GATES)
    assert admissible is False
    assert reason == "no_gate_passed"


def test_evaluate_admissibility_gates_are_inclusive_at_the_threshold():
    # ">=" not ">": a candidate sitting exactly on the gate is kept, because
    # this stage optimizes recall.
    on_iou = PairFeatures(LOW_IOU_GATE, 10_000.0, 0.0, 0.0, 0.5, 0.02)
    assert evaluate_admissibility(on_iou, None, GATES)[0] is True

    on_center = PairFeatures(0.0, 5.0, CENTER_GATE, 0.0, 0.5, 0.08)
    assert evaluate_admissibility(on_center, None, GATES)[0] is True


def test_max_center_distance_veto_overrides_a_passing_iou_gate():
    # The hard veto is the one gate that is an AND, so a configured pixel budget
    # can override an otherwise-passing OR.
    features = PairFeatures(
        iou=0.9,
        center_distance_px=120.0,
        center_score=0.9,
        area_ratio=1.0,
        confidence_score=0.9,
        diagnostic_score=0.9,
    )
    gates = dict(GATES, max_center_distance_px=100.0)
    admissible, reason = evaluate_admissibility(features, None, gates)
    assert admissible is False
    assert reason == "center_distance>100.0px"

    # Without the veto configured the same pair is admissible.
    assert evaluate_admissibility(features, None, GATES)[0] is True


def test_infinite_center_distance_trips_a_configured_veto():
    features = PairFeatures(0.0, float("inf"), 0.0, 0.0, 0.5, 0.0)
    gates = dict(GATES, max_center_distance_px=100.0)
    assert evaluate_admissibility(features, None, gates)[0] is False


def test_geo_gate_is_skipped_when_either_side_lacks_a_geo_center():
    # A missing geo center must not be treated as distance 0 (a free pass).
    features = PairFeatures(0.0, 10_000.0, 0.0, 0.0, 0.5, 0.05)
    assert evaluate_admissibility(features, None, GATES)[0] is False


# ----------------------------------------------------------------------
# far-away objects
# ----------------------------------------------------------------------
def test_far_away_object_yields_no_real_edge_but_the_observation_keeps_null():
    observation = _observation("m_far", (0.0, 0.0, 10.0, 10.0))
    far = _object("k_far", (900.0, 900.0, 940.0, 940.0))

    features = compute_pair_features([0.0, 0.0, 10.0, 10.0], [900.0, 900.0, 940.0, 940.0])
    assert features.iou < LOW_IOU_GATE
    assert features.center_score < CENTER_GATE

    edges = build_candidate_graph([observation], [far])
    assert all(e.is_null for e in edges)
    assert observation.candidate_object_ids == [NULL_OBJECT_ID]


def test_inadmissible_pairs_are_omitted_entirely_rather_than_kept_as_false():
    # Inadmissible pairs must never appear, because anything present could
    # receive probability mass.
    observation = _observation("m_mixed", (0.0, 0.0, 10.0, 10.0))
    objects = [_object("k_near", (1.0, 1.0, 11.0, 11.0)), _object("k_far", (900.0, 900.0, 940.0, 940.0))]
    edges = build_candidate_graph([observation], objects)

    assert {e.object_id for e in edges} == {"k_near", NULL_OBJECT_ID}
    assert all(e.admissible for e in edges)
    assert "k_far" not in observation.candidate_object_ids


def test_multiple_observations_are_gated_independently():
    near = _observation("m_near", (0.0, 0.0, 10.0, 10.0))
    far = _observation("m_far", (900.0, 900.0, 940.0, 940.0))
    edges = build_candidate_graph([near, far], [_object("k1", (0.0, 0.0, 10.0, 10.0))])

    assert near.candidate_object_ids == ["k1", NULL_OBJECT_ID]
    assert far.candidate_object_ids == [NULL_OBJECT_ID]
    assert len([e for e in edges if e.observation_id == "m_far"]) == 1


# ----------------------------------------------------------------------
# candidate cap
# ----------------------------------------------------------------------
def test_max_candidates_caps_real_edges_and_keeps_the_highest_scoring_ones():
    observation = _observation("m_cap", (0.0, 0.0, 10.0, 10.0), detection_confidence=None)
    objects = [
        _object("k_exact", (0.0, 0.0, 10.0, 10.0)),
        _object("k_close", (1.0, 1.0, 11.0, 11.0)),
        _object("k_looser", (2.0, 2.0, 12.0, 12.0)),
        _object("k_loosest", (3.0, 3.0, 13.0, 13.0)),
    ]
    uncapped = build_candidate_graph(
        [observation], objects, config=load_config(overrides={"physical": {"max_candidates_per_observation": 99}})
    )
    assert len(_real(uncapped)) == 4

    capped = build_candidate_graph(
        [observation], objects, config=load_config(overrides={"physical": {"max_candidates_per_observation": 2}})
    )
    real = _real(capped)
    assert len(real) == 2
    # The cap keeps the best candidates, so recall loss is minimized.
    assert [e.object_id for e in real] == ["k_exact", "k_close"]
    scores = [e.features["diagnostic_score"] for e in real]
    assert scores == sorted(scores, reverse=True)


def test_null_edge_is_exempt_from_the_candidate_cap():
    # Even a cap of 1 must not consume the NULL slot.
    observation = _observation("m_cap1", (0.0, 0.0, 10.0, 10.0))
    objects = [_object("k1", (0.0, 0.0, 10.0, 10.0)), _object("k2", (1.0, 1.0, 11.0, 11.0))]
    edges = build_candidate_graph(
        [observation], objects, config=load_config(overrides={"physical": {"max_candidates_per_observation": 1}})
    )
    assert len(_real(edges)) == 1
    assert len(_nulls(edges)) == 1
    assert observation.candidate_object_ids == ["k1", NULL_OBJECT_ID]


def test_candidate_cap_applies_per_observation_not_globally():
    observations = [
        _observation("m_a", (0.0, 0.0, 10.0, 10.0)),
        _observation("m_b", (0.0, 0.0, 10.0, 10.0)),
    ]
    objects = [_object("k1", (0.0, 0.0, 10.0, 10.0)), _object("k2", (1.0, 1.0, 11.0, 11.0))]
    edges = build_candidate_graph(
        observations, objects, config=load_config(overrides={"physical": {"max_candidates_per_observation": 1}})
    )
    for observation_id in ("m_a", "m_b"):
        row = [e for e in edges if e.observation_id == observation_id]
        assert len(row) == 2  # one real + NULL


@pytest.mark.parametrize("disabled", [None, 0, ""])
def test_candidate_cap_is_disabled_by_none_zero_or_empty(disabled):
    observation = _observation("m_uncapped", (0.0, 0.0, 10.0, 10.0))
    objects = [
        _object("k1", (0.0, 0.0, 10.0, 10.0)),
        _object("k2", (1.0, 1.0, 11.0, 11.0)),
        _object("k3", (2.0, 2.0, 12.0, 12.0)),
    ]
    edges = build_candidate_graph(
        [observation],
        objects,
        config=load_config(overrides={"physical": {"max_candidates_per_observation": disabled}}),
    )
    assert len(_real(edges)) == 3


# ----------------------------------------------------------------------
# candidate_mask
# ----------------------------------------------------------------------
def test_candidate_mask_shape_and_false_for_non_candidate_cells():
    observations = [
        _observation("m1", (0.0, 0.0, 10.0, 10.0)),
        _observation("m2", (900.0, 900.0, 940.0, 940.0)),
    ]
    objects = [_object("k1", (0.0, 0.0, 10.0, 10.0)), _object("k_far", (500.0, 500.0, 510.0, 510.0))]
    edges = build_candidate_graph(observations, objects)

    observation_ids = ["m1", "m2"]
    object_ids = ["k1", "k_far", NULL_OBJECT_ID]
    mask = candidate_mask(edges, observation_ids, object_ids)

    assert len(mask) == len(observation_ids)
    assert all(len(row) == len(object_ids) for row in mask)
    assert all(isinstance(cell, bool) for row in mask for cell in row)

    assert mask[0] == [True, False, True]   # m1 matches k1, never k_far
    assert mask[1] == [False, False, True]  # m2 matches nothing real
    # NULL is always available, in every row.
    assert all(row[object_ids.index(NULL_OBJECT_ID)] for row in mask)


def test_candidate_mask_respects_the_caller_index_order():
    edges = build_candidate_graph(
        [_observation("m1", (0.0, 0.0, 10.0, 10.0))],
        [_object("k1", (0.0, 0.0, 10.0, 10.0)), _object("k_far", (500.0, 500.0, 510.0, 510.0))],
    )
    forward = candidate_mask(edges, ["m1"], ["k1", "k_far"])
    reversed_order = candidate_mask(edges, ["m1"], ["k_far", "k1"])
    assert forward == [[True, False]]
    assert reversed_order == [[False, True]]


def test_candidate_mask_ignores_ids_absent_from_the_index_and_inadmissible_edges():
    edges = build_candidate_graph(
        [_observation("m1", (0.0, 0.0, 10.0, 10.0))], [_object("k1", (0.0, 0.0, 10.0, 10.0))]
    )
    # An object the caller did not index must not raise or leak into the mask.
    assert candidate_mask(edges, ["m1"], [NULL_OBJECT_ID]) == [[True]]
    # An observation with no edges is all-False, never missing.
    assert candidate_mask(edges, ["m1", "m_unknown"], ["k1"]) == [[True], [False]]

    for edge in edges:
        edge.admissible = False
    assert candidate_mask(edges, ["m1"], ["k1", NULL_OBJECT_ID]) == [[False, False]]


def test_candidate_mask_is_all_false_with_no_edges():
    assert candidate_mask([], ["m1", "m2"], ["k1"]) == [[False], [False]]


# ----------------------------------------------------------------------
# candidate_recall
# ----------------------------------------------------------------------
def test_candidate_recall_counts_covered_total_and_missing():
    observations = [
        _observation("m_hit", (0.0, 0.0, 10.0, 10.0)),
        _observation("m_miss", (900.0, 900.0, 940.0, 940.0)),
    ]
    objects = [_object("k1", (0.0, 0.0, 10.0, 10.0)), _object("k2", (500.0, 500.0, 510.0, 510.0))]
    edges = build_candidate_graph(observations, objects)

    # m_hit's true object is retained; m_miss's true object was gated out.
    report = candidate_recall(edges, {"m_hit": "k1", "m_miss": "k2"})
    assert report["total"] == 2
    assert report["covered"] == 1
    assert report["missing"] == ["m_miss"]
    assert report["recall"] == pytest.approx(0.5)


def test_candidate_recall_is_one_when_every_truth_is_retained():
    observations = [
        _observation("m1", (0.0, 0.0, 10.0, 10.0)),
        _observation("m2", (1.0, 1.0, 11.0, 11.0)),
    ]
    objects = [_object("k1", (0.0, 0.0, 10.0, 10.0)), _object("k2", (1.0, 1.0, 11.0, 11.0))]
    edges = build_candidate_graph(observations, objects)

    report = candidate_recall(edges, {"m1": "k1", "m2": "k2"})
    assert report["recall"] == pytest.approx(1.0)
    assert report["covered"] == 2
    assert report["missing"] == []


def test_candidate_recall_counts_a_null_ground_truth_as_covered():
    # A NULL-truth observation is always covered, because the NULL edge is
    # unconditional. Recall must not be penalized for correct non-association.
    observation = _observation("m_null", (0.0, 0.0, 10.0, 10.0))
    edges = build_candidate_graph([observation], [_object("k_far", (900.0, 900.0, 940.0, 940.0))])

    report = candidate_recall(edges, {"m_null": NULL_OBJECT_ID})
    assert report["recall"] == pytest.approx(1.0)
    assert report["covered"] == 1
    assert report["total"] == 1
    assert report["missing"] == []
    assert report["mean_candidates"] == pytest.approx(1.0)


def test_candidate_recall_mixes_null_and_real_ground_truth():
    observations = [
        _observation("m_real", (0.0, 0.0, 10.0, 10.0)),
        _observation("m_null", (900.0, 900.0, 940.0, 940.0)),
    ]
    edges = build_candidate_graph(observations, [_object("k1", (0.0, 0.0, 10.0, 10.0))])

    report = candidate_recall(edges, {"m_real": "k1", "m_null": NULL_OBJECT_ID})
    assert report["covered"] == 2
    assert report["recall"] == pytest.approx(1.0)
    # m_real has {k1, NULL} = 2 candidates, m_null has {NULL} = 1.
    assert report["mean_candidates"] == pytest.approx(1.5)


def test_candidate_recall_flags_an_observation_with_no_edges_at_all():
    report = candidate_recall([], {"m_ghost": "k1"})
    assert report["recall"] == pytest.approx(0.0)
    assert report["covered"] == 0
    assert report["missing"] == ["m_ghost"]
    assert report["mean_candidates"] == pytest.approx(0.0)


def test_candidate_recall_of_empty_ground_truth_is_one_by_convention():
    # Vacuous truth: nothing to retain means nothing was lost.
    edges = build_candidate_graph([_observation("m1")], [_object("k1")])
    report = candidate_recall(edges, {})
    assert report["recall"] == pytest.approx(1.0)
    assert report["total"] == 0


def test_candidate_recall_ignores_inadmissible_edges():
    edges = build_candidate_graph(
        [_observation("m1", (0.0, 0.0, 10.0, 10.0))], [_object("k1", (0.0, 0.0, 10.0, 10.0))]
    )
    for edge in edges:
        if not edge.is_null:
            edge.admissible = False

    report = candidate_recall(edges, {"m1": "k1"})
    assert report["covered"] == 0
    assert report["missing"] == ["m1"]


# ----------------------------------------------------------------------
# determinism
# ----------------------------------------------------------------------
def test_build_candidate_graph_is_deterministic():
    def build():
        return build_candidate_graph(
            [_observation("m1", (0.0, 0.0, 10.0, 10.0)), _observation("m2", (5.0, 5.0, 15.0, 15.0))],
            [_object("k1", (0.0, 0.0, 10.0, 10.0)), _object("k2", (4.0, 4.0, 14.0, 14.0))],
        )

    first = [(e.observation_id, e.object_id, e.features, e.gate_reason) for e in build()]
    second = [(e.observation_id, e.object_id, e.features, e.gate_reason) for e in build()]
    assert first == second
