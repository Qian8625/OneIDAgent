"""Physical prior (paper Eq. 2): softmax over d^T f plus d_null.

The behaviour these tests lock in:

* a row is a softmax over ``exp(prior_scale * diagnostic_score)`` and ``exp(d_null)``,
  so an absolute weak candidate keeps most of the row's NULL mass;
* ``null_prior`` is the mass a perfect single candidate leaves on NULL, which is
  how ``null_logit = null`` derives ``d_null``;
* inadmissible edges carry exactly zero mass and never enter the normalisation;
* ``prior_floor`` (the cascade's independent-data channel) lifts a row's real
  budget after normalisation.
"""
from __future__ import annotations

import math

import pytest

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.physical_prior import (
    DEFAULT_PRIOR_SCALE,
    compute_physical_prior,
    default_null_logit,
    prior_matrix,
)
from DS_VI.models.shared_identity.schema import NULL_OBJECT_ID, CandidateEdge

CONFIG = load_config()
PHYSICAL = dict(CONFIG["physical"])
NULL_PRIOR = float(PHYSICAL["null_prior"])
SCALE = float(PHYSICAL.get("prior_scale", DEFAULT_PRIOR_SCALE))


def _edges(*specs, observation_id="m1", admissible=True):
    """One row's edges: (object_id, diagnostic_score) pairs plus a NULL edge."""
    edges = [
        CandidateEdge(
            observation_id=observation_id,
            object_id=object_id,
            features={"diagnostic_score": score},
            admissible=admissible,
        )
        for object_id, score in specs
    ]
    edges.append(CandidateEdge(observation_id=observation_id, object_id=NULL_OBJECT_ID))
    return edges


def _config(**physical):
    config = load_config()
    config["physical"].update(physical)
    return config


def test_config_exposes_the_prior_form_knobs():
    assert NULL_PRIOR == 0.15
    assert SCALE == pytest.approx(6.0)
    assert "null_logit" in PHYSICAL
    assert PHYSICAL.get("null_logit") is None          # derived from null_prior
    assert "prior_quality_scaling" not in PHYSICAL     # removed: paper has no such term
    assert "min_edge_score" not in PHYSICAL


def test_null_logit_makes_a_perfect_single_candidate_keep_null_prior():
    edges = _edges(("k1", 1.0))
    prior = compute_physical_prior(edges, CONFIG)
    assert prior["m1"][NULL_OBJECT_ID] == pytest.approx(NULL_PRIOR, abs=1e-9)
    assert prior["m1"]["k1"] == pytest.approx(1.0 - NULL_PRIOR, abs=1e-9)


def test_prior_matches_the_closed_form():
    specs = [("k1", 0.9), ("k2", 0.4), ("k3", 0.05)]
    prior = compute_physical_prior(_edges(*specs), CONFIG)
    null_logit = default_null_logit(NULL_PRIOR, SCALE)
    logits = [SCALE * score for _, score in specs]
    peak = max(max(logits), null_logit)
    weights = [math.exp(v - peak) for v in logits] + [math.exp(null_logit - peak)]
    total = sum(weights)
    expected = {object_id: weight / total for (object_id, _), weight in zip(specs, weights)}
    for object_id, value in expected.items():
        assert prior["m1"][object_id] == pytest.approx(value, abs=1e-12)
    assert prior["m1"][NULL_OBJECT_ID] == pytest.approx(weights[-1] / total, abs=1e-12)


def test_every_row_is_an_exact_simplex():
    for specs in ((), (("k1", 0.5),), (("k1", 0.9), ("k2", 0.1)),
                  (("k1", 0.2), ("k2", 0.6), ("k3", 0.99))):
        prior = compute_physical_prior(_edges(*specs), CONFIG)
        assert sum(prior["m1"].values()) == pytest.approx(1.0, abs=1e-12)


def test_edges_are_updated_in_place_and_agree_with_the_row():
    edges = _edges(("k1", 0.8), ("k2", 0.3))
    prior = compute_physical_prior(edges, CONFIG)
    for edge in edges:
        assert edge.prior_probability == pytest.approx(prior["m1"][edge.object_id], abs=1e-12)
    assert sum(edge.prior_probability for edge in edges) == pytest.approx(1.0, abs=1e-12)


def test_empty_candidate_set_puts_all_mass_on_null():
    edges = [CandidateEdge(observation_id="m_lonely", object_id=NULL_OBJECT_ID)]
    prior = compute_physical_prior(edges, CONFIG)
    assert prior["m_lonely"] == {NULL_OBJECT_ID: 1.0}


def test_all_inadmissible_edges_put_all_mass_on_null():
    edges = _edges(("k_bad", 0.9), admissible=False)
    prior = compute_physical_prior(edges, CONFIG)
    assert prior["m1"][NULL_OBJECT_ID] == 1.0
    assert edges[0].prior_probability == 0.0


def test_inadmissible_edges_do_not_dilute_the_admissible_ones():
    admissible = CandidateEdge(observation_id="m1", object_id="k_ok",
                               features={"diagnostic_score": 0.8})
    inadmissible = CandidateEdge(observation_id="m1", object_id="k_bad",
                                 features={"diagnostic_score": 0.99}, admissible=False)
    null_edge = CandidateEdge(observation_id="m1", object_id=NULL_OBJECT_ID)
    prior = compute_physical_prior([admissible, inadmissible, null_edge], CONFIG)
    assert "k_bad" not in prior["m1"]
    assert prior["m1"]["k_ok"] == pytest.approx(
        compute_physical_prior(_edges(("k_ok", 0.8)), CONFIG)["m1"]["k_ok"], abs=1e-12)


def test_better_absolute_quality_leaves_less_null_mass():
    masses = []
    for score in (0.0, 0.35, 0.5, 0.75, 0.95, 1.0):
        prior = compute_physical_prior(_edges(("k1", score)), CONFIG)
        masses.append(prior["m1"][NULL_OBJECT_ID])
    assert masses == sorted(masses, reverse=True)
    assert masses[0] > 0.9      # a zero-score sole candidate is nearly all NULL
    assert masses[-1] == pytest.approx(NULL_PRIOR, abs=1e-9)


def test_a_zero_score_candidate_is_almost_null():
    prior = compute_physical_prior(_edges(("k1", 0.0)), CONFIG)
    assert prior["m1"][NULL_OBJECT_ID] > 0.9


def test_missing_diagnostic_score_is_treated_as_zero():
    edge = CandidateEdge(observation_id="m1", object_id="k1")
    null_edge = CandidateEdge(observation_id="m1", object_id=NULL_OBJECT_ID)
    prior = compute_physical_prior([edge, null_edge], CONFIG)
    expected = compute_physical_prior(_edges(("k1", 0.0)), CONFIG)
    assert prior["m1"]["k1"] == pytest.approx(expected["m1"]["k1"], abs=1e-12)


def test_prior_scale_controls_the_separation():
    # Pin d_null so the comparison isolates |d| rather than the derived offset.
    row_flat = compute_physical_prior(
        _edges(("k1", 0.6), ("k2", 0.2)), _config(prior_scale=1.0, null_logit=-2.0))
    row_sharp = compute_physical_prior(
        _edges(("k1", 0.6), ("k2", 0.2)), _config(prior_scale=12.0, null_logit=-2.0))
    flat_ratio = row_flat["m1"]["k1"] / row_flat["m1"]["k2"]
    sharp_ratio = row_sharp["m1"]["k1"] / row_sharp["m1"]["k2"]
    assert sharp_ratio > flat_ratio


def test_explicit_null_logit_overrides_the_derived_one():
    row_default = compute_physical_prior(_edges(("k1", 0.6)), CONFIG)
    row_low = compute_physical_prior(_edges(("k1", 0.6)), _config(null_logit=-6.0))
    assert row_low["m1"]["k1"] > row_default["m1"]["k1"]


def test_prior_floor_lifts_the_real_budget():
    edge = CandidateEdge(observation_id="m1", object_id="k1",
                         features={"diagnostic_score": 0.01})
    null_edge = CandidateEdge(observation_id="m1", object_id=NULL_OBJECT_ID)
    prior = compute_physical_prior(
        [edge, null_edge], CONFIG, objects=[_StubObject("k1", 0.5)])
    assert prior["m1"]["k1"] == pytest.approx(0.5, abs=1e-9)
    assert prior["m1"][NULL_OBJECT_ID] == pytest.approx(0.5, abs=1e-9)


def test_prior_matrix_orders_rows_and_columns():
    edges = _edges(("k1", 0.9), ("k2", 0.1), observation_id="m1")
    prior = compute_physical_prior(edges, CONFIG)
    matrix = prior_matrix(prior, ["m1", "m2"], ["k1", "k2", NULL_OBJECT_ID])
    assert matrix[0][0] == pytest.approx(prior["m1"]["k1"])
    assert matrix[1] == [0.0, 0.0, 0.0]


def test_empty_edge_list_is_an_empty_mapping():
    assert compute_physical_prior([]) == {}


class _StubObject:
    def __init__(self, object_id: str, floor: float):
        self.object_id = object_id
        self.context = {"prior_floor": floor}
