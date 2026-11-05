"""Truth-strength scaling in the decision layer.

``decision.truth_scale: absolute`` is the shipped behavior and must stay
bit-identical. ``uniform_lift`` rescales the same posterior by the attribute
domain size so one threshold works across attributes whose additive prior
smoothing keeps absolute probabilities in different narrow bands.
"""
from __future__ import annotations

import pytest

from DS_VI.models.shared_identity.decision import (
    decide_object_truth,
    truth_strength,
)
from DS_VI.models.shared_identity.schema import (
    EvidenceState,
    ObjectEvidenceSummary,
)

CATEGORY_DOMAIN = [
    "vehicle",
    "tank",
    "ship",
    "aircraft",
    "bridge",
    "harbor",
    "other",
]


def _belief(probability: float = 0.24, attribute: str = "category"):
    return {
        "attribute": attribute,
        "time_key": "t0",
        "value": "ship",
        "probability": probability,
        "distribution": {"ship": probability, "harbor": 0.05, "other": 0.03},
    }


def _state(probability: float = 0.24) -> EvidenceState:
    summary = ObjectEvidenceSummary(
        object_id="OBJ-1",
        association_support=2.0,
        supporting_observations=["o1", "o2"],
        truth_beliefs={"OBJ-1|category|t0": _belief(probability)},
        coverage={
            "modalities": {"optical": 1, "sar": 1},
            "time_keys": {"t0": 2},
            "modality_count": 2,
            "time_count": 1,
        },
        provenance_summary={},
        ambiguity={"association_entropy": 0.1},
        contradictions=[],
    )
    return EvidenceState(
        objects={"OBJ-1": summary},
        observation_count=2,
        claim_count=2,
        coverage={},
        null_mass={"o1": 0.1, "o2": 0.1},
        inference_diagnostics={},
    )


def _config(scale: str = "absolute", truth: float = 0.6, qualified: float = 0.4):
    return {
        "decision": {
            "truth_scale": scale,
            "truth_threshold": truth,
            "qualified_truth_threshold": qualified,
        },
        "value_domains": {"category": list(CATEGORY_DOMAIN)},
    }


# ======================================================================
# pure scaling
# ======================================================================
def test_absolute_scale_returns_the_probability_unchanged():
    value, mode, domain = truth_strength(0.25, _belief(0.25), {"decision": {}})

    assert value == 0.25
    assert mode == "absolute"
    assert domain is None


def test_uniform_lift_multiplies_by_the_configured_domain_size():
    value, mode, domain = truth_strength(0.2, _belief(0.2), _config("uniform_lift"))

    assert mode == "uniform_lift"
    assert domain == len(CATEGORY_DOMAIN)
    assert value == pytest.approx(0.2 * len(CATEGORY_DOMAIN))


def test_uniform_lift_falls_back_to_observed_distribution():
    config = {"decision": {"truth_scale": "uniform_lift"}, "value_domains": {}}
    belief = {
        "attribute": "wingspan",
        "probability": 0.5,
        "distribution": {"a": 0.5, "b": 0.3},
    }

    value, mode, domain = truth_strength(0.5, belief, config)

    assert mode == "uniform_lift"
    assert domain == 2
    assert value == pytest.approx(1.0)


def test_uniform_lift_without_any_domain_downgrades_to_absolute():
    value, mode, domain = truth_strength(
        0.3, {"attribute": "wingspan", "distribution": {}}, _config("uniform_lift")
    )

    assert mode == "absolute"
    assert domain is None
    assert value == 0.3


# ======================================================================
# decision-layer behavior
# ======================================================================
def test_absolute_thresholds_fall_back_to_weak_support():
    """The shipped rule cannot commit at p=0.24, which is the observed scale."""
    decision = decide_object_truth(
        _state(0.24), "OBJ-1", "category", None, config=_config("absolute", 0.6, 0.4)
    )

    assert decision.status == "Weak Support"
    assert decision.value == "ship"
    assert decision.evidence_refs["truth_scale"] == "absolute"


def test_uniform_lift_thresholds_commit_on_the_same_posterior():
    decision = decide_object_truth(
        _state(0.24), "OBJ-1", "category", None, config=_config("uniform_lift", 2.0, 1.5)
    )

    assert decision.status == "Weak Support"
    assert decision.value == "ship"
    assert decision.evidence_refs["truth_scale"] == "uniform_lift"
    assert decision.evidence_refs["truth_domain_size"] == len(CATEGORY_DOMAIN)
    assert decision.evidence_refs["truth_lift"] == pytest.approx(0.24 * len(CATEGORY_DOMAIN))


def test_uniform_lift_still_returns_weak_support_below_its_own_threshold():
    decision = decide_object_truth(
        _state(0.24), "OBJ-1", "category", None, config=_config("uniform_lift", 2.4, 2.1)
    )

    assert decision.status == "Weak Support"


def test_other_gates_still_fire_under_lift_scale():
    """The gate order is unchanged; lift only replaces the truth comparison."""
    state = _state(0.24)
    state.objects["OBJ-1"].ambiguity = {"association_entropy": 1.5}

    decision = decide_object_truth(
        state, "OBJ-1", "category", None, config=_config("uniform_lift", 2.0, 1.5)
    )

    assert decision.status == "Uncertain"
