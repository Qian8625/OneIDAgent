"""Optional mechanisms: damping schedule and provenance informativeness budget."""
from __future__ import annotations

from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.inference import run_joint_inference
from DS_VI.models.shared_identity.physical_prior import compute_physical_prior
from DS_VI.models.shared_identity.provenance import compute_group_weights
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    CandidateEdge,
    Claim,
    InferenceProblem,
    Modality,
    ObjectHypothesis,
    PrimitiveObservation,
    SpatialSupport,
    TimeScope,
)


# ------------------------------------------------------------------- builders
def _config(**physical):
    config = load_config()
    config["physical"].update(physical)
    return config


def _edges(*specs):
    """Edges for one row, always including the NULL candidate (as the graph does)."""
    edges = [
        CandidateEdge(observation_id="obs", object_id=object_id, features={"diagnostic_score": score})
        for object_id, score in specs
    ]
    edges.append(CandidateEdge(observation_id="obs", object_id=NULL_OBJECT_ID))
    return edges


def _observation(observation_id: str, bbox, channel: str) -> PrimitiveObservation:
    return PrimitiveObservation(
        observation_id=observation_id,
        modality=Modality.coerce("optical"),
        source_tool="SyntheticSensor",
        time_scope=TimeScope(key="t0"),
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        detection_confidence=0.9,
        input_roots=["root"],
        context={"channel": channel, "image_group_id": channel},
    )


def _object(object_id: str, bbox) -> ObjectHypothesis:
    return ObjectHypothesis(
        object_id=object_id,
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        attribute_hints={"category": "ship"},
    )


def _problem(config):
    observations = [_observation("obs_a", [0, 0, 10, 10], "a"), _observation("obs_b", [1, 1, 11, 11], "b")]
    objects = [_object("obj_1", [0, 0, 10, 10]), _object("obj_2", [50, 50, 60, 60])]
    claims = [
        Claim(claim_id=f"c_{obs.observation_id}", observation_id=obs.observation_id,
              attribute="category", value="ship", time_scope=TimeScope(key="t0"),
              source_channel=obs.context["channel"])
        for obs in observations
    ]
    edges = build_candidate_graph(observations, objects, config)
    return InferenceProblem(
        observations=observations, claims=claims, objects=objects, candidate_edges=edges
    )


def _state(config):
    config.setdefault("decision", {})["accept_betp"] = 0.3
    return run_joint_inference(_problem(config), config)


# --------------------------------------------------------------- Dirichlet prior
def test_damping_schedule_defaults_to_constant():
    state = _state(load_config())
    assert state.diagnostics["damping_schedule"] == "constant"
    assert state.diagnostics["final_damping"] == state.diagnostics["feedback_damping"]


def test_ramp_reduces_early_evidence_and_ends_at_the_configured_damping():
    config = load_config()
    config["inference"].update({"damping_schedule": "ramp", "damping_ramp_iterations": 3})
    state = _state(config)
    assert state.diagnostics["damping_schedule"] == "ramp"
    assert state.diagnostics["final_damping"] == state.diagnostics["feedback_damping"]
    # The first round carries no evidence, so the ramp path differs from the
    # constant one but must stay in the same basin.
    constant = _state(load_config())
    assert state.association["obs_b"]["obj_1"] > 0.5
    assert constant.association["obs_b"]["obj_1"] > 0.5


def test_ramp_with_one_iteration_equals_constant():
    base = load_config()
    base["inference"]["max_iter"] = 2
    ramp = load_config()
    ramp["inference"].update(
        {"max_iter": 2, "damping_schedule": "ramp", "damping_ramp_iterations": 1}
    )
    assert _state(base).association == _state(ramp).association


# -------------------------------------------------- informativeness provenance
def _claims(*triples):
    return [
        Claim(
            claim_id=claim_id,
            observation_id=observation_id,
            attribute=attribute,
            value=value,
            provenance_group_id="grp_1",
        )
        for claim_id, observation_id, attribute, value in triples
    ]


def test_informativeness_concentrates_the_budget_on_the_discriminative_claim():
    claims = _claims(
        ("c1", "obs", "category", "ship"),
        ("c2", "obs", "orientation", "north"),
        ("c3", "obs", "orientation", "east"),
    )
    weights = compute_group_weights(
        claims, informativeness={"c1": 0.99, "c2": 0.0, "c3": 0.0}, weighting="informativeness"
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["c1"] == 1.0
    assert weights["c2"] == 0.0 and weights["c3"] == 0.0


def test_informativeness_falls_back_to_uniform_when_nothing_is_informative():
    claims = _claims(("c1", "obs", "orientation", "north"), ("c2", "obs", "orientation", "east"))
    weights = compute_group_weights(
        claims, informativeness={"c1": 0.0, "c2": 0.0}, weighting="informativeness"
    )
    assert weights["c1"] == weights["c2"] == 0.5


def test_uniform_weighting_is_the_default():
    claims = _claims(("c1", "obs", "category", "ship"), ("c2", "obs", "subtype", "cargo"))
    weights = compute_group_weights(claims, informativeness={"c1": 0.99, "c2": 0.1})
    assert weights["c1"] == weights["c2"] == 0.5
