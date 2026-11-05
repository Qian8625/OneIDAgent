"""Association-matrix (A) invariants for the shared-identity inference layer.

A has exactly one row per *observation*.  Every assertion here is about that row
being a legal probability distribution over admissible objects, and about it
responding to evidence in the right direction and magnitude.
"""
from __future__ import annotations

import math

from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph, candidate_mask
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.inference import (
    build_value_domains,
    initialize_truth,
    run_joint_inference,
    update_association,
)
from DS_VI.models.shared_identity.physical_features import bbox_iou
from DS_VI.models.shared_identity.physical_prior import compute_physical_prior, prior_matrix
from DS_VI.models.shared_identity.provenance import assign_group_ids, compute_group_weights
from DS_VI.models.shared_identity.reliability import ReliabilityModel
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    Claim,
    InferenceProblem,
    Modality,
    ObjectHypothesis,
    PrimitiveObservation,
    SpatialSupport,
    TimeScope,
)

EPSILON = 1e-12


# ----------------------------------------------------------------------
# builders (no conftest in this repo, so each test module is self-contained)
# ----------------------------------------------------------------------
def make_observation(
    observation_id: str,
    modality: str,
    bbox,
    time_key: str = "t0",
    input_root: str = "root_a",
    score: float = 0.9,
    source_tool: str = "SyntheticSensor",
) -> PrimitiveObservation:
    return PrimitiveObservation(
        observation_id=observation_id,
        modality=Modality.coerce(modality),
        source_tool=source_tool,
        time_scope=TimeScope(key=time_key),
        # SpatialSupport is frozen, so it is always constructed, never mutated.
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        detection_confidence=float(score),
        input_roots=[input_root],
    )


def make_claim(
    claim_id: str,
    observation_id: str,
    attribute: str,
    value,
    time_key: str = "t0",
    channel: str = "optical",
    source_tool: str = "SyntheticSensor",
) -> Claim:
    return Claim(
        claim_id=claim_id,
        observation_id=observation_id,
        attribute=attribute,
        value=value,
        time_scope=TimeScope(key=time_key),
        source_tool=source_tool,
        source_channel=channel,
    )


def make_object(object_id: str, label: str, bbox) -> ObjectHypothesis:
    return ObjectHypothesis(
        object_id=object_id,
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        attribute_hints={"category": label},
    )


def build_problem(observations, claims, objects, config=None, capacities=None):
    """Assemble an InferenceProblem with a freshly built candidate graph."""
    cfg = dict(config or load_config())
    edges = build_candidate_graph(observations, objects, cfg)
    problem = InferenceProblem(
        observations=list(observations),
        claims=list(claims),
        objects=list(objects),
        candidate_edges=edges,
        capacities=dict(capacities or {}),
    )
    return problem, cfg


def dense_association(problem, config):
    """Run exactly one ``update_association`` step and return the dense matrix.

    ``run_joint_inference`` only reports strictly positive cells, so the dense
    view is the only way to assert that masked-out cells are *exactly* 0.0.
    """
    cfg = dict(config)
    inference_cfg = dict(cfg.get("inference") or {})
    eta = float(inference_cfg.get("eta", 1.0))
    smoothing = float(inference_cfg.get("truth_prior_smoothing", 1.0))

    observations = {obs.observation_id: obs for obs in problem.observations}
    assign_group_ids(problem.claims, observations)
    weights = compute_group_weights(problem.claims, None, 1.0)

    observation_ids = [obs.observation_id for obs in problem.observations]
    object_ids = problem.object_ids(include_null=True)

    prior = prior_matrix(
        compute_physical_prior(problem.candidate_edges, cfg), observation_ids, object_ids
    )
    mask = candidate_mask(problem.candidate_edges, observation_ids, object_ids)
    reliabilities = ReliabilityModel(cfg).build_table(problem.claims, observations)
    domains = build_value_domains(problem, cfg)
    truth = initialize_truth(problem, object_ids, domains, smoothing)

    association = update_association(
        problem, observation_ids, object_ids, prior, mask, truth,
        reliabilities, weights, domains, eta, EPSILON,
    )
    return association, mask, observation_ids, object_ids


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------
def test_association_rows_are_probability_distributions():
    """Every A row must be a legal distribution: non-negative and summing to 1."""
    obs_near = make_observation("m_near", "optical", [100, 100, 140, 140], input_root="r1")
    obs_mid = make_observation("m_mid", "sar", [120, 100, 160, 140], input_root="r2")
    obs_far = make_observation("m_far", "optical", [900, 900, 940, 940], input_root="r3")
    objects = [
        make_object("obj_a", "ship", [100, 100, 140, 140]),
        make_object("obj_b", "ship", [140, 100, 180, 140]),
    ]
    claims = [
        make_claim("c_near", "m_near", "category", "ship"),
        make_claim("c_mid", "m_mid", "category", "ship", channel="sar"),
        make_claim("c_far", "m_far", "category", "vehicle"),
    ]
    problem, cfg = build_problem([obs_near, obs_mid, obs_far], claims, objects)

    association, _, observation_ids, _ = dense_association(problem, cfg)
    for m, observation_id in enumerate(observation_ids):
        row = association[m]
        assert all(value >= 0.0 for value in row), f"negative mass in row {observation_id}"
        # 1e-9 rather than == 1.0: the row comes out of a softmax + renormalization.
        assert math.isclose(sum(row), 1.0, abs_tol=1e-9), f"row {observation_id} sums to {sum(row)}"

    state = run_joint_inference(problem, cfg)
    for observation_id, row in state.association.items():
        assert all(p >= 0.0 for p in row.values())
        assert math.isclose(sum(row.values()), 1.0, abs_tol=1e-9), observation_id


def test_non_candidate_cells_are_exactly_zero():
    """Inadmissible cells must be 0.0 exactly, not merely small.

    A gated-out edge that keeps epsilon mass would let a physically impossible
    association leak into the truth update, which is exactly what the candidate
    mask exists to prevent.
    """
    obs_near = make_observation("m_near", "optical", [100, 100, 140, 140], input_root="r1")
    obs_far = make_observation("m_far", "optical", [900, 900, 940, 940], input_root="r2")
    objects = [
        make_object("obj_a", "ship", [100, 100, 140, 140]),
        make_object("obj_b", "ship", [300, 300, 340, 340]),
    ]
    claims = [
        make_claim("c_near", "m_near", "category", "ship"),
        make_claim("c_far", "m_far", "category", "vehicle"),
    ]
    problem, cfg = build_problem([obs_near, obs_far], claims, objects)

    association, mask, observation_ids, object_ids = dense_association(problem, cfg)

    masked_out = [
        (m, k)
        for m in range(len(observation_ids))
        for k in range(len(object_ids))
        if not mask[m][k]
    ]
    assert masked_out, "test is vacuous unless at least one cell is gated out"
    for m, k in masked_out:
        assert association[m][k] == 0.0, (
            f"gated cell ({observation_ids[m]}, {object_ids[k]}) has mass {association[m][k]}"
        )

    # The sparse view from run_joint_inference must simply omit those objects.
    state = run_joint_inference(problem, cfg)
    for m, k in masked_out:
        assert object_ids[k] not in state.association_row(observation_ids[m])


def test_null_only_row_gets_full_null_mass():
    """An observation outside every gate has NULL as its only admissible edge."""
    obs_far = make_observation("m_far", "optical", [900, 900, 940, 940])
    objects = [make_object("obj_near", "ship", [10, 10, 50, 50])]
    claims = [make_claim("c_far", "m_far", "category", "vehicle")]
    problem, cfg = build_problem([obs_far], claims, objects)

    admissible = {
        edge.object_id for edge in problem.candidate_edges
        if edge.observation_id == "m_far" and edge.admissible
    }
    assert admissible == {NULL_OBJECT_ID}, "gating premise broken; test would be vacuous"

    state = run_joint_inference(problem, cfg)
    row = state.association_row("m_far")
    assert row == {NULL_OBJECT_ID: 1.0}
    # No real object may collect any mass when none was admissible.
    assert state.top_object("m_far") == (NULL_OBJECT_ID, 1.0)


def test_association_is_monotone_in_geometric_quality():
    """Better overlap with the object must yield more mass on that object.

    The prior is quality-scaled, so a sole weak candidate must not win its row
    by default just because it has no competitor.
    """
    object_bbox = [10, 20, 50, 80]
    offsets = ([10, 20, 50, 80], [16, 26, 56, 86], [24, 34, 64, 94])

    ious = []
    probabilities = []
    for index, bbox in enumerate(offsets):
        observation = make_observation(f"m{index}", "optical", bbox)
        objects = [make_object("obj", "ship", object_bbox)]
        claims = [make_claim(f"c{index}", f"m{index}", "category", "ship")]
        problem, cfg = build_problem([observation], claims, objects)
        state = run_joint_inference(problem, cfg)
        ious.append(bbox_iou(bbox, object_bbox))
        probabilities.append(state.association_row(f"m{index}").get("obj", 0.0))

    # Premise: the three geometries really are ordered by IoU.
    assert ious[0] > ious[1] > ious[2]
    # Conclusion: association probability follows the same order strictly.
    assert probabilities[0] > probabilities[1] > probabilities[2], probabilities
    # And the worst geometry must leave the majority of the mass on NULL.
    assert probabilities[2] < 0.5


def test_joint_inference_converges_with_clean_diagnostics():
    """A simple 2-observation / 1-object problem must converge cleanly."""
    obs_optical = make_observation("m_opt", "optical", [10, 20, 50, 80], input_root="root_optical")
    obs_sar = make_observation(
        "m_sar", "sar", [12, 22, 52, 82], input_root="root_sar", source_tool="SyntheticSAR"
    )
    objects = [make_object("obj_1", "ship", [11, 21, 51, 81])]
    claims = [
        make_claim("c1", "m_opt", "category", "ship"),
        make_claim("c2", "m_sar", "category", "ship", channel="sar", source_tool="SyntheticSAR"),
    ]
    problem, cfg = build_problem([obs_optical, obs_sar], claims, objects)

    state = run_joint_inference(problem, cfg)

    assert state.converged is True
    assert 0 < state.iterations <= int(dict(cfg["inference"])["max_iter"])
    # Convergence means the last A step barely moved.
    assert state.max_delta_history[-1] <= float(dict(cfg["inference"])["tol"])
    # Structural invariant, recorded on every run; anything but 0.0 is a bug.
    assert state.diagnostics["identity_contradiction_rate"] == 0.0
    # The capacity projection must report a feasible solution, not a fallback.
    assert state.diagnostics["projection_feasible"] is True
    assert state.diagnostics["projection"]["max_violation"] <= 1e-5

    # Both views of one ship should land on it.
    assert state.top_object("m_opt")[0] == "obj_1"
    assert state.top_object("m_sar")[0] == "obj_1"


def test_symmetric_candidates_stay_tied():
    """Two indistinguishable candidates must keep identical probability.

    Breaking the tie here would be a silent arbitrary choice: the model would
    report confidence it has no evidence for. Tolerance is 1e-6 because the row
    goes through iterative KL scaling, not closed-form arithmetic.
    """
    observation = make_observation("m_sym", "sar", [110, 100, 150, 140], input_root="root_sar")
    objects = [
        make_object("obj_sym_1", "ship", [100, 100, 140, 140]),
        make_object("obj_sym_2", "ship", [120, 100, 160, 140]),
    ]
    claims = [make_claim("c_sym", "m_sym", "category", "ship", channel="sar")]
    problem, cfg = build_problem([observation], claims, objects)

    state = run_joint_inference(problem, cfg)
    row = state.association_row("m_sym")

    assert "obj_sym_1" in row and "obj_sym_2" in row
    assert abs(row["obj_sym_1"] - row["obj_sym_2"]) < 1e-6, row
    # Neither candidate may be reported as a confident identity.
    assert row["obj_sym_1"] < 0.5 and row["obj_sym_2"] < 0.5
    # Unexplained mass belongs to NULL, which is the honest answer here.
    assert row[NULL_OBJECT_ID] > max(row["obj_sym_1"], row["obj_sym_2"])
