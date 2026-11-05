"""Capacity-constrained KL projection of the association matrix.

The projection is what stops ten observations from all claiming one object.  It
must do so while keeping every row on the simplex, keeping gated cells at exactly
zero, and leaving ``__NULL__`` unconstrained.
"""
from __future__ import annotations

import math

import pytest

from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.inference import run_joint_inference
from DS_VI.models.shared_identity.projection import (
    ProjectionInfeasibleError,
    build_capacities,
    project_association_block_kl,
    project_association_kl,
    verify_projection,
)
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

# The projection stops scaling once the violation is below its own tolerance, so
# assertions on column sums use the same slack the implementation guarantees.
CAPACITY_TOL = 1e-5


def column_sums(matrix):
    columns = len(matrix[0]) if matrix else 0
    return [sum(row[k] for row in matrix) for k in range(columns)]


def test_block_projection_resets_capacity_per_acquisition_block():
    matrix, diagnostics = project_association_block_kl(
        [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
        [1.0, None],
        [[True, True], [True, True], [True, True]],
        ["acq_a", "acq_a", "acq_b"],
        block_capacity=1.0,
    )

    assert diagnostics["mode"] == "block_object_column"
    assert sum(row[0] for row in matrix[:2]) <= 1.0 + CAPACITY_TOL
    assert sum(row[0] for row in matrix[2:]) <= 1.0 + CAPACITY_TOL
    assert sum(row[0] for row in matrix) > 1.0


# ----------------------------------------------------------------------
# tests on project_association_kl / build_capacities / verify_projection
# ----------------------------------------------------------------------
def test_projection_keeps_rows_normalized_and_respects_capacity():
    """Four observations fight over a capacity-1 object; rows stay on the simplex."""
    a_tilde = [[0.90, 0.05, 0.05] for _ in range(4)]
    capacities = [1.0, 2.0, None]  # obj0 cap 1, obj1 cap 2, NULL unconstrained
    mask = [[True, True, True] for _ in range(4)]

    projected, diagnostics = project_association_kl(a_tilde, capacities, mask)

    assert diagnostics["converged"] is True
    for m, row in enumerate(projected):
        assert math.isclose(sum(row), 1.0, abs_tol=1e-9), f"row {m} sums to {sum(row)}"
        assert all(value >= 0.0 for value in row)

    sums = column_sums(projected)
    assert sums[0] <= 1.0 + CAPACITY_TOL, sums
    assert sums[1] <= 2.0 + CAPACITY_TOL, sums
    # The over-subscribed column really was pushed down from its raw 3.6 total.
    assert sums[0] < 3.6
    assert verify_projection(projected, capacities, mask)["feasible"] is True


def test_null_capacity_is_effectively_unlimited():
    """Every observation may route to NULL; a ``None`` capacity is never scaled."""
    rows = 6
    a_tilde = [[0.0, 1.0] for _ in range(rows)]
    capacities = [1.0, None]
    mask = [[True, True] for _ in range(rows)]

    projected, diagnostics = project_association_kl(a_tilde, capacities, mask)

    assert diagnostics["converged"] is True
    assert diagnostics["max_violation"] == 0.0
    sums = column_sums(projected)
    # All 6 units of mass sit on NULL. Any finite NULL capacity would have
    # forced some of it onto the real object, inventing an association.
    assert math.isclose(sums[1], float(rows), abs_tol=1e-9), sums
    assert math.isclose(sums[0], 0.0, abs_tol=1e-12), sums


def test_build_capacities_defaults_and_null_column():
    """NULL gets ``None`` (unlimited); unset real objects get the default."""
    object_ids = ["obj_1", "obj_2", NULL_OBJECT_ID]
    capacities = build_capacities(object_ids, NULL_OBJECT_ID, {"obj_2": 5}, default_capacity=2)

    assert capacities[0] == 2.0  # unset -> default
    assert capacities[1] == 5.0  # explicit override honoured
    assert capacities[2] is None  # NULL must be unconstrained, not a large number

    # Ordering follows object_ids, so a NULL column anywhere is handled.
    reordered = build_capacities([NULL_OBJECT_ID, "obj_1"], NULL_OBJECT_ID, {}, 3)
    assert reordered == [None, 3.0]


def test_verify_projection_detects_violations_and_accepts_valid_matrices():
    """verify_projection is the independent checker, so it must actually fail."""
    capacities = [1.0, 1.0, None]
    mask = [[True, True, True], [True, True, True]]

    over_capacity = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    report = verify_projection(over_capacity, capacities, mask)
    assert report["feasible"] is False
    assert any("exceeds capacity" in problem for problem in report["problems"])

    unnormalized = [[0.5, 0.0, 0.0], [0.2, 0.3, 0.1]]
    assert verify_projection(unnormalized, capacities, mask)["feasible"] is False

    # Mass on a gated-out cell must be reported even if rows and columns are fine.
    gated_mask = [[False, True, True], [True, True, True]]
    leaking = [[0.5, 0.25, 0.25], [0.5, 0.25, 0.25]]
    leak_report = verify_projection(leaking, [None, None, None], gated_mask)
    assert leak_report["feasible"] is False
    assert any("non-candidate" in problem for problem in leak_report["problems"])

    valid = [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]]
    assert verify_projection(valid, [1.0, 1.0, None], mask)["feasible"] is True


def test_empty_candidate_row_falls_back_to_null():
    """A row with no real candidate must not crash and must land fully on NULL."""
    a_tilde = [[0.0, 0.0, 0.0]]
    capacities = [1.0, 1.0, None]
    mask = [[False, False, True]]  # only the NULL column is admissible

    projected, diagnostics = project_association_kl(a_tilde, capacities, mask)

    assert diagnostics["converged"] is True
    assert projected[0][2] == 1.0
    # Gated columns stay exactly zero even though the input row was all zeros.
    assert projected[0][0] == 0.0 and projected[0][1] == 0.0

    # With no admissible column at all the projection must refuse rather than
    # emit an illegal distribution.
    with pytest.raises(ProjectionInfeasibleError):
        project_association_kl([[0.0, 0.0]], [1.0, None], [[False, False]])

    # No NULL column and not enough capacity is also infeasible, not silently
    # over-subscribed.
    with pytest.raises(ProjectionInfeasibleError):
        project_association_kl([[0.5, 0.5]] * 3, [1.0, 1.0], [[True, True]] * 3)


def test_projection_is_idempotent_on_feasible_input():
    """Re-projecting a feasible matrix must not move it.

    The projection runs once per inference iteration, so a non-idempotent
    projection would keep drifting A and prevent convergence.
    """
    feasible = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]
    capacities = [2.0, 2.0, None]
    mask = [[True, True, True], [True, True, True]]
    assert verify_projection(feasible, capacities, mask)["feasible"] is True

    once, first_diagnostics = project_association_kl(feasible, capacities, mask)
    assert first_diagnostics["converged"] is True
    assert max(
        abs(once[m][k] - feasible[m][k]) for m in range(2) for k in range(3)
    ) < 1e-9

    twice, _ = project_association_kl(once, capacities, mask)
    assert max(abs(twice[m][k] - once[m][k]) for m in range(2) for k in range(3)) < 1e-9

    # Also idempotent after a genuine correction: projecting the output again
    # must be a no-op.
    corrected, _ = project_association_kl(
        [[0.9, 0.05, 0.05]] * 4, [1.0, 2.0, None], [[True] * 3] * 4
    )
    reprojected, _ = project_association_kl(corrected, [1.0, 2.0, None], [[True] * 3] * 4)
    assert max(
        abs(reprojected[m][k] - corrected[m][k]) for m in range(4) for k in range(3)
    ) < 1e-9


def test_end_to_end_capacity_is_enforced_by_run_joint_inference():
    """Capacity survives the full A/projection/Q loop, not just a direct call."""
    observations = [
        PrimitiveObservation(
            observation_id=f"m{index}",
            modality=Modality.coerce("optical"),
            source_tool="SyntheticSensor",
            time_scope=TimeScope(key="t0"),
            spatial_support=SpatialSupport(bbox=[10.0, 20.0, 50.0, 80.0]),
            detection_confidence=0.9,
            input_roots=[f"root_{index}"],
        )
        for index in range(5)
    ]
    claims = [
        Claim(
            claim_id=f"c{index}",
            observation_id=f"m{index}",
            attribute="category",
            value="ship",
            time_scope=TimeScope(key="t0"),
            source_tool="SyntheticSensor",
            source_channel="optical",
        )
        for index in range(5)
    ]
    objects = [
        ObjectHypothesis(
            object_id="obj",
            spatial_support=SpatialSupport(bbox=[10.0, 20.0, 50.0, 80.0]),
            attribute_hints={"category": "ship"},
        )
    ]
    # Paper Eq.(29) caps the row sum per acquisition/channel block, so the five
    # observations are placed in one block.
    for observation in observations:
        observation.context["image_group_id"] = "img_same_block"
    cfg = load_config()
    edges = build_candidate_graph(observations, objects, cfg)
    problem = InferenceProblem(
        observations=observations,
        claims=claims,
        objects=objects,
        candidate_edges=edges,
    )

    state = run_joint_inference(problem, cfg)

    assert state.diagnostics["projection_feasible"] is True
    object_column = sum(row.get("obj", 0.0) for row in state.association.values())
    # One block, one slot: the object cannot absorb more than 1 unit.
    assert object_column <= 1.0 + CAPACITY_TOL, object_column
    # The overflow has to go somewhere legal, and NULL is the only place.
    null_column = sum(row.get(NULL_OBJECT_ID, 0.0) for row in state.association.values())
    assert null_column > 1.0
    for observation_id, row in state.association.items():
        assert math.isclose(sum(row.values()), 1.0, abs_tol=1e-9), observation_id


def test_build_capacities_none_default_is_unconstrained():
    """``default_capacity=None`` removes the constraint instead of applying 2.

    The capacity ablation needs a real "off": falling back to the shipped
    default would make the flag a no-op whenever the data uses that default.
    """
    object_ids = ["obj_1", "obj_2", NULL_OBJECT_ID]

    capacities = build_capacities(
        object_ids, NULL_OBJECT_ID, {"obj_2": 5}, default_capacity=None
    )

    assert capacities[0] is None
    assert capacities[1] == 5.0
    assert capacities[2] is None


def _over_subscribed_problem() -> InferenceProblem:
    """Four observations whose only candidate is one capacity-1 object."""
    observations = [
        PrimitiveObservation(
            observation_id=f"m{index}",
            modality=Modality.coerce("optical"),
            source_tool="SyntheticSensor",
            time_scope=TimeScope(key="t0"),
            spatial_support=SpatialSupport(bbox=[10.0, 20.0, 50.0, 80.0]),
            detection_confidence=0.9,
            input_roots=[f"root_{index}"],
        )
        for index in range(4)
    ]
    claims = [
        Claim(
            claim_id=f"c{index}",
            observation_id=f"m{index}",
            attribute="category",
            value="ship",
            time_scope=TimeScope(key="t0"),
            source_tool="SyntheticSensor",
            source_channel="optical",
        )
        for index in range(4)
    ]
    objects = [
        ObjectHypothesis(
            object_id="obj",
            spatial_support=SpatialSupport(bbox=[10.0, 20.0, 50.0, 80.0]),
            attribute_hints={"category": "ship"},
        )
    ]
    for observation in observations:
        observation.context["image_group_id"] = "img_same_block"
    cfg = load_config()
    edges = build_candidate_graph(observations, objects, cfg)
    return InferenceProblem(
        observations=observations,
        claims=claims,
        objects=objects,
        candidate_edges=edges,
    )


def test_capacity_projection_flag_controls_the_constraint():
    """``capacity_projection=false`` must actually drop the cap."""
    cfg_on = load_config(overrides={"inference": {"capacity_projection": True}})
    cfg_off = load_config(overrides={"inference": {"capacity_projection": False}})

    state_on = run_joint_inference(_over_subscribed_problem(), cfg_on)
    state_off = run_joint_inference(_over_subscribed_problem(), cfg_off)

    assert state_on.diagnostics["capacity_projection"] is True
    assert state_off.diagnostics["capacity_projection"] is False

    column_on = sum(row.get("obj", 0.0) for row in state_on.association.values())
    column_off = sum(row.get("obj", 0.0) for row in state_off.association.values())

    assert column_on <= 1.0 + CAPACITY_TOL, column_on
    assert column_off > 1.0 + CAPACITY_TOL, column_off
