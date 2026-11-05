"""OT primitives: balanced/unbalanced log Sinkhorn, dustbin, extraction."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from DS_VI.models.opt.torch_ot import (
    assign_from_plan,
    log_sinkhorn,
    sinkhorn_unbalanced,
    with_dustbin,
)


def test_balanced_sinkhorn_respects_both_marginals():
    cost = np.array([[0.1, 0.9], [0.8, 0.2]])
    result = log_sinkhorn(cost, reg=0.05)
    # A sharply peaked kernel converges the *marginals* slowly, so the check is
    # on the marginal violation rather than on log-scaling deltas.
    assert result["converged"]
    assert np.allclose(result["row_mass"], np.ones(2), atol=1e-5)
    assert np.allclose(result["col_mass"], np.ones(2), atol=1e-6)
    assert result["total_mass"] == pytest.approx(2.0, abs=1e-6)


def test_masked_entries_stay_exactly_zero():
    cost = np.array([[0.1, 0.9], [0.8, 0.2]])
    mask = np.array([[True, False], [False, True]])
    result = log_sinkhorn(cost, reg=0.05, mask=mask)
    assert result["plan"][0, 1] == 0.0
    assert result["plan"][1, 0] == 0.0
    assert result["plan"][0, 0] > 0.0


def test_unbalanced_relaxes_the_marginals():
    cost = np.array([[0.1, 0.9], [0.8, 0.2]])
    balanced = log_sinkhorn(cost, reg=0.05)
    relaxed = sinkhorn_unbalanced(cost, reg=0.05, tau_a=0.05, tau_b=0.05)
    assert relaxed["total_mass"] < balanced["total_mass"]
    assert np.any(np.abs(relaxed["row_mass"] - 1.0) > 1e-3)


def test_dustbin_absorbs_a_hopeless_row():
    # Row 0 has a good counterpart, row 1 does not; the dustbin must take row 1.
    cost, _ = with_dustbin(np.array([[0.0], [0.9]]), dustbin_cost=0.5)
    result = sinkhorn_unbalanced(cost, reg=0.05, tau_a=0.5, tau_b=0.5)
    assignment = assign_from_plan(result["plan"], min_mass=0.5, real_columns=1)
    assert assignment["pairs"] == [{"row": 0, "col": 0, "mass": assignment["pairs"][0]["mass"]}]
    assert assignment["unmatched_rows"] == [1]


def test_assignment_is_one_to_one():
    # Two rows both prefer column 0 but the second-best for row 1 is column 1.
    plan = np.array([[0.9, 0.1], [0.6, 0.4]])
    assignment = assign_from_plan(plan, min_mass=0.3, real_columns=2)
    cols = [pair["col"] for pair in assignment["pairs"]]
    assert len(cols) == len(set(cols)) == 2


def test_extraction_returns_everything_below_the_mass_floor():
    plan = np.array([[0.2, 0.1], [0.1, 0.2]])
    assignment = assign_from_plan(plan, min_mass=0.5, real_columns=2)
    assert assignment["pairs"] == []
    assert assignment["unmatched_rows"] == [0, 1]
    assert assignment["unmatched_cols"] == [0, 1]


def test_runs_are_deterministic():
    cost = np.array([[0.1, 0.4], [0.5, 0.2], [0.9, 0.8]])
    first = log_sinkhorn(cost, reg=0.1)
    second = log_sinkhorn(cost, reg=0.1)
    assert np.array_equal(first["plan"], second["plan"])


def test_shape_and_mass_validation():
    with pytest.raises(ValueError):
        log_sinkhorn(np.zeros((2, 3, 4)))
    with pytest.raises(ValueError):
        log_sinkhorn(np.zeros((2, 3)), row_mass=[1.0])
    with pytest.raises(ValueError):
        log_sinkhorn(np.zeros((2, 3)), mask=np.ones((3, 2), dtype=bool))
