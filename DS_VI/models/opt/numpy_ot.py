"""Numpy reference implementation of the same (un)balanced Sinkhorn.

Purpose: (1) a torch-free reference for the equivalence tests, so the torch port is
verified against readable code rather than against itself; (2) an explicit
``backend="numpy"`` escape hatch for machines without torch (a *choice*, never a
silent fallback — the caller names the backend).

The algorithm mirrors :mod:`DS_VI.models.opt.torch_ot` exactly: log-domain scaling
with marginal exponents ``tau / (tau + reg)``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:  # scipy is a shipped dependency
    from scipy.special import logsumexp as _logsumexp
except Exception:  # pragma: no cover - scipy is in requirements.txt
    def _logsumexp(values, axis):
        peak = np.max(values, axis=axis, keepdims=True)
        return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(values - peak), axis=axis))


def log_sinkhorn(
    cost: Any,
    reg: float = 0.05,
    row_mass: Optional[Any] = None,
    col_mass: Optional[Any] = None,
    mask: Optional[Any] = None,
    tau_a: Optional[float] = None,
    tau_b: Optional[float] = None,
    iterations: int = 2000,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"cost must be 2-D, got shape {matrix.shape}")
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return {
            "plan": np.zeros((rows, cols)),
            "iterations": 0,
            "converged": True,
            "row_mass": np.zeros(rows),
            "col_mass": np.zeros(cols),
            "total_mass": 0.0,
        }

    log_kernel = -matrix / float(reg)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != matrix.shape:
            raise ValueError(f"mask shape {mask_array.shape} != cost {matrix.shape}")
        log_kernel = np.where(mask_array, log_kernel, -np.inf)

    def log_mass(values: Optional[Any], size: int) -> np.ndarray:
        if values is None:
            return np.zeros(size)
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size != size:
            raise ValueError(f"mass must have {size} entries, got {array.size}")
        return np.log(np.clip(array, 1e-30, None))

    log_a = log_mass(row_mass, rows)
    log_b = log_mass(col_mass, cols)
    exponent_a = 1.0 if tau_a is None else float(tau_a) / (float(tau_a) + float(reg))
    exponent_b = 1.0 if tau_b is None else float(tau_b) / (float(tau_b) + float(reg))

    log_u = np.zeros(rows)
    log_v = np.zeros(cols)
    converged = False
    iteration = 0
    balanced = tau_a is None and tau_b is None
    target_a = np.exp(log_a)
    target_b = np.exp(log_b)
    for iteration in range(1, int(iterations) + 1):
        previous_u = log_u
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            log_u = exponent_a * (log_a - _logsumexp(log_kernel + log_v[None, :], axis=1))
            log_v = exponent_b * (log_b - _logsumexp(log_kernel + log_u[:, None], axis=0))
        log_u = np.nan_to_num(log_u, nan=0.0, posinf=0.0, neginf=-1e12)
        log_v = np.nan_to_num(log_v, nan=0.0, posinf=0.0, neginf=-1e12)
        if balanced and (iteration % 5 == 0 or iteration == 1):
            probe = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
            probe = np.nan_to_num(probe, nan=0.0, posinf=0.0, neginf=0.0)
            violation = max(
                float(np.max(np.abs(probe.sum(axis=1) - target_a))),
                float(np.max(np.abs(probe.sum(axis=0) - target_b))),
            )
            if violation <= float(tol):
                converged = True
                break
        elif not balanced and float(np.max(np.abs(log_u - previous_u))) <= float(tol):
            converged = True
            break

    with np.errstate(over="ignore", invalid="ignore"):
        plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
    plan = np.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "plan": plan,
        "iterations": iteration,
        "converged": converged,
        "row_mass": plan.sum(axis=1),
        "col_mass": plan.sum(axis=0),
        "total_mass": float(plan.sum()),
    }


def sinkhorn_unbalanced(
    cost: Any,
    reg: float = 0.05,
    tau_a: float = 0.5,
    tau_b: float = 0.5,
    **kwargs: Any,
) -> Dict[str, Any]:
    return log_sinkhorn(cost, reg=reg, tau_a=tau_a, tau_b=tau_b, **kwargs)


__all__ = ["log_sinkhorn", "sinkhorn_unbalanced"]
