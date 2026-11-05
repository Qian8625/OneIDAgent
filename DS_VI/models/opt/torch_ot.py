"""Log-domain Sinkhorn and unbalanced optimal transport on torch.

The cascade's association is a *one-to-one* assignment with a gate. Two things
that assignment cannot express are exactly what the over-merge analysis pointed at:

* "no good counterpart" — a detection should be allowed to match a **dustbin**
  instead of being forced onto the least-bad partner;
* "the counts do not balance" — one side may have more detections than the other,
  so the marginal constraints are wrong by construction.

Both are standard unbalanced-OT ideas. ``sinkhorn_unbalanced`` here follows the
KL-relaxed marginals parameterisation (exponent ``tau / (tau + reg)``): larger
``tau`` tends to balanced Sinkhorn, smaller ``tau`` lets mass be destroyed and
created, which is what a dustbin encodes explicitly.

Everything is log-domain for stability, CPU float64 by default so an OT experiment
stays deterministic and reproducible without a GPU.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from DS_VI.models.opt import pick_device, pick_dtype, require_torch


def _to_tensor(values: Any, dtype: Any, device: Any) -> Any:
    torch = require_torch()
    array = np.asarray(values, dtype=np.float64)
    return torch.as_tensor(array, dtype=dtype, device=device)


def _log_mass(mass: Optional[Any], size: int, dtype: Any, device: Any) -> Any:
    torch = require_torch()
    if mass is None:
        return torch.zeros(size, dtype=dtype, device=device)
    values = _to_tensor(mass, dtype, device).reshape(-1)
    if values.numel() != size:
        raise ValueError(f"mass must have {size} entries, got {values.numel()}")
    return torch.log(torch.clamp(values, min=1e-30))


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
    device: str = "cpu",
) -> Dict[str, Any]:
    """Solve the (un)balanced entropic transport problem in the log domain.

    Args:
        cost: ``[m, n]`` cost matrix (finite for admissible entries).
        reg: entropic regularisation (``epsilon``).
        row_mass / col_mass: target marginals (default: all ones).
        mask: ``[m, n]`` boolean admissibility mask; ``False`` entries are forced
            to exactly zero transport.
        tau_a / tau_b: KL penalty on the row/column marginals. ``None`` means
            balanced Sinkhorn. Finite values relax the marginals with exponent
            ``tau / (tau + reg)`` (Chizat et al. 2017 / POT convention).
        iterations, tol: scaling sweeps and convergence tolerance on the update.

    Returns:
        ``{"plan", "iterations", "converged", "row_mass", "col_mass",
        "total_mass", "device", "dtype"}`` where the mass entries are numpy.
    """
    torch = require_torch()
    device_obj = pick_device(device)
    dtype = pick_dtype(device_obj, exact=True)

    cost_t = _to_tensor(cost, dtype, device_obj)
    if cost_t.dim() != 2:
        raise ValueError(f"cost must be 2-D, got shape {tuple(cost_t.shape)}")
    rows, cols = int(cost_t.shape[0]), int(cost_t.shape[1])
    if rows == 0 or cols == 0:
        return {
            "plan": np.zeros((rows, cols), dtype=float),
            "iterations": 0,
            "converged": True,
            "row_mass": np.zeros(rows),
            "col_mass": np.zeros(cols),
            "total_mass": 0.0,
            "device": str(device_obj),
            "dtype": str(dtype),
        }

    log_kernel = -cost_t / float(reg)
    if mask is not None:
        mask_t = torch.as_tensor(np.asarray(mask, dtype=bool), device=device_obj)
        if mask_t.shape != cost_t.shape:
            raise ValueError(f"mask shape {tuple(mask_t.shape)} != cost {tuple(cost_t.shape)}")
        log_kernel = log_kernel.masked_fill(~mask_t, float("-inf"))

    log_a = _log_mass(row_mass, rows, dtype, device_obj)
    log_b = _log_mass(col_mass, cols, dtype, device_obj)
    exponent_a = 1.0 if tau_a is None else float(tau_a) / (float(tau_a) + float(reg))
    exponent_b = 1.0 if tau_b is None else float(tau_b) / (float(tau_b) + float(reg))

    log_u = torch.zeros(rows, dtype=dtype, device=device_obj)
    log_v = torch.zeros(cols, dtype=dtype, device=device_obj)
    # Balanced problems are checked on the *marginal violation*, not on the log
    # scaling deltas: a sharply peaked kernel (small ``reg``) converges the
    # marginals long before the logarithms stop drifting, so a delta test would
    # report "not converged" on a perfectly valid plan.
    balanced = tau_a is None and tau_b is None
    target_a = torch.exp(log_a)
    target_b = torch.exp(log_b)
    converged = False
    iteration = 0
    for iteration in range(1, int(iterations) + 1):
        previous_u = log_u
        previous_v = log_v
        log_u = exponent_a * (log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1))
        log_v = exponent_b * (log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0))
        if balanced and (iteration % 5 == 0 or iteration == 1):
            probe = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
            violation = float(
                torch.maximum(
                    torch.max(torch.abs(probe.sum(dim=1) - target_a)),
                    torch.max(torch.abs(probe.sum(dim=0) - target_b)),
                )
            )
            if violation <= float(tol):
                converged = True
                break
        elif not balanced:
            delta = float(torch.maximum(
                torch.max(torch.abs(log_u - previous_u)),
                torch.max(torch.abs(log_v - previous_v)),
            ))
            if delta <= float(tol):
                converged = True
                break

    plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    plan = torch.nan_to_num(plan, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "plan": plan.detach().cpu().numpy(),
        "iterations": iteration,
        "converged": converged,
        "row_mass": plan.detach().sum(dim=1).cpu().numpy(),
        "col_mass": plan.detach().sum(dim=0).cpu().numpy(),
        "total_mass": float(plan.detach().sum()),
        "device": str(device_obj),
        "dtype": str(dtype),
        "exponents": (exponent_a, exponent_b),
    }


def sinkhorn_unbalanced(
    cost: Any,
    reg: float = 0.05,
    tau_a: float = 0.5,
    tau_b: float = 0.5,
    row_mass: Optional[Any] = None,
    col_mass: Optional[Any] = None,
    mask: Optional[Any] = None,
    iterations: int = 2000,
    tol: float = 1e-5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """KL-relaxed marginals: mass may be created/destroyed (dustbin in disguise)."""
    return log_sinkhorn(
        cost, reg=reg, row_mass=row_mass, col_mass=col_mass, mask=mask,
        tau_a=tau_a, tau_b=tau_b, iterations=iterations, tol=tol, device=device,
    )


def with_dustbin(cost: Any, dustbin_cost: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Append one dustbin column: everything can go unmatched at `dustbin_cost`.

    Returns ``(augmented_cost, mask)`` where the mask forbids the dustbin column
    only when the caller explicitly passes a mask afterwards.
    """
    array = np.asarray(cost, dtype=float)
    if array.ndim != 2:
        raise ValueError("cost must be 2-D")
    rows, cols = array.shape
    augmented = np.full((rows, cols + 1), float(dustbin_cost), dtype=float)
    augmented[:, :cols] = array
    mask = np.ones((rows, cols + 1), dtype=bool)
    mask[:, cols] = True
    return augmented, mask


def with_dustbin_square(cost: Any, dustbin_cost: float = 0.5, forbid: float = 2.0) -> np.ndarray:
    """Square dustbin augmentation: every row and column gets its own exit.

    Layout ``(m+n) x (m+n)``: top-left is the real cost, the diagonal blocks let a
    detection leave unmatched (row -> own dustbin column) or an object stay
    unclaimed (own dustbin row -> column), and the bottom-right corner is free
    dustbin-to-dustbin transport. Balanced Sinkhorn on this matrix converges in tens
    of iterations and never underflows, unlike a single shared dustbin column whose
    marginal has to absorb every unmatched row at once.
    """
    array = np.asarray(cost, dtype=float)
    if array.ndim != 2:
        raise ValueError("cost must be 2-D")
    rows, cols = array.shape
    augmented = np.zeros((rows + cols, rows + cols), dtype=float)
    augmented[:rows, :cols] = array
    off_diagonal = np.ones((rows, rows)) * float(forbid)
    np.fill_diagonal(off_diagonal, float(dustbin_cost))
    augmented[:rows, cols:] = off_diagonal
    off_diagonal_cols = np.ones((cols, cols)) * float(forbid)
    np.fill_diagonal(off_diagonal_cols, float(dustbin_cost))
    augmented[rows:, :cols] = off_diagonal_cols
    augmented[rows:, cols:] = 0.0
    return augmented


def assign_from_plan(
    plan: Any,
    min_mass: float = 0.5,
    real_columns: Optional[int] = None,
) -> Dict[str, Any]:
    """Greedy one-to-one extraction from a soft plan.

    A row is matched when its best **real** column carries at least ``min_mass``
    and beats the dustbin column (the last one when ``real_columns`` is None).
    Conflicts are resolved by plan mass so no object is claimed twice.
    """
    transport = np.asarray(plan, dtype=float)
    rows, cols = transport.shape
    real = cols if real_columns is None else int(real_columns)
    if real <= 0:
        return {"pairs": [], "unmatched_rows": list(range(rows)), "unmatched_cols": []}

    # Every (row, column) entry above the floor is a candidate, not only each
    # row's argmax: if a row's best column is taken by a stronger row it must be
    # able to fall back to its second choice instead of being reported unmatched.
    candidates = []
    for i in range(rows):
        real_slice = transport[i, :real]
        if real_slice.size == 0:
            continue
        dustbin_mass = float(transport[i, real]) if real < cols else 0.0
        for j in range(real):
            mass = float(real_slice[j])
            if mass >= float(min_mass) and mass >= dustbin_mass:
                candidates.append((mass, i, j))
    candidates.sort(reverse=True)

    used_rows: set = set()
    used_cols: set = set()
    pairs = []
    for mass, i, j in candidates:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        pairs.append({"row": i, "col": j, "mass": round(mass, 6)})
    return {
        "pairs": pairs,
        "unmatched_rows": [i for i in range(rows) if i not in used_rows],
        "unmatched_cols": [j for j in range(real) if j not in used_cols],
    }


__all__ = [
    "assign_from_plan",
    "log_sinkhorn",
    "sinkhorn_unbalanced",
    "with_dustbin",
    "with_dustbin_square",
]
