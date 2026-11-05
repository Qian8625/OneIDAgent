"""Capacity-constrained projection of the association matrix.

The unconstrained softmax update gives each observation a valid probability row,
but nothing stops ten observations from all claiming the same object.  This
module projects the matrix back onto the feasible set via iterative KL scaling
(Sinkhorn-style, with inequality column constraints):

* every row sums to 1
* all entries are non-negative
* non-candidate entries stay exactly 0
* each real object's column sum is at most its capacity
* the ``__NULL__`` column is unconstrained
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class ProjectionInfeasibleError(RuntimeError):
    """Raised when no assignment can satisfy the capacity/candidate structure."""


def project_association_kl(
    a_tilde: Sequence[Sequence[float]],
    capacities: Sequence[Optional[float]],
    candidate_mask: Sequence[Sequence[bool]],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Project ``a_tilde`` onto the capacity-feasible set by KL scaling.

    Args:
        a_tilde: ``[m][k]`` unconstrained association probabilities.
        capacities: Per-column capacity; ``None`` means unconstrained (used for
            the ``__NULL__`` column).
        candidate_mask: ``[m][k]`` admissibility mask.  ``False`` entries are
            forced to 0.
        max_iter: Maximum scaling sweeps.
        tol: Convergence tolerance on the maximum constraint violation.

    Returns:
        ``(projected, diagnostics)``.  ``diagnostics`` records iterations,
        final violation, and convergence.

    Raises:
        ProjectionInfeasibleError: When a row has no admissible candidate at all,
            or total capacity cannot cover the observations that require a real
            object.  Never silently emits an illegal distribution.
    """
    rows = len(a_tilde)
    cols = len(a_tilde[0]) if rows else 0

    if rows == 0:
        return ([], {"iterations": 0, "converged": True, "max_violation": 0.0})

    for m in range(rows):
        if not any(candidate_mask[m][k] for k in range(cols)):
            raise ProjectionInfeasibleError(
                f"Observation row {m} has no admissible candidate, not even __NULL__"
            )

    unconstrained_cols = [k for k in range(cols) if capacities[k] is None]
    if not unconstrained_cols:
        total_capacity = sum(float(c) for c in capacities if c is not None)
        if total_capacity + 1e-9 < rows:
            raise ProjectionInfeasibleError(
                f"Total capacity {total_capacity} cannot cover {rows} observations "
                "and no unconstrained (NULL) column is available"
            )

    # Start from the masked, row-normalized matrix.
    matrix: List[List[float]] = []
    for m in range(rows):
        row = [
            max(0.0, float(a_tilde[m][k])) if candidate_mask[m][k] else 0.0
            for k in range(cols)
        ]
        total = sum(row)
        if total <= 0.0:
            admissible = [k for k in range(cols) if candidate_mask[m][k]]
            share = 1.0 / len(admissible)
            row = [share if candidate_mask[m][k] else 0.0 for k in range(cols)]
        else:
            row = [v / total for v in row]
        matrix.append(row)

    iterations = 0
    max_violation = 0.0
    converged = False

    for iteration in range(1, int(max_iter) + 1):
        iterations = iteration
        # Column step: shrink over-subscribed real objects.
        max_violation = 0.0
        for k in range(cols):
            capacity = capacities[k]
            if capacity is None:
                continue
            column_sum = sum(matrix[m][k] for m in range(rows))
            violation = column_sum - float(capacity)
            if violation > tol:
                max_violation = max(max_violation, violation)
                scale = float(capacity) / column_sum
                for m in range(rows):
                    matrix[m][k] *= scale

        # Row step: restore each row to the simplex.
        for m in range(rows):
            total = sum(matrix[m][k] for k in range(cols))
            if total <= 0.0:
                admissible = [k for k in range(cols) if candidate_mask[m][k]]
                share = 1.0 / len(admissible)
                for k in range(cols):
                    matrix[m][k] = share if candidate_mask[m][k] else 0.0
            else:
                for k in range(cols):
                    matrix[m][k] /= total

        if max_violation <= tol:
            converged = True
            break

    # Final feasibility report after the last row renormalization.
    final_violation = 0.0
    for k in range(cols):
        capacity = capacities[k]
        if capacity is None:
            continue
        column_sum = sum(matrix[m][k] for m in range(rows))
        final_violation = max(final_violation, column_sum - float(capacity))

    diagnostics = {
        "iterations": iterations,
        "converged": bool(converged and final_violation <= max(tol, 1e-6) * 10),
        "max_violation": max(0.0, final_violation),
        "tol": float(tol),
    }
    return (matrix, diagnostics)


def build_capacities(
    object_ids: Sequence[str],
    null_object_id: str,
    capacity_map: Mapping[str, int],
    default_capacity: Optional[float],
) -> List[Optional[float]]:
    """Per-column capacity vector; ``None`` for the NULL column.

    ``default_capacity=None`` means "no capacity constraint": real objects
    without an explicit entry are left unconstrained as well.  This is the
    representation an ablation uses to switch the capacity projection off
    completely, rather than silently falling back to the shipped default.
    """
    capacities: List[Optional[float]] = []
    for object_id in object_ids:
        if object_id == null_object_id:
            capacities.append(None)
        elif object_id in capacity_map:
            capacities.append(float(capacity_map[object_id]))
        elif default_capacity is None:
            capacities.append(None)
        else:
            capacities.append(float(default_capacity))
    return capacities


def verify_projection(
    matrix: Sequence[Sequence[float]],
    capacities: Sequence[Optional[float]],
    candidate_mask: Sequence[Sequence[bool]],
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Check every invariant the projection is supposed to guarantee."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    problems: List[str] = []

    for m in range(rows):
        total = sum(matrix[m])
        if abs(total - 1.0) > max(tol, 1e-9) * 100:
            problems.append(f"row {m} sums to {total}")
        for k in range(cols):
            if matrix[m][k] < -1e-12:
                problems.append(f"entry ({m},{k}) is negative: {matrix[m][k]}")
            if not candidate_mask[m][k] and matrix[m][k] > 1e-12:
                problems.append(f"non-candidate entry ({m},{k}) has mass {matrix[m][k]}")

    for k in range(cols):
        capacity = capacities[k]
        if capacity is None:
            continue
        column_sum = sum(matrix[m][k] for m in range(rows))
        if column_sum > float(capacity) + max(tol, 1e-6) * 10:
            problems.append(f"column {k} sum {column_sum} exceeds capacity {capacity}")

    return {"feasible": not problems, "problems": problems}


def project_association_block_kl(
    a_tilde: Sequence[Sequence[float]],
    capacities: Sequence[Optional[float]],
    candidate_mask: Sequence[Sequence[bool]],
    block_ids: Sequence[str],
    block_capacity: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Project rows with a capacity of one per object inside each block.

    Rows sharing a block compete for each real object.  The NULL column remains
    unconstrained.  This explicit API prevents the legacy global object-column
    projection from being silently presented as block capacity.
    """
    if len(block_ids) != len(a_tilde):
        raise ValueError("block_ids must contain one entry per association row")
    if block_capacity <= 0.0:
        raise ValueError("block_capacity must be positive")
    rows = len(a_tilde)
    cols = len(a_tilde[0]) if rows else 0
    blocks: Dict[str, List[int]] = {}
    for row_index, block_id in enumerate(block_ids):
        blocks.setdefault(str(block_id), []).append(row_index)
    matrix: List[List[float]] = []
    for row_index, row in enumerate(a_tilde):
        masked = [
            max(0.0, float(value)) if candidate_mask[row_index][column] else 0.0
            for column, value in enumerate(row)
        ]
        total = sum(masked)
        if total <= 0.0:
            admissible = [column for column in range(cols) if candidate_mask[row_index][column]]
            if not admissible:
                raise ProjectionInfeasibleError(f"row {row_index} has no admissible candidate")
            masked = [1.0 / len(admissible) if column in admissible else 0.0 for column in range(cols)]
        else:
            masked = [value / total for value in masked]
        matrix.append(masked)

    converged = False
    max_violation = 0.0
    for _iteration in range(1, int(max_iter) + 1):
        max_violation = 0.0
        for row_indices in blocks.values():
            for column in range(cols):
                if capacities[column] is None:
                    continue
                column_capacity = min(block_capacity, float(capacities[column]))
                column_sum = sum(matrix[row_index][column] for row_index in row_indices)
                if column_sum <= column_capacity + tol:
                    continue
                violation = column_sum - column_capacity
                max_violation = max(max_violation, violation)
                scale = column_capacity / column_sum
                for row_index in row_indices:
                    matrix[row_index][column] *= scale
        for row_index in range(rows):
            total = sum(matrix[row_index])
            if total <= 0.0:
                admissible = [column for column in range(cols) if candidate_mask[row_index][column]]
                matrix[row_index] = [
                    1.0 / len(admissible) if column in admissible else 0.0
                    for column in range(cols)
                ]
            else:
                matrix[row_index] = [value / total for value in matrix[row_index]]
        if max_violation <= tol:
            converged = True
            break

    final_violation = 0.0
    block_column_sums: Dict[str, Dict[str, float]] = {}
    for block_id, row_indices in blocks.items():
        block_column_sums[block_id] = {}
        for column in range(cols):
            if capacities[column] is None:
                continue
            value = sum(matrix[row_index][column] for row_index in row_indices)
            block_column_sums[block_id][str(column)] = value
            final_violation = max(final_violation, value - block_capacity)
    diagnostics = {
        "mode": "block_object_column",
        "block_capacity": float(block_capacity),
        "block_count": len(blocks),
        "block_ids": dict((block_id, list(indices)) for block_id, indices in blocks.items()),
        "block_column_sums": block_column_sums,
        "iterations": _iteration if rows else 0,
        "converged": bool(converged and final_violation <= max(tol, 1e-6) * 10),
        "max_violation": max(0.0, final_violation),
        "tol": float(tol),
    }
    return matrix, diagnostics


__all__ = [
    "ProjectionInfeasibleError",
    "project_association_kl",
    "build_capacities",
    "verify_projection",
    "project_association_block_kl",
]
