"""Convert candidate-edge diagnostic scores into a normalized physical prior.

``p[m, k]`` is the probability, before any claim evidence is considered, that
observation ``m`` belongs to object ``k``.  Each row sums to exactly 1 and
reserves NULL mass, following the paper's physical-prior form (Eq. 2):

.. code-block:: text

    p[m, k]  = exp(d^T f[m, k]) / (exp(d_null) + sum_h exp(d^T f[m, h]))
    p[m, 0]  = 1 - sum_k p[m, k] > 0

``d^T f`` is the diagnostic score the candidate graph already computes (a
weighted, unit-clamped agreement score over IoU / center / area / confidence),
multiplied by ``physical.prior_scale`` so one configuration knob carries the
magnitude of ``d``.  ``d_null`` is ``physical.null_logit``.  Both are frozen
constants: the paper fits them on independently adjudicated correspondences,
which this repository does not ship, so the shipped default is calibrated to
keep ``null_prior`` mass on a perfect single candidate.

Edge objects are updated in place with their ``prior_probability``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.schema import NULL_OBJECT_ID, CandidateEdge

#: Magnitude of ``d`` when the configuration does not set ``prior_scale``.
DEFAULT_PRIOR_SCALE = 6.0


def default_null_logit(null_prior: float, scale: float = DEFAULT_PRIOR_SCALE) -> float:
    """``d_null`` that leaves ``null_prior`` mass on a perfect single candidate.

    With one candidate at ``d^T f = scale`` the NULL share is
    ``1 / (1 + exp(scale - d_null))``; solving for ``null_prior`` gives
    ``d_null = scale + log(null_prior / (1 - null_prior))``.
    """
    p = min(max(float(null_prior), 1e-6), 1.0 - 1e-6)
    return float(scale) + math.log(p / (1.0 - p))


def compute_physical_prior(
    edges: Sequence[CandidateEdge],
    config: Optional[Mapping[str, Any]] = None,
    objects: Optional[Sequence[Any]] = None,
) -> Dict[str, Dict[str, float]]:
    """Build the row-normalized physical prior from candidate edges.

    Rows are a softmax over ``exp(d^T f)`` (real candidates) plus
    ``exp(d_null)`` (NULL), so an absolute weak candidate keeps substantial NULL
    mass instead of winning its row by default.  Inadmissible edges carry zero
    mass and never enter the normalisation.  A candidate that carries
    ``context['prior_floor']`` (the independent-data channel of the cascade)
    raises its row's real budget to that floor after normalisation, so the
    evidence layer — not the physical prior — decides those weak candidates.
    """
    cfg = dict(config or load_config())
    physical = dict(cfg.get("physical") or {})
    weights = dict(physical.get("weights") or {})
    null_prior = min(max(float(physical.get("null_prior", 0.15)), 0.0), 1.0)
    scale = float(physical.get("prior_scale", DEFAULT_PRIOR_SCALE))
    raw_null_logit = physical.get("null_logit")
    null_logit = (
        float(raw_null_logit)
        if raw_null_logit is not None
        else default_null_logit(null_prior, scale)
    )

    floors: Dict[str, float] = {}
    for obj in objects or []:
        floor = (getattr(obj, "context", None) or {}).get("prior_floor")
        if floor is not None:
            try:
                floors[str(obj.object_id)] = min(max(float(floor), 0.0), 1.0)
            except (TypeError, ValueError):
                continue

    grouped: Dict[str, List[CandidateEdge]] = {}
    for edge in edges:
        grouped.setdefault(edge.observation_id, []).append(edge)

    prior: Dict[str, Dict[str, float]] = {}
    for observation_id, row_edges in grouped.items():
        real_edges = [e for e in row_edges if e.admissible and not e.is_null]
        null_edges = [e for e in row_edges if e.is_null]

        if not real_edges:
            for edge in null_edges:
                edge.prior_probability = 1.0
            prior[observation_id] = {NULL_OBJECT_ID: 1.0}
            continue

        # d^T f: the graph's weighted agreement score, scaled by |d|.
        logits = [
            scale * min(max(float(e.features.get("diagnostic_score", 0.0)), 0.0), 1.0)
            for e in real_edges
        ]
        peak = max(max(logits), null_logit)
        real_weights = [math.exp(value - peak) for value in logits]
        null_weight = math.exp(null_logit - peak) if null_edges else 0.0
        denominator = sum(real_weights) + null_weight
        if denominator <= 0.0:  # pragma: no cover - exp is strictly positive
            denominator = 1.0

        row: Dict[str, float] = {}
        for edge, weight in zip(real_edges, real_weights, strict=False):
            row[edge.object_id] = weight / denominator
        row_null = (null_weight / denominator) if null_edges else 0.0

        # Independent-data channel: lift the real budget to the declared floor.
        row_floor = max((floors.get(str(e.object_id), 0.0) for e in real_edges), default=0.0)
        real_total = sum(row.values())
        if row_floor > 0.0 and real_total < row_floor:
            lift = (row_floor / real_total) if real_total > 0.0 else 0.0
            if lift > 0.0:
                for object_id in row:
                    row[object_id] = min(1.0, row[object_id] * lift)
            else:  # every real weight underflowed; split the floor evenly
                share = row_floor / len(real_edges)
                for edge in real_edges:
                    row[edge.object_id] = share
            row_null = max(0.0, 1.0 - sum(row.values()))

        if null_edges:
            row[NULL_OBJECT_ID] = max(0.0, row_null)

        # Re-normalise after the floor adjustment so the row is an exact simplex.
        total = sum(row.values())
        if total > 0.0 and abs(total - 1.0) > 1e-12:
            row = {key: value / total for key, value in row.items()}
        if not null_edges:
            row[NULL_OBJECT_ID] = 0.0

        for edge in row_edges:
            edge.prior_probability = (
                row.get(edge.object_id, 0.0) if edge.admissible else 0.0
            )
        if not null_edges:
            row.pop(NULL_OBJECT_ID, None)

        prior[observation_id] = row

    return prior


def prior_matrix(
    prior: Mapping[str, Mapping[str, float]],
    observation_ids: Sequence[str],
    object_ids: Sequence[str],
) -> List[List[float]]:
    """Dense ``[m][k]`` view of the prior in the given index order."""
    return [
        [float((prior.get(obs_id) or {}).get(obj_id, 0.0)) for obj_id in object_ids]
        for obs_id in observation_ids
    ]


__all__ = [
    "DEFAULT_PRIOR_SCALE",
    "compute_physical_prior",
    "default_null_logit",
    "prior_matrix",
]
