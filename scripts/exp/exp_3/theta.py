"""Fitting half of the contextual reliability model ``r_i = sigmoid(b_{s,a} + beta_a^T x_i)``.

The repository only ships the *inference* half (``contextual_reliability.py``);
the fitting code lived in ``oneid_agent.tf_eval.shared_identity`` and is not part
of this checkout.  This module re-implements it under the same artifact contract
so a fold's theta can be loaded by the unmodified engine.

Estimator: logistic regression with an L2 penalty on the feature coefficients
only (intercepts are never penalised), one model per attribute with a
modality-specific intercept, which is exactly the functional form the artifact
schema declares::

    intercepts      "optical|category" -> b
    coefficients    "category" -> {feature: beta}

Only (modality, attribute) cells with at least ``min_support`` claims get an
intercept; a cell below the floor stays unsupported so the engine falls back to
the shipped reliability table instead of trusting a coefficient fitted on a
handful of rows.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from DS_VI.models.shared_identity.contextual_reliability import (
    CONTINUOUS_FEATURES,
    FEATURE_SET_V1,
    extract_features,
)

from .data import ClaimRow, REPO_ROOT

KIND = "shared_identity_contextual_theta"

#: Shipped artifact, used only as the clamp reference and as the comparison arm.
SHIPPED_THETA = REPO_ROOT / "DS_VI/configs/shared_identity/artifacts/contextual_theta_v1.json"

#: Fallback clamp bounds, matching ``reliability.min`` / ``reliability.max``.
DEFAULT_CLAMP = (0.35, 0.97)

#: Feature -> missingness indicator (mirrors the inference module).
MISSING_INDICATORS = {"registration_offset_norm": "registration_missing"}


def shipped_clamp() -> Tuple[float, float]:
    try:
        payload = json.loads(SHIPPED_THETA.read_text(encoding="utf-8"))
        clamp = payload.get("clamp") or {}
        low = float(clamp.get("min", DEFAULT_CLAMP[0]))
        high = float(clamp.get("max", DEFAULT_CLAMP[1]))
        if 0.0 < low < high < 1.0:
            return low, high
    except Exception:  # pragma: no cover - artifact is optional for the fit
        pass
    return DEFAULT_CLAMP


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _raw_features(row: ClaimRow) -> Dict[str, float]:
    """Feature vector exactly as the engine will compute it at inference time."""
    return extract_features({"context": row.context()}, None)  # type: ignore[arg-type]


def _scaling(rows: Sequence[ClaimRow]) -> Dict[str, Dict[str, float]]:
    scaling: Dict[str, Dict[str, float]] = {}
    for name in CONTINUOUS_FEATURES:
        values = [float(_raw_features(row).get(name, 0.0)) for row in rows]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(var)
        scaling[name] = {"mean": mean, "std": std if std > 1e-9 else 1.0}
    return scaling


def _design(
    rows: Sequence[ClaimRow],
    modalities: Sequence[str],
    features: Sequence[str],
    scaling: Mapping[str, Mapping[str, float]],
) -> Tuple[List[List[float]], List[float]]:
    """Rows of ``[modality one-hots..., scaled features...]`` and labels."""
    matrix: List[List[float]] = []
    labels: List[float] = []
    for row in rows:
        raw = _raw_features(row)
        vector: List[float] = [1.0 if row.modality == modality else 0.0 for modality in modalities]
        for name in features:
            value = float(raw.get(name, 0.0))
            indicator = MISSING_INDICATORS.get(name)
            if indicator and float(raw.get(indicator, 0.0)) >= 0.5:
                value = 0.0
            elif name in scaling:
                stats = scaling[name]
                value = (value - float(stats.get("mean", 0.0))) / (float(stats.get("std", 1.0)) or 1.0)
            vector.append(value)
        matrix.append(vector)
        labels.append(float(row.correct))
    return matrix, labels


def _solve(matrix: List[List[float]], labels: List[float],
           penalty: Sequence[float], max_iter: int = 100, tol: float = 1e-10) -> List[float]:
    """Newton / IRLS for logistic regression with diagonal ridge ``penalty``."""
    n_features = len(matrix[0])
    beta = [0.0] * n_features
    for _ in range(max_iter):
        gradient = [-float(penalty[j]) * beta[j] for j in range(n_features)]
        hessian = [[0.0] * n_features for _ in range(n_features)]
        for vector, label in zip(matrix, labels):
            eta = sum(b * x for b, x in zip(beta, vector))
            prob = _sigmoid(eta)
            weight = max(prob * (1.0 - prob), 1e-9)
            residual = label - prob
            for i in range(n_features):
                xi = vector[i]
                if xi == 0.0:
                    continue
                gradient[i] += residual * xi
                for j in range(n_features):
                    xj = vector[j]
                    if xj != 0.0:
                        hessian[i][j] += weight * xi * xj
        for i in range(n_features):
            hessian[i][i] += float(penalty[i]) + 1e-9
        step = _newton_step(hessian, gradient)
        if step is None:
            break
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < tol:
            break
    return beta


def _newton_step(hessian: List[List[float]], gradient: List[float]) -> Optional[List[float]]:
    """Solve ``H d = g`` by Gauss-Jordan elimination with partial pivoting."""
    n = len(gradient)
    augmented = [row[:] + [gradient[i]] for i, row in enumerate(hessian)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [value - factor * base
                              for value, base in zip(augmented[row], augmented[column])]
    return [augmented[i][n] for i in range(n)]


def fit_theta(
    rows: Sequence[ClaimRow],
    attributes: Sequence[str],
    *,
    min_support: int = 30,
    l2: float = 1.0,
    features: Sequence[str] = FEATURE_SET_V1,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit one artifact over ``rows`` (already restricted to the training folds)."""
    by_attribute: Dict[str, List[ClaimRow]] = {}
    for row in rows:
        if row.attribute in attributes:
            by_attribute.setdefault(row.attribute, []).append(row)

    scaling = _scaling(rows)
    clamp_min, clamp_max = shipped_clamp()
    intercepts: Dict[str, float] = {}
    coefficients: Dict[str, Dict[str, float]] = {}
    fit_cells: Dict[str, Any] = {}

    for attribute in attributes:
        attribute_rows = by_attribute.get(attribute) or []
        if not attribute_rows:
            continue
        per_modality: Dict[str, List[ClaimRow]] = {}
        for row in attribute_rows:
            per_modality.setdefault(row.modality, []).append(row)
        kept = [modality for modality, subset in per_modality.items() if len(subset) >= min_support]
        if not kept:
            fit_cells[attribute] = {"support": len(attribute_rows), "fitted": False,
                                    "reason": f"no modality reaches min_support={min_support}"}
            continue
        subset = [row for row in attribute_rows if row.modality in kept]
        kept = sorted(kept)
        matrix, labels = _design(subset, kept, features, scaling)
        penalty = [0.0] * len(kept) + [float(l2)] * len(features)
        beta = _solve(matrix, labels, penalty)
        for index, modality in enumerate(kept):
            intercepts[f"{modality}|{attribute}"] = round(beta[index], 9)
        coefficients[attribute] = {
            name: round(beta[len(kept) + index], 9) for index, name in enumerate(features)
        }
        fit_cells[attribute] = {
            "support": len(subset),
            "modalities": kept,
            "fitted": True,
            "accuracy": round(sum(labels) / len(labels), 6) if labels else None,
        }

    if not intercepts:
        raise ValueError(
            "fit_theta produced no usable (modality, attribute) cell: every cell is "
            f"below min_support={min_support}. Training rows={len(rows)}, "
            f"pairs={len({row.pair_id for row in rows})}. Use more training pairs "
            "or lower --min-support; refusing to emit an artifact the engine would reject."
        )

    artifact: Dict[str, Any] = {
        "schema_version": "1.0",
        "kind": KIND,
        "features": list(features),
        "scaling": scaling,
        "missing_indicators": dict(MISSING_INDICATORS),
        "intercepts": intercepts,
        "coefficients": coefficients,
        "clamp": {"min": clamp_min, "max": clamp_max},
        "fit": {
            "feature_set": "v1",
            "regularization_lambda": float(l2),
            "min_support": int(min_support),
            "row_count": len(rows),
            "attributes": fit_cells,
            "protocol": "exp_3: 5-fold cross-fitting (out-of-fold theta)",
            "dataset": "rebuilt from tool/truth property workbooks (see exp_3/data.py)",
            **(dict(metadata) if metadata else {}),
        },
    }
    return artifact


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def artifact_log_loss(artifact: Mapping[str, Any], rows: Sequence[ClaimRow]) -> Dict[str, Any]:
    """Out-of-fold predictive quality of a fitted artifact (claim level).

    ``auc`` is the probability that a correct claim scores above an incorrect
    one; ``brier`` and ``log_loss`` are proper scoring rules.  These are the
    numbers that prove the fold-theta generalises rather than memorising.
    """
    from DS_VI.models.shared_identity.contextual_reliability import ThetaArtifact

    theta = ThetaArtifact(
        features=list(artifact["features"]),
        scaling={k: dict(v) for k, v in artifact["scaling"].items()},
        intercepts={k: float(v) for k, v in artifact["intercepts"].items()},
        coefficients={k: {n: float(b) for n, b in v.items()}
                      for k, v in artifact["coefficients"].items()},
        clamp_min=float(artifact["clamp"]["min"]),
        clamp_max=float(artifact["clamp"]["max"]),
        metadata=dict(artifact.get("fit") or {}),
        missing_indicators={k: str(v) for k, v in (artifact.get("missing_indicators") or {}).items()},
    )
    correct_scores: List[float] = []
    wrong_scores: List[float] = []
    losses: List[float] = []
    briers: List[float] = []
    scored = 0
    for row in rows:
        raw = _raw_features(row)
        probability = theta.predict(row.modality, row.attribute, raw)
        if probability is None:
            continue
        scored += 1
        probability = min(max(probability, 1e-6), 1 - 1e-6)
        label = float(row.correct)
        losses.append(-(label * math.log(probability) + (1 - label) * math.log(1 - probability)))
        briers.append((probability - label) ** 2)
        (correct_scores if row.correct else wrong_scores).append(probability)
    auc = None
    if correct_scores and wrong_scores:
        wins = 0.0
        for value in correct_scores:
            for other in wrong_scores:
                wins += 1.0 if value > other else 0.5 if value == other else 0.0
        auc = wins / (len(correct_scores) * len(wrong_scores))
    return {
        "scored": scored,
        "unsupported": len(rows) - scored,
        "log_loss": round(sum(losses) / len(losses), 6) if losses else None,
        "brier": round(sum(briers) / len(briers), 6) if briers else None,
        "auc": round(auc, 6) if auc is not None else None,
    }


__all__ = [
    "DEFAULT_CLAMP",
    "KIND",
    "SHIPPED_THETA",
    "artifact_log_loss",
    "fit_theta",
    "shipped_clamp",
    "write_artifact",
]
