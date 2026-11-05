"""Contextual claim reliability ``r_i = sigmoid(b_{s,a} + beta_a^T x_i)``.

Eq. (3) / Table 14 of the paper replace the fixed per-(modality, attribute)
reliability table with a logistic channel over measured context.  This module
owns the *inference-time* half of that contract:

* the feature vector ``x_i`` is extracted from an observation's context,
  independently of any ground truth;
* a fitted ``theta`` artifact supplies per-(source, attribute) intercepts and
  per-attribute coefficients;
* :class:`ContextualReliabilityModel` evaluates the sigmoid and clamps it into a
  bounded interval so a single claim can never become a logical certainty.

The fitting half lives in
``oneid_agent.tf_eval.shared_identity.fit_contextual_reliability``.

Missing context never crashes the model: Table 14 specifies that missing fields
receive indicators, so every feature block contributes an explicit
``*_missing`` column and the absent covariate enters as zero after scaling.
A cell whose ``(modality, attribute)`` pair was never fitted is reported as
unsupported so the caller can fall back to the shipped table.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from DS_VI.models.shared_identity.schema import (
    Claim,
    Modality,
    PrimitiveObservation,
    SchemaValidationError,
)

#: Minimal staged feature set: visibility degradation, registration residual,
#: and missingness indicators for both blocks.  Stage 2 adds evidence quality,
#: target scale and clutter proxies as separate, ablatable features.
FEATURE_SET_V1 = (
    "visibility_partial",
    "visibility_poor",
    "visibility_missing",
    "registration_offset_norm",
    "registration_missing",
)

#: Staged extension: add measured annotation quality and local detection
#: density (a clutter proxy) as separate, ablatable covariates.  Kept as its
#: own named set so a v2 artifact is never confused with a v1 one.
FEATURE_SET_V2 = FEATURE_SET_V1 + (
    "evidence_quality_medium",
    "evidence_quality_low",
    "evidence_quality_missing",
    "detection_count",
    "detection_count_missing",
)

FEATURE_SETS = {"v1": FEATURE_SET_V1, "v2": FEATURE_SET_V2}

#: Features fitted on a real-valued scale.  Binary indicators are never
#: standardized, so their coefficients stay directly interpretable as log-odds
#: shifts relative to the clear/complete reference level.
CONTINUOUS_FEATURES = ("registration_offset_norm", "detection_count")

#: Continuous feature -> the missingness indicator that gates it.  When the
#: indicator is active the covariate is mean-imputed (standardized value 0)
#: instead of treating its raw placeholder zero as a measurement.
MISSING_INDICATORS = {
    "registration_offset_norm": "registration_missing",
    "detection_count": "detection_count_missing",
}

_EVIDENCE_QUALITY_LEVELS = {
    "medium": "evidence_quality_medium",
    "low": "evidence_quality_low",
}

_VISIBILITY_LEVELS = {
    "partial": "visibility_partial",
    "poor": "visibility_poor",
}


def extract_features(observation: Mapping[str, Any], claim: Claim) -> Dict[str, float]:
    """Extract the raw (unscaled) context feature vector for one claim.

    ``observation`` is a mapping with ``context``; the claim only selects the
    attribute and is otherwise never read.  Absent fields become explicit
    missingness columns rather than an error.
    """
    del claim  # attribute-specific applicability is handled by the coefficients
    context = dict((observation or {}).get("context") or {})
    # Seed the union of staged sets so every artifact finds the keys it knows;
    # prediction only reads the features its own artifact declares.
    features = {name: 0.0 for name in FEATURE_SET_V2}

    visibility = str(context.get("visibility") or "").strip().lower()
    if visibility == "clear":
        pass
    elif visibility in _VISIBILITY_LEVELS:
        features[_VISIBILITY_LEVELS[visibility]] = 1.0
    else:
        features["visibility_missing"] = 1.0

    quality = str(context.get("evidence_quality") or "").strip().lower()
    if quality == "high":
        pass
    elif quality in _EVIDENCE_QUALITY_LEVELS:
        features[_EVIDENCE_QUALITY_LEVELS[quality]] = 1.0
    else:
        features["evidence_quality_missing"] = 1.0

    count = context.get("detection_count")
    if count is None:
        features["detection_count_missing"] = 1.0
    else:
        try:
            features["detection_count"] = float(count)
        except (TypeError, ValueError):
            features["detection_count_missing"] = 1.0

    offset = context.get("registration_offset_norm")
    counterpart = context.get("registration_counterpart_present")
    if offset is None or counterpart is False:
        features["registration_missing"] = 1.0
    else:
        try:
            features["registration_offset_norm"] = float(offset)
        except (TypeError, ValueError):
            features["registration_missing"] = 1.0
    return features


@dataclass(frozen=True)
class ThetaArtifact:
    """A fitted contextual reliability parameter vector."""

    features: Sequence[str]
    scaling: Mapping[str, Mapping[str, float]]
    intercepts: Mapping[str, float]
    coefficients: Mapping[str, Mapping[str, float]]
    clamp_min: float
    clamp_max: float
    metadata: Mapping[str, Any]
    #: Continuous feature -> missingness indicator.  When the indicator is
    #: active the covariate is mean-imputed (standardized value 0) instead of
    #: treating its raw placeholder zero as a measurement.
    missing_indicators: Mapping[str, str] = field(default_factory=dict)

    @property
    def modality_attributes(self) -> Sequence[str]:
        return tuple(sorted(self.intercepts))

    def supports(self, modality: str, attribute: str) -> bool:
        return f"{modality}|{attribute}" in self.intercepts

    def predict(
        self,
        modality: str,
        attribute: str,
        raw_features: Mapping[str, float],
    ) -> Optional[float]:
        """Return ``sigmoid(b + beta . x)`` or ``None`` for unsupported cells."""
        intercept = self.intercepts.get(f"{modality}|{attribute}")
        if intercept is None:
            return None
        beta = self.coefficients.get(attribute) or {}
        total = float(intercept)
        for name in self.features:
            if name not in beta:
                continue
            value = float(raw_features.get(name, 0.0))
            indicator = self.missing_indicators.get(name)
            if indicator and float(raw_features.get(indicator, 0.0)) >= 0.5:
                value = 0.0
            elif name in self.scaling:
                value = self._scale(name, value)
            total += float(beta[name]) * value
        return self._clamp(_sigmoid(total))

    def _scale(self, name: str, value: float) -> float:
        stats = self.scaling.get(name) or {}
        mean = float(stats.get("mean", 0.0))
        std = float(stats.get("std", 1.0)) or 1.0
        return (value - mean) / std

    def _clamp(self, value: float) -> float:
        return max(self.clamp_min, min(self.clamp_max, value))

    def describe(self) -> Dict[str, Any]:
        return {
            "features": list(self.features),
            "missing_indicators": dict(self.missing_indicators),
            "intercepts": dict(self.intercepts),
            "coefficients": {k: dict(v) for k, v in self.coefficients.items()},
            "clamp": {"min": self.clamp_min, "max": self.clamp_max},
            "fit": dict(self.metadata),
        }


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def load_theta_artifact(path: str | Path) -> ThetaArtifact:
    """Load and validate a fitted artifact.

    Unreadable, non-numeric, or kind-mismatched files raise instead of silently
    falling back to the shipped table: a flag that changes nothing would make
    the contextual ablation indistinguishable from the baseline.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise SchemaValidationError(
            f"contextual theta artifact not found: {artifact_path}. "
            "Run fit_contextual_reliability to produce one."
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"{artifact_path}: not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SchemaValidationError(f"{artifact_path}: artifact must be an object")
    kind = payload.get("kind")
    if kind != "shared_identity_contextual_theta":
        raise SchemaValidationError(
            f"{artifact_path}: unexpected kind {kind!r}; expected "
            "'shared_identity_contextual_theta'"
        )
    features = payload.get("features")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or not features:
        raise SchemaValidationError(f"{artifact_path}: 'features' must be a non-empty list")
    intercepts = payload.get("intercepts")
    if not isinstance(intercepts, Mapping) or not intercepts:
        raise SchemaValidationError(f"{artifact_path}: 'intercepts' must be a non-empty mapping")
    coefficients = payload.get("coefficients") or {}
    if not isinstance(coefficients, Mapping):
        raise SchemaValidationError(f"{artifact_path}: 'coefficients' must be a mapping")
    scaling = payload.get("scaling") or {}
    if not isinstance(scaling, Mapping):
        raise SchemaValidationError(f"{artifact_path}: 'scaling' must be a mapping")
    missing_indicators = payload.get("missing_indicators") or {}
    if not isinstance(missing_indicators, Mapping):
        raise SchemaValidationError(
            f"{artifact_path}: 'missing_indicators' must be a mapping"
        )
    clamp = payload.get("clamp") or {}
    clamp_min = float(clamp.get("min", 1e-3))
    clamp_max = float(clamp.get("max", 1.0 - 1e-3))
    if not 0.0 < clamp_min < clamp_max < 1.0:
        raise SchemaValidationError(
            f"{artifact_path}: clamp must satisfy 0 < min < max < 1, got "
            f"[{clamp_min}, {clamp_max}]"
        )
    return ThetaArtifact(
        features=[str(name) for name in features],
        scaling={
            str(name): {str(k): float(v) for k, v in dict(stats).items()}
            for name, stats in scaling.items()
        },
        intercepts={str(key): float(value) for key, value in intercepts.items()},
        coefficients={
            str(attribute): {str(name): float(beta) for name, beta in dict(cell).items()}
            for attribute, cell in coefficients.items()
        },
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        metadata=dict(payload.get("fit") or {}),
        missing_indicators={
            str(name): str(indicator) for name, indicator in missing_indicators.items()
        },
    )


def modality_key(modality: Any) -> str:
    if isinstance(modality, Modality):
        return modality.value
    return str(modality or "").lower()


def claim_modality(
    claim: Claim,
    observation: Optional[PrimitiveObservation] = None,
) -> str:
    if observation is not None:
        return modality_key(observation.modality)
    return modality_key(Modality.coerce(claim.source_channel))


class ContextualReliabilityModel:
    """Evaluate a fitted :class:`ThetaArtifact` per claim."""

    def __init__(self, artifact: ThetaArtifact):
        self.artifact = artifact

    def for_claim(
        self,
        claim: Claim,
        observation: Optional[PrimitiveObservation] = None,
    ) -> Optional[float]:
        """Contextual reliability, or ``None`` when the cell is unsupported."""
        if observation is None:
            return None
        modality = claim_modality(claim, observation)
        features = extract_features({"context": observation.context}, claim)
        return self.artifact.predict(modality, str(claim.attribute), features)

    def describe(self) -> Dict[str, Any]:
        return self.artifact.describe()


__all__ = [
    "CONTINUOUS_FEATURES",
    "FEATURE_SETS",
    "FEATURE_SET_V1",
    "FEATURE_SET_V2",
    "MISSING_INDICATORS",
    "ContextualReliabilityModel",
    "ThetaArtifact",
    "claim_modality",
    "extract_features",
    "load_theta_artifact",
    "modality_key",
]
