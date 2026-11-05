"""Fixed per-(modality, attribute) claim reliability.

``r_i`` is the probability that claim ``i`` reports the true value.  It is a
property of the *sensor and attribute*, not of the individual observation.

Explicitly forbidden inputs: detector confidence, match score, association
probability.  Reusing any of those would let the model reinforce its own beliefs.

Contextual calibration (``reliability.contextual``) optionally replaces the
shipped table with one fitted from labelled claim-vs-truth agreement, as emitted
by ``tf_eval.shared_identity.calibrate_reliability --emit-config``.  Enabling it
without a readable table is a hard error: a flag that silently falls back to the
default would make the calibrated ablation indistinguishable from the baseline.

A second, paper-faithful path loads a fitted ``theta`` artifact and computes
``r_i = sigmoid(b_{s_i,a_i} + beta_{a_i}^T x_i)`` from observation context.  The
theta path takes precedence and, by default, is never shrunk or re-clamped by the
fixed-table guards: those guards exist for a table whose cells claim to be
probabilities, whereas the sigmoid is bounded by construction.  Setting
``reliability.shrinkage_applies_to_theta`` to true opts the theta cells into the
same shrinkage/min/max guards; it is off by default so shipped attribute-arm
numbers stay reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:  # pragma: no cover - yaml is a hard dependency of the repo
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from DS_VI.models.shared_identity.config import load_config, project_path
from DS_VI.models.shared_identity.contextual_reliability import (
    ContextualReliabilityModel,
    load_theta_artifact,
)
from DS_VI.models.shared_identity.schema import (
    Claim,
    Modality,
    PrimitiveObservation,
    SchemaValidationError,
)


def load_contextual_table(path: str | Path) -> Dict[str, Dict[str, float]]:
    """Load and validate a calibration table for ``reliability.contextual``.

    Accepts either a full config document (``reliability.by_modality_attribute``)
    or a bare ``by_modality_attribute`` mapping, in YAML or JSON.  Missing,
    unreadable, empty or non-numeric tables raise instead of silently falling
    back to the shipped defaults.
    """
    table_path = Path(path)
    if not table_path.exists():
        raise SchemaValidationError(
            f"contextual reliability table not found: {table_path}. "
            "Run calibrate_reliability --emit-config to produce one."
        )
    text = table_path.read_text(encoding="utf-8")
    if table_path.suffix.lower() in (".yaml", ".yml") and yaml is not None:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, Mapping):
        raise SchemaValidationError(f"{table_path}: calibration table must be a mapping")
    section = loaded.get("reliability") if "reliability" in loaded else loaded
    if not isinstance(section, Mapping):
        raise SchemaValidationError(f"{table_path}: 'reliability' section must be a mapping")
    raw_table = section.get("by_modality_attribute")
    if not isinstance(raw_table, Mapping) or not raw_table:
        raise SchemaValidationError(
            f"{table_path}: no 'by_modality_attribute' cells to apply"
        )
    table: Dict[str, Dict[str, float]] = {}
    for modality, attributes in raw_table.items():
        if not isinstance(attributes, Mapping):
            raise SchemaValidationError(
                f"{table_path}: modality '{modality}' must map to attributes"
            )
        cells: Dict[str, float] = {}
        for attribute, value in attributes.items():
            try:
                cells[str(attribute)] = float(value)
            except (TypeError, ValueError) as exc:
                raise SchemaValidationError(
                    f"{table_path}: cell {modality}|{attribute} is not numeric: {value!r}"
                ) from exc
        table[str(modality)] = cells
    return table


class ReliabilityModel:
    """Look up fixed reliabilities from configuration."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        cfg = dict(config or load_config())
        self._config = cfg
        section = dict(cfg.get("reliability") or {})
        self.default = float(section.get("default", 0.70))
        self.min = float(section.get("min", 0.35))
        self.max = float(section.get("max", 0.97))
        shrinkage = float(section.get("shrinkage", 1.0))
        if not 0.0 <= shrinkage <= 1.0:
            raise SchemaValidationError(
                f"reliability.shrinkage must be in [0, 1], got {shrinkage}"
            )
        self.shrinkage = shrinkage
        outlier_mixture = float(section.get("outlier_mixture", 0.0))
        if not 0.0 <= outlier_mixture < 1.0:
            raise SchemaValidationError(
                f"reliability.outlier_mixture must be in [0, 1), got {outlier_mixture}"
            )
        self.outlier_mixture = outlier_mixture
        applies_to_theta = section.get("shrinkage_applies_to_theta", False)
        if not isinstance(applies_to_theta, bool):
            raise SchemaValidationError(
                "reliability.shrinkage_applies_to_theta must be a boolean, got "
                f"{applies_to_theta!r}"
            )
        self.shrinkage_applies_to_theta = bool(applies_to_theta)
        self.table: Dict[str, Dict[str, float]] = {
            str(modality): {str(attr): float(value) for attr, value in (attrs or {}).items()}
            for modality, attrs in (section.get("by_modality_attribute") or {}).items()
        }
        contextual = dict(section.get("contextual") or {})
        self.contextual_enabled = bool(contextual.get("enabled", False))
        # Relative artifact paths resolve against the project root, never the
        # process working directory: the controller and the skill workers are
        # started from different places and must load the same frozen artifact.
        raw_config_path = contextual.get("config_path")
        raw_theta_path = contextual.get("theta_path")
        self.contextual_config_path = (
            str(project_path(raw_config_path)) if raw_config_path else None
        )
        self.contextual_theta_path = (
            str(project_path(raw_theta_path)) if raw_theta_path else None
        )
        self.contextual_table: Dict[str, Dict[str, float]] = {}
        self.contextual_theta: Optional[ContextualReliabilityModel] = None
        if self.contextual_enabled:
            if self.contextual_theta_path:
                self.contextual_theta = ContextualReliabilityModel(
                    load_theta_artifact(self.contextual_theta_path)
                )
            elif self.contextual_config_path:
                self.contextual_table = load_contextual_table(self.contextual_config_path)
            else:
                raise SchemaValidationError(
                    "reliability.contextual.enabled is true but neither theta_path nor "
                    "config_path is set; refusing to run with a flag that changes nothing."
                )

    def clamp(self, value: float) -> float:
        """Keep reliability strictly inside ``(0, 1)`` and within config bounds."""
        return max(self.min, min(self.max, float(value)))

    def shrink(self, value: float) -> float:
        """Pull a reliability toward 0.5 by ``reliability.shrinkage``.

        ``r' = 0.5 + s * (r - 0.5)``.  ``s = 1`` returns the table value
        unchanged; ``s = 0`` returns 0.5, i.e. a claim that says nothing.  This
        is a deliberate accuracy/robustness trade: an empirically well calibrated
        cell close to 1.0 lets one corrupted claim decide a truth variable, and
        that damage propagates into the association through the feedback term.
        """
        if self.shrinkage >= 1.0:
            return float(value)
        return 0.5 + self.shrinkage * (float(value) - 0.5)

    def reliability_for(self, modality: Any, attribute: str) -> float:
        """Fixed reliability for a ``(modality, attribute)`` pair.

        The calibrated table wins when it has the cell; otherwise the shipped
        table applies.  A calibration may legitimately cover only the cells it
        had support to fit.
        """
        modality_key = (
            modality.value if isinstance(modality, Modality) else str(modality or "").lower()
        )
        value: Optional[float] = None
        if self.contextual_enabled:
            calibrated = self.contextual_table.get(modality_key) or {}
            if str(attribute) in calibrated:
                value = calibrated[str(attribute)]
        if value is None:
            by_modality = self.table.get(modality_key) or {}
            value = by_modality.get(str(attribute), self.default)
        return self.clamp(self.shrink(value))

    def for_claim(
        self,
        claim: Claim,
        observation: Optional[PrimitiveObservation] = None,
    ) -> float:
        """Reliability of one claim.

        The contextual theta artifact wins when it supports the claim's cell;
        unsupported cells fall back to the fixed/calibrated table, matching the
        partial-coverage contract.  The modality comes from the owning
        observation when available, falling back to the claim's
        ``source_channel``.  ``raw_confidence`` is read only for diagnostics and
        never influences the returned value.
        """
        if self.contextual_theta is not None and observation is not None:
            contextual_value = self.contextual_theta.for_claim(claim, observation)
            if contextual_value is not None:
                if self.shrinkage_applies_to_theta:
                    # Opt-in: the sigmoid is bounded by construction, so the
                    # fixed-table guards are off for theta cells by default.
                    return self.clamp(self.shrink(float(contextual_value)))
                return float(contextual_value)
        if observation is not None:
            modality = observation.modality
        else:
            modality = Modality.coerce(claim.source_channel)
        return self.reliability_for(modality, claim.attribute)

    def build_table(
        self,
        claims: Sequence[Claim],
        observations: Mapping[str, PrimitiveObservation],
    ) -> Dict[str, float]:
        """Build a ``claim_id -> r_i`` table for a whole inference problem."""
        return {
            claim.claim_id: self.for_claim(claim, observations.get(claim.observation_id))
            for claim in claims
        }

    def describe(self) -> Dict[str, Any]:
        """Config snapshot for provenance and evaluation reports."""
        cells = sum(len(attrs) for attrs in self.contextual_table.values())
        if self.contextual_theta is not None:
            source = "contextual_theta"
        elif self.contextual_enabled and cells:
            source = "calibrated_shrunk" if self.shrinkage < 1.0 else "calibrated"
        else:
            source = "fixed_config_shrunk" if self.shrinkage < 1.0 else "fixed_config"
        return {
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "shrinkage": self.shrinkage,
            "shrinkage_applies_to_theta": self.shrinkage_applies_to_theta,
            "outlier_mixture": self.outlier_mixture,
            "by_modality_attribute": {k: dict(v) for k, v in self.table.items()},
            "contextual_enabled": self.contextual_enabled,
            "contextual_config_path": self.contextual_config_path,
            "contextual_table": {k: dict(v) for k, v in self.contextual_table.items()},
            "contextual_cells": cells,
            "contextual_theta_path": self.contextual_theta_path,
            "theta_cells": (
                len(self.contextual_theta.artifact.intercepts)
                if self.contextual_theta is not None
                else 0
            ),
            "contextual_theta": (
                self.contextual_theta.describe() if self.contextual_theta is not None else None
            ),
            "source": source,
        }


__all__ = ["ReliabilityModel"]
