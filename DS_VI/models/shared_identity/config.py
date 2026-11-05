"""Configuration loading for the shared-identity layer.

All thresholds live in ``configs/shared_identity/default.yaml``.  This module is
the only place that knows where that file is and how to merge overrides, so no
inference or worker code needs to hardcode a number.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dependency of the repo
    yaml = None

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "DS_VI/configs" / "shared_identity" / "default.yaml"


def project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is already absolute.

    Relative paths are independent of the process working directory.
    """
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return candidate
    return _REPO_ROOT / candidate


def resolve_config_path(path: Optional[str | Path] = None) -> Path:
    """Return the shared-identity YAML path using the documented precedence.

    ``explicit path`` > ``DEFAULT_CONFIG_PATH``. A user-supplied path that does not exist raises:
    silently falling back to the shipped defaults would make a tuned
    experiment indistinguishable from the baseline.
    """
    if path is not None:
        candidate = project_path(path)
        if not candidate.is_file():
            raise FileNotFoundError(
                f"shared-identity config not found: {candidate} "
                "(explicit path; refusing to fall back to defaults)"
            )
        return candidate

    return DEFAULT_CONFIG_PATH

# Fallback mirror of the YAML defaults, used when PyYAML or the file is missing
# so that unit tests never depend on filesystem layout.
_FALLBACK_CONFIG: Dict[str, Any] = {
    "schema_version": "1.0",
    "physical": {
        "weights": {"iou": 0.45, "center": 0.30, "area": 0.15, "confidence": 0.10},
        "center_scale_factor": 0.75,
        "gates": {
            "low_iou_gate": 0.05,
            "center_gate": 0.25,
            "geo_gate_m": 250.0,
            "max_center_distance_px": None,
        },
        "null_prior": 0.15,
        # Paper Eq.(2): p[m,k] ~ exp(d^T f), with d = prior_scale * weights and
        # d_null = null_logit.  null_logit: null derives it from null_prior.
        "prior_scale": 6.0,
        "null_logit": None,
        "max_candidates_per_observation": 12,
    },
    "association": {
        "mode": "cross_image_independent",
        "strict_scope": True,
        "forbid_same_image": True,
        "pair_capacity": None,
    },
    "inference": {
        # Objective default (paper Table 6); every shipped config sets 1.5.
        "eta": 1.0,
        "max_iter": 50,
        "tol": 1.0e-5,
        "epsilon": 1.0e-12,
        "projection": {"max_iter": 100, "tol": 1.0e-6},
        "default_capacity": None,
        "capacity_projection": True,
        "capacity_mode": "block_object_column",
        "block_capacity": 1.0,
        "truth_prior_smoothing": 1.0,
        # Neutral robustness defaults; see the YAML for the rationale.
        "evidence_cap": None,
        "feedback_damping": 1.0,
        # Attributes pooled over time (empty = every axis is per-time).
        "time_invariant_attributes": [],
    },
    "reliability": {
        "default": 0.70,
        "by_modality_attribute": {
            "optical": {"category": 0.88, "subtype": 0.72, "color": 0.73, "heading": 0.65, "orientation": 0.24},
            "sar": {"category": 0.78, "subtype": 0.60, "heading": 0.55, "orientation": 0.63},
            "temporal": {"presence": 0.80, "motion": 0.70, "orientation": 0.21},
            "cross_modal": {"category": 0.82, "subtype": 0.66, "heading": 0.60, "orientation": 0.24},
        },
        "min": 0.35,
        "max": 0.97,
        "shrinkage": 1.0,
        "shrinkage_applies_to_theta": False,
        "outlier_mixture": 0.0,
        "contextual": {"enabled": False, "config_path": None, "theta_path": None},
    },
    "provenance": {
        "group_budget": 1.0,
        "dedup_exact_replay": True,
        "grouping_mode": "connected_component",
        "remove_deterministic_restatements": True,
        "budget_enabled": True,
    },
    "cascade": {
        "residual_threshold": 0.5,
        "max_links": {"match": 4, "time": 6},
        "center_scale_factor": {"match": 0.75, "time": 1.5},
        "extra_links": 2,
        "extra_mode": {"match": "pruned", "time": "mutual"},
        "extra_prior_floor": 0.2,
        "extra_margin": {"match": 0.30, "time": 0.20},
        "extra_consistency": {"match": [], "time": ["subtype", "color", "size_bin", "orientation"]},
        "consistency_mode": "weighted",
        "consistency_tau": 0.0,
        "consistency_min_weight": 0.5,
        "policy": "blend",
        "blend_weight": 0.5,
        "extra_policy": "confirmed",
    },
    "decision": {
        "identity_threshold": 0.60,
        "truth_threshold": 0.60,
        "qualified_truth_threshold": 0.40,
        "ambiguity_entropy": 0.90,
        "min_coverage": 0.50,
        "max_null_mass": 0.60,
        "contradiction_margin": 0.15,
    },
    "acquisition": {
        "association_entropy_trigger": 0.60,
        "truth_entropy_trigger": 0.60,
        "max_requests": 5,
    },
    # subtype / state / color / size_bin mirror the annotation vocabulary
    # declared in the property workbooks (the label space offered to annotators),
    # not values harvested from evaluation gold.  ``state`` belongs to the
    # pre-2026-11 sheets and is kept so an older workbook still has a domain;
    # ``color`` is the axis that replaced it.
    "value_domains": {
        "category": ["vehicle", "tank", "ship", "aircraft", "bridge", "harbor", "other"],
        "subtype": [
            "civilian_airliner",
            "military_aircraft",
            "small_aircraft",
            "cargo_ship",
            "fishing_boat",
            "warship",
            "small_boat",
            "storage_tank",
            "oil_tank",
            "road_bridge",
            "rail_bridge",
            "pier",
            "dock",
            "car",
            "truck",
            "bus",
            "unknown",
        ],
        "color": [
            "white",
            "gray",
            "black",
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "brown",
            "multi",
        ],
        "state": [
            "moving",
            "static",
            "moored",
            "parked",
            "under_construction",
            "damaged",
            "occluded",
            "unknown",
        ],
        "size_bin": ["small", "medium", "large", "unknown"],
        "heading": ["north", "east", "south", "west", "unknown"],
        "orientation": ["horizontal", "vertical", "diagonal"],
        "presence": ["present", "absent"],
        "motion": ["static", "moving", "unknown"],
    },
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_overrides(assignments: Sequence[str]) -> Dict[str, Any]:
    """Parse ``a.b=value`` command-line assignments into a nested mapping.

    Values are JSON where possible (``null``, ``0.5``, ``true``, ``"x"``) and
    fall back to the raw string so YAML-style words such as
    ``grouping_mode=connected_component`` keep working.
    """
    override: Dict[str, Any] = {}
    for assignment in assignments or ():
        if "=" not in assignment:
            raise ValueError(f"override expects key=value, got {assignment!r}")
        key, raw = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"override has an empty key: {assignment!r}")
        try:
            value: Any = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            value = raw
        parts = key.split(".")
        cursor = override
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return override


def load_config(
    path: Optional[str | Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the shared-identity config, merged over the built-in defaults.

    Args:
        path: Optional YAML path.  Defaults to
            ``configs/shared_identity/default.yaml``.
        overrides: Optional nested mapping applied last.

    Returns:
        A fully populated config dict.  Missing file or missing PyYAML falls
        back to the in-module defaults rather than raising, keeping algorithm
        unit tests independent of the repo layout.
    """
    config = copy.deepcopy(_FALLBACK_CONFIG)
    config_path = resolve_config_path(path)
    if yaml is not None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, Mapping):
            config = deep_merge(config, loaded)
    if overrides:
        config = deep_merge(config, overrides)
    return config


def effective_config_path() -> str:
    """Path of the shared-identity YAML that :func:`load_config` would read."""
    return str(resolve_config_path())


def section(config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """Return one top-level config section as a plain dict."""
    value = config.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "deep_merge",
    "effective_config_path",
    "load_config",
    "parse_overrides",
    "project_path",
    "resolve_config_path",
    "section",
]
