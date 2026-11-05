"""Plugin-dispatched skill parameter normalization.

Normalizers are owned by individual skill plugins. Each plugin can declare a
``normalizer.module`` in its manifest and register one or more functions with
``register_normalizer`` from this module.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


NormalizerFn = Callable[[Dict[str, Any]], Tuple[Dict[str, Any], List[Dict[str, Any]]]]

_NORMALIZER_REGISTRY: Dict[str, NormalizerFn] = {}


class NormalizerRegistryError(RuntimeError):
    """Raised when normalizer registration or lookup fails."""


def register_normalizer(name: str, *, override: bool = False):
    """Register a normalizer function by symbolic manifest name."""
    if not name or not isinstance(name, str):
        raise NormalizerRegistryError("Normalizer name must be a non-empty string")

    def decorator(fn: NormalizerFn) -> NormalizerFn:
        if not callable(fn):
            raise NormalizerRegistryError(f"Normalizer for {name} is not callable")
        if name in _NORMALIZER_REGISTRY and not override:
            raise NormalizerRegistryError(f"Duplicate normalizer registration: {name}")
        _NORMALIZER_REGISTRY[name] = fn
        return fn

    return decorator


def get_normalizer(name: Optional[str]) -> Optional[NormalizerFn]:
    if not name:
        return None
    return _NORMALIZER_REGISTRY.get(name)


def has_normalizer(name: Optional[str]) -> bool:
    return bool(name and name in _NORMALIZER_REGISTRY)


def list_normalizers() -> List[str]:
    return sorted(_NORMALIZER_REGISTRY.keys())


def clear_normalizers() -> None:
    """Clear the registry. Intended for tests only."""
    _NORMALIZER_REGISTRY.clear()


def normalize_skill_params(
    skill_name: str,
    spec: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    params = dict(params or {})
    normalizer_name = spec.get("normalizer")

    if not normalizer_name:
        return params, []

    trace: List[Dict[str, Any]] = [
        {
            "stage": "normalize_payload",
            "status": "start",
            "skill_name": skill_name,
            "normalizer": normalizer_name,
        }
    ]

    normalizer = get_normalizer(normalizer_name)
    if normalizer is None:
        trace.append(
            {
                "stage": "normalize_payload",
                "status": "skipped",
                "reason": f"Unknown normalizer: {normalizer_name}",
            }
        )
        return params, trace

    normalized, normalizer_trace = normalizer(params)
    trace.extend(normalizer_trace)
    return normalized, trace


__all__ = [
    "NormalizerFn",
    "NormalizerRegistryError",
    "register_normalizer",
    "get_normalizer",
    "has_normalizer",
    "list_normalizers",
    "clear_normalizers",
    "normalize_skill_params",
]
