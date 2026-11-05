"""Dempster-Shafer evidence primitives for two-source ship fusion.

Frame of discernment ``Theta = {S, B}``: ``S`` = ship, ``B`` = background.
A BPA is a plain dict with keys ``"S"``, ``"B"``, ``"SB"`` holding the masses
on ``{S}``, ``{B}`` and ``Theta``; the empty set always carries zero mass.

Only numpy is required so the rules stay unit-testable without the detection
stack. ``RDC_fusion`` and ``PCR6_fusion`` keep the names used by the pipeline
skeleton; if the paper defines different rules, replace those two functions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

FRAME: Tuple[str, ...] = ("S", "B")
MASS_KEYS: Tuple[str, ...] = ("S", "B", "SB")
VACUOUS: Dict[str, float] = {"S": 0.0, "B": 0.0, "SB": 1.0}
ELEMENT_KEYS: Tuple[str, ...] = ("S", "B", "SB")
ELEMENT_SETS: Dict[str, frozenset] = {
    "S": frozenset({"S"}),
    "B": frozenset({"B"}),
    "SB": frozenset({"S", "B"}),
}
EPS = 1e-12


def clamp01(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return max(0.0, min(1.0, out))


def make_bpa(
    score: Any,
    reliability: float = 1.0,
    support: Optional[float] = None,
    support_weight: float = 0.5,
) -> Dict[str, float]:
    """Base BPA of one detection.

    ``m(S) = alpha * s``, ``m(B) = alpha * (1 - s)``, ``m(SB) = 1 - alpha``
    with ``s`` the detector score and ``alpha`` the source reliability. When a
    geometry support ``g`` is given, ``s`` is replaced by the convex blend
    ``(1 - w) * s + w * g`` before the masses are built.
    """
    s = clamp01(score, 0.5)
    alpha = clamp01(reliability, 1.0)
    if support is not None:
        w = clamp01(support_weight, 0.0)
        s = (1.0 - w) * s + w * clamp01(support, 0.5)
    return normalize_bpa({"S": alpha * s, "B": alpha * (1.0 - s), "SB": 1.0 - alpha})


def normalize_bpa(mass: Mapping[str, Any]) -> Dict[str, float]:
    out = {key: max(0.0, float(mass.get(key, 0.0) or 0.0)) for key in MASS_KEYS}
    total = sum(out.values())
    if total <= EPS:
        return dict(VACUOUS)
    return {key: value / total for key, value in out.items()}


def validate_bpa(mass: Mapping[str, Any], tol: float = 1e-6) -> Dict[str, float]:
    """Return the mass as floats, raising ValueError when it is not a valid BPA."""
    if not isinstance(mass, Mapping):
        raise ValueError(f"BPA must be a mapping, got {type(mass).__name__}")
    unknown = set(mass) - set(MASS_KEYS)
    if unknown:
        raise ValueError(f"BPA contains unknown keys: {sorted(unknown)}")
    out: Dict[str, float] = {}
    for key in MASS_KEYS:
        value = mass.get(key, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"BPA mass {key!r} is not numeric: {value!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"BPA mass {key!r} is not finite: {value!r}")
        if value < -tol:
            raise ValueError(f"BPA mass {key!r} is negative: {value!r}")
        out[key] = max(0.0, value)
    total = sum(out.values())
    if abs(total - 1.0) > tol:
        raise ValueError(f"BPA does not sum to 1 (sum={total:.12f})")
    return out


def _mass_vector(mass: Mapping[str, float]) -> np.ndarray:
    return np.array([0.0, float(mass.get("S", 0.0)), float(mass.get("B", 0.0)), float(mass.get("SB", 0.0))])


def _jaccard(left: frozenset, right: frozenset) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def jousselme_distance(m1: Mapping[str, float], m2: Mapping[str, float]) -> float:
    """Jousselme et al. evidence distance between two BPAs, in [0, 1]."""
    v1 = _mass_vector(m1)
    v2 = _mass_vector(m2)
    elements = (frozenset(), ELEMENT_SETS["S"], ELEMENT_SETS["B"], ELEMENT_SETS["SB"])
    jac = np.array([[_jaccard(a, b) for b in elements] for a in elements], dtype=float)
    diff = v1 - v2
    squared = float(diff @ jac @ diff)
    return math.sqrt(max(0.0, 0.5 * squared))


def _conjunctive(m1: Mapping[str, float], m2: Mapping[str, float]) -> Tuple[Dict[str, float], float]:
    conj = {key: 0.0 for key in MASS_KEYS}
    conflict = 0.0
    for k1 in ELEMENT_KEYS:
        for k2 in ELEMENT_KEYS:
            product = float(m1.get(k1, 0.0)) * float(m2.get(k2, 0.0))
            if product == 0.0:
                continue
            inter = ELEMENT_SETS[k1] & ELEMENT_SETS[k2]
            if not inter:
                conflict += product
            elif inter == ELEMENT_SETS["S"]:
                conj["S"] += product
            elif inter == ELEMENT_SETS["B"]:
                conj["B"] += product
            else:
                conj["SB"] += product
    return conj, conflict


def dempster_combine(
    m1: Mapping[str, float],
    m2: Mapping[str, float],
) -> Tuple[Optional[Dict[str, float]], float]:
    """Dempster rule. Returns ``(mass, conflict)`` or ``(None, conflict)`` when
    the conflict mass would make the normalization singular."""
    conj, conflict = _conjunctive(m1, m2)
    if conflict >= 1.0 - EPS:
        return None, conflict
    scale = 1.0 / (1.0 - conflict)
    return normalize_bpa({key: value * scale for key, value in conj.items()}), conflict


def discount_bpa(mass: Mapping[str, float], reliability: float) -> Dict[str, float]:
    """Delete mass proportionally: ``m' = a * m + (1 - a) * m(Theta)``."""
    alpha = clamp01(reliability, 0.0)
    return normalize_bpa({
        "S": alpha * float(mass.get("S", 0.0)),
        "B": alpha * float(mass.get("B", 0.0)),
        "SB": alpha * float(mass.get("SB", 0.0)) + (1.0 - alpha),
    })


def RDC_fusion(
    m1: Mapping[str, float],
    m2: Mapping[str, float],
    conflict_fallback: bool = True,
    with_info: bool = False,
):
    """Reliability-discounting combination (default RDC definition).

    Each source is discounted by the mutual agreement ``a = 1 - d(m1, m2)``
    and the discounted BPAs are combined with Dempster. When the conflict is
    singular the result falls back to PCR6 (``conflict_fallback``).
    """
    distance = jousselme_distance(m1, m2)
    agreement = clamp01(1.0 - distance, 0.0)
    combined, conflict = dempster_combine(discount_bpa(m1, agreement), discount_bpa(m2, agreement))
    fallback = False
    if combined is None:
        if not conflict_fallback:
            combined = normalize_bpa(discount_bpa(m1, agreement))
        else:
            combined = PCR6_fusion(m1, m2)
            fallback = True
    if with_info:
        return combined, {
            "rule": "rdc",
            "distance": distance,
            "agreement": agreement,
            "conflict": conflict,
            "fallback": fallback,
        }
    return combined


def PCR6_fusion(m1: Mapping[str, float], m2: Mapping[str, float]) -> Dict[str, float]:
    """Proportional Conflict Redistribution rule #6 (DSmT) for two sources."""
    conj, _conflict = _conjunctive(m1, m2)
    out = dict(conj)
    for key_x in ELEMENT_KEYS:
        for key_y in ELEMENT_KEYS:
            if key_x == key_y:
                continue
            if ELEMENT_SETS[key_x] & ELEMENT_SETS[key_y]:
                continue
            cx = float(m1.get(key_x, 0.0))
            cy = float(m2.get(key_y, 0.0))
            denom = cx + cy
            if denom <= EPS:
                continue
            out[key_x] += cx * cx * cy / denom
            out[key_y] += cy * cy * cx / denom
    return normalize_bpa(out)


def adaptive_fusion(
    m1: Mapping[str, float],
    m2: Mapping[str, float],
    theta: float = 0.5,
    rule: str = "adaptive",
    conflict_fallback: bool = True,
    with_info: bool = False,
):
    mode = str(rule or "adaptive").strip().lower()
    distance = jousselme_distance(m1, m2)
    if mode in {"rdc", "reliability", "discount"}:
        mass, info = RDC_fusion(m1, m2, conflict_fallback=conflict_fallback, with_info=True)
        info.update({"distance": distance, "selected_by": "fixed"})
    elif mode in {"pcr6", "dsmt"}:
        mass = PCR6_fusion(m1, m2)
        info = {"rule": "pcr6", "distance": distance, "selected_by": "fixed", "fallback": False}
    elif mode in {"adaptive", "auto"}:
        if distance < float(theta):
            mass, info = RDC_fusion(m1, m2, conflict_fallback=conflict_fallback, with_info=True)
            info.update({"distance": distance, "selected_by": f"adaptive(theta={float(theta):g})"})
        else:
            mass = PCR6_fusion(m1, m2)
            info = {
                "rule": "pcr6",
                "distance": distance,
                "selected_by": f"adaptive(theta={float(theta):g})",
                "fallback": False,
            }
    else:
        raise ValueError(f"Unsupported fusion rule: {rule!r}")
    if with_info:
        return mass, info
    return mass


def betp(mass: Mapping[str, float]) -> Dict[str, float]:
    """Pignistic transform of a BPA on ``{S, B}``."""
    theta = float(mass.get("SB", 0.0))
    return {"S": float(mass.get("S", 0.0)) + 0.5 * theta, "B": float(mass.get("B", 0.0)) + 0.5 * theta}


def belief_plausibility(mass: Mapping[str, float]) -> Dict[str, float]:
    theta = float(mass.get("SB", 0.0))
    ship = float(mass.get("S", 0.0))
    background = float(mass.get("B", 0.0))
    return {
        "belief_ship": ship,
        "plausibility_ship": ship + theta,
        "belief_background": background,
        "plausibility_background": background + theta,
    }


def make_decision(
    mass: Mapping[str, float],
    accept_betp: float = 0.5,
    abstain_band: float = 0.0,
) -> Dict[str, Any]:
    validate_bpa(mass)
    pignistic = betp(mass)
    threshold = float(accept_betp)
    band = max(0.0, float(abstain_band))
    ship_probability = pignistic["S"]
    if band > 0.0 and abs(ship_probability - threshold) < band:
        decision = "abstain"
    elif ship_probability >= threshold:
        decision = "ship"
    else:
        decision = "background"
    out = {
        "decision": decision,
        "betp_ship": ship_probability,
        "threshold": threshold,
        "mass": dict(mass),
    }
    out.update(belief_plausibility(mass))
    return out


def mean_conflict(pairs: Sequence[Tuple[Mapping[str, float], Mapping[str, float]]]) -> float:
    if not pairs:
        return 0.0
    return sum(jousselme_distance(a, b) for a, b in pairs) / len(pairs)
