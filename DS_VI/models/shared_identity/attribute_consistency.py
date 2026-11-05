"""Reliability-weighted attribute consistency for cross-view identity candidates.

A cross-view pair is only worth merging when the two boxes *agree* on attributes
that both views can actually judge. Hard equality cannot express that: it treats a
subtype agreement from the SAR view (measured 0.19 accuracy on a 17-value domain)
exactly like a size-bucket agreement (0.95), so it both keeps junk and rejects good
pairs on a coin-flip signal.

The score is weighted by measured informativeness:

.. code-block:: text

    info(modality, attribute) = max(0, acc - 1/C) / (1 - 1/C)

with ``acc`` the accuracy among *filled* predictions measured on the property
workbooks and ``C`` the attribute's domain size. ``info = 0`` means the channel is
at or below chance and must not vote. A pair's weight for an attribute is ``min``
over its two views: agreement is only evidence when *both* sides can produce it.

Decision:

.. code-block:: text

    asserted = sum of weights where both members report the attribute
    net      = (agree - disagree) / asserted        in [-1, 1]
    asserted < min_weight -> neutral (nothing to judge, keep the candidate)
    otherwise            -> keep iff net >= tau
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

#: ``modality -> attribute -> measured informativeness`` (chance-normalized).
#: Measured on the 2026-11 property workbooks by
#: ``python -B -m DS_VI.experiments.metrics.attribute_informativeness`` (see that
#: module for the definition); accuracy among *filled* predictions, pooled per
#: modality, chance = 1/domain size.
#:
#: Re-measured because the annotation release replaced ``state`` with ``color``,
#: filled ``subtype`` on every instance, and reissued ``orientation``.  Numbers
#: that no longer describe the data made the gate weigh a channel at zero weight.
ATTRIBUTE_INFORMATIVENESS: Dict[str, Dict[str, float]] = {
    "optical": {
        "category": 0.9985,
        "subtype": 0.7961,
        "color": 0.7097,
        "size_bin": 0.9988,
        # Chance-level on the new sheets (0.313 accuracy over 3 values): it must
        # abstain, not vote against the pair.
        "orientation": 0.0,
        # No ``state`` column in the 2026-11 release; the axis is kept at 0 so a
        # legacy workbook cannot resurrect a stale weight.
        "state": 0.0,
    },
    "sar": {
        "category": 0.9767,
        "subtype": 0.2906,
        "size_bin": 0.9216,
        "orientation": 0.4291,
        # SAR carries no colour annotation (every cell empty) -> abstain.
        "color": 0.0,
        "state": 0.0,
    },
}

#: Attributes a weighted consistency check reads by default.
DEFAULT_ATTRIBUTES: Tuple[str, ...] = ("subtype", "color", "size_bin", "orientation")

#: View -> modality per family, mirroring the detection provider and VI.
VIEW_MODALITY: Dict[str, Dict[str, str]] = {
    "match": {"a": "optical", "b": "sar"},
    "time": {"a": "optical", "b": "optical"},
}


def attribute_weight(attribute: str, modality_a: str, modality_b: str) -> float:
    """Weight of one attribute for a pair whose views carry these modalities."""
    left = ATTRIBUTE_INFORMATIVENESS.get(modality_a, {}).get(attribute, 0.0)
    right = ATTRIBUTE_INFORMATIVENESS.get(modality_b, {}).get(attribute, 0.0)
    return min(float(left), float(right))


def pair_weights(family: str, attributes: Sequence[str] = DEFAULT_ATTRIBUTES) -> Dict[str, float]:
    """Per-attribute weights for one family's two views."""
    modalities = VIEW_MODALITY.get(family, {"a": "optical", "b": "sar"})
    return {
        attribute: attribute_weight(attribute, modalities["a"], modalities["b"])
        for attribute in attributes
    }


def consistency(
    attributes_a: Mapping[str, Any],
    attributes_b: Mapping[str, Any],
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    """Weighted agreement of one candidate pair.

    A blank cell contributes nothing: the tools leave subtype/size empty on a
    large fraction of detections, and a missing value is not a disagreement.
    """
    asserted = 0.0
    agree = 0.0
    disagree = 0.0
    detail: Dict[str, str] = {}
    for attribute, weight in weights.items():
        if weight <= 0.0:
            continue
        left = attributes_a.get(attribute)
        right = attributes_b.get(attribute)
        if not left or not right:
            continue
        asserted += weight
        if str(left) == str(right):
            agree += weight
            detail[attribute] = "agree"
        else:
            disagree += weight
            detail[attribute] = "disagree"
    net = (agree - disagree) / asserted if asserted > 0.0 else 0.0
    return {
        "asserted": round(asserted, 6),
        "agree": round(agree, 6),
        "disagree": round(disagree, 6),
        "net": round(net, 6),
        "detail": detail,
    }


def accept(score: Mapping[str, Any], tau: float = 0.0, min_weight: float = 0.5) -> bool:
    """Decision for one candidate: neutral when nothing is asserted."""
    if float(score.get("asserted", 0.0)) < float(min_weight):
        return True
    return float(score.get("net", 0.0)) >= float(tau)


__all__ = [
    "ATTRIBUTE_INFORMATIVENESS",
    "DEFAULT_ATTRIBUTES",
    "VIEW_MODALITY",
    "accept",
    "attribute_weight",
    "consistency",
    "pair_weights",
]
