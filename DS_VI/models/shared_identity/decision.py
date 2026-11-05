"""Decision layer: turn an EvidenceState into a task-level verdict.

This is the *only* place that produces status labels.  The inference module must
never emit ``Confirmed`` / ``not found`` / ``inconsistent``, because a decision
threshold feeding back into a likelihood would corrupt the posterior.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.schema import (
    DecisionResult,
    EvidenceState,
    ObjectEvidenceSummary,
)


def _expected_coverage_cells(config: Mapping[str, Any]) -> int:
    """How many (modality, time) cells a well-covered object should have."""
    return int(dict(config.get("decision") or {}).get("expected_coverage_cells", 2))


def coverage_ratio(summary: ObjectEvidenceSummary, expected_cells: int) -> float:
    """Observed (modality, time) cells over expected, capped at 1."""
    modality_count = int(summary.coverage.get("modality_count", 0))
    time_count = int(summary.coverage.get("time_count", 0))
    observed = max(modality_count, 0) + max(time_count - 1, 0)
    if expected_cells <= 0:
        return 1.0
    return min(1.0, observed / float(expected_cells))


def truth_strength(
    probability: float,
    belief: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple:
    """Return ``(value, mode, domain_size)`` compared against the commit thresholds.

    ``absolute`` (shipped) compares the posterior probability directly.
    ``uniform_lift`` compares ``p * |D|``, i.e. how many times more mass the
    top value carries than one value under a uniform distribution of the same
    attribute domain. Lift is invariant to the additive
    ``inference.truth_prior_smoothing`` that keeps absolute posteriors in a
    narrow band, so one threshold remains meaningful across attributes whose
    domain sizes differ. When the domain is unknown the absolute probability is
    returned and the mode downgrades to ``absolute``.

    This function only rescales the comparison; it never changes the posterior.
    """
    mode = str((config.get("decision") or {}).get("truth_scale", "absolute") or "absolute")
    if mode != "uniform_lift":
        return float(probability), "absolute", None
    attribute = str(belief.get("attribute") or "")
    labels = list(dict(config.get("value_domains") or {}).get(attribute) or [])
    domain_size = len(labels) or len(belief.get("distribution") or {}) or 0
    if domain_size <= 0:
        return float(probability), "absolute", None
    return float(probability) * float(domain_size), "uniform_lift", int(domain_size)


def decide_object_truth(
    evidence_state: EvidenceState,
    object_id: str,
    attribute: Optional[str] = None,
    time_key: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> DecisionResult:
    """Decide the truth of one object attribute from accumulated evidence.

    Decision order, most specific first:

    1. ``insufficient_evidence`` — no supporting observations, or coverage below
       ``min_coverage``, or NULL mass above ``max_null_mass``.
    2. ``contradictory_evidence`` — competing values with near-equal support on
       the requested axis.
    3. ``ambiguous_identity`` — association entropy above ``ambiguity_entropy``,
       or top association below ``identity_threshold``.
    4. ``Confirmed`` — truth strength at or above ``truth_threshold``.
    5. ``Weak Support`` — truth strength in
       [``qualified_truth_threshold``, ``truth_threshold``), and every
       remaining case that is neither a validated conflict nor unresolved.
    6. ``abstain`` — otherwise.

    Truth strength is the posterior probability under ``decision.truth_scale:
    absolute`` or ``p * |domain|`` under ``uniform_lift``; see
    :func:`truth_strength`.
    """
    cfg = dict(config or load_config())
    thresholds = dict(cfg.get("decision") or {})
    identity_threshold = float(thresholds.get("identity_threshold", 0.60))
    truth_threshold = float(thresholds.get("truth_threshold", 0.60))
    qualified_threshold = float(thresholds.get("qualified_truth_threshold", 0.40))
    ambiguity_entropy = float(thresholds.get("ambiguity_entropy", 0.90))
    min_coverage = float(thresholds.get("min_coverage", 0.50))
    max_null_mass = float(thresholds.get("max_null_mass", 0.60))

    summary = evidence_state.objects.get(object_id)
    if summary is None:
        result = DecisionResult(
            status="Uncertain",
            object_id=object_id,
            attribute=attribute,
            reasons=[f"object '{object_id}' is not present in the evidence state"],
        )
        result.validate()
        return result

    reasons: List[str] = []
    evidence_refs: Dict[str, Any] = {
        "supporting_observations": list(summary.supporting_observations),
        "association_support": summary.association_support,
        "coverage": dict(summary.coverage),
        "provenance": dict(summary.provenance_summary),
    }

    # 1. Evidence sufficiency.
    ratio = coverage_ratio(summary, _expected_coverage_cells(cfg))
    evidence_refs["coverage_ratio"] = ratio
    if not summary.supporting_observations:
        reasons.append("no observation is associated with this object above 0.5")
        result = DecisionResult(
            status="Uncertain",
            object_id=object_id,
            attribute=attribute,
            reasons=reasons,
            evidence_refs=evidence_refs,
        )
        result.validate()
        return result

    mean_null = (
        sum(
            evidence_state.null_mass.get(observation_id, 0.0)
            for observation_id in summary.supporting_observations
        )
        / len(summary.supporting_observations)
    )
    evidence_refs["mean_null_mass"] = mean_null
    if ratio < min_coverage or mean_null > max_null_mass:
        if ratio < min_coverage:
            reasons.append(f"coverage ratio {ratio:.2f} < min_coverage {min_coverage:.2f}")
        if mean_null > max_null_mass:
            reasons.append(f"mean NULL mass {mean_null:.2f} > max {max_null_mass:.2f}")
        result = DecisionResult(
            status="Uncertain",
            object_id=object_id,
            attribute=attribute,
            reasons=reasons,
            evidence_refs=evidence_refs,
        )
        result.validate()
        return result

    # Locate the requested truth axis.
    belief = None
    belief_key = None
    for key, item in summary.truth_beliefs.items():
        if attribute is not None and item.get("attribute") != attribute:
            continue
        if time_key is not None and item.get("time_key") != time_key:
            continue
        if belief is None or float(item.get("probability", 0.0)) > float(
            belief.get("probability", 0.0)
        ):
            belief = item
            belief_key = key

    # 2. Contradictions on the requested axis.
    relevant = [
        item
        for item in summary.contradictions
        if attribute is None or item.get("attribute") == attribute
    ]
    if relevant:
        evidence_refs["contradictions"] = relevant
        reasons.append(
            "competing values with near-equal association support: "
            + ", ".join(
                f"{c['attribute']}={c['competing_values']}" for c in relevant
            )
        )
        result = DecisionResult(
            status="Conflict",
            object_id=object_id,
            attribute=attribute,
            value=belief.get("value") if belief else None,
            confidence=float(belief.get("probability", 0.0)) if belief else 0.0,
            reasons=reasons,
            evidence_refs=evidence_refs,
        )
        result.validate()
        return result

    # 3. Identity ambiguity.
    association_entropy = float(summary.ambiguity.get("association_entropy", 0.0))
    top_association = max(
        (
            evidence_state.objects[object_id].association_support
            / max(len(summary.supporting_observations), 1),
            0.0,
        )
    )
    evidence_refs["association_entropy"] = association_entropy
    evidence_refs["mean_association"] = top_association
    if association_entropy > ambiguity_entropy or top_association < identity_threshold:
        if association_entropy > ambiguity_entropy:
            reasons.append(
                f"association entropy {association_entropy:.3f} > {ambiguity_entropy:.3f}"
            )
        if top_association < identity_threshold:
            reasons.append(
                f"mean association {top_association:.3f} < identity threshold "
                f"{identity_threshold:.3f}"
            )
        result = DecisionResult(
            status="Uncertain",
            object_id=object_id,
            attribute=attribute,
            value=belief.get("value") if belief else None,
            confidence=float(belief.get("probability", 0.0)) if belief else 0.0,
            reasons=reasons,
            evidence_refs=evidence_refs,
        )
        result.validate()
        return result

    # 4/5/6. Truth strength.
    if belief is None:
        reasons.append(
            f"no truth belief for attribute={attribute!r} time={time_key!r}"
        )
        result = DecisionResult(
            status="Uncertain",
            object_id=object_id,
            attribute=attribute,
            reasons=reasons,
            evidence_refs=evidence_refs,
        )
        result.validate()
        return result

    probability = float(belief.get("probability", 0.0))
    strength, scale_mode, domain_size = truth_strength(probability, belief, cfg)
    evidence_refs["truth_axis"] = belief_key
    evidence_refs["truth_distribution"] = belief.get("distribution")
    evidence_refs["truth_scale"] = scale_mode
    if domain_size:
        evidence_refs["truth_domain_size"] = domain_size
        evidence_refs["truth_lift"] = round(strength, 6)
    strength_label = "probability" if scale_mode == "absolute" else "lift"

    if strength >= truth_threshold:
        status = "Confirmed"
        reasons.append(f"truth {strength_label} {strength:.3f} >= {truth_threshold:.3f}")
    elif strength >= qualified_threshold:
        status = "Weak Support"
        reasons.append(
            f"truth {strength_label} {strength:.3f} in "
            f"[{qualified_threshold:.3f}, {truth_threshold:.3f})"
        )
    else:
        status = "Weak Support"
        reasons.append(
            f"truth {strength_label} {strength:.3f} < qualified threshold "
            f"{qualified_threshold:.3f}"
        )

    result = DecisionResult(
        status=status,
        object_id=object_id,
        attribute=belief.get("attribute"),
        value=belief.get("value"),
        confidence=probability,
        reasons=reasons,
        evidence_refs=evidence_refs,
    )
    result.validate()
    return result


def decide_best_object(
    evidence_state: EvidenceState,
    attribute: Optional[str] = None,
    time_key: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> DecisionResult:
    """Decide over the best-supported object; abstain when there is none."""
    if not evidence_state.objects:
        result = DecisionResult(
            status="Uncertain",
            reasons=["no object hypotheses exist in the evidence state"],
        )
        result.validate()
        return result
    best = max(
        evidence_state.objects.values(), key=lambda item: item.association_support
    )
    return decide_object_truth(
        evidence_state, best.object_id, attribute, time_key, config
    )


__all__ = [
    "coverage_ratio",
    "decide_best_object",
    "decide_object_truth",
    "truth_strength",
]
