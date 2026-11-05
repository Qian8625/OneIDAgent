"""Roll the store plus inference result into a per-object diagnostic summary.

This layer produces *numbers and diagnostics only*: association support, truth
beliefs, coverage, provenance, ambiguity, contradictions.  Task labels such as
``supported`` or ``not found`` are produced downstream in :mod:`decision`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from DS_VI.models.shared_identity.likelihood import entropy, normalized_entropy
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    EvidenceState,
    InferenceState,
    ObjectEvidenceSummary,
)

if TYPE_CHECKING:  # pragma: no cover
    from DS_VI.models.shared_identity.evidence_store import EvidenceStore


def _truth_summary(
    state: InferenceState,
    object_id: str,
) -> Dict[str, Dict[str, Any]]:
    """Best value, probability, entropy and full distribution per attribute/time."""
    summary: Dict[str, Dict[str, Any]] = {}
    for (obj, attribute, time_key), distribution in state.truth.items():
        if obj != object_id or not distribution:
            continue
        best_value, best_probability = max(distribution.items(), key=lambda item: item[1])
        summary[f"{attribute}@{time_key}"] = {
            "attribute": attribute,
            "time_key": time_key,
            "value": best_value,
            "probability": float(best_probability),
            "entropy": entropy(distribution),
            "normalized_entropy": normalized_entropy(distribution),
            "distribution": {str(k): float(v) for k, v in distribution.items()},
        }
    return summary


def _contradictions(
    store: EvidenceStore,
    state: InferenceState,
    object_id: str,
    margin: float,
) -> List[Dict[str, Any]]:
    """Attribute axes where high-association claims disagree on the value.

    A contradiction means two well-associated observations assert different
    values for the same attribute while the truth posterior remains undecided.
    """
    found: List[Dict[str, Any]] = []
    by_axis: Dict[tuple, Dict[str, float]] = {}
    for claim in store.claims.values():
        association = float(state.association_row(claim.observation_id).get(object_id, 0.0))
        if association <= 0.0:
            continue
        axis = (claim.attribute, claim.time_scope.key)
        bucket = by_axis.setdefault(axis, {})
        key = str(claim.value).strip().lower()
        bucket[key] = bucket.get(key, 0.0) + association

    for (attribute, time_key), bucket in by_axis.items():
        if len(bucket) < 2:
            continue
        ordered = sorted(bucket.items(), key=lambda item: item[1], reverse=True)
        top, second = ordered[0], ordered[1]
        if abs(top[1] - second[1]) <= float(margin) and second[1] > 0.0:
            found.append(
                {
                    "attribute": attribute,
                    "time_key": time_key,
                    "competing_values": [top[0], second[0]],
                    "support": [float(top[1]), float(second[1])],
                }
            )
    return found


def build_evidence_state(
    store: EvidenceStore,
    state: Optional[InferenceState] = None,
) -> EvidenceState:
    """Summarize a store and inference result into an :class:`EvidenceState`."""
    inference = state or store.inference_state
    if inference is None:
        return EvidenceState(
            observation_count=len(store.observations),
            claim_count=len(store.claims),
            coverage=dict(store.coverage),
            inference_diagnostics={"status": "no_inference"},
        )

    decision_cfg = dict(store.config.get("decision") or {})
    margin = float(decision_cfg.get("contradiction_margin", 0.15))

    objects: Dict[str, ObjectEvidenceSummary] = {}
    for object_id in store.objects:
        supporting: List[str] = []
        support_mass = 0.0
        for observation_id in store.observations:
            probability = float(state_probability(inference, observation_id, object_id))
            if probability > 0.0:
                support_mass += probability
            if probability >= 0.5:
                supporting.append(observation_id)

        modalities: Dict[str, int] = {}
        times: Dict[str, int] = {}
        for observation_id in supporting:
            observation = store.observations[observation_id]
            modalities[observation.modality.value] = (
                modalities.get(observation.modality.value, 0) + 1
            )
            times[observation.time_scope.key] = times.get(observation.time_scope.key, 0) + 1

        association_distribution = {
            observation_id: float(state_probability(inference, observation_id, object_id))
            for observation_id in store.observations
        }
        nonzero = {k: v for k, v in association_distribution.items() if v > 0.0}

        truth_beliefs = _truth_summary(inference, object_id)
        ambiguity = {
            "association_entropy": entropy(nonzero) if nonzero else 0.0,
            "max_truth_entropy": max(
                (item["entropy"] for item in truth_beliefs.values()), default=0.0
            ),
            "mean_truth_entropy": (
                sum(item["entropy"] for item in truth_beliefs.values()) / len(truth_beliefs)
                if truth_beliefs
                else 0.0
            ),
        }

        provenance_nodes = [
            store.provenance.nodes[observation_id]
            for observation_id in supporting
            if observation_id in store.provenance.nodes
        ]
        provenance_summary = {
            "node_count": len(provenance_nodes),
            "duplicate_count": sum(1 for n in provenance_nodes if n.duplicate_of),
            "input_roots": sorted(
                {root for node in provenance_nodes for root in node.input_roots}
            ),
            "tools": sorted({node.tool_name for node in provenance_nodes}),
        }

        objects[object_id] = ObjectEvidenceSummary(
            object_id=object_id,
            association_support=support_mass,
            supporting_observations=supporting,
            truth_beliefs=truth_beliefs,
            coverage={
                "modalities": modalities,
                "time_keys": times,
                "modality_count": len(modalities),
                "time_count": len(times),
            },
            provenance_summary=provenance_summary,
            ambiguity=ambiguity,
            contradictions=_contradictions(store, inference, object_id, margin),
        )

    null_mass = {
        observation_id: float(state_probability(inference, observation_id, NULL_OBJECT_ID))
        for observation_id in store.observations
    }

    return EvidenceState(
        objects=objects,
        observation_count=len(store.observations),
        claim_count=len(store.claims),
        coverage=dict(store.coverage),
        null_mass=null_mass,
        inference_diagnostics={
            "iterations": inference.iterations,
            "converged": inference.converged,
            "final_objective": inference.diagnostics.get("final_objective"),
            "final_max_delta": inference.diagnostics.get("final_max_delta"),
            "projection": inference.diagnostics.get("projection"),
            "reliability_source": (
                inference.diagnostics.get("reliability_model") or {}
            ).get("source"),
        },
    )


def state_probability(state: InferenceState, observation_id: str, object_id: str) -> float:
    """``A[observation_id, object_id]``, defaulting to 0."""
    return float(state.association_row(observation_id).get(object_id, 0.0))


def compact_summary(evidence_state: EvidenceState, top_k: int = 3) -> Dict[str, Any]:
    """Small, LLM-context-friendly digest of an :class:`EvidenceState`."""
    ranked = sorted(
        evidence_state.objects.values(),
        key=lambda item: item.association_support,
        reverse=True,
    )[: max(1, int(top_k))]
    return {
        "observation_count": evidence_state.observation_count,
        "claim_count": evidence_state.claim_count,
        "coverage": dict(evidence_state.coverage),
        "mean_null_mass": (
            sum(evidence_state.null_mass.values()) / len(evidence_state.null_mass)
            if evidence_state.null_mass
            else 0.0
        ),
        "inference": dict(evidence_state.inference_diagnostics),
        "top_objects": [
            {
                "object_id": item.object_id,
                "association_support": round(item.association_support, 4),
                "supporting_observations": len(item.supporting_observations),
                "association_entropy": round(
                    float(item.ambiguity.get("association_entropy", 0.0)), 4
                ),
                "truth": {
                    key: {
                        "value": belief["value"],
                        "probability": round(float(belief["probability"]), 4),
                    }
                    for key, belief in item.truth_beliefs.items()
                },
                "contradictions": len(item.contradictions),
            }
            for item in ranked
        ],
    }


__all__ = ["build_evidence_state", "state_probability", "compact_summary"]
