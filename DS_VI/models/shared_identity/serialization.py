"""Versioned JSON serialization of the evidence store.

Deliberately plain JSON rather than a database, so multi-round agent state can be
checkpointed and restored without introducing migration machinery.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Mapping, Type

from DS_VI.models.shared_identity.schema import (
    SCHEMA_VERSION,
    CandidateEdge,
    Claim,
    InferenceState,
    Modality,
    ObjectHypothesis,
    PrimitiveObservation,
    ProvenanceNode,
    Relation,
    RelationType,
    SpatialSupport,
    TimeScope,
)

if TYPE_CHECKING:  # pragma: no cover
    from DS_VI.models.shared_identity.evidence_store import EvidenceStore


def store_to_dict(store: EvidenceStore) -> Dict[str, Any]:
    """Serialize an :class:`EvidenceStore` into a JSON-safe dict."""
    return {
        "schema_version": SCHEMA_VERSION,
        "observations": [o.to_dict() for o in store.observations.values()],
        "claims": [c.to_dict() for c in store.claims.values()],
        "objects": [o.to_dict() for o in store.objects.values()],
        "candidate_edges": [e.to_dict() for e in store.candidate_edges],
        "relations": [r.to_dict() for r in store.relations.values()],
        "provenance_nodes": [n.to_dict() for n in store.provenance.nodes.values()],
        "coverage": dict(store.coverage),
        "ingest_log": list(store.ingest_log),
        "duplicate_ingests": int(store.duplicate_ingests),
        "call_fingerprints": sorted(store._call_fingerprints),
        "call_owners": dict(store._call_owners),
        "inference_state": (
            store.inference_state.to_dict() if store.inference_state else None
        ),
        "evidence_revision": int(getattr(store, "_revision", 0)),
        "inference_revision": int(getattr(store, "_inference_revision", -1)),
        "autoseed_objects": bool(getattr(store, "autoseed_objects", True)),
        "config": dict(store.config),
    }


def _time_scope(payload: Mapping[str, Any]) -> TimeScope:
    data = dict(payload or {})
    return TimeScope(
        key=str(data.get("key", "t0")),
        timestamp=data.get("timestamp"),
        label=data.get("label"),
    )


def _spatial_support(payload: Mapping[str, Any]) -> SpatialSupport:
    data = dict(payload or {})
    return SpatialSupport(
        bbox=list(data["bbox"]) if data.get("bbox") else None,
        bbox_format=str(data.get("bbox_format", "xyxy")),
        frame_id=data.get("frame_id"),
        geo_center=list(data["geo_center"]) if data.get("geo_center") else None,
        pixel_size_m=data.get("pixel_size_m"),
    )


def observation_from_dict(payload: Mapping[str, Any]) -> PrimitiveObservation:
    data = dict(payload)
    observation = PrimitiveObservation(
        observation_id=str(data["observation_id"]),
        modality=Modality.coerce(data.get("modality")),
        source_tool=str(data.get("source_tool", "unknown")),
        acquisition_id=data.get("acquisition_id"),
        time_scope=_time_scope(data.get("time_scope") or {}),
        spatial_support=_spatial_support(data.get("spatial_support") or {}),
        context=dict(data.get("context") or {}),
        claim_ids=list(data.get("claim_ids") or []),
        input_roots=list(data.get("input_roots") or []),
        derivation_parents=list(data.get("derivation_parents") or []),
        candidate_object_ids=list(data.get("candidate_object_ids") or []),
        detection_confidence=data.get("detection_confidence"),
        round_id=data.get("round_id"),
    )
    observation.validate()
    return observation


def claim_from_dict(payload: Mapping[str, Any]) -> Claim:
    data = dict(payload)
    claim = Claim(
        claim_id=str(data["claim_id"]),
        observation_id=str(data["observation_id"]),
        attribute=str(data["attribute"]),
        value=data.get("value"),
        time_scope=_time_scope(data.get("time_scope") or {}),
        source_tool=str(data.get("source_tool", "unknown")),
        source_channel=str(data.get("source_channel", "unknown")),
        raw_confidence=data.get("raw_confidence"),
        context=dict(data.get("context") or {}),
        provenance_group_id=data.get("provenance_group_id"),
    )
    claim.validate()
    return claim


def object_from_dict(payload: Mapping[str, Any]) -> ObjectHypothesis:
    data = dict(payload)
    hypothesis = ObjectHypothesis(
        object_id=str(data["object_id"]),
        spatial_support=_spatial_support(data.get("spatial_support") or {}),
        time_scopes=list(data.get("time_scopes") or []),
        attribute_hints=dict(data.get("attribute_hints") or {}),
        capacity=data.get("capacity"),
        active=bool(data.get("active", True)),
        confidence=data.get("confidence"),
        context=dict(data.get("context") or {}),
    )
    hypothesis.validate()
    return hypothesis


def edge_from_dict(payload: Mapping[str, Any]) -> CandidateEdge:
    data = dict(payload)
    edge = CandidateEdge(
        observation_id=str(data["observation_id"]),
        object_id=str(data["object_id"]),
        features={str(k): float(v) for k, v in (data.get("features") or {}).items()},
        physical_logit=float(data.get("physical_logit", 0.0)),
        prior_probability=float(data.get("prior_probability", 0.0)),
        admissible=bool(data.get("admissible", True)),
        gate_reason=str(data.get("gate_reason", "")),
    )
    edge.validate()
    return edge


def relation_from_dict(payload: Mapping[str, Any]) -> Relation:
    data = dict(payload)
    relation = Relation(
        relation_id=str(data["relation_id"]),
        relation_type=RelationType.coerce(data.get("relation_type")),
        source_observation_id=data.get("source_observation_id"),
        target_observation_id=data.get("target_observation_id"),
        strength=float(data.get("strength", 0.5)),
        source_tool=str(data.get("source_tool", "unknown")),
        context=dict(data.get("context") or {}),
    )
    relation.validate()
    return relation


def inference_state_from_dict(payload: Mapping[str, Any]) -> InferenceState:
    data = dict(payload or {})
    truth: Dict[Any, Dict[Any, float]] = {}
    for key, distribution in (data.get("truth") or {}).items():
        parts = str(key).split("|")
        if len(parts) == 3:
            truth[(parts[0], parts[1], parts[2])] = {
                k: float(v) for k, v in (distribution or {}).items()
            }
    return InferenceState(
        association={
            str(k): {str(ok): float(ov) for ok, ov in (v or {}).items()}
            for k, v in (data.get("association") or {}).items()
        },
        truth=truth,
        reliability={str(k): float(v) for k, v in (data.get("reliability") or {}).items()},
        objective_history=[float(v) for v in (data.get("objective_history") or [])],
        max_delta_history=[float(v) for v in (data.get("max_delta_history") or [])],
        iterations=int(data.get("iterations", 0)),
        converged=bool(data.get("converged", False)),
        diagnostics=dict(data.get("diagnostics") or {}),
    )


def store_from_dict(
    payload: Mapping[str, Any],
    store_cls: Type[EvidenceStore],
) -> EvidenceStore:
    """Restore an :class:`EvidenceStore` from :func:`store_to_dict` output."""
    data = dict(payload)
    version = str(data.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported evidence store schema_version {version!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )

    store = store_cls(config=data.get("config"))
    for item in data.get("observations") or []:
        observation = observation_from_dict(item)
        store.observations[observation.observation_id] = observation
    for item in data.get("claims") or []:
        claim = claim_from_dict(item)
        store.claims[claim.claim_id] = claim
    for item in data.get("objects") or []:
        hypothesis = object_from_dict(item)
        store.objects[hypothesis.object_id] = hypothesis
    store.candidate_edges = [edge_from_dict(item) for item in data.get("candidate_edges") or []]
    for item in data.get("relations") or []:
        relation = relation_from_dict(item)
        store.relations[relation.relation_id] = relation
    for item in data.get("provenance_nodes") or []:
        node_data = dict(item)
        node = ProvenanceNode(
            node_id=str(node_data["node_id"]),
            kind=str(node_data.get("kind", "observation")),
            tool_name=str(node_data.get("tool_name", "unknown")),
            fingerprint=str(node_data.get("fingerprint", "")),
            parents=list(node_data.get("parents") or []),
            input_roots=list(node_data.get("input_roots") or []),
            group_id=node_data.get("group_id"),
            round_id=node_data.get("round_id"),
            duplicate_of=node_data.get("duplicate_of"),
        )
        node.validate()
        store.provenance.nodes[node.node_id] = node
        if node.fingerprint and not node.duplicate_of:
            store.provenance._fingerprint_owner.setdefault(node.fingerprint, node.node_id)
    store.coverage = {str(k): int(v) for k, v in (data.get("coverage") or {}).items()}
    store.ingest_log = list(data.get("ingest_log") or [])
    store.duplicate_ingests = int(data.get("duplicate_ingests", 0))
    store._call_fingerprints = set(data.get("call_fingerprints") or [])
    store._call_owners = {str(k): str(v) for k, v in (data.get("call_owners") or {}).items()}
    # Defaults are the pre-revision behavior (0 / -1), so an older checkpoint
    # simply re-runs inference once instead of trusting a possibly stale state.
    store._revision = int(data.get("evidence_revision", 0))
    store._inference_revision = int(data.get("inference_revision", -1))
    # Old checkpoints predate the flag; the conservative default preserves the
    # historical freezing behavior only when objects were explicitly supplied.
    store.autoseed_objects = bool(
        data.get("autoseed_objects", not data.get("objects"))
    )
    if data.get("inference_state"):
        store.inference_state = inference_state_from_dict(data["inference_state"])
    return store


__all__ = [
    "store_to_dict",
    "store_from_dict",
    "observation_from_dict",
    "claim_from_dict",
    "object_from_dict",
    "edge_from_dict",
    "relation_from_dict",
    "inference_state_from_dict",
]
