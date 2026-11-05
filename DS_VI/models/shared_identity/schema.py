"""Object-level evidence schema for shared-identity truth discovery.

The central structural invariant enforced here: a :class:`Claim` carries only an
``observation_id``.  It has no ``object_id`` and no association vector.  All
claims produced by one :class:`PrimitiveObservation` therefore read the same row
``A[m, :]`` of the association matrix.

Only the standard library is used so that these types can be imported in
GPU-free, network-free unit tests.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Reserved object id for "this observation belongs to no tracked object".
NULL_OBJECT_ID = "__NULL__"

SCHEMA_VERSION = "1.0"


class SchemaValidationError(ValueError):
    """Raised when evidence objects violate a structural or numeric invariant."""


class Modality(str, Enum):
    """Sensing modality of an observation."""

    OPTICAL = "optical"
    SAR = "sar"
    TEMPORAL = "temporal"
    CROSS_MODAL = "cross_modal"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: Any) -> Modality:
        text = str(value or "").strip().lower()
        aliases = {
            "rgb": cls.OPTICAL,
            "optical": cls.OPTICAL,
            "eo": cls.OPTICAL,
            "visible": cls.OPTICAL,
            "sar": cls.SAR,
            "radar": cls.SAR,
            "temporal": cls.TEMPORAL,
            "change": cls.TEMPORAL,
            "time": cls.TEMPORAL,
            "cross_modal": cls.CROSS_MODAL,
            "crossmodal": cls.CROSS_MODAL,
        }
        return aliases.get(text, cls.UNKNOWN)


class ClaimAttribute(str, Enum):
    """Attribute axes that claims and object truth are defined over."""

    CATEGORY = "category"
    SUBTYPE = "subtype"
    SIZE_BIN = "size_bin"
    STATE = "state"
    #: Surface colour of the target (``white`` / ``gray`` / ``multi`` / ...).
    #: The 2026-11 property workbooks annotate this axis; it replaced ``state``
    #: on the instances sheet, so both spellings stay recognised.
    COLOR = "color"
    HEADING = "heading"
    #: Image-plane orientation of the target box, as the tools report it:
    #: ``horizontal`` / ``vertical`` / ``diagonal``.  Distinct from ``heading``
    #: (a compass direction) — the property workbooks and detectors use this
    #: axis, so mapping it onto ``heading`` would put two different label spaces
    #: into one truth variable.
    ORIENTATION = "orientation"
    PRESENCE = "presence"
    MOTION = "motion"

    @classmethod
    def coerce(cls, value: Any) -> str:
        """Return a canonical attribute string.

        Unknown attributes are allowed (returned normalized) so that new tools
        can contribute evidence without a schema change, but the known axes get
        stable spellings.
        """
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "class": cls.CATEGORY.value,
            "label": cls.CATEGORY.value,
            "category": cls.CATEGORY.value,
            "type": cls.CATEGORY.value,
            "subtype": cls.SUBTYPE.value,
            "subclass": cls.SUBTYPE.value,
            "fine_grained": cls.SUBTYPE.value,
            "color": cls.COLOR.value,
            "color_name": cls.COLOR.value,
            "colour": cls.COLOR.value,
            "state": cls.STATE.value,
            "status": cls.STATE.value,
            "heading": cls.HEADING.value,
            "direction": cls.HEADING.value,
            "orientation": cls.ORIENTATION.value,
            "exists": cls.PRESENCE.value,
            "existence": cls.PRESENCE.value,
            "motion": cls.MOTION.value,
            "moved": cls.MOTION.value,
            "movement": cls.MOTION.value,
        }
        if not text:
            raise SchemaValidationError("Claim attribute must be a non-empty string")
        return aliases.get(text, text)


class RelationType(str, Enum):
    """Temporal / structural relation kinds between observations."""

    STAYED = "stayed"
    LEFT = "left"
    APPEARED = "appeared"
    MOVED = "moved"
    SAME_SCENE = "same_scene"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: Any) -> RelationType:
        text = str(value or "").strip().lower()
        try:
            return cls(text)
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class TimeScope:
    """Discrete time bucket an observation or truth variable refers to.

    ``key`` is the canonical identifier used when indexing ``Q[k, a, t, v]``.
    """

    key: str = "t0"
    timestamp: Optional[float] = None
    label: Optional[str] = None

    def validate(self) -> None:
        if not str(self.key or "").strip():
            raise SchemaValidationError("TimeScope.key must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpatialSupport:
    """Where an observation lives in image and (optionally) world coordinates."""

    bbox: Optional[List[float]] = None
    bbox_format: str = "xyxy"
    frame_id: Optional[str] = None
    geo_center: Optional[List[float]] = None
    pixel_size_m: Optional[float] = None

    def validate(self) -> None:
        if self.bbox is not None:
            if not isinstance(self.bbox, (list, tuple)) or len(self.bbox) < 4:
                raise SchemaValidationError("SpatialSupport.bbox must have 4 numbers")
            try:
                x1, y1, x2, y2 = (float(v) for v in self.bbox[:4])
            except (TypeError, ValueError) as exc:
                raise SchemaValidationError(f"SpatialSupport.bbox is not numeric: {exc}") from exc
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                raise SchemaValidationError("SpatialSupport.bbox must be finite")
            if x2 <= x1 or y2 <= y1:
                raise SchemaValidationError(f"SpatialSupport.bbox is degenerate: {self.bbox}")
        if self.geo_center is not None:
            if not isinstance(self.geo_center, (list, tuple)) or len(self.geo_center) < 2:
                raise SchemaValidationError("SpatialSupport.geo_center must have 2 numbers")
        if self.pixel_size_m is not None and float(self.pixel_size_m) <= 0.0:
            raise SchemaValidationError("SpatialSupport.pixel_size_m must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrimitiveObservation:
    """One indivisible perception act: a detection box, a crop, a direct answer.

    An observation is the *identity unit*.  ``A`` has exactly one row per
    observation, and every claim derived from this observation shares that row.
    """

    observation_id: str
    modality: Modality = Modality.UNKNOWN
    source_tool: str = "unknown"
    acquisition_id: Optional[str] = None
    time_scope: TimeScope = field(default_factory=TimeScope)
    spatial_support: SpatialSupport = field(default_factory=SpatialSupport)
    context: Dict[str, Any] = field(default_factory=dict)
    claim_ids: List[str] = field(default_factory=list)
    input_roots: List[str] = field(default_factory=list)
    derivation_parents: List[str] = field(default_factory=list)
    candidate_object_ids: List[str] = field(default_factory=list)
    detection_confidence: Optional[float] = None
    round_id: Optional[int] = None

    def validate(self) -> None:
        if not str(self.observation_id or "").strip():
            raise SchemaValidationError("PrimitiveObservation.observation_id must be non-empty")
        if not isinstance(self.modality, Modality):
            raise SchemaValidationError("PrimitiveObservation.modality must be a Modality")
        self.time_scope.validate()
        self.spatial_support.validate()
        if self.detection_confidence is not None:
            conf = float(self.detection_confidence)
            if not 0.0 <= conf <= 1.0:
                raise SchemaValidationError(
                    f"detection_confidence out of [0,1]: {self.detection_confidence}"
                )
        if NULL_OBJECT_ID in self.candidate_object_ids[:-1]:
            # NULL is allowed but must not shadow a real candidate ordering.
            pass

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["modality"] = self.modality.value
        return data


@dataclass
class Claim:
    """A single attribute assertion produced by one observation.

    Deliberately has **no** ``object_id`` and **no** association field.  Identity
    is resolved once per observation, never per claim.
    """

    claim_id: str
    observation_id: str
    attribute: str
    value: Any
    time_scope: TimeScope = field(default_factory=TimeScope)
    source_tool: str = "unknown"
    source_channel: str = "unknown"
    raw_confidence: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    provenance_group_id: Optional[str] = None

    # Guard against reintroducing the forbidden fields via **kwargs plumbing.
    FORBIDDEN_FIELDS = ("object_id", "association", "association_vector", "assigned_object")

    def validate(self) -> None:
        if not str(self.claim_id or "").strip():
            raise SchemaValidationError("Claim.claim_id must be non-empty")
        if not str(self.observation_id or "").strip():
            raise SchemaValidationError("Claim.observation_id must be non-empty")
        self.attribute = ClaimAttribute.coerce(self.attribute)
        if self.value is None or str(self.value).strip() == "":
            raise SchemaValidationError(f"Claim {self.claim_id} has an empty value")
        self.time_scope.validate()
        if self.raw_confidence is not None:
            conf = float(self.raw_confidence)
            if not 0.0 <= conf <= 1.0:
                raise SchemaValidationError(
                    f"Claim {self.claim_id} raw_confidence out of [0,1]: {conf}"
                )
        for forbidden in self.FORBIDDEN_FIELDS:
            if forbidden in self.context:
                raise SchemaValidationError(
                    f"Claim {self.claim_id} must not carry '{forbidden}'. "
                    "Identity is shared per observation, not per claim."
                )

    @property
    def truth_key(self) -> Tuple[str, str]:
        """``(attribute, time_key)`` addressing of the truth variable."""
        return (self.attribute, self.time_scope.key)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectHypothesis:
    """A candidate physical object that observations may be associated with."""

    object_id: str
    spatial_support: SpatialSupport = field(default_factory=SpatialSupport)
    time_scopes: List[str] = field(default_factory=list)
    attribute_hints: Dict[str, Any] = field(default_factory=dict)
    capacity: Optional[int] = None
    active: bool = True
    confidence: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.object_id or "").strip():
            raise SchemaValidationError("ObjectHypothesis.object_id must be non-empty")
        if self.object_id == NULL_OBJECT_ID:
            raise SchemaValidationError(
                f"'{NULL_OBJECT_ID}' is reserved and must not be a real ObjectHypothesis"
            )
        self.spatial_support.validate()
        if self.capacity is not None and int(self.capacity) < 1:
            raise SchemaValidationError("ObjectHypothesis.capacity must be >= 1 when set")
        if self.confidence is not None and not 0.0 <= float(self.confidence) <= 1.0:
            raise SchemaValidationError("ObjectHypothesis.confidence must be in [0,1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEdge:
    """A potential observation -> object association with its physical prior."""

    observation_id: str
    object_id: str
    features: Dict[str, float] = field(default_factory=dict)
    physical_logit: float = 0.0
    prior_probability: float = 0.0
    admissible: bool = True
    gate_reason: str = ""

    @property
    def is_null(self) -> bool:
        return self.object_id == NULL_OBJECT_ID

    def validate(self) -> None:
        if not str(self.observation_id or "").strip():
            raise SchemaValidationError("CandidateEdge.observation_id must be non-empty")
        if not str(self.object_id or "").strip():
            raise SchemaValidationError("CandidateEdge.object_id must be non-empty")
        if not 0.0 <= float(self.prior_probability) <= 1.0 + 1e-9:
            raise SchemaValidationError(
                f"CandidateEdge prior_probability out of [0,1]: {self.prior_probability}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Relation:
    """Relation evidence between two observations (e.g. temporal continuity).

    Relations adjust priors and coverage.  They never create a deterministic
    identity link.
    """

    relation_id: str
    relation_type: RelationType
    source_observation_id: Optional[str] = None
    target_observation_id: Optional[str] = None
    strength: float = 0.5
    source_tool: str = "unknown"
    context: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.relation_id or "").strip():
            raise SchemaValidationError("Relation.relation_id must be non-empty")
        if not isinstance(self.relation_type, RelationType):
            raise SchemaValidationError("Relation.relation_type must be a RelationType")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise SchemaValidationError(f"Relation.strength out of [0,1]: {self.strength}")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["relation_type"] = self.relation_type.value
        return data


@dataclass
class ProvenanceNode:
    """A node in the evidence derivation DAG."""

    node_id: str
    kind: str = "observation"
    tool_name: str = "unknown"
    fingerprint: str = ""
    parents: List[str] = field(default_factory=list)
    input_roots: List[str] = field(default_factory=list)
    group_id: Optional[str] = None
    round_id: Optional[int] = None
    duplicate_of: Optional[str] = None

    def validate(self) -> None:
        if not str(self.node_id or "").strip():
            raise SchemaValidationError("ProvenanceNode.node_id must be non-empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceState:
    """Result of one joint A-Q inference run.

    ``association`` maps ``observation_id -> {object_id: probability}`` with each
    row summing to 1.  ``truth`` maps ``(object_id, attribute, time_key) ->
    {value: probability}``, each normalized independently.
    """

    association: Dict[str, Dict[str, float]] = field(default_factory=dict)
    truth: Dict[Tuple[str, str, str], Dict[Any, float]] = field(default_factory=dict)
    reliability: Dict[str, float] = field(default_factory=dict)
    objective_history: List[float] = field(default_factory=list)
    max_delta_history: List[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def association_row(self, observation_id: str) -> Dict[str, float]:
        return dict(self.association.get(observation_id) or {})

    def top_object(self, observation_id: str) -> Tuple[Optional[str], float]:
        row = self.association.get(observation_id) or {}
        if not row:
            return (None, 0.0)
        object_id, probability = max(row.items(), key=lambda item: item[1])
        return (object_id, float(probability))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "association": {k: dict(v) for k, v in self.association.items()},
            "truth": {
                f"{key[0]}|{key[1]}|{key[2]}": {str(val): float(p) for val, p in dist.items()}
                for key, dist in self.truth.items()
            },
            "reliability": dict(self.reliability),
            "objective_history": list(self.objective_history),
            "max_delta_history": list(self.max_delta_history),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class ObjectEvidenceSummary:
    """Per-object rollup used by the decision layer."""

    object_id: str
    association_support: float = 0.0
    supporting_observations: List[str] = field(default_factory=list)
    truth_beliefs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    provenance_summary: Dict[str, Any] = field(default_factory=dict)
    ambiguity: Dict[str, float] = field(default_factory=dict)
    contradictions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceState:
    """Diagnostic snapshot of the evidence store after inference."""

    objects: Dict[str, ObjectEvidenceSummary] = field(default_factory=dict)
    observation_count: int = 0
    claim_count: int = 0
    coverage: Dict[str, Any] = field(default_factory=dict)
    null_mass: Dict[str, float] = field(default_factory=dict)
    inference_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "observation_count": int(self.observation_count),
            "claim_count": int(self.claim_count),
            "coverage": dict(self.coverage),
            "null_mass": dict(self.null_mass),
            "inference_diagnostics": dict(self.inference_diagnostics),
        }


@dataclass
class DecisionResult:
    """Task-level decision derived from an :class:`EvidenceState`."""

    status: str
    object_id: Optional[str] = None
    attribute: Optional[str] = None
    value: Optional[Any] = None
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    evidence_refs: Dict[str, Any] = field(default_factory=dict)

    #: The four-valued evidence state of the paper (Appendix C.4):
    #: Confirmed (complete support), Weak Support (the remainder),
    #: Conflict (validated active conflict), Uncertain (unresolved).
    VALID_STATUSES = (
        "Confirmed",
        "Weak Support",
        "Conflict",
        "Uncertain",
    )

    def validate(self) -> None:
        if self.status not in self.VALID_STATUSES:
            raise SchemaValidationError(
                f"DecisionResult.status must be one of {self.VALID_STATUSES}, got {self.status!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AcquisitionRequest:
    """Rule-based suggestion for the next piece of evidence to acquire."""

    request_type: str
    target_object_id: Optional[str] = None
    attribute: Optional[str] = None
    modality: Optional[str] = None
    time_key: Optional[str] = None
    rationale: str = ""
    priority: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceProblem:
    """Self-contained input bundle for :func:`inference.run_joint_inference`."""

    observations: List[PrimitiveObservation] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    objects: List[ObjectHypothesis] = field(default_factory=list)
    candidate_edges: List[CandidateEdge] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    provenance_weights: Dict[str, float] = field(default_factory=dict)
    provenance_budget_enabled: bool = True
    value_domains: Dict[str, List[Any]] = field(default_factory=dict)
    capacities: Dict[str, int] = field(default_factory=dict)
    truth_priors: Dict[Tuple[str, str, str], Dict[Any, float]] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate every element and all cross-references."""
        observation_ids = set()
        for observation in self.observations:
            observation.validate()
            if observation.observation_id in observation_ids:
                raise SchemaValidationError(
                    f"Duplicate observation_id: {observation.observation_id}"
                )
            observation_ids.add(observation.observation_id)

        object_ids = set()
        for obj in self.objects:
            obj.validate()
            if obj.object_id in object_ids:
                raise SchemaValidationError(f"Duplicate object_id: {obj.object_id}")
            object_ids.add(obj.object_id)

        claim_ids = set()
        groups: Dict[str, set] = {}
        for claim in self.claims:
            claim.validate()
            if claim.claim_id in claim_ids:
                raise SchemaValidationError(f"Duplicate claim_id: {claim.claim_id}")
            claim_ids.add(claim.claim_id)
            if claim.observation_id not in observation_ids:
                raise SchemaValidationError(
                    f"Claim {claim.claim_id} references unknown observation "
                    f"{claim.observation_id}"
                )
            if claim.provenance_group_id:
                groups.setdefault(claim.provenance_group_id, set()).add(claim.observation_id)

        for edge in self.candidate_edges:
            edge.validate()
            if edge.observation_id not in observation_ids:
                raise SchemaValidationError(
                    f"CandidateEdge references unknown observation {edge.observation_id}"
                )
            if not edge.is_null and edge.object_id not in object_ids:
                raise SchemaValidationError(
                    f"CandidateEdge references unknown object {edge.object_id}"
                )

        for relation in self.relations:
            relation.validate()
            for ref in (relation.source_observation_id, relation.target_observation_id):
                if ref is not None and ref not in observation_ids:
                    raise SchemaValidationError(
                        f"Relation {relation.relation_id} references unknown observation {ref}"
                    )

        for group_id, weight_sum in _group_weight_sums(self.claims, self.provenance_weights).items():
            if self.provenance_budget_enabled and weight_sum > 1.0 + 1e-6:
                raise SchemaValidationError(
                    f"Provenance group '{group_id}' weight sum {weight_sum:.6f} exceeds 1.0"
                )

    def claims_by_observation(self) -> Dict[str, List[Claim]]:
        """Group claims by their observation, preserving insertion order."""
        grouped: Dict[str, List[Claim]] = {
            observation.observation_id: [] for observation in self.observations
        }
        for claim in self.claims:
            grouped.setdefault(claim.observation_id, []).append(claim)
        return grouped

    def object_ids(self, include_null: bool = True) -> List[str]:
        ids = [obj.object_id for obj in self.objects if obj.active]
        if include_null:
            ids.append(NULL_OBJECT_ID)
        return ids


def _group_weight_sums(
    claims: Sequence[Claim],
    provenance_weights: Mapping[str, float],
) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    for claim in claims:
        group_id = claim.provenance_group_id
        if not group_id:
            continue
        weight = float(provenance_weights.get(claim.claim_id, 0.0))
        sums[group_id] = sums.get(group_id, 0.0) + weight
    return sums


def stable_id(prefix: str, payload: Any) -> str:
    """Deterministic short id derived from ``payload``.

    Used everywhere ids must be reproducible across runs so that exact replay
    of the same tool call yields the same observation and claim ids.
    """
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def fingerprint(payload: Any) -> str:
    """Stable content fingerprint for replay detection."""
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


__all__ = [
    "NULL_OBJECT_ID",
    "SCHEMA_VERSION",
    "SchemaValidationError",
    "Modality",
    "ClaimAttribute",
    "RelationType",
    "TimeScope",
    "SpatialSupport",
    "PrimitiveObservation",
    "Claim",
    "ObjectHypothesis",
    "CandidateEdge",
    "Relation",
    "ProvenanceNode",
    "InferenceState",
    "ObjectEvidenceSummary",
    "EvidenceState",
    "DecisionResult",
    "AcquisitionRequest",
    "InferenceProblem",
    "stable_id",
    "fingerprint",
]
