"""Shared-identity truth discovery layer for OneID.

An object-level evidence reasoning layer sitting on top of the existing
perception workers:

.. code-block:: text

    tool output
        -> PrimitiveObservation + Claim
        -> Physical Candidate Graph + NULL
        -> Shared Identity A[m,k] <-> Object Truth Q[k,a,t,v]
        -> Reliability + Provenance
        -> Persistent Evidence State
        -> Decision / Acquisition

The invariant that defines this layer: all claims produced by one
``PrimitiveObservation`` read the same row ``A[m, :]``.  A ``Claim`` never owns an
object association.
"""
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    SCHEMA_VERSION,
    AcquisitionRequest,
    CandidateEdge,
    Claim,
    ClaimAttribute,
    DecisionResult,
    EvidenceState,
    InferenceProblem,
    InferenceState,
    Modality,
    ObjectEvidenceSummary,
    ObjectHypothesis,
    PrimitiveObservation,
    Relation,
    RelationType,
    SchemaValidationError,
    SpatialSupport,
    TimeScope,
)

__all__ = [
    "NULL_OBJECT_ID",
    "SCHEMA_VERSION",
    "AcquisitionRequest",
    "CandidateEdge",
    "Claim",
    "ClaimAttribute",
    "DecisionResult",
    "EvidenceState",
    "InferenceProblem",
    "InferenceState",
    "Modality",
    "ObjectEvidenceSummary",
    "ObjectHypothesis",
    "PrimitiveObservation",
    "Relation",
    "RelationType",
    "SchemaValidationError",
    "SpatialSupport",
    "TimeScope",
    "load_config",
]
