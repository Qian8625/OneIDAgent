"""Rule-based next-evidence suggestions.

Version 1 is deliberately three simple rules — no information-gain optimization:

* high association entropy -> ask for an attribute that separates the competing
  candidate identities
* high truth entropy -> ask for more evidence on that attribute
* missing coverage -> ask for the absent modality or time observation
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.schema import (
    AcquisitionRequest,
    EvidenceState,
    Modality,
)

EXPECTED_MODALITIES = (Modality.OPTICAL.value, Modality.SAR.value)

#: Signature of one suggestion for de-duplication against already attempted
#: acquisitions: ``(request_type, modality, attribute, time_key)``.
RequestSignature = Tuple[str, Optional[str], Optional[str], Optional[str]]


def request_signature(request: AcquisitionRequest) -> RequestSignature:
    return (
        str(request.request_type),
        request.modality,
        request.attribute,
        request.time_key,
    )


def _capability_matches(
    request: AcquisitionRequest,
    capabilities: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    """Return the cheapest matching action cost, or None when unsupported."""
    best: Optional[float] = None
    for capability in capabilities:
        if str(capability.get("request_type") or "") != request.request_type:
            continue
        if capability.get("modality") and capability.get("modality") != request.modality:
            continue
        if capability.get("attribute") and capability.get("attribute") != request.attribute:
            continue
        cost = float(capability.get("cost", 0.0) or 0.0)
        if best is None or cost < best:
            best = cost
    return best


def suggest_acquisitions(
    evidence_state: EvidenceState,
    config: Optional[Mapping[str, Any]] = None,
    *,
    capabilities: Optional[Sequence[Mapping[str, Any]]] = None,
    budget: Optional[float] = None,
    queried: Optional[Iterable[RequestSignature]] = None,
) -> List[AcquisitionRequest]:
    """Produce prioritized :class:`AcquisitionRequest` objects.

    Requests are sorted by descending priority and truncated to
    ``acquisition.max_requests``. When a public action catalogue is supplied the
    list is restricted to requests some action can satisfy and the remaining
    budget can afford; ``queried`` drops signatures already attempted, so the
    same missing-modality or attribute request is not repeated forever.
    """
    cfg = dict(config or load_config())
    section = dict(cfg.get("acquisition") or {})
    association_trigger = float(section.get("association_entropy_trigger", 0.60))
    truth_trigger = float(section.get("truth_entropy_trigger", 0.60))
    max_requests = int(section.get("max_requests", 5))

    requests: List[AcquisitionRequest] = []

    for object_id, summary in evidence_state.objects.items():
        association_entropy = float(summary.ambiguity.get("association_entropy", 0.0))

        # Rule 1: identity is ambiguous -> request a discriminative attribute.
        if association_entropy >= association_trigger:
            discriminative = _most_discriminative_attribute(summary.truth_beliefs)
            requests.append(
                AcquisitionRequest(
                    request_type="discriminative_attribute",
                    target_object_id=object_id,
                    attribute=discriminative,
                    rationale=(
                        f"association entropy {association_entropy:.3f} >= "
                        f"{association_trigger:.3f}; a '{discriminative}' observation "
                        "would separate the competing candidate identities"
                    ),
                    priority=association_entropy,
                )
            )

        # Rule 2: a truth axis is undecided -> request more evidence for it.
        for belief in summary.truth_beliefs.values():
            normalized = float(belief.get("normalized_entropy", 0.0))
            if normalized >= truth_trigger:
                requests.append(
                    AcquisitionRequest(
                        request_type="attribute_evidence",
                        target_object_id=object_id,
                        attribute=belief.get("attribute"),
                        time_key=belief.get("time_key"),
                        rationale=(
                            f"normalized truth entropy {normalized:.3f} >= "
                            f"{truth_trigger:.3f} for "
                            f"{belief.get('attribute')}@{belief.get('time_key')}"
                        ),
                        priority=normalized,
                    )
                )

        # Rule 3: a modality or time cell is missing -> request that observation.
        observed_modalities = set((summary.coverage.get("modalities") or {}).keys())
        for modality in EXPECTED_MODALITIES:
            if modality not in observed_modalities:
                requests.append(
                    AcquisitionRequest(
                        request_type="missing_modality",
                        target_object_id=object_id,
                        modality=modality,
                        rationale=f"no {modality} observation supports this object",
                        priority=0.5,
                    )
                )

    requests.sort(key=lambda item: item.priority, reverse=True)
    requests = requests[:max_requests]

    if capabilities is not None:
        filtered: List[AcquisitionRequest] = []
        for request in requests:
            cost = _capability_matches(request, capabilities)
            if cost is None:
                continue
            if budget is not None and cost > budget + 1e-9:
                continue
            filtered.append(request)
        requests = filtered

    if queried:
        attempted: Set[RequestSignature] = set(queried)
        requests = [
            request for request in requests if request_signature(request) not in attempted
        ]
    return requests


def _most_discriminative_attribute(truth_beliefs: Mapping[str, Mapping[str, Any]]) -> str:
    """Pick the attribute whose belief is most concentrated.

    A sharply-known attribute is the best discriminator: comparing it against a
    candidate object is informative, whereas asking about an attribute the model
    is already unsure of adds little.
    """
    best_attribute = "subtype"
    best_probability = -1.0
    for belief in truth_beliefs.values():
        probability = float(belief.get("probability", 0.0))
        if probability > best_probability:
            best_probability = probability
            best_attribute = str(belief.get("attribute") or best_attribute)
    return best_attribute


def summarize_requests(requests: List[AcquisitionRequest]) -> Dict[str, Any]:
    """Compact digest for logs and LLM context.

    Keeps the target, time axis and rationale: dropping them made the digest
    unusable for auditing why an acquisition was proposed.
    """
    return {
        "count": len(requests),
        "requests": [
            {
                "type": request.request_type,
                "object_id": request.target_object_id,
                "attribute": request.attribute,
                "modality": request.modality,
                "time_key": request.time_key,
                "priority": round(float(request.priority), 4),
                "rationale": request.rationale,
            }
            for request in requests
        ],
    }


__all__ = [
    "suggest_acquisitions",
    "summarize_requests",
    "request_signature",
    "EXPECTED_MODALITIES",
]
