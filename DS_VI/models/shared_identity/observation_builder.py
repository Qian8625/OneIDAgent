"""Helpers for turning raw tool output into observations and claims.

Adapters in :mod:`shared_identity.adapters` call into this module so that id
generation, bbox normalization, and claim construction stay consistent across
optical, SAR, cross-modal, and temporal sources.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from DS_VI.models.shared_identity.physical_features import clamp_unit, normalize_bbox
from DS_VI.models.shared_identity.schema import (
    ObjectHypothesis,
    Claim,
    ClaimAttribute,
    Modality,
    PrimitiveObservation,
    SpatialSupport,
    TimeScope,
    stable_id,
)
from DS_VI.models.shared_identity.scope import (
    image_group_id,
    observations_are_related,
    relation_group_id,
)


def build_observation(
    source_tool: str,
    modality: Any,
    bbox: Optional[Sequence[float]] = None,
    time_key: str = "t0",
    acquisition_id: Optional[str] = None,
    detection_confidence: Optional[float] = None,
    frame_id: Optional[str] = None,
    geo_center: Optional[Sequence[float]] = None,
    pixel_size_m: Optional[float] = None,
    input_roots: Optional[Sequence[str]] = None,
    derivation_parents: Optional[Sequence[str]] = None,
    context: Optional[Mapping[str, Any]] = None,
    round_id: Optional[int] = None,
    index: Optional[int] = None,
) -> PrimitiveObservation:
    """Construct one :class:`PrimitiveObservation` with a deterministic id.

    The id is a stable hash over the identifying content (tool, modality, bbox,
    time, acquisition, index) so that replaying the same tool call produces the
    same ``observation_id`` and the replay dedup logic can recognize it.
    """
    normalized_bbox = normalize_bbox(bbox)
    resolved_modality = (
        modality if isinstance(modality, Modality) else Modality.coerce(modality)
    )
    observation_id = stable_id(
        "obs",
        {
            "tool": str(source_tool),
            "modality": resolved_modality.value,
            "bbox": normalized_bbox,
            "time": str(time_key),
            "acquisition": acquisition_id,
            "frame": frame_id,
            "index": index,
        },
    )
    observation = PrimitiveObservation(
        observation_id=observation_id,
        modality=resolved_modality,
        source_tool=str(source_tool),
        acquisition_id=acquisition_id,
        time_scope=TimeScope(key=str(time_key)),
        spatial_support=SpatialSupport(
            bbox=normalized_bbox,
            frame_id=frame_id,
            geo_center=list(geo_center) if geo_center else None,
            pixel_size_m=float(pixel_size_m) if pixel_size_m else None,
        ),
        context=dict(context or {}),
        input_roots=list(input_roots or []),
        derivation_parents=list(derivation_parents or []),
        detection_confidence=(
            clamp_unit(detection_confidence) if detection_confidence is not None else None
        ),
        round_id=round_id,
    )
    observation.validate()
    return observation


def build_claim(
    observation: PrimitiveObservation,
    attribute: str,
    value: Any,
    source_channel: str = "unknown",
    raw_confidence: Optional[float] = None,
    time_key: Optional[str] = None,
    context: Optional[Mapping[str, Any]] = None,
    provenance_group_id: Optional[str] = None,
    source_tool: Optional[str] = None,
) -> Claim:
    """Construct one :class:`Claim` bound to ``observation`` by id only.

    The claim records ``observation_id`` and nothing about object identity.  Its
    id is appended to ``observation.claim_ids`` so the shared row can be found
    from either direction.

    ``source_tool`` overrides the observation's tool for claims produced by a
    different primitive source (for example a VLM describing the same crop). It
    defaults to ``observation.source_tool``, and it only enters ``claim_id`` when
    supplied, so existing claim ids are unchanged.
    """
    resolved_attribute = ClaimAttribute.coerce(attribute)
    resolved_time = str(time_key or observation.time_scope.key)
    resolved_tool = str(source_tool or observation.source_tool)
    material = {
        "observation": observation.observation_id,
        "attribute": resolved_attribute,
        "value": value,
        "time": resolved_time,
        "channel": str(source_channel),
    }
    if source_tool:
        material["tool"] = resolved_tool
    claim_id = stable_id("clm", material)
    claim = Claim(
        claim_id=claim_id,
        observation_id=observation.observation_id,
        attribute=resolved_attribute,
        value=value,
        time_scope=TimeScope(key=resolved_time),
        source_tool=resolved_tool,
        source_channel=str(source_channel),
        raw_confidence=(
            clamp_unit(raw_confidence) if raw_confidence is not None else None
        ),
        context=dict(context or {}),
        provenance_group_id=provenance_group_id,
    )
    claim.validate()
    if claim.claim_id not in observation.claim_ids:
        observation.claim_ids.append(claim.claim_id)
    return claim


def extract_detections(payload: Any) -> List[Dict[str, Any]]:
    """Pull a detection list out of the many shapes tool responses take.

    Accepts a bare list, a ``DetectionSet``-style dict, a full worker response
    with a ``structured`` block, or the legacy ``data`` alias.  Returns ``[]``
    rather than raising when nothing detection-like is present.
    """
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, Mapping):
        return []
    for key in ("detections", "selected_detections", "sar_candidates", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    structured = payload.get("structured")
    if isinstance(structured, Mapping):
        return extract_detections(structured)
    return []


def detection_bbox(detection: Mapping[str, Any], *names: str) -> Optional[List[float]]:
    """First valid bbox among ``names`` (defaults to the usual aliases)."""
    keys = names or ("bbox", "rgb_bbox", "sar_bbox", "box")
    for key in keys:
        normalized = normalize_bbox(detection.get(key))
        if normalized is not None:
            return normalized
    return None


def detection_label(detection: Mapping[str, Any], *names: str) -> Optional[str]:
    """First non-empty label among ``names``."""
    keys = names or ("label", "class_name", "category", "name", "rgb_label", "sar_label")
    for key in keys:
        value = detection.get(key)
        if value not in (None, "") and str(value).strip().lower() != "object":
            return str(value).strip()
    return None


def detection_score(detection: Mapping[str, Any], *names: str) -> Optional[float]:
    """First numeric score among ``names``, clamped to ``[0, 1]``."""
    keys = names or ("score", "confidence", "rgb_score", "sar_score")
    for key in keys:
        value = detection.get(key)
        if value in (None, ""):
            continue
        try:
            return clamp_unit(float(value))
        except (TypeError, ValueError):
            continue
    return None


def claims_from_detection(
    observation: PrimitiveObservation,
    detection: Mapping[str, Any],
    source_channel: str,
    attribute_keys: Optional[Mapping[str, Sequence[str]]] = None,
) -> List[Claim]:
    """Derive all attribute claims a single detection supports.

    Category comes from the label; subtype, heading and image-plane orientation
    are read from their usual field names when the detector supplies them. Every
    returned claim shares ``observation``'s identity row.
    """
    keys = dict(
        attribute_keys
        or {
            ClaimAttribute.CATEGORY.value: ("label", "class_name", "category", "name"),
            ClaimAttribute.SUBTYPE.value: ("subtype", "subclass", "fine_grained_label"),
            ClaimAttribute.HEADING.value: ("heading",),
            ClaimAttribute.ORIENTATION.value: ("orientation",),
        }
    )
    claims: List[Claim] = []
    for attribute, field_names in keys.items():
        value: Any = None
        if attribute == ClaimAttribute.CATEGORY.value:
            value = detection_label(detection, *field_names)
        else:
            for name in field_names:
                candidate = detection.get(name)
                if candidate not in (None, ""):
                    value = candidate
                    break
        if value in (None, ""):
            continue
        claims.append(
            build_claim(
                observation,
                attribute,
                value,
                source_channel=source_channel,
                raw_confidence=detection_score(detection),
            )
        )
    return claims


def object_hypotheses_from_observations(
    observations: Sequence[PrimitiveObservation],
    iou_merge_threshold: float = 0.5,
) -> List[Tuple[str, List[float]]]:
    """Seed object hypotheses by greedily clustering observation supports.

    This is a *bootstrap* only: it proposes where objects might be so that the
    candidate graph has columns.  It performs no identity decision — the
    resulting hypotheses all compete softly inside the inference.
    """
    from DS_VI.models.shared_identity.physical_features import bbox_iou

    clusters: List[List[List[float]]] = []
    for observation in observations:
        bbox = observation.spatial_support.bbox
        if bbox is None:
            continue
        placed = False
        for cluster in clusters:
            if any(bbox_iou(bbox, member) >= iou_merge_threshold for member in cluster):
                cluster.append(list(bbox))
                placed = True
                break
        if not placed:
            clusters.append([list(bbox)])

    hypotheses: List[Tuple[str, List[float]]] = []
    for index, cluster in enumerate(clusters):
        count = len(cluster)
        mean_bbox = [sum(member[i] for member in cluster) / count for i in range(4)]
        object_id = stable_id("obj", {"index": index, "bbox": [round(v, 3) for v in mean_bbox]})
        hypotheses.append((object_id, mean_bbox))
    return hypotheses


def cross_image_object_hypotheses(
    observations: Sequence[PrimitiveObservation],
    capacity: int = 2,
) -> List["ObjectHypothesis"]:
    """Build pair candidates using only related, different image groups.

    A candidate object represents one possible cross-image identity.  It has
    exactly one member observation from each side of the relation, so two
    detections from the same image can never become members of one candidate.
    The candidate graph later uses the stored pair features instead of the
    union bbox for its physical prior.
    """
    from DS_VI.models.shared_identity.physical_features import compute_pair_features

    candidates: List[ObjectHypothesis] = []
    for left_index, left in enumerate(observations):
        if left.spatial_support.bbox is None:
            continue
        for right in observations[left_index + 1 :]:
            if right.spatial_support.bbox is None:
                continue
            if not observations_are_related(left, right):
                continue

            features = compute_pair_features(
                left.spatial_support.bbox,
                right.spatial_support.bbox,
                observation_confidence=left.detection_confidence,
                object_confidence=right.detection_confidence,
            )
            candidates.append(
                ObjectHypothesis(
                    object_id=stable_id(
                        "cross_image_object",
                        {
                            "relation_group_id": relation_group_id(left),
                            "left": left.observation_id,
                            "right": right.observation_id,
                        },
                    ),
                    # The bbox is retained for compatibility and diagnostics.
                    # Candidate scoring uses pair_features below because the two
                    # image supports may not share one coordinate frame.
                    spatial_support=SpatialSupport(
                        bbox=list(left.spatial_support.bbox)
                    ),
                    # One identity may be observed once per related image.  The
                    # pair candidate therefore has one capacity slot per member,
                    # while same-image alternatives are absent altogether.
                    capacity=None if capacity is None else max(1, int(capacity)),
                    context={
                        "candidate_type": "cross_image_pair",
                        "relation_group_id": relation_group_id(left),
                        "member_observation_ids": [
                            left.observation_id,
                            right.observation_id,
                        ],
                        "member_image_group_ids": [
                            image_group_id(left),
                            image_group_id(right),
                        ],
                        "pair_features": features.to_dict(),
                    },
                )
            )
    return candidates


__all__ = [
    "build_observation",
    "build_claim",
    "extract_detections",
    "detection_bbox",
    "detection_label",
    "detection_score",
    "claims_from_detection",
    "object_hypotheses_from_observations",
    "cross_image_object_hypotheses",
]
