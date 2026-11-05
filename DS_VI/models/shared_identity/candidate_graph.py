"""Candidate graph construction: enumerate observation -> object edges.

This module answers "which objects *could* this observation be?" with high
recall.  It deliberately does not answer "which object *is* it?" — that is the
job of the joint A-Q inference in :mod:`inference`.

Every observation always receives a ``__NULL__`` edge so that its association
row can be normalized even when no real object is admissible.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.physical_features import (
    PairFeatures,
    compute_pair_features,
)
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    CandidateEdge,
    ObjectHypothesis,
    PrimitiveObservation,
)
from DS_VI.models.shared_identity.scope import image_group_id, relation_group_id


def _geo_distance_m(
    observation: PrimitiveObservation,
    obj: ObjectHypothesis,
) -> Optional[float]:
    """Great-circle-free planar distance between geo centers, in meters.

    Returns ``None`` when either side lacks a geo center.  Coordinates are
    treated as (lon, lat) degrees and converted with a local flat-earth
    approximation, which is accurate enough for gating at these scales.
    """
    a = observation.spatial_support.geo_center
    b = obj.spatial_support.geo_center
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    lon_a, lat_a = float(a[0]), float(a[1])
    lon_b, lat_b = float(b[0]), float(b[1])
    mean_lat = math.radians((lat_a + lat_b) / 2.0)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * max(math.cos(mean_lat), 1e-6)
    dx = (lon_a - lon_b) * meters_per_deg_lon
    dy = (lat_a - lat_b) * meters_per_deg_lat
    return math.hypot(dx, dy)


def evaluate_admissibility(
    features: PairFeatures,
    geo_distance_m: Optional[float],
    gates: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Decide whether an edge is admissible under high-recall OR-gating.

    An edge passes when *any* of the IoU, center-score, or geo-distance gates is
    satisfied.  A hard ``max_center_distance_px`` veto, when configured,
    overrides all of them.

    Returns:
        ``(admissible, reason)`` where ``reason`` names the gate that decided.
    """
    max_center = gates.get("max_center_distance_px")
    if max_center not in (None, "") and features.center_distance_px > float(max_center):
        return (False, f"center_distance>{max_center}px")

    low_iou_gate = float(gates.get("low_iou_gate", 0.05))
    center_gate = float(gates.get("center_gate", 0.25))
    geo_gate_m = gates.get("geo_gate_m")

    if features.iou >= low_iou_gate:
        return (True, f"iou>={low_iou_gate}")
    if features.center_score >= center_gate:
        return (True, f"center_score>={center_gate}")
    if geo_gate_m not in (None, "") and geo_distance_m is not None:
        if geo_distance_m <= float(geo_gate_m):
            return (True, f"geo_distance<={geo_gate_m}m")
    return (False, "no_gate_passed")


def _pair_features_from_object(obj: ObjectHypothesis) -> Optional[PairFeatures]:
    """Read the precomputed cross-image score of a pair candidate."""
    payload = (obj.context or {}).get("pair_features")
    if not isinstance(payload, Mapping):
        return None
    try:
        return PairFeatures(
            iou=float(payload.get("iou", 0.0)),
            center_distance_px=float(payload.get("center_distance_px", 0.0)),
            center_score=float(payload.get("center_score", 0.0)),
            area_ratio=float(payload.get("area_ratio", 0.0)),
            confidence_score=float(payload.get("confidence_score", 0.0)),
            diagnostic_score=float(payload.get("diagnostic_score", 0.0)),
        )
    except (TypeError, ValueError):
        return None


def _scope_allows(
    observation: PrimitiveObservation,
    obj: ObjectHypothesis,
    strict_scope: bool,
) -> bool:
    """Apply image/relation scope before geometry is evaluated."""
    context = obj.context or {}
    candidate_type = str(context.get("candidate_type") or "")
    member_observation_ids = {
        str(value)
        for value in (context.get("member_observation_ids") or [])
    }

    # Pair candidates are always exact: an observation can only use a pair
    # candidate if it is one of that pair's two members.
    if candidate_type == "cross_image_pair":
        if observation.observation_id not in member_observation_ids:
            return False
        member_images = {
            str(value)
            for value in (context.get("member_image_group_ids") or [])
            if value not in (None, "")
        }
        if len(member_images) < 2:
            return False
        return True

    if not strict_scope:
        return True

    object_relation = context.get("relation_group_id")
    if object_relation not in (None, ""):
        if str(object_relation) != str(relation_group_id(observation)):
            return False

    member_images = {
        str(value)
        for value in (context.get("member_image_group_ids") or [])
        if value not in (None, "")
    }
    observation_image = image_group_id(observation)
    if member_images and observation_image not in member_images:
        return False

    # Legacy hypotheses without scope metadata remain usable for compatibility;
    # new producers should emit pair candidates or member image metadata.
    return True


def build_candidate_graph(
    observations: Sequence[PrimitiveObservation],
    objects: Sequence[ObjectHypothesis],
    config: Optional[Mapping[str, Any]] = None,
) -> List[CandidateEdge]:
    """Enumerate candidate edges for every observation.

    Args:
        observations: Observations needing an identity.
        objects: Active object hypotheses.
        config: Optional shared-identity config; loaded from disk when omitted.

    Returns:
        A list of :class:`CandidateEdge`.  Inadmissible pairs are omitted
        entirely (they must never receive probability mass).  Each observation
        contributes exactly one ``__NULL__`` edge, appended last.
    """
    cfg = dict(config or load_config())
    physical = dict(cfg.get("physical") or {})
    gates = dict(physical.get("gates") or {})
    weights = dict(physical.get("weights") or {})
    center_scale_factor = float(physical.get("center_scale_factor", 0.75))
    max_candidates = physical.get("max_candidates_per_observation")
    association_cfg = dict(cfg.get("association") or {})
    strict_scope = bool(association_cfg.get("strict_scope", False))
    scope_mode = str(association_cfg.get("mode", "legacy_global"))

    active_objects = [obj for obj in objects if obj.active]
    edges: List[CandidateEdge] = []

    for observation in observations:
        scored: List[Tuple[float, CandidateEdge]] = []
        for obj in active_objects:
            if scope_mode == "cross_image_independent" and not _scope_allows(
                observation, obj, strict_scope
            ):
                continue

            features = _pair_features_from_object(obj)
            if features is None:
                features = compute_pair_features(
                    observation.spatial_support.bbox,
                    obj.spatial_support.bbox,
                    observation_confidence=observation.detection_confidence,
                    object_confidence=obj.confidence,
                    center_scale_factor=center_scale_factor,
                    weights=weights,
                )
            geo_distance = _geo_distance_m(observation, obj)
            if (obj.context or {}).get("bypass_gates"):
                # Independent-data candidates (cascade extras) are weak by
                # construction: the D-S stage already declined them. Their
                # selectivity comes from the k-nearest construction and the VI
                # confirmation, so geometry must not veto them here.
                admissible, reason = True, "extra_bypass_gates"
            else:
                admissible, reason = evaluate_admissibility(features, geo_distance, gates)
            if not admissible:
                continue
            feature_dict = features.to_dict()
            if geo_distance is not None:
                feature_dict["geo_distance_m"] = geo_distance
            scored.append(
                (
                    features.diagnostic_score,
                    CandidateEdge(
                        observation_id=observation.observation_id,
                        object_id=obj.object_id,
                        features=feature_dict,
                        admissible=True,
                        gate_reason=reason,
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        if max_candidates not in (None, "") and int(max_candidates) > 0:
            scored = scored[: int(max_candidates)]

        observation.candidate_object_ids = [edge.object_id for _, edge in scored]
        edges.extend(edge for _, edge in scored)

        # NULL is unconditional and never capacity-limited.
        edges.append(
            CandidateEdge(
                observation_id=observation.observation_id,
                object_id=NULL_OBJECT_ID,
                features={},
                admissible=True,
                gate_reason="null_always_admissible",
            )
        )
        observation.candidate_object_ids.append(NULL_OBJECT_ID)

    return edges


def candidate_mask(
    edges: Sequence[CandidateEdge],
    observation_ids: Sequence[str],
    object_ids: Sequence[str],
) -> List[List[bool]]:
    """Boolean ``[m][k]`` mask of admissible edges, in the given index order."""
    index_of_object = {object_id: k for k, object_id in enumerate(object_ids)}
    index_of_observation = {obs_id: m for m, obs_id in enumerate(observation_ids)}
    mask = [[False] * len(object_ids) for _ in observation_ids]
    for edge in edges:
        if not edge.admissible:
            continue
        m = index_of_observation.get(edge.observation_id)
        k = index_of_object.get(edge.object_id)
        if m is None or k is None:
            continue
        mask[m][k] = True
    return mask


def candidate_recall(
    edges: Sequence[CandidateEdge],
    ground_truth: Mapping[str, str],
) -> Dict[str, Any]:
    """Fraction of ground-truth associations retained in the candidate graph.

    Args:
        edges: The candidate graph.
        ground_truth: ``observation_id -> true object_id`` (``__NULL__`` allowed).

    Returns:
        Dict with ``recall``, ``covered``, ``total``, ``missing`` and
        ``mean_candidates``.
    """
    by_observation: Dict[str, set] = {}
    for edge in edges:
        if edge.admissible:
            by_observation.setdefault(edge.observation_id, set()).add(edge.object_id)

    covered = 0
    missing: List[str] = []
    for observation_id, true_object in ground_truth.items():
        if true_object in by_observation.get(observation_id, set()):
            covered += 1
        else:
            missing.append(observation_id)

    total = len(ground_truth)
    counts = [len(v) for v in by_observation.values()]
    return {
        "recall": (covered / total) if total else 1.0,
        "covered": covered,
        "total": total,
        "missing": missing,
        "mean_candidates": (sum(counts) / len(counts)) if counts else 0.0,
    }


__all__ = [
    "build_candidate_graph",
    "candidate_mask",
    "candidate_recall",
    "evaluate_admissibility",
]
