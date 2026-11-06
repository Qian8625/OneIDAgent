"""Single source of truth for cross-modal / cross-view pair geometry features.

The cross-modal worker historically implemented its own IoU / center-distance /
area-ratio math with weights that drifted from the shared implementation
(``0.50/0.30/0.10/0.10`` versus ``0.45/0.30/0.15/0.10``).  This module replaces it
so that a given pair of boxes always yields identical features regardless of which
component asks for them.

The diagnostic score produced here is intentionally *not* a decision: it feeds
``physical_prior`` as a soft prior only.  Hard matching belongs to the legacy
compatibility paths.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

# Canonical diagnostic-score weights.  These are the weights specified by the
# revision report; the legacy CrossModalInterpret weights are kept only inside
# that worker's ``legacy_hard`` path.
W_IOU = 0.45
W_CENTER = 0.30
W_AREA = 0.15
W_CONFIDENCE = 0.10

# Fraction of the larger box diagonal used to normalize center distance.
CENTER_SCALE_FACTOR = 0.75

DEFAULT_CONFIDENCE = 0.5

BBox = Sequence[float]


@dataclass(frozen=True)
class PairFeatures:
    """Normalized geometric/photometric agreement between two supports."""

    iou: float
    center_distance_px: float
    center_score: float
    area_ratio: float
    confidence_score: float
    diagnostic_score: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def clamp_unit(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` into ``[0, 1]``, falling back to ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if math.isnan(number) or math.isinf(number):
        number = default
    return max(0.0, min(1.0, number))


def normalize_bbox(bbox: Optional[BBox]) -> Optional[list]:
    """Return a validated ``[x1, y1, x2, y2]`` list, or ``None`` when degenerate.

    A box is degenerate when it is missing, non-numeric, or has non-positive
    width or height.  Degenerate boxes never raise; callers get ``None`` and
    decide how to handle it.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    for v in (x1, y1, x2, y2):
        if math.isnan(v) or math.isinf(v):
            return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_area(bbox: BBox) -> float:
    """Area of ``bbox``; ``0.0`` for degenerate boxes."""
    box = normalize_bbox(bbox)
    if box is None:
        return 0.0
    return (box[2] - box[0]) * (box[3] - box[1])


def bbox_center(bbox: BBox) -> Optional[list]:
    """Center point of ``bbox``, or ``None`` when degenerate."""
    box = normalize_bbox(bbox)
    if box is None:
        return None
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def bbox_diagonal(bbox: BBox) -> float:
    """Diagonal length of ``bbox``, floored at ``1.0`` to keep ratios finite."""
    box = normalize_bbox(bbox)
    if box is None:
        return 1.0
    return max(1.0, math.hypot(box[2] - box[0], box[3] - box[1]))


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-union of two boxes; ``0.0`` if either is degenerate."""
    box_a = normalize_bbox(a)
    box_b = normalize_bbox(b)
    if box_a is None or box_b is None:
        return 0.0
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    if union <= 0.0:
        return 0.0
    return max(0.0, min(1.0, inter / union))


def bbox_center_distance(a: BBox, b: BBox) -> float:
    """Euclidean center distance in pixels; ``inf`` if either box is degenerate."""
    center_a = bbox_center(a)
    center_b = bbox_center(b)
    if center_a is None or center_b is None:
        return float("inf")
    return math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


def center_score_from_distance(
    distance_px: float,
    scale_px: float,
) -> float:
    """Map a center distance to ``[0, 1]`` using ``scale_px`` as the zero point."""
    if math.isinf(distance_px) or math.isnan(distance_px):
        return 0.0
    scale = max(float(scale_px), 1.0)
    return max(0.0, min(1.0, 1.0 - distance_px / scale))


def center_scale(
    a: BBox,
    b: BBox,
    scale_factor: float = CENTER_SCALE_FACTOR,
) -> float:
    """Reference distance scale for a pair: a fraction of the larger diagonal."""
    return max(1.0, max(bbox_diagonal(a), bbox_diagonal(b)) * float(scale_factor))


def bbox_area_ratio(a: BBox, b: BBox) -> float:
    """Ratio of the smaller to the larger area; ``0.0`` when either is empty."""
    area_a = bbox_area(a)
    area_b = bbox_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)


def diagnostic_score(
    iou: float,
    center_score: float,
    area_ratio: float,
    confidence_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted agreement score used only as a soft physical prior."""
    w = weights or {}
    w_iou = float(w.get("iou", W_IOU))
    w_center = float(w.get("center", W_CENTER))
    w_area = float(w.get("area", W_AREA))
    w_conf = float(w.get("confidence", W_CONFIDENCE))
    total = w_iou + w_center + w_area + w_conf
    raw = (
        w_iou * clamp_unit(iou)
        + w_center * clamp_unit(center_score)
        + w_area * clamp_unit(area_ratio)
        + w_conf * clamp_unit(confidence_score)
    )
    if total <= 0.0:
        return 0.0
    return max(0.0, min(1.0, raw / total)) if abs(total - 1.0) > 1e-9 else max(0.0, min(1.0, raw))


def compute_pair_features(
    observation_support: Optional[BBox],
    object_support: Optional[BBox],
    observation_confidence: Optional[float] = None,
    object_confidence: Optional[float] = None,
    center_scale_factor: float = CENTER_SCALE_FACTOR,
    weights: Optional[Dict[str, float]] = None,
) -> PairFeatures:
    """Compute the canonical :class:`PairFeatures` for one observation/object pair.

    Args:
        observation_support: ``[x1, y1, x2, y2]`` support of the observation.
        object_support: ``[x1, y1, x2, y2]`` support of the object hypothesis.
        observation_confidence: Optional detector confidence for the observation.
        object_confidence: Optional confidence attached to the object hypothesis.
        center_scale_factor: Fraction of the larger diagonal used to normalize
            center distance.
        weights: Optional override of the diagnostic-score weights.

    Returns:
        A :class:`PairFeatures` instance.  Degenerate supports yield an
        all-zero feature vector with ``center_distance_px`` set to ``inf``
        rather than raising.
    """
    box_obs = normalize_bbox(observation_support)
    box_obj = normalize_bbox(object_support)

    conf_values = [
        clamp_unit(c, DEFAULT_CONFIDENCE)
        for c in (observation_confidence, object_confidence)
        if c is not None
    ]
    confidence = sum(conf_values) / len(conf_values) if conf_values else DEFAULT_CONFIDENCE

    if box_obs is None or box_obj is None:
        return PairFeatures(
            iou=0.0,
            center_distance_px=float("inf"),
            center_score=0.0,
            area_ratio=0.0,
            confidence_score=confidence,
            diagnostic_score=0.0,
        )

    iou = bbox_iou(box_obs, box_obj)
    distance = bbox_center_distance(box_obs, box_obj)
    scale = center_scale(box_obs, box_obj, center_scale_factor)
    center = center_score_from_distance(distance, scale)
    area_ratio = bbox_area_ratio(box_obs, box_obj)
    score = diagnostic_score(iou, center, area_ratio, confidence, weights)

    return PairFeatures(
        iou=iou,
        center_distance_px=distance,
        center_score=center,
        area_ratio=area_ratio,
        confidence_score=confidence,
        diagnostic_score=score,
    )


def legacy_center_score(a: BBox, b: BBox) -> float:
    """Center score using the legacy ``max-diagonal`` scale (no 0.75 factor).

    Kept so that ``legacy_hard`` code paths can reproduce their historic
    numbers bit-for-bit while still routing through this module.
    """
    distance = bbox_center_distance(a, b)
    scale = max(bbox_diagonal(a), bbox_diagonal(b), 1.0)
    return center_score_from_distance(distance, scale)


__all__ = [
    "W_IOU",
    "W_CENTER",
    "W_AREA",
    "W_CONFIDENCE",
    "CENTER_SCALE_FACTOR",
    "DEFAULT_CONFIDENCE",
    "PairFeatures",
    "clamp_unit",
    "normalize_bbox",
    "bbox_area",
    "bbox_center",
    "bbox_diagonal",
    "bbox_iou",
    "bbox_center_distance",
    "center_score_from_distance",
    "center_scale",
    "bbox_area_ratio",
    "diagnostic_score",
    "compute_pair_features",
    "legacy_center_score",
]
