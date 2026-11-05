"""Official HOTA (Higher Order Tracking Accuracy) for the two-view task.

The task maps onto tracking exactly: **each image pair is a sequence** and **each
view is a frame**.  A GT object with a box in view a and/or view b is a track of
length 1-2 whose ID is the annotated object; a predicted object is a track whose
ID is the emitted `object_id`.  Association success therefore measures precisely
"were these two detections put in the same identity", which is what the cascade
is supposed to do.

Definitions follow Luiten et al., IJCV 2020 (and TrackEval's implementation):

.. code-block:: text

    match at threshold alpha      one-to-one per frame, IoU >= alpha,
                                  maximising the number of matches (Hungarian)
    TP / FN / FP                  matched GT, unmatched GT, unmatched prediction
    DetA_alpha = TP / (TP + FN + FP)
    for each matched pair c:
        TPA(c) = matched pairs sharing gtID(c) AND prID(c)
        FNA(c) = GT detections of gtID(c) that are not in TPA(c)
        FPA(c) = predicted detections of prID(c) that are not in TPA(c)
        A(c)   = |TPA| / (|TPA| + |FNA| + |FPA|)
    AssA_alpha = mean over c of A(c)
    HOTA_alpha = sqrt(DetA_alpha * AssA_alpha)
    HOTA       = mean over alpha in 0.05 .. 0.95 step 0.05

Counts are summed over all pairs before the ratios are taken (TrackEval
aggregates the dataset, it does not average per-sequence ratios), and the
per-pair counters are returned so a paired bootstrap can recompute the whole
statistic exactly.

This is the *official* metric.  ``property_metrics.composite_association`` keeps
the earlier object-level approximation under a different key, and the two must
never be mixed in one table.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from DS_VI.experiments.metrics.property_metrics import bbox_of, object_members

#: Localisation thresholds, identical to TrackEval's HOTA defaults.
THRESHOLDS: Tuple[float, ...] = tuple(round(0.05 * step, 2) for step in range(1, 20))

NULL_TRACK = "__none__"


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def _track_of_instances(objects: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """``instance_id -> track id`` for one side (predicted or truth objects)."""
    owner: Dict[str, str] = {}
    for position, row in enumerate(objects):
        track = str(row.get("object_id") or f"{NULL_TRACK}:{position}")
        for instance_id in object_members(row).values():
            owner[instance_id] = track
    return owner


def _frame_detections(
    instances: Mapping[str, Dict[str, Any]],
    owner: Mapping[str, str],
    view: str,
) -> List[Tuple[str, str, List[float]]]:
    """``[(instance_id, track_id, bbox)]`` for one view, in a stable order."""
    out: List[Tuple[str, str, List[float]]] = []
    for instance_id in sorted(owner):
        row = instances.get(instance_id)
        if row is None or str(row.get("view")) != view:
            continue
        box = bbox_of(row)
        if box is None:
            continue
        out.append((instance_id, owner[instance_id], box))
    return out


def _match_frame(
    truth: Sequence[Tuple[str, str, List[float]]],
    predicted: Sequence[Tuple[str, str, List[float]]],
    alpha: float,
) -> List[Tuple[int, int]]:
    """One-to-one IoU matching maximising the number of admissable matches."""
    if not truth or not predicted:
        return []
    from scipy.optimize import linear_sum_assignment

    import numpy as np

    overlaps = np.zeros((len(truth), len(predicted)), dtype=float)
    for gi, (_gid, _gtrack, gbox) in enumerate(truth):
        for pi, (_pid, _ptrack, pbox) in enumerate(predicted):
            overlaps[gi, pi] = _iou(gbox, pbox)
    admissible = overlaps >= float(alpha)
    if not admissible.any():
        return []
    # Maximise the number of matches first (TrackEval's HOTA matching), then IoU.
    weight = np.where(admissible, 2.0 + overlaps, 0.0)
    rows, cols = linear_sum_assignment(-weight)
    return [
        (int(gi), int(pi))
        for gi, pi in zip(rows.tolist(), cols.tolist())
        if admissible[gi, pi]
    ]


def _alignment(
    matched: Sequence[Tuple[str, str, str, str]],
    truth_track_counts: Mapping[str, int],
    pred_track_counts: Mapping[str, int],
) -> Tuple[float, int]:
    """``(sum of A(c), number of matched pairs)`` for one threshold."""
    pair_counts: Dict[Tuple[str, str], int] = {}
    for _gid, gtrack, _pid, ptrack in matched:
        pair_counts[(gtrack, ptrack)] = pair_counts.get((gtrack, ptrack), 0) + 1
    total = 0.0
    for _gid, gtrack, _pid, ptrack in matched:
        tpa = pair_counts[(gtrack, ptrack)]
        fna = truth_track_counts.get(gtrack, 0) - tpa
        fpa = pred_track_counts.get(ptrack, 0) - tpa
        denominator = tpa + max(fna, 0) + max(fpa, 0)
        total += (tpa / denominator) if denominator > 0 else 0.0
    return total, len(matched)


def pair_hota(
    predicted_objects: Sequence[Mapping[str, Any]],
    truth_objects: Sequence[Mapping[str, Any]],
    tool_instances: Mapping[str, Dict[str, Any]],
    truth_instances: Mapping[str, Dict[str, Any]],
    views: Optional[Sequence[str]] = None,
    thresholds: Sequence[float] = THRESHOLDS,
) -> List[Dict[str, float]]:
    """Per-threshold ``{tp, fn, fp, alignment_sum}`` for one image pair.

    ``views=None`` derives the frames from the member instances, so callers do not
    have to know the family's view names.
    """
    pred_owner = _track_of_instances(predicted_objects)
    truth_owner = _track_of_instances(truth_objects)
    pred_counts: Dict[str, int] = {}
    for track in pred_owner.values():
        pred_counts[track] = pred_counts.get(track, 0) + 1
    truth_counts: Dict[str, int] = {}
    for track in truth_owner.values():
        truth_counts[track] = truth_counts.get(track, 0) + 1

    if views is None:
        member_ids = set(truth_owner) | set(pred_owner)
        discovered = {
            str(instance.get("view"))
            for instance_id in member_ids
            for instance in (truth_instances.get(instance_id), tool_instances.get(instance_id))
            if instance is not None and instance.get("view") not in (None, "")
        }
        views = sorted(discovered)
    frames = {
        view: (
            _frame_detections(truth_instances, truth_owner, view),
            _frame_detections(tool_instances, pred_owner, view),
        )
        for view in views
    }
    out: List[Dict[str, float]] = []
    for alpha in thresholds:
        matched: List[Tuple[str, str, str, str]] = []
        truth_total = pred_total = 0
        for view in views:
            truth_dets, pred_dets = frames[view]
            truth_total += len(truth_dets)
            pred_total += len(pred_dets)
            for gi, pi in _match_frame(truth_dets, pred_dets, alpha):
                matched.append(
                    (
                        truth_dets[gi][0], truth_dets[gi][1],
                        pred_dets[pi][0], pred_dets[pi][1],
                    )
                )
        tp = len(matched)
        fn = truth_total - tp
        fp = pred_total - tp
        alignment, _count = _alignment(matched, truth_counts, pred_counts)
        out.append(
            {"tp": float(tp), "fn": float(fn), "fp": float(fp), "alignment_sum": float(alignment)}
        )
    return out


def aggregate(counts: Sequence[Sequence[Mapping[str, float]]]) -> Dict[str, Any]:
    """Dataset HOTA from per-pair per-threshold counters."""
    if not counts:
        return {"hota": 0.0, "deta": 0.0, "assa": 0.0, "thresholds": [], "per_threshold": []}
    depth = len(counts[0])
    deta_values: List[float] = []
    assa_values: List[float] = []
    hota_values: List[float] = []
    for index in range(depth):
        tp = sum(float(entry[index]["tp"]) for entry in counts)
        fn = sum(float(entry[index]["fn"]) for entry in counts)
        fp = sum(float(entry[index]["fp"]) for entry in counts)
        alignment = sum(float(entry[index]["alignment_sum"]) for entry in counts)
        denominator = tp + fn + fp
        deta = (tp / denominator) if denominator else 0.0
        assa = (alignment / tp) if tp else 0.0
        product = deta * assa
        deta_values.append(deta)
        assa_values.append(assa)
        hota_values.append(math.sqrt(product) if product > 0 else 0.0)
    return {
        "hota": round(sum(hota_values) / len(hota_values), 6),
        "deta": round(sum(deta_values) / len(deta_values), 6),
        "assa": round(sum(assa_values) / len(assa_values), 6),
        "thresholds": list(THRESHOLDS[:depth]),
        "per_threshold": [
            {"alpha": THRESHOLDS[i], "hota": round(hota_values[i], 6),
             "deta": round(deta_values[i], 6), "assa": round(assa_values[i], 6)}
            for i in range(depth)
        ],
    }


def hota_from_counts(counts: Mapping[str, Any]) -> float:
    """Convenience: aggregate a ``{pair_id: per-threshold counts}`` mapping."""
    return float(aggregate(list(counts.values()))["hota"])


__all__ = [
    "THRESHOLDS",
    "aggregate",
    "hota_from_counts",
    "pair_hota",
]
