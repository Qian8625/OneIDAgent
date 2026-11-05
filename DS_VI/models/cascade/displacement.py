"""Scene displacement-field gate for independent candidates.

An independent candidate is built from detections the D-S stage left unused, so it
has no fused evidence behind it; its only protection is geometry plus (optionally)
attribute agreement.  On scenes where the whole frame shifts between the two
views (time family), a *crossed* pairing (A1-B2, A2-B1) produces a shift that is
far from the scene's own displacement field, which is a structural cue the
confirmation score does not see.

The field is the robust location of the **accepted D-S pair** shifts (median) and
its scale is their median absolute deviation.  A candidate is kept when its own
member shift lies within ``threshold`` scale units of that location.  With fewer
than three reference shifts the field cannot be estimated and everything is kept
(no silent filtering).

This lives in the package (not in the comparison harness) so the production entry
point and the arms share one implementation.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple


def _centre(box: Sequence[float]) -> Tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def filter_extras_by_displacement(
    records: Sequence[Mapping[str, Any]],
    shifts: Sequence[Sequence[float]],
    threshold: float,
) -> Tuple[List[Mapping[str, Any]], int]:
    """``(kept, rejected_count)`` for one image pair."""
    reference = [tuple(float(value) for value in shift) for shift in shifts]
    if len(reference) < 3:
        return list(records), 0
    import numpy as np

    field = np.asarray(reference, dtype=float)
    median = np.median(field, axis=0)
    scale = float(np.median(np.linalg.norm(field - median, axis=1))) + 1e-6
    kept: List[Mapping[str, Any]] = []
    rejected = 0
    for record in records:
        boxes = [box for box in (record.get("member_bboxes") or {}).values() if box]
        if len(boxes) < 2:
            kept.append(record)
            continue
        left, right = _centre(boxes[0]), _centre(boxes[1])
        delta = np.asarray([right[0] - left[0], right[1] - left[1]])
        if float(np.linalg.norm(delta - median)) <= float(threshold) * scale:
            kept.append(record)
        else:
            rejected += 1
    return kept, rejected


def shifts_from_matched(matched: Sequence[Mapping[str, Any]]) -> List[Tuple[float, float]]:
    """Member shifts of the accepted D-S pairs, for the reference field."""
    out: List[Tuple[float, float]] = []
    for item in matched:
        bbox_a = item.get("view_a_bbox")
        bbox_b = item.get("view_b_bbox")
        if not bbox_a or not bbox_b:
            continue
        left, right = _centre(bbox_a), _centre(bbox_b)
        out.append((right[0] - left[0], right[1] - left[1]))
    return out


__all__ = ["filter_extras_by_displacement", "shifts_from_matched"]
