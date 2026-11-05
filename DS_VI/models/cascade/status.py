"""Reported per-object status vocabulary (single source of truth).

An output object answers two questions at once: *which views saw it* and (for the
time family) *did it move between them*.  The pipeline, the comparison arms and
the evaluator must agree on the wording, so the vocabulary lives here and nowhere
else.

match (RGB = view a, SAR = view b)::

    match      both views
    rgb-only   view a only
    sar-only   view b only
    unknown    neither view (degenerate; callers must not emit such an object)

time (time1 = view a, time2 = view b)::

    static     present at both times, displacement within the stayed threshold
    moved      present at both times, displacement above it
    time1-only present at time1 only (the object left / was removed)
    time2-only present at time2 only (the object appeared / was added)

``unknown`` exists for match because a candidate can, in principle, lose both
member indices (for example a VI candidate whose members were re-assigned).  The
time vocabulary is closed: every emitted object carries exactly one of its four
labels, and :func:`object_status` raises rather than inventing a fifth.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

MATCH_STATUSES: Tuple[str, ...] = ("match", "rgb-only", "sar-only", "unknown")
TIME_STATUSES: Tuple[str, ...] = ("static", "moved", "time1-only", "time2-only")

STATUSES_BY_FAMILY = {"match": MATCH_STATUSES, "time": TIME_STATUSES}

#: Status -> evaluator change class (time family only).  ``static``/``moved`` are
#: the two "present at both times" cases; the single-side labels are the change
#: label the evaluator scores.
STATUS_TO_CHANGE = {
    "static": "stayed",
    "moved": "moved",
    "time1-only": "disappeared",
    "time2-only": "appeared",
}

#: Change label -> status, used when assembling an object from a D-S record.
CHANGE_TO_STATUS = {value: key for key, value in STATUS_TO_CHANGE.items()}

MATCH_VIEW_STATUS = {"a": "rgb-only", "b": "sar-only"}
TIME_VIEW_STATUS = {"a": "time1-only", "b": "time2-only"}


def object_status(
    family: str,
    has_view_a: bool,
    has_view_b: bool,
    change: Optional[Any] = None,
) -> str:
    """Status label for one output object.

    Args:
        family: ``match`` or ``time``.
        has_view_a: the object has a view-a (RGB / time1) member.
        has_view_b: the object has a view-b (SAR / time2) member.
        change: change label of a two-view object (``stayed``/``moved``); only
            the time family reads it. Missing or unrecognised values fall back to
            ``static`` for a two-view object, which is the majority class.

    Raises:
        ValueError: for an unknown family, or for a time object with no member at
            all (the vocabulary has no label for it).
    """
    name = str(family or "").strip().lower()
    if name == "match":
        if has_view_a and has_view_b:
            return "match"
        if has_view_a:
            return MATCH_VIEW_STATUS["a"]
        if has_view_b:
            return MATCH_VIEW_STATUS["b"]
        return "unknown"
    if name == "time":
        if has_view_a and has_view_b:
            text = str(change or "").strip().lower()
            return "moved" if text == "moved" else "static"
        if has_view_a:
            return TIME_VIEW_STATUS["a"]
        if has_view_b:
            return TIME_VIEW_STATUS["b"]
        raise ValueError("time object has no view member; cannot label its status")
    raise ValueError(f"Unsupported family for object_status: {family!r}")


def status_from_change(family: str, change: Optional[Any], has_view_a: bool, has_view_b: bool) -> str:
    """Backwards-compatible alias kept for call sites that carry a change label."""
    return object_status(family, has_view_a, has_view_b, change)


__all__ = [
    "CHANGE_TO_STATUS",
    "MATCH_STATUSES",
    "MATCH_VIEW_STATUS",
    "STATUSES_BY_FAMILY",
    "STATUS_TO_CHANGE",
    "TIME_STATUSES",
    "TIME_VIEW_STATUS",
    "object_status",
    "status_from_change",
]
