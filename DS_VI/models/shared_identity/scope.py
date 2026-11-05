"""Image and relation scope helpers for shared-identity association.

The shared-identity model operates on observations, but an observation also has
an image provenance boundary.  These helpers keep that boundary explicit so a
candidate graph can be restricted to related images without using modality as
a proxy (``time1`` and ``time2`` are both optical, for example).
"""
from __future__ import annotations

from typing import Any, Optional, Set


def image_group_id(observation: Any) -> Optional[str]:
    """Return the stable image identity for one observation.

    ``acquisition_root`` is deliberately not used here: it can be shared by
    RGB and SAR views of the same acquisition.  The fallback chain preserves
    compatibility with older serialized observations.
    """
    context = dict(getattr(observation, "context", {}) or {})
    value = (
        context.get("image_group_id")
        or getattr(observation, "acquisition_id", None)
        or getattr(getattr(observation, "spatial_support", None), "frame_id", None)
    )
    if value in (None, ""):
        roots = list(getattr(observation, "input_roots", []) or [])
        value = roots[0] if roots else None
    return None if value in (None, "") else str(value)


def relation_group_id(observation: Any) -> Optional[str]:
    """Return the independent matching group for one observation."""
    context = dict(getattr(observation, "context", {}) or {})
    value = (
        context.get("relation_group_id")
        or context.get("pair_id")
        or context.get("scene_group_id")
    )
    if value in (None, ""):
        # Legacy callers normally build one inference problem per pair.  Keep
        # those callers in one implicit group while allowing explicit metadata
        # to partition real multi-image runs.
        return "__implicit_relation_group__"
    return str(value)


def related_image_group_ids(observation: Any) -> Set[str]:
    """Return explicitly related image ids, if the producer supplied them."""
    values = (getattr(observation, "context", {}) or {}).get(
        "related_image_group_ids", []
    )
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values if value not in (None, "")}


def same_image(left: Any, right: Any) -> bool:
    """Whether two observations originate from the same image group."""
    left_id = image_group_id(left)
    right_id = image_group_id(right)
    return left_id is not None and left_id == right_id


def observations_are_related(left: Any, right: Any) -> bool:
    """Return whether two observations may participate in one association."""
    if relation_group_id(left) != relation_group_id(right):
        return False
    if same_image(left, right):
        return False

    left_image = image_group_id(left)
    right_image = image_group_id(right)
    left_related = related_image_group_ids(left)
    right_related = related_image_group_ids(right)

    if left_related and right_image not in left_related:
        return False
    if right_related and left_image not in right_related:
        return False
    return True


__all__ = [
    "image_group_id",
    "relation_group_id",
    "related_image_group_ids",
    "same_image",
    "observations_are_related",
]
