"""Ambiguity rejection in the independent-candidate builder."""
from __future__ import annotations

from DS_VI.models.cascade.vi import _pruned_cross_view_hypotheses
from DS_VI.models.shared_identity.observation_builder import build_observation


def _obs(bbox, image: str):
    return build_observation(
        source_tool="tool",
        modality="optical",
        bbox=bbox,
        acquisition_id=image,
        time_key="t0",
        context={"image_group_id": image, "channel": "ab"[0 if image.endswith("a") else 1]},
    )


def test_symmetric_candidates_are_rejected_when_ambiguous():
    left = _obs([0, 0, 10, 10], "img_a")
    # Two equally good counterparts on either side of the left box: the geometry
    # cannot pick one, so a margin must discard the observation entirely.
    right_near = _obs([11, 0, 21, 10], "img_b")
    right_symmetric = _obs([-11, 0, -1, 10], "img_b")

    plain = _pruned_cross_view_hypotheses(
        [left], [right_near, right_symmetric], 2, "ship", "g", max_links=2,
    )
    gated = _pruned_cross_view_hypotheses(
        [left], [right_near, right_symmetric], 2, "ship", "g", max_links=2, margin=0.2,
    )
    assert len(plain) == 2
    assert gated == []


def test_clear_winner_survives_the_margin():
    left = _obs([0, 0, 10, 10], "img_a")
    right_close = _obs([1, 1, 11, 11], "img_b")
    right_far = _obs([90, 90, 100, 100], "img_b")

    gated = _pruned_cross_view_hypotheses(
        [left], [right_close, right_far], 2, "ship", "g", max_links=1, margin=0.2,
    )
    assert len(gated) == 1
    assert gated[0].spatial_support.bbox == [0.0, 0.0, 10.0, 10.0]


def test_margin_zero_is_a_no_op():
    left = _obs([0, 0, 10, 10], "img_a")
    right = _obs([1, 1, 11, 11], "img_b")
    assert len(_pruned_cross_view_hypotheses([left], [right], 2, "ship", "g", margin=0.0)) == 1
