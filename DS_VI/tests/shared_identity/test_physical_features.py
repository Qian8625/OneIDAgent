"""Geometry/agreement features shared by every cross-modal component.

This module exists because two components previously implemented their own IoU
and hybrid-score math with different weights.  These tests pin the single
canonical implementation, including the exact 0.45/0.30/0.15/0.10 weighting and
the legacy compatibility path.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from DS_VI.models.shared_identity.physical_features import (
    CENTER_SCALE_FACTOR,
    DEFAULT_CONFIDENCE,
    W_AREA,
    W_CENTER,
    W_CONFIDENCE,
    W_IOU,
    PairFeatures,
    bbox_area,
    bbox_area_ratio,
    bbox_center,
    bbox_center_distance,
    bbox_diagonal,
    bbox_iou,
    center_scale,
    center_score_from_distance,
    clamp_unit,
    compute_pair_features,
    diagnostic_score,
    legacy_center_score,
    normalize_bbox,
)

UNIT = [0.0, 0.0, 10.0, 10.0]


def test_canonical_weights_sum_to_one():
    # diagnostic_score() only skips its renormalization branch when the weights
    # already sum to 1, so this is a precondition of the weighted-sum tests.
    assert W_IOU + W_CENTER + W_AREA + W_CONFIDENCE == pytest.approx(1.0, abs=1e-12)
    assert (W_IOU, W_CENTER, W_AREA, W_CONFIDENCE) == (0.45, 0.30, 0.15, 0.10)


# ----------------------------------------------------------------------
# bbox_iou
# ----------------------------------------------------------------------
def test_bbox_iou_identical_boxes_is_one():
    assert bbox_iou(UNIT, list(UNIT)) == pytest.approx(1.0, abs=1e-12)


def test_bbox_iou_disjoint_boxes_is_zero():
    assert bbox_iou(UNIT, [100.0, 100.0, 110.0, 110.0]) == 0.0


def test_bbox_iou_edge_touching_boxes_is_zero():
    # Boxes sharing only an edge have zero-area intersection, not a sliver.
    assert bbox_iou(UNIT, [10.0, 0.0, 20.0, 10.0]) == 0.0


def test_bbox_iou_half_overlap_matches_hand_computation():
    # [0,0,10,10] vs [5,0,15,10]: intersection 5*10=50, union 100+100-50=150.
    other = [5.0, 0.0, 15.0, 10.0]
    assert bbox_iou(UNIT, other) == pytest.approx(50.0 / 150.0, abs=1e-12)
    assert bbox_iou(UNIT, other) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_bbox_iou_quarter_overlap_matches_hand_computation():
    # [0,0,10,10] vs [5,5,15,15]: intersection 5*5=25, union 200-25=175.
    assert bbox_iou(UNIT, [5.0, 5.0, 15.0, 15.0]) == pytest.approx(25.0 / 175.0, abs=1e-12)


def test_bbox_iou_nested_box_matches_area_ratio():
    # Inner box fully contained: intersection == inner area, union == outer area.
    assert bbox_iou(UNIT, [2.0, 2.0, 4.0, 4.0]) == pytest.approx(4.0 / 100.0, abs=1e-12)


def test_bbox_iou_is_symmetric():
    a, b = UNIT, [5.0, 0.0, 15.0, 10.0]
    assert bbox_iou(a, b) == bbox_iou(b, a)


@pytest.mark.parametrize("bad", [None, [1.0, 2.0, 3.0], [10.0, 10.0, 0.0, 0.0], "nope"])
def test_bbox_iou_returns_zero_for_degenerate_inputs(bad):
    # Degenerate supports must never raise; callers rely on a 0.0 feature.
    assert bbox_iou(UNIT, bad) == 0.0
    assert bbox_iou(bad, UNIT) == 0.0


# ----------------------------------------------------------------------
# bbox_area / center / diagonal / area_ratio
# ----------------------------------------------------------------------
def test_bbox_area_known_values_and_degenerate_zero():
    assert bbox_area(UNIT) == pytest.approx(100.0)
    assert bbox_area([2.0, 3.0, 7.0, 9.0]) == pytest.approx(5.0 * 6.0)
    assert bbox_area([10.0, 10.0, 0.0, 0.0]) == 0.0
    assert bbox_area(None) == 0.0


def test_bbox_center_known_value_and_none_for_degenerate():
    assert bbox_center(UNIT) == [5.0, 5.0]
    assert bbox_center([2.0, 4.0, 8.0, 10.0]) == [5.0, 7.0]
    # None (not a sentinel point) so callers cannot mistake it for the origin.
    assert bbox_center([5.0, 5.0, 1.0, 1.0]) is None
    assert bbox_center(None) is None


def test_bbox_diagonal_known_value():
    # 3-4-5 triangle: width 3, height 4 -> diagonal 5.
    assert bbox_diagonal([0.0, 0.0, 3.0, 4.0]) == pytest.approx(5.0, abs=1e-12)
    assert bbox_diagonal(UNIT) == pytest.approx(math.hypot(10.0, 10.0), abs=1e-12)


@pytest.mark.parametrize("bbox", [None, [5.0, 5.0, 1.0, 1.0], [0.0, 0.0, 0.1, 0.1]])
def test_bbox_diagonal_is_floored_at_one(bbox):
    # The diagonal is a denominator in center_scale(); flooring at 1.0 keeps
    # ratios finite for degenerate and sub-pixel boxes.
    assert bbox_diagonal(bbox) == 1.0


def test_bbox_area_ratio_is_smaller_over_larger_and_never_exceeds_one():
    # The implementation always divides min area by max area, so the ratio is
    # symmetric and bounded by 1 regardless of argument order.
    small, large = [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 20.0, 20.0]
    assert bbox_area_ratio(small, large) == pytest.approx(100.0 / 400.0, abs=1e-12)
    assert bbox_area_ratio(large, small) == pytest.approx(0.25, abs=1e-12)
    assert bbox_area_ratio(small, large) == bbox_area_ratio(large, small)
    assert bbox_area_ratio(small, list(small)) == pytest.approx(1.0, abs=1e-12)


def test_bbox_area_ratio_is_one_for_congruent_but_displaced_boxes():
    # Area ratio measures size agreement only; position is the center term's job.
    assert bbox_area_ratio(UNIT, [500.0, 500.0, 510.0, 510.0]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("bad", [None, [10.0, 10.0, 0.0, 0.0]])
def test_bbox_area_ratio_is_zero_when_either_box_is_empty(bad):
    assert bbox_area_ratio(UNIT, bad) == 0.0
    assert bbox_area_ratio(bad, UNIT) == 0.0


# ----------------------------------------------------------------------
# bbox_center_distance
# ----------------------------------------------------------------------
def test_bbox_center_distance_known_pythagorean_value():
    # Centers (5,5) and (8,9) -> 3-4-5 triangle -> distance 5.
    assert bbox_center_distance(UNIT, [3.0, 4.0, 13.0, 14.0]) == pytest.approx(5.0, abs=1e-12)


def test_bbox_center_distance_zero_for_concentric_boxes():
    assert bbox_center_distance(UNIT, [2.0, 2.0, 8.0, 8.0]) == 0.0


def test_bbox_center_distance_pure_horizontal_shift():
    assert bbox_center_distance(UNIT, [7.0, 0.0, 17.0, 10.0]) == pytest.approx(7.0, abs=1e-12)


@pytest.mark.parametrize("bad", [None, [10.0, 10.0, 0.0, 0.0]])
def test_bbox_center_distance_is_infinite_for_degenerate_boxes(bad):
    # inf (not 0.0) so a missing box can never look like a perfect positional
    # match, and so max_center_distance_px vetoes it.
    assert math.isinf(bbox_center_distance(UNIT, bad))


# ----------------------------------------------------------------------
# normalize_bbox
# ----------------------------------------------------------------------
def test_normalize_bbox_returns_floats_for_a_valid_box():
    assert normalize_bbox((1, 2, 3, 4)) == [1.0, 2.0, 3.0, 4.0]
    assert all(isinstance(v, float) for v in normalize_bbox((1, 2, 3, 4)))


def test_normalize_bbox_rejects_reversed_coordinates_rather_than_swapping():
    # Documented behavior: reversed coords are treated as degenerate and return
    # None. It does not silently repair them, so upstream bugs stay visible.
    assert normalize_bbox([10.0, 10.0, 0.0, 0.0]) is None
    assert normalize_bbox([10.0, 0.0, 0.0, 10.0]) is None
    assert normalize_bbox([0.0, 10.0, 10.0, 0.0]) is None


def test_normalize_bbox_rejects_zero_width_or_height():
    assert normalize_bbox([5.0, 0.0, 5.0, 10.0]) is None
    assert normalize_bbox([0.0, 5.0, 10.0, 5.0]) is None


@pytest.mark.parametrize(
    "bad",
    [None, [], [1.0, 2.0, 3.0], "abc", 42, {"x": 1}, ["a", "b", "c", "d"], [0.0, 0.0, None, 5.0]],
)
def test_normalize_bbox_returns_none_for_missing_short_or_non_numeric_input(bad):
    assert normalize_bbox(bad) is None


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_normalize_bbox_rejects_non_finite_coordinates(bad_value):
    assert normalize_bbox([0.0, 0.0, bad_value, 5.0]) is None


def test_normalize_bbox_truncates_extra_coordinates():
    assert normalize_bbox([1.0, 2.0, 3.0, 4.0, 99.0]) == [1.0, 2.0, 3.0, 4.0]


# ----------------------------------------------------------------------
# clamp_unit
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,expected",
    [(5.0, 1.0), (1.0001, 1.0), (-2.0, 0.0), (-0.0001, 0.0), (0.0, 0.0), (1.0, 1.0), (0.42, 0.42)],
)
def test_clamp_unit_clamps_out_of_range_values(value, expected):
    assert clamp_unit(value) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("bad", [None, "abc", float("nan"), float("inf"), float("-inf"), object()])
def test_clamp_unit_falls_back_to_default_for_unusable_values(bad):
    # A NaN leaking into the weighted sum would poison the whole prior, so NaN
    # and inf resolve to the caller's default rather than propagating.
    assert clamp_unit(bad, default=0.25) == 0.25
    assert clamp_unit(bad) == 0.0


def test_clamp_unit_clamps_the_default_path_result_too():
    assert clamp_unit("abc", default=9.0) == 1.0
    assert clamp_unit("abc", default=-9.0) == 0.0


def test_clamp_unit_accepts_numeric_strings():
    assert clamp_unit("0.5") == pytest.approx(0.5)


# ----------------------------------------------------------------------
# center scale / center score
# ----------------------------------------------------------------------
def test_center_scale_is_the_larger_diagonal_times_the_factor():
    a, b = UNIT, [0.0, 0.0, 30.0, 40.0]  # diagonals ~14.14 and 50.0
    assert center_scale(a, b) == pytest.approx(50.0 * CENTER_SCALE_FACTOR, abs=1e-12)
    assert CENTER_SCALE_FACTOR == 0.75


def test_center_scale_is_floored_at_one():
    assert center_scale([0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]) == 1.0


def test_center_score_from_distance_known_values():
    assert center_score_from_distance(0.0, 10.0) == pytest.approx(1.0, abs=1e-12)
    assert center_score_from_distance(5.0, 10.0) == pytest.approx(0.5, abs=1e-12)
    # Beyond the scale the score saturates at 0 instead of going negative.
    assert center_score_from_distance(20.0, 10.0) == 0.0


def test_center_score_from_distance_handles_infinite_distance_and_tiny_scale():
    assert center_score_from_distance(float("inf"), 10.0) == 0.0
    assert center_score_from_distance(float("nan"), 10.0) == 0.0
    # Scale is floored at 1.0, so a 0 scale cannot divide by zero.
    assert center_score_from_distance(2.0, 0.0) == 0.0


# ----------------------------------------------------------------------
# diagnostic_score
# ----------------------------------------------------------------------
def test_diagnostic_score_is_the_documented_weighted_sum():
    iou, center, area, conf = 0.4, 0.6, 0.8, 0.9
    expected = 0.45 * iou + 0.30 * center + 0.15 * area + 0.10 * conf
    assert diagnostic_score(iou, center, area, conf) == pytest.approx(expected, abs=1e-12)


def test_diagnostic_score_is_one_for_a_perfect_match_and_zero_for_a_total_miss():
    assert diagnostic_score(1.0, 1.0, 1.0, 1.0) == pytest.approx(1.0, abs=1e-12)
    assert diagnostic_score(0.0, 0.0, 0.0, 0.0) == 0.0


def test_diagnostic_score_clamps_each_component_before_weighting():
    # Out-of-range components are clamped individually, so an inflated IoU
    # cannot buy more than its 0.45 share.
    assert diagnostic_score(5.0, 0.0, 0.0, 0.0) == pytest.approx(W_IOU, abs=1e-12)
    assert diagnostic_score(-5.0, 1.0, 0.0, 0.0) == pytest.approx(W_CENTER, abs=1e-12)


def test_diagnostic_score_renormalizes_weights_that_do_not_sum_to_one():
    # With non-unit weights the raw sum is divided by the weight total, keeping
    # the score comparable across weight sets.
    weights = {"iou": 2.0, "center": 0.0, "area": 0.0, "confidence": 0.0}
    assert diagnostic_score(0.5, 1.0, 1.0, 1.0, weights=weights) == pytest.approx(0.5, abs=1e-12)


def test_diagnostic_score_is_zero_when_all_weights_are_zero():
    weights = {"iou": 0.0, "center": 0.0, "area": 0.0, "confidence": 0.0}
    assert diagnostic_score(1.0, 1.0, 1.0, 1.0, weights=weights) == 0.0


# ----------------------------------------------------------------------
# compute_pair_features
# ----------------------------------------------------------------------
def test_compute_pair_features_matches_the_documented_weighting_for_a_concrete_pair():
    # Fully hand-checkable pair:
    #   a = [0,0,10,10], b = [5,0,15,10]
    #   iou            = 50/150 = 1/3
    #   center dist    = |5-10| = 5.0
    #   center scale   = max(diag)=14.142... * 0.75
    #   center score   = 1 - 5/(14.142*0.75)
    #   area ratio     = 100/100 = 1.0
    #   confidence     = mean(0.8, 0.6) = 0.7
    a, b = UNIT, [5.0, 0.0, 15.0, 10.0]
    features = compute_pair_features(a, b, observation_confidence=0.8, object_confidence=0.6)

    assert isinstance(features, PairFeatures)
    assert features.iou == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert features.center_distance_px == pytest.approx(5.0, abs=1e-12)

    expected_scale = math.hypot(10.0, 10.0) * 0.75
    expected_center = 1.0 - 5.0 / expected_scale
    assert features.center_score == pytest.approx(expected_center, abs=1e-12)
    assert features.area_ratio == pytest.approx(1.0, abs=1e-12)
    assert features.confidence_score == pytest.approx(0.7, abs=1e-12)

    expected_score = (
        0.45 * (1.0 / 3.0) + 0.30 * expected_center + 0.15 * 1.0 + 0.10 * 0.7
    )
    assert features.diagnostic_score == pytest.approx(expected_score, abs=1e-12)
    # Pinned numeric value so an accidental reweighting is caught immediately.
    assert features.diagnostic_score == pytest.approx(0.5285786437626905, abs=1e-12)


def test_compute_pair_features_perfect_overlap_scores_the_confidence_complement():
    # Identical boxes give iou=center=area=1, so the only shortfall is the
    # 0.10-weighted confidence term at its 0.5 default.
    features = compute_pair_features(UNIT, list(UNIT))
    assert features.confidence_score == pytest.approx(DEFAULT_CONFIDENCE, abs=1e-12)
    assert features.diagnostic_score == pytest.approx(0.95, abs=1e-12)


def test_compute_pair_features_averages_only_the_supplied_confidences():
    # One-sided confidence must not be averaged against the default, otherwise
    # a detector score would be silently halved toward 0.5.
    only_obs = compute_pair_features(UNIT, list(UNIT), observation_confidence=0.9)
    assert only_obs.confidence_score == pytest.approx(0.9, abs=1e-12)

    only_obj = compute_pair_features(UNIT, list(UNIT), object_confidence=0.3)
    assert only_obj.confidence_score == pytest.approx(0.3, abs=1e-12)

    neither = compute_pair_features(UNIT, list(UNIT))
    assert neither.confidence_score == pytest.approx(DEFAULT_CONFIDENCE, abs=1e-12)


def test_compute_pair_features_clamps_out_of_range_confidences_toward_the_default():
    # clamp_unit(value, DEFAULT_CONFIDENCE) clamps numeric out-of-range values
    # to the unit interval; the default only applies to unusable values.
    assert compute_pair_features(UNIT, list(UNIT), 5.0).confidence_score == pytest.approx(1.0)
    assert compute_pair_features(UNIT, list(UNIT), -5.0).confidence_score == pytest.approx(0.0)
    assert compute_pair_features(
        UNIT, list(UNIT), float("nan")
    ).confidence_score == pytest.approx(DEFAULT_CONFIDENCE)


@pytest.mark.parametrize("bad", [None, [1.0, 2.0, 3.0], [10.0, 10.0, 0.0, 0.0]])
def test_compute_pair_features_degenerate_support_yields_zeros_and_infinite_distance(bad):
    # Degenerate input must not raise, and must not look like a match. The
    # confidence term survives because it does not depend on geometry.
    features = compute_pair_features(bad, UNIT, observation_confidence=0.8)
    assert features.iou == 0.0
    assert features.center_score == 0.0
    assert features.area_ratio == 0.0
    assert features.diagnostic_score == 0.0
    assert math.isinf(features.center_distance_px)
    assert features.confidence_score == pytest.approx(0.8, abs=1e-12)


def test_compute_pair_features_honors_a_center_scale_factor_override():
    # A larger scale factor forgives the same displacement more.
    tight = compute_pair_features(UNIT, [5.0, 0.0, 15.0, 10.0], center_scale_factor=0.5)
    loose = compute_pair_features(UNIT, [5.0, 0.0, 15.0, 10.0], center_scale_factor=2.0)
    assert loose.center_score > tight.center_score
    # Geometry-only terms are unaffected by the scale factor.
    assert tight.iou == pytest.approx(loose.iou, abs=1e-12)


def test_compute_pair_features_honors_a_weights_override():
    a, b = UNIT, [5.0, 0.0, 15.0, 10.0]
    baseline = compute_pair_features(a, b)
    iou_only = compute_pair_features(
        a, b, weights={"iou": 1.0, "center": 0.0, "area": 0.0, "confidence": 0.0}
    )
    assert iou_only.diagnostic_score == pytest.approx(baseline.iou, abs=1e-12)
    assert iou_only.diagnostic_score != pytest.approx(baseline.diagnostic_score, abs=1e-9)


def test_compute_pair_features_is_symmetric_in_its_two_supports():
    a, b = UNIT, [5.0, 2.0, 17.0, 12.0]
    forward = compute_pair_features(a, b, 0.8, 0.4)
    backward = compute_pair_features(b, a, 0.4, 0.8)
    assert forward.to_dict() == backward.to_dict()


def test_pair_features_is_frozen_and_dict_round_trips_all_fields():
    features = compute_pair_features(UNIT, list(UNIT))
    with pytest.raises(dataclasses.FrozenInstanceError):
        features.iou = 0.0  # frozen dataclass
    assert set(features.to_dict()) == {
        "iou",
        "center_distance_px",
        "center_score",
        "area_ratio",
        "confidence_score",
        "diagnostic_score",
    }


# ----------------------------------------------------------------------
# legacy compatibility
# ----------------------------------------------------------------------
def test_legacy_center_score_uses_the_full_diagonal_with_no_scale_factor():
    # Legacy divides by max(diagonal); the canonical path divides by
    # max(diagonal) * 0.75. The legacy scale is therefore larger, so the same
    # displacement is penalized less and legacy scores higher.
    a, b = UNIT, [5.0, 0.0, 15.0, 10.0]
    distance = bbox_center_distance(a, b)
    full_diagonal = max(bbox_diagonal(a), bbox_diagonal(b), 1.0)

    legacy = legacy_center_score(a, b)
    canonical = center_score_from_distance(distance, center_scale(a, b))

    assert legacy == pytest.approx(1.0 - distance / full_diagonal, abs=1e-12)
    assert canonical == pytest.approx(
        1.0 - distance / (full_diagonal * CENTER_SCALE_FACTOR), abs=1e-12
    )
    assert legacy > canonical

    # The shortfalls differ by exactly 1/CENTER_SCALE_FACTOR while both scores
    # stay unsaturated, which is the precise algebraic relationship.
    assert (1.0 - canonical) == pytest.approx(
        (1.0 - legacy) / CENTER_SCALE_FACTOR, abs=1e-12
    )
    assert legacy == pytest.approx(0.6464466094067263, abs=1e-12)
    assert canonical == pytest.approx(0.5285954792089683, abs=1e-12)


def test_legacy_center_score_agrees_with_canonical_only_at_zero_distance():
    # With no displacement both scales are irrelevant, so the two paths agree.
    concentric = [2.0, 2.0, 8.0, 8.0]
    assert legacy_center_score(UNIT, concentric) == pytest.approx(1.0, abs=1e-12)
    assert center_score_from_distance(
        bbox_center_distance(UNIT, concentric), center_scale(UNIT, concentric)
    ) == pytest.approx(1.0, abs=1e-12)


def test_legacy_center_score_reaches_zero_for_far_boxes_like_the_canonical_path():
    far = [500.0, 500.0, 510.0, 510.0]
    assert legacy_center_score(UNIT, far) == 0.0


def test_legacy_center_score_is_zero_for_degenerate_boxes():
    assert legacy_center_score(UNIT, None) == 0.0


# ----------------------------------------------------------------------
# determinism
# ----------------------------------------------------------------------
def test_repeated_calls_on_the_same_boxes_are_bit_identical():
    # The stated purpose of this module is that a given pair of boxes always
    # yields identical features, no matter who asks or how often.
    a, b = [3.0, 7.0, 41.0, 63.0], [9.0, 11.0, 52.0, 70.0]
    first = compute_pair_features(a, b, 0.77, 0.31)
    for _ in range(5):
        again = compute_pair_features(a, b, 0.77, 0.31)
        assert again == first
        assert again.to_dict() == first.to_dict()


def test_equal_valued_boxes_from_different_containers_give_identical_features():
    # Tuple vs list, int vs float must not change a single feature.
    from_lists = compute_pair_features([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0], 0.5, 0.5)
    from_tuples = compute_pair_features((0, 0, 10, 10), (5, 0, 15, 10), 0.5, 0.5)
    assert from_lists == from_tuples


def test_primitive_helpers_are_deterministic_across_calls():
    a, b = [1.0, 2.0, 33.0, 44.0], [5.0, 6.0, 40.0, 50.0]
    assert [bbox_iou(a, b) for _ in range(3)] == [bbox_iou(a, b)] * 3
    assert [bbox_area_ratio(a, b) for _ in range(3)] == [bbox_area_ratio(a, b)] * 3
    assert [bbox_center_distance(a, b) for _ in range(3)] == [bbox_center_distance(a, b)] * 3
