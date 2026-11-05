"""Unit tests for the weighted attribute-consistency score."""
from __future__ import annotations

from DS_VI.models.shared_identity import attribute_consistency as consistency


def test_orientation_never_votes():
    # Measured 0.18 (optical) / 0.31 (sar) on a 3-value domain: at or below chance.
    for family in ("match", "time"):
        assert consistency.pair_weights(family)["orientation"] == 0.0


def test_channel_weight_is_the_weaker_view():
    # Match pairs one optical and one SAR view: the SAR subtype channel is the
    # limiting one, so subtype must weigh less there than for time (optical/optical).
    assert consistency.pair_weights("match")["subtype"] < consistency.pair_weights("time")["subtype"]
    assert consistency.pair_weights("match")["size_bin"] > consistency.pair_weights("match")["subtype"]


def test_weak_channel_disagreement_abstains_in_match():
    weights = consistency.pair_weights("match", ("subtype",))
    score = consistency.consistency({"subtype": "cargo_ship"}, {"subtype": "warship"}, weights)
    assert score["asserted"] < 0.5
    # Below the asserted floor the candidate is neutral, not rejected.
    assert consistency.accept(score)


def test_strong_channel_disagreement_rejects_in_time():
    weights = consistency.pair_weights("time", ("subtype",))
    score = consistency.consistency({"subtype": "cargo_ship"}, {"subtype": "warship"}, weights)
    assert score["asserted"] >= 0.5
    assert not consistency.accept(score)


def test_size_conflict_rejects_everywhere():
    for family in ("match", "time"):
        weights = consistency.pair_weights(family)
        score = consistency.consistency({"size_bin": "large"}, {"size_bin": "small"}, weights)
        assert not consistency.accept(score)


def test_blanks_abstain_and_full_agreement_passes():
    weights = consistency.pair_weights("time")
    blank = consistency.consistency({"size_bin": "large"}, {}, weights)
    assert blank["asserted"] == 0.0
    assert consistency.accept(blank)

    agreed = consistency.consistency(
        {"subtype": "cargo_ship", "size_bin": "large", "state": "static"},
        {"subtype": "cargo_ship", "size_bin": "large", "state": "static"},
        weights,
    )
    assert agreed["net"] == 1.0
    assert consistency.accept(agreed, tau=0.5)


def test_partial_agreement_lands_between_the_margins():
    weights = consistency.pair_weights("time")
    score = consistency.consistency(
        {"subtype": "cargo_ship", "size_bin": "large"},
        {"subtype": "warship", "size_bin": "large"},
        weights,
    )
    # size agrees, subtype disagrees: net is positive but below 0.5.
    assert 0.0 < score["net"] < 0.5
    assert consistency.accept(score, tau=0.0)
    assert not consistency.accept(score, tau=0.5)
