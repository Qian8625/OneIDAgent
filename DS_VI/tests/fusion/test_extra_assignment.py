"""Independent candidates: one-to-one assignment vs the historical many-to-one."""
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


def _pairs(hypotheses):
    return [tuple(item.context["member_observation_ids"]) + (round(
        float(item.context["pair_features"]["diagnostic_score"]), 6),) for item in hypotheses]


def test_many_to_one_allows_a_shared_partner():
    left_near = _obs([0, 0, 10, 10], "img_a")
    left_far = _obs([4, 0, 14, 10], "img_a")
    right = _obs([1, 0, 11, 10], "img_b")

    many = _pruned_cross_view_hypotheses(
        [left_near, left_far], [right], 2, "ship", "g", max_links=1,
    )
    assert len(many) == 2
    assert {pair[1] for pair in _pairs(many)} == {right.observation_id}


def test_one_to_one_consumes_each_member_once():
    left_near = _obs([0, 0, 10, 10], "img_a")
    left_far = _obs([4, 0, 14, 10], "img_a")
    right = _obs([1, 0, 11, 10], "img_b")

    one = _pruned_cross_view_hypotheses(
        [left_near, left_far], [right], 2, "ship", "g", max_links=1,
        assignment="one_to_one",
    )
    assert len(one) == 1
    chosen = _pairs(one)[0]
    # The closer left box wins; the other one stays unused.
    assert chosen[0] == left_near.observation_id
    assert chosen[1] == right.observation_id


def test_one_to_one_keeps_disjoint_pairs():
    left_one = _obs([0, 0, 10, 10], "img_a")
    left_two = _obs([100, 100, 110, 110], "img_a")
    right_one = _obs([1, 0, 11, 10], "img_b")
    right_two = _obs([101, 100, 111, 110], "img_b")

    one = _pruned_cross_view_hypotheses(
        [left_one, left_two], [right_one, right_two], 2, "ship", "g", max_links=2,
        assignment="one_to_one",
    )
    pairs = _pairs(one)
    assert len(pairs) == 2
    assert sorted(pair[1] for pair in pairs) == sorted([right_one.observation_id, right_two.observation_id])
    lefts = [pair[0] for pair in pairs]
    assert len(set(lefts)) == 2


def test_assignment_is_deterministic():
    lefts = [_obs([index * 12, 0, index * 12 + 10, 10], "img_a") for index in range(4)]
    rights = [_obs([index * 12 + 1, 0, index * 12 + 11, 10], "img_b") for index in range(2)]
    first = _pairs(_pruned_cross_view_hypotheses(
        lefts, rights, 2, "ship", "g", max_links=2, assignment="one_to_one"))
    second = _pairs(_pruned_cross_view_hypotheses(
        lefts, rights, 2, "ship", "g", max_links=2, assignment="one_to_one"))
    assert first == second
