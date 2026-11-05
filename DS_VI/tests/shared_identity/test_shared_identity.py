"""The shared-identity structural invariant.

One :class:`PrimitiveObservation` is the identity unit.  All claims derived from
it read the same row ``A[m, :]``; there is no per-claim identity anywhere.  These
tests pin that property down and contrast it against the untied (B2) setup where
per-claim rows are free to diverge.
"""
from __future__ import annotations

import copy
import math

from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.inference import (
    identity_contradiction_rate,
    run_joint_inference,
)
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    Claim,
    InferenceProblem,
    Modality,
    ObjectHypothesis,
    PrimitiveObservation,
    SpatialSupport,
    TimeScope,
)

OBJECT_A_BBOX = [100, 100, 140, 140]
OBJECT_B_BBOX = [140, 100, 180, 140]
# Sits exactly between A and B, so geometry alone cannot decide identity.
MID_BBOX = [120, 100, 160, 140]


# ----------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------
def make_observation(
    observation_id: str,
    modality: str,
    bbox,
    time_key: str = "t0",
    input_root: str = "root_a",
    score: float = 0.9,
    source_tool: str = "SyntheticSensor",
) -> PrimitiveObservation:
    return PrimitiveObservation(
        observation_id=observation_id,
        modality=Modality.coerce(modality),
        source_tool=source_tool,
        time_scope=TimeScope(key=time_key),
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        detection_confidence=float(score),
        input_roots=[input_root],
    )


def make_claim(
    claim_id: str,
    observation_id: str,
    attribute: str,
    value,
    time_key: str = "t0",
    channel: str = "optical",
    source_tool: str = "SyntheticSensor",
) -> Claim:
    return Claim(
        claim_id=claim_id,
        observation_id=observation_id,
        attribute=attribute,
        value=value,
        time_scope=TimeScope(key=time_key),
        source_tool=source_tool,
        source_channel=channel,
    )


def make_object(object_id: str, label: str, bbox) -> ObjectHypothesis:
    return ObjectHypothesis(
        object_id=object_id,
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        attribute_hints={"category": label},
    )


def build_problem(observations, claims, objects, config=None, capacities=None):
    cfg = dict(config or load_config())
    edges = build_candidate_graph(observations, objects, cfg)
    problem = InferenceProblem(
        observations=list(observations),
        claims=list(claims),
        objects=list(objects),
        candidate_edges=edges,
        capacities=dict(capacities or {}),
    )
    return problem, cfg


def untie_claims(observations, claims):
    """Give every claim its own pseudo-observation (the B2 ablation).

    Mirrors ``tf_eval.shared_identity.evaluator._untie_claims``: each claim gets a
    deep copy of its parent observation under a new id, which deliberately breaks
    the shared-identity invariant so claim-wise identity becomes observable.
    """
    parents = {observation.observation_id: observation for observation in observations}
    untied_observations = []
    untied_claims = []
    for claim in claims:
        parent = parents.get(claim.observation_id)
        if parent is None:
            continue
        pseudo_id = f"{claim.observation_id}__{claim.claim_id}"
        pseudo = copy.deepcopy(parent)
        pseudo.observation_id = pseudo_id
        pseudo.claim_ids = [claim.claim_id]
        untied_observations.append(pseudo)
        untied_claim = copy.deepcopy(claim)
        untied_claim.observation_id = pseudo_id
        untied_claims.append(untied_claim)
    return untied_observations, untied_claims


def discriminative_scene(mid_category: str, untied: bool = False):
    """Two distinguishable objects plus one ambiguous observation between them.

    ``obj_a`` is pinned to ``ship`` / ``civilian`` and ``obj_b`` to ``aircraft`` /
    ``cargo`` by anchor observations that sit exactly on them.  The ``mid``
    observation is geometrically tied between the two, so its identity can only
    be resolved by its own semantic claims.

    Capacity is raised to 3 per object on purpose. At the default capacity of 2
    the two anchors alone saturate an object's column, so any mass ``mid`` gains
    is taken straight back off the anchors by the projection and the object's
    total support cannot grow. That is correct projection behaviour, but it would
    mask the effect under test, so the scene leaves room for a third observation.
    """
    objects = [
        make_object("obj_a", "ship", OBJECT_A_BBOX),
        make_object("obj_b", "aircraft", OBJECT_B_BBOX),
    ]
    anchor_a1 = make_observation("a1", "optical", OBJECT_A_BBOX, input_root="root_a1")
    anchor_a2 = make_observation("a2", "optical", OBJECT_A_BBOX, input_root="root_a2")
    anchor_b1 = make_observation("b1", "optical", OBJECT_B_BBOX, input_root="root_b1")
    anchor_b2 = make_observation("b2", "optical", OBJECT_B_BBOX, input_root="root_b2")
    mid = make_observation("mid", "optical", MID_BBOX, input_root="root_mid")

    observations = [anchor_a1, anchor_a2, anchor_b1, anchor_b2, mid]
    claims = [
        make_claim("a1_cat", "a1", "category", "ship"),
        make_claim("a2_sub", "a2", "subtype", "civilian"),
        make_claim("b1_cat", "b1", "category", "aircraft"),
        make_claim("b2_sub", "b2", "subtype", "cargo"),
        # The three claims of the ambiguous observation. `mid_cat` is the varied
        # discriminative claim. `mid_sub=cargo` deliberately points the *other*
        # way (obj_b owns 'cargo'), which is what lets the untied contrast test
        # observe two fragments of one perception act disagreeing. `mid_head` is
        # not asserted by either anchor, so it is pure passenger evidence.
        make_claim("mid_cat", "mid", "category", mid_category),
        make_claim("mid_sub", "mid", "subtype", "cargo"),
        make_claim("mid_head", "mid", "heading", "north"),
    ]
    if untied:
        observations, claims = untie_claims(observations, claims)
    return build_problem(
        observations, claims, objects, capacities={"obj_a": 3, "obj_b": 3}
    )


def symmetric_scene(attribute: str, value):
    """Two identical ship hypotheses with one observation exactly between them."""
    objects = [
        make_object("obj_c1", "ship", [80, 100, 120, 140]),
        make_object("obj_c2", "ship", [120, 100, 160, 140]),
    ]
    anchor_1 = make_observation("s1", "optical", [80, 100, 120, 140], input_root="root_s1")
    anchor_2 = make_observation("s2", "optical", [120, 100, 160, 140], input_root="root_s2")
    mid = make_observation("mid", "optical", [100, 100, 140, 140], input_root="root_mid")
    claims = [
        make_claim("s1_cat", "s1", "category", "ship"),
        make_claim("s2_cat", "s2", "category", "ship"),
        make_claim("mid_claim", "mid", attribute, value),
    ]
    return build_problem([anchor_1, anchor_2, mid], claims, objects)


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------
def test_three_claims_share_exactly_one_association_row():
    """A has one row per observation, never one per claim.

    This is the whole revision in one assertion: 3 claims on 1 observation must
    produce 1 association row, and every claim must read that same row.
    """
    observation = make_observation("m1", "optical", [10, 20, 50, 80])
    objects = [make_object("obj", "ship", [10, 20, 50, 80])]
    claims = [
        make_claim("c1", "m1", "category", "ship"),
        make_claim("c2", "m1", "subtype", "cargo"),
        make_claim("c3", "m1", "heading", "north"),
    ]
    problem, cfg = build_problem([observation], claims, objects)

    state = run_joint_inference(problem, cfg)

    assert len(problem.claims) == 3
    # The count is tied to observations, not claims. If identity were per claim
    # this would be 3.
    assert len(state.association) == len(problem.observations) == 1
    assert set(state.association) == {"m1"}

    # Each claim resolves to the identical row object, by identity of content.
    rows = {
        claim.claim_id: tuple(sorted(state.association_row(claim.observation_id).items()))
        for claim in problem.claims
    }
    assert len(set(rows.values())) == 1, rows

    # All three claims were pooled into one provenance group with a shared
    # budget, so correlated evidence is not triple-counted.
    group_ids = {claim.provenance_group_id for claim in problem.claims}
    assert len(group_ids) == 1
    weights = state.diagnostics["provenance_weights"]
    assert math.isclose(sum(weights[claim.claim_id] for claim in problem.claims), 1.0, abs_tol=1e-9)


def test_identity_contradiction_rate_is_exactly_zero_for_multi_claim_problems():
    """The invariant metric must be 0.0 exactly, not merely small.

    A non-zero value would mean claims of one observation disagreed about
    identity, which the data structure makes impossible; it is a bug signal, not
    a tunable threshold.
    """
    problem, cfg = discriminative_scene("ship")
    state = run_joint_inference(problem, cfg)

    # Premise: at least one observation carries more than one claim, otherwise
    # the metric has nothing to consider and 0.0 would be trivially true.
    multi_claim = [
        observation_id
        for observation_id, claims in problem.claims_by_observation().items()
        if len(claims) >= 2
    ]
    assert multi_claim, "scene must contain a multi-claim observation"

    assert identity_contradiction_rate(state, problem) == 0.0
    # Also recorded automatically on every run so a regression surfaces there too.
    assert state.diagnostics["identity_contradiction_rate"] == 0.0


def test_claims_never_gain_an_identity_field():
    """Claims must stay identity-free before and after inference.

    ``Claim.FORBIDDEN_FIELDS`` names the attributes whose reintroduction would
    silently restore per-claim identity.
    """
    problem, cfg = discriminative_scene("ship")

    for claim in problem.claims:
        for forbidden in Claim.FORBIDDEN_FIELDS:
            assert not hasattr(claim, forbidden), (claim.claim_id, forbidden)

    state = run_joint_inference(problem, cfg)

    for claim in problem.claims:
        serialized = claim.to_dict()
        for forbidden in Claim.FORBIDDEN_FIELDS:
            assert not hasattr(claim, forbidden), (claim.claim_id, forbidden)
            # Nor smuggled in through the free-form context dict.
            assert forbidden not in claim.context, (claim.claim_id, forbidden)
            # Nor present in the serialized form that crosses process boundaries.
            assert forbidden not in serialized, (claim.claim_id, forbidden)

    # And nothing about a claim id appears as an association key.
    claim_ids = {claim.claim_id for claim in problem.claims}
    assert not (claim_ids & set(state.association))


def test_discriminative_claim_moves_the_whole_shared_row():
    """One informative claim must move every claim of its observation together.

    Only ``mid_cat`` changes between the two runs, yet the entire shared row for
    ``mid`` flips. Because the row is shared, ``mid_sub`` and ``mid_head`` are
    carried along; they cannot stay behind on the other object.
    """
    ship_problem, cfg = discriminative_scene("ship")
    ship_state = run_joint_inference(ship_problem, cfg)
    ship_row = ship_state.association_row("mid")

    aircraft_problem, cfg = discriminative_scene("aircraft")
    aircraft_state = run_joint_inference(aircraft_problem, cfg)
    aircraft_row = aircraft_state.association_row("mid")

    # The discriminative claim decides which real object leads.
    assert ship_row["obj_a"] > ship_row["obj_b"], ship_row
    assert aircraft_row["obj_b"] > aircraft_row["obj_a"], aircraft_row

    # There is exactly one row for `mid`, and all three of its claims read it.
    mid_claims = [claim for claim in ship_problem.claims if claim.observation_id == "mid"]
    assert len(mid_claims) == 3
    per_claim_rows = {
        claim.claim_id: tuple(sorted(ship_state.association_row(claim.observation_id).items()))
        for claim in mid_claims
    }
    assert len(set(per_claim_rows.values())) == 1, per_claim_rows

    # The move is visible in the shared row, not in per-claim rows: there simply
    # are no per-claim rows to inspect.
    assert set(ship_state.association) == {"a1", "a2", "b1", "b2", "mid"}
    assert identity_contradiction_rate(ship_state, ship_problem) == 0.0
    assert identity_contradiction_rate(aircraft_state, aircraft_problem) == 0.0

    # The truth side confirms the *other two* claims of `mid` moved with it: the
    # `heading=north` claim is unique to `mid`, so whichever object the shared row
    # favours is the one whose heading belief it lifts. This is the load-bearing
    # check that identity moved for the whole observation, not just for the
    # discriminative claim.
    ship_north_a = ship_state.truth[("obj_a", "heading", "t0")]["north"]
    ship_north_b = ship_state.truth[("obj_b", "heading", "t0")]["north"]
    aircraft_north_a = aircraft_state.truth[("obj_a", "heading", "t0")]["north"]
    aircraft_north_b = aircraft_state.truth[("obj_b", "heading", "t0")]["north"]
    assert ship_north_a > aircraft_north_a
    assert aircraft_north_b > ship_north_b

    # Same for the `subtype=cargo` claim of `mid`, which is also carried along.
    ship_cargo_a = ship_state.truth[("obj_a", "subtype", "t0")]["cargo"]
    ship_cargo_b = ship_state.truth[("obj_b", "subtype", "t0")]["cargo"]
    aircraft_cargo_a = aircraft_state.truth[("obj_a", "subtype", "t0")]["cargo"]
    aircraft_cargo_b = aircraft_state.truth[("obj_b", "subtype", "t0")]["cargo"]
    assert ship_cargo_a > aircraft_cargo_a
    assert aircraft_cargo_b > ship_cargo_b


def test_untied_claims_can_diverge_which_is_what_shared_identity_prevents():
    """Contrast: split the claims into pseudo-observations and rows diverge.

    Same scene, same config, same evidence; the only change is the identity unit.
    Under the untied (B2) setup ``mid_cat`` and ``mid_sub`` prefer *different*
    objects, i.e. one perception act ends up split across two physical objects.
    The tied run cannot express that, which is the shared-identity contribution.
    """
    tied_problem, cfg = discriminative_scene("ship")
    tied_state = run_joint_inference(tied_problem, cfg)

    untied_problem, cfg = discriminative_scene("ship", untied=True)
    untied_state = run_joint_inference(untied_problem, cfg)

    # Structural difference: one row per claim instead of one row per observation.
    assert len(tied_state.association) == len(tied_problem.observations) == 5
    assert len(untied_state.association) == len(untied_problem.claims) == 7

    def best_real_object(row):
        real = {k: v for k, v in row.items() if k != NULL_OBJECT_ID}
        return max(real, key=real.get) if real else None

    cat_row = untied_state.association_row("mid__mid_cat")
    sub_row = untied_state.association_row("mid__mid_sub")

    # The two fragments of one perception act now disagree about identity.
    assert best_real_object(cat_row) == "obj_a", cat_row
    assert best_real_object(sub_row) == "obj_b", sub_row
    # And the rows are genuinely far apart, not a rounding difference.
    assert max(abs(cat_row.get(k, 0.0) - sub_row.get(k, 0.0)) for k in set(cat_row) | set(sub_row)) > 0.1

    # In the tied run there is only one row, so this divergence is unrepresentable.
    tied_row = tied_state.association_row("mid")
    assert best_real_object(tied_row) == "obj_a", tied_row
    for claim_id in ("mid_cat", "mid_sub", "mid_head"):
        assert f"mid__{claim_id}" not in tied_state.association


def test_common_attribute_does_not_shift_odds_between_symmetric_candidates():
    """A non-discriminative claim must leave the candidate ratio at 1.

    ``category=ship`` fits both hypotheses equally, so it may raise total real
    mass but must not prefer either object. Any drift would be evidence invented
    out of a shared property.
    """
    common_problem, cfg = symmetric_scene("category", "ship")
    common_state = run_joint_inference(common_problem, cfg)
    common_row = common_state.association_row("mid")

    unrelated_problem, cfg = symmetric_scene("heading", "north")
    unrelated_state = run_joint_inference(unrelated_problem, cfg)
    unrelated_row = unrelated_state.association_row("mid")

    for label, row in (("common", common_row), ("unrelated", unrelated_row)):
        assert abs(row["obj_c1"] - row["obj_c2"]) < 1e-9, (label, row)
        # Odds ratio stays 1: neither candidate is preferred.
        assert math.isclose(row["obj_c1"] / row["obj_c2"], 1.0, rel_tol=1e-9), (label, row)
        # Neither may be reported as a confident identity.
        assert row["obj_c1"] < 0.5 and row["obj_c2"] < 0.5, (label, row)

    # The matching category does raise the total real-object mass (it makes the
    # scene more plausible overall) while still refusing to break the tie.
    common_real = common_row["obj_c1"] + common_row["obj_c2"]
    unrelated_real = unrelated_row["obj_c1"] + unrelated_row["obj_c2"]
    assert common_real > unrelated_real
    assert identity_contradiction_rate(common_state, common_problem) == 0.0
