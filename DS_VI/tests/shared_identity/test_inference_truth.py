"""Object-truth (Q) invariants for the shared-identity inference layer.

``Q[k, a, t, v]`` is indexed by (object, attribute, time).  Each such variable is
normalized on its own, ``__NULL__`` never gets one, and finite evidence never
produces a probability of exactly 1.
"""
from __future__ import annotations

import math

from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.inference import (
    build_value_domains,
    run_joint_inference,
    truth_priors_for,
    update_truth,
)
from DS_VI.models.shared_identity.reliability import ReliabilityModel
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

EPSILON = 1e-12
OBJECT_BBOX = [10, 20, 50, 80]


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


def build_problem(observations, claims, objects, config=None):
    cfg = dict(config or load_config())
    edges = build_candidate_graph(observations, objects, cfg)
    problem = InferenceProblem(
        observations=list(observations),
        claims=list(claims),
        objects=list(objects),
        candidate_edges=edges,
    )
    return problem, cfg


def single_object_scene(claim_specs, config=None):
    """One object, one observation per claim spec, all boxes on the object.

    ``claim_specs`` is a sequence of ``(attribute, value, channel, time_key)``.
    Separate observations are used so the claims are independent evidence rather
    than one shared derivation group.
    """
    observations = []
    claims = []
    for index, (attribute, value, channel, time_key) in enumerate(claim_specs):
        observation_id = f"m{index}"
        observations.append(
            make_observation(
                observation_id,
                channel,
                OBJECT_BBOX,
                time_key=time_key,
                input_root=f"root_{index}",
            )
        )
        claims.append(
            make_claim(
                f"c{index}", observation_id, attribute, value,
                time_key=time_key, channel=channel,
            )
        )
    objects = [make_object("obj", "ship", OBJECT_BBOX)]
    return build_problem(observations, claims, objects, config)


# ----------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------
def test_every_truth_variable_normalizes_independently():
    """Each (object, attribute, time) is its own distribution summing to 1."""
    problem, cfg = single_object_scene(
        [
            ("category", "ship", "optical", "t0"),
            ("subtype", "cargo", "optical", "t0"),
            ("presence", "present", "temporal", "t0"),
            ("presence", "present", "temporal", "t1"),
        ]
    )
    state = run_joint_inference(problem, cfg)

    # Truth is keyed by the full triple, so distinct attributes and distinct
    # times must appear as distinct keys.
    keys = set(state.truth)
    assert ("obj", "category", "t0") in keys
    assert ("obj", "subtype", "t0") in keys
    assert ("obj", "presence", "t0") in keys
    assert ("obj", "presence", "t1") in keys

    domains = state.diagnostics["value_domains"]
    for key, distribution in state.truth.items():
        _, attribute, _ = key
        assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-9), key
        assert all(p >= 0.0 for p in distribution.values()), key
        # The support must be the full configured domain, so a value no tool
        # mentioned is still representable (and gets non-zero prior mass).
        assert set(distribution) == set(domains[attribute]), key


def test_null_object_has_no_truth_distribution():
    """``__NULL__`` is 'no object', so it cannot own attribute beliefs.

    Giving NULL a truth variable would let unassigned observations vote on a
    phantom object and then feed that back into association.
    """
    obs_near = make_observation("m_near", "optical", OBJECT_BBOX, input_root="r1")
    obs_far = make_observation("m_far", "optical", [900, 900, 940, 940], input_root="r2")
    objects = [make_object("obj", "ship", OBJECT_BBOX)]
    claims = [
        make_claim("c_near", "m_near", "category", "ship"),
        make_claim("c_far", "m_far", "category", "vehicle"),
    ]
    problem, cfg = build_problem([obs_near, obs_far], claims, objects)

    state = run_joint_inference(problem, cfg)

    # The far observation really is on NULL, so NULL carries claim mass here.
    assert state.top_object("m_far")[0] == NULL_OBJECT_ID
    assert NULL_OBJECT_ID in state.diagnostics["object_ids"]
    assert not [key for key in state.truth if key[0] == NULL_OBJECT_ID]


def test_two_agreeing_claims_beat_one():
    """Corroboration must raise the posterior, not merely leave it unchanged."""
    one_problem, cfg = single_object_scene([("category", "ship", "optical", "t0")])
    one = run_joint_inference(one_problem, cfg).truth[("obj", "category", "t0")]["ship"]

    two_problem, cfg = single_object_scene(
        [("category", "ship", "optical", "t0"), ("category", "ship", "optical", "t0")]
    )
    two = run_joint_inference(two_problem, cfg).truth[("obj", "category", "t0")]["ship"]

    assert two > one, (one, two)
    # Still not certain: two independent noisy sensors are not a proof.
    assert two < 1.0


def test_wrong_minority_does_not_overturn_majority():
    """A confident wrong claim must lose to the majority and never reach 1.0."""
    problem, cfg = single_object_scene(
        [
            ("category", "ship", "optical", "t0"),
            ("category", "ship", "optical", "t0"),
            ("category", "aircraft", "optical", "t0"),
        ]
    )
    state = run_joint_inference(problem, cfg)
    distribution = state.truth[("obj", "category", "t0")]

    assert max(distribution, key=distribution.get) == "ship"
    assert distribution["ship"] > distribution["aircraft"]
    # Strictly below 1.0: reliability is clamped at max 0.97, so no finite amount
    # of evidence may become logical certainty.
    assert distribution["ship"] < 1.0
    # The wrong value keeps a non-zero share, which is what makes the posterior
    # revisable by later evidence instead of latched.
    assert distribution["aircraft"] > 0.0


def test_higher_reliability_modality_wins_a_conflict():
    """With optical.category 0.88 vs sar.category 0.78, optical must win.

    Then the reliabilities are swapped via config override and the winner must
    flip, proving the outcome is driven by reliability and not by claim order or
    by anything geometric.
    """
    specs = [("category", "ship", "optical", "t0"), ("category", "aircraft", "sar", "t0")]

    default_cfg = load_config()
    reliability_table = default_cfg["reliability"]["by_modality_attribute"]
    assert reliability_table["optical"]["category"] == 0.88
    assert reliability_table["sar"]["category"] == 0.78

    problem, cfg = single_object_scene(specs, default_cfg)
    state = run_joint_inference(problem, cfg)
    assert state.reliability["c0"] == 0.88 and state.reliability["c1"] == 0.78
    distribution = state.truth[("obj", "category", "t0")]
    assert distribution["ship"] > distribution["aircraft"], distribution

    swapped_cfg = load_config(
        overrides={
            "reliability": {
                "by_modality_attribute": {
                    "optical": {"category": 0.78},
                    "sar": {"category": 0.88},
                }
            }
        }
    )
    swapped_problem, swapped_cfg = single_object_scene(specs, swapped_cfg)
    swapped_state = run_joint_inference(swapped_problem, swapped_cfg)
    assert swapped_state.reliability["c0"] == 0.78 and swapped_state.reliability["c1"] == 0.88
    swapped_distribution = swapped_state.truth[("obj", "category", "t0")]
    assert swapped_distribution["aircraft"] > swapped_distribution["ship"], swapped_distribution


def test_truth_at_different_times_are_independent_variables():
    """t0 and t1 beliefs must not overwrite each other.

    Collapsing them would make 'present at t0, absent at t1' look like a
    contradiction instead of a change, which is the whole point of the time axis.
    """
    obs_t0 = make_observation(
        "m_t0", "temporal", [10, 20, 50, 80], time_key="t0", input_root="root_before"
    )
    obs_t1 = make_observation(
        "m_t1", "temporal", [12, 22, 52, 82], time_key="t1", input_root="root_after"
    )
    objects = [make_object("obj_temporal", "vehicle", [11, 21, 51, 81])]
    claims = [
        make_claim("ct0", "m_t0", "presence", "present", time_key="t0", channel="temporal"),
        make_claim("ct1", "m_t1", "presence", "absent", time_key="t1", channel="temporal"),
    ]
    problem, cfg = build_problem([obs_t0, obs_t1], claims, objects)

    state = run_joint_inference(problem, cfg)
    at_t0 = state.truth[("obj_temporal", "presence", "t0")]
    at_t1 = state.truth[("obj_temporal", "presence", "t1")]

    assert math.isclose(sum(at_t0.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(sum(at_t1.values()), 1.0, abs_tol=1e-9)
    assert max(at_t0, key=at_t0.get) == "present"
    assert max(at_t1, key=at_t1.get) == "absent"
    # Neither variable was flattened toward the other's evidence.
    assert at_t0["present"] > 0.5 and at_t1["absent"] > 0.5

    # And a t1-only claim must leave the t0 variable at its prior.
    only_t1_problem, cfg = build_problem(
        [obs_t1],
        [make_claim("ct1", "m_t1", "presence", "absent", time_key="t1", channel="temporal")],
        [make_object("obj_temporal", "vehicle", [11, 21, 51, 81])],
    )
    only_t1_state = run_joint_inference(only_t1_problem, cfg)
    assert ("obj_temporal", "presence", "t0") not in only_t1_state.truth


def test_update_truth_direction_isolated_from_association():
    """Reliability alone decides the winner when association is held identical.

    Calling ``update_truth`` directly pins A to 1.0 on the object for both
    claims, so no geometric difference can explain the result.
    """
    problem, cfg = single_object_scene(
        [("category", "ship", "optical", "t0"), ("category", "aircraft", "sar", "t0")]
    )
    observation_ids = ["m0", "m1"]
    object_ids = ["obj", NULL_OBJECT_ID]
    association = [[1.0, 0.0], [1.0, 0.0]]
    weights = {"c0": 1.0, "c1": 1.0}
    domains = build_value_domains(problem, cfg)
    priors = truth_priors_for(problem, domains, 1.0)

    observations = {obs.observation_id: obs for obs in problem.observations}
    reliabilities = ReliabilityModel(cfg).build_table(problem.claims, observations)
    assert reliabilities == {"c0": 0.88, "c1": 0.78}

    truth = update_truth(
        problem, observation_ids, object_ids, association,
        reliabilities, weights, domains, priors, EPSILON,
    )
    distribution = truth[("obj", "category", "t0")]
    assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-9)
    assert distribution["ship"] > distribution["aircraft"]

    flipped = update_truth(
        problem, observation_ids, object_ids, association,
        {"c0": 0.78, "c1": 0.88}, weights, domains, priors, EPSILON,
    )
    flipped_distribution = flipped[("obj", "category", "t0")]
    assert flipped_distribution["aircraft"] > flipped_distribution["ship"]
    # NULL still owns no truth variable even when called directly.
    assert not [key for key in flipped if key[0] == NULL_OBJECT_ID]
