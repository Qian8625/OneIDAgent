"""Decision-layer contract: every verdict branch, and why each one is reachable.

The decision layer is the only place allowed to emit a status label. These tests
pin the documented ordering in :func:`decide_object_truth` (insufficient ->
contradictory -> ambiguous -> supported/qualified/abstain) against thresholds
read from ``configs/shared_identity/default.yaml``.

Most cases build an :class:`EvidenceState` directly instead of running inference:
the decision layer is a pure function of that state, so hand-built states let a
single threshold be crossed in isolation without an inference run also moving.
"""
from __future__ import annotations

import math

import pytest

from DS_VI.models.shared_identity.acquisition import (
    suggest_acquisitions,
    summarize_requests,
)
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.decision import (
    coverage_ratio,
    decide_best_object,
    decide_object_truth,
)
from DS_VI.models.shared_identity.evidence_state import compact_summary
from DS_VI.models.shared_identity.evidence_store import EvidenceStore
from DS_VI.models.shared_identity.schema import (
    DecisionResult,
    EvidenceState,
    ObjectEvidenceSummary,
)

CONFIG = load_config()
THRESHOLDS = CONFIG["decision"]
# decision.expected_coverage_cells is not in the shipped YAML; the module default
# is 2 (one optical plus one SAR cell, or one modality across two epochs).
EXPECTED_COVERAGE_CELLS = int(THRESHOLDS.get("expected_coverage_cells", 2))


# ----------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------
def make_belief(
    attribute: str = "category",
    value: str = "ship",
    probability: float = 0.80,
    time_key: str = "t0",
    normalized_entropy: float = 0.20,
):
    return {
        attribute + "@" + time_key: {
            "attribute": attribute,
            "time_key": time_key,
            "value": value,
            "probability": float(probability),
            "entropy": 0.3,
            "normalized_entropy": float(normalized_entropy),
            "distribution": {value: float(probability), "other": 1.0 - float(probability)},
        }
    }


def make_summary(
    object_id: str = "obj_1",
    association_support: float = 1.80,
    supporting_observations=("m1", "m2"),
    truth_beliefs=None,
    modality_count: int = 2,
    time_count: int = 1,
    association_entropy: float = 0.40,
    contradictions=None,
) -> ObjectEvidenceSummary:
    """A well-supported object by default; each test perturbs one field."""
    return ObjectEvidenceSummary(
        object_id=object_id,
        association_support=float(association_support),
        supporting_observations=list(supporting_observations),
        truth_beliefs=make_belief() if truth_beliefs is None else truth_beliefs,
        coverage={
            "modalities": {"optical": 1, "sar": 1},
            "time_keys": {"t0": 2},
            "modality_count": int(modality_count),
            "time_count": int(time_count),
        },
        provenance_summary={"node_count": len(supporting_observations)},
        ambiguity={"association_entropy": float(association_entropy)},
        contradictions=list(contradictions or []),
    )


def make_state(*summaries: ObjectEvidenceSummary, null_mass: float = 0.05) -> EvidenceState:
    observations = sorted(
        {obs for summary in summaries for obs in summary.supporting_observations}
    )
    return EvidenceState(
        objects={summary.object_id: summary for summary in summaries},
        observation_count=len(observations),
        claim_count=len(observations),
        null_mass={observation: float(null_mass) for observation in observations},
    )


def assert_well_formed(result: DecisionResult) -> None:
    """Every branch must pass validate() and explain itself in words.

    An empty or machine-only reason list would leave the agent unable to report
    *why* it did not commit, which is the whole point of a non-committal verdict.
    """
    result.validate()
    assert result.status in DecisionResult.VALID_STATUSES
    assert result.reasons, f"{result.status} produced no reason"
    for reason in result.reasons:
        assert isinstance(reason, str) and reason.strip()
        # Human-readable: prose, not a bare token or number.
        assert len(reason) > 12, reason
        assert " " in reason.strip(), reason


# ----------------------------------------------------------------------
# 1. insufficient evidence
# ----------------------------------------------------------------------
def test_unknown_object_is_insufficient_evidence_not_a_crash():
    """Asking about an object that was never hypothesized must not invent one."""
    result = decide_object_truth(make_state(make_summary()), "obj_missing", "category")

    assert result.status == "Uncertain"
    assert result.object_id == "obj_missing"
    assert result.value is None
    assert_well_formed(result)


def test_no_supporting_observation_is_insufficient_evidence():
    """No observation associates above 0.5, so there is nothing to reason from.

    This case is checked before the truth beliefs are even consulted; a strong
    truth posterior attached to an unsupported object must not leak out as an
    answer.
    """
    orphan = make_summary(
        association_support=0.0,
        supporting_observations=(),
        truth_beliefs=make_belief(probability=0.99),
    )

    result = decide_object_truth(make_state(orphan), "obj_1", "category")

    assert result.status == "Uncertain"
    # Despite p=0.99 on the axis, no value is reported.
    assert result.value is None
    assert result.confidence == 0.0
    assert "no observation" in " ".join(result.reasons)
    assert_well_formed(result)


def test_coverage_below_min_coverage_is_insufficient_evidence():
    """Thin (modality, time) coverage blocks a verdict even with strong support."""
    thin = make_summary(modality_count=0, time_count=0)
    ratio = coverage_ratio(thin, EXPECTED_COVERAGE_CELLS)
    # Premise of the test: we really are under the configured floor.
    assert ratio < THRESHOLDS["min_coverage"]

    result = decide_object_truth(make_state(thin), "obj_1", "category")

    assert result.status == "Uncertain"
    assert any("coverage ratio" in reason for reason in result.reasons)
    assert result.evidence_refs["coverage_ratio"] == pytest.approx(ratio)
    assert_well_formed(result)


def test_mean_null_mass_above_max_is_insufficient_evidence():
    """High NULL mass means the supporting observations probably belong to nothing.

    The observations exist and coverage is fine, so only the NULL channel can
    veto here; that is the check under test.
    """
    null_mass = float(THRESHOLDS["max_null_mass"]) + 0.2
    state = make_state(make_summary(), null_mass=null_mass)

    result = decide_object_truth(state, "obj_1", "category")

    assert result.status == "Uncertain"
    assert any("NULL mass" in reason for reason in result.reasons)
    assert result.evidence_refs["mean_null_mass"] == pytest.approx(null_mass)
    # Coverage was not the problem, which isolates the NULL veto.
    assert result.evidence_refs["coverage_ratio"] >= THRESHOLDS["min_coverage"]
    assert_well_formed(result)


def test_missing_truth_axis_is_insufficient_evidence_not_a_guess():
    """A well-identified object with no belief on the asked axis cannot answer.

    Identity being resolved does not entitle the layer to answer a question no
    claim ever addressed.
    """
    result = decide_object_truth(make_state(make_summary()), "obj_1", "motion")

    assert result.status == "Uncertain"
    assert result.value is None
    assert any("no truth belief" in reason for reason in result.reasons)
    assert_well_formed(result)


# ----------------------------------------------------------------------
# 2. contradictory evidence
# ----------------------------------------------------------------------
def test_near_equal_competing_values_are_contradictory_evidence():
    """Two values with near-equal association support cannot be averaged away."""
    contested = make_summary(
        contradictions=[
            {
                "attribute": "category",
                "time_key": "t0",
                "competing_values": ["ship", "aircraft"],
                "support": [0.81, 0.80],
            }
        ]
    )

    result = decide_object_truth(make_state(contested), "obj_1", "category")

    assert result.status == "Conflict"
    # The competing values are surfaced so a caller can report the conflict.
    assert "ship" in " ".join(result.reasons) and "aircraft" in " ".join(result.reasons)
    assert result.evidence_refs["contradictions"][0]["competing_values"] == ["ship", "aircraft"]
    assert_well_formed(result)


def test_contradiction_on_another_attribute_does_not_block_this_one():
    """The contradiction filter is per-axis, so an unrelated clash is ignored."""
    contested = make_summary(
        contradictions=[
            {
                "attribute": "heading",
                "time_key": "t0",
                "competing_values": ["north", "south"],
                "support": [0.5, 0.49],
            }
        ]
    )

    result = decide_object_truth(make_state(contested), "obj_1", "category")

    assert result.status == "Confirmed"
    assert_well_formed(result)


def test_contradiction_outranks_ambiguous_identity():
    """Documented ordering: contradiction is reported before identity ambiguity.

    Both triggers are armed here. The more specific diagnosis must win, so a
    caller is told the values disagree rather than the vaguer 'identity unclear'.
    """
    both = make_summary(
        association_entropy=float(THRESHOLDS["ambiguity_entropy"]) + 0.5,
        contradictions=[
            {
                "attribute": "category",
                "time_key": "t0",
                "competing_values": ["ship", "aircraft"],
                "support": [0.8, 0.8],
            }
        ],
    )

    result = decide_object_truth(make_state(both), "obj_1", "category")

    assert result.status == "Conflict"
    assert_well_formed(result)


# ----------------------------------------------------------------------
# 3. ambiguous identity
# ----------------------------------------------------------------------
def test_high_association_entropy_is_ambiguous_identity_not_an_attribute_answer():
    """When identity is unresolved the attribute question is not yet answerable.

    The truth posterior is deliberately strong (p=0.95 >= truth_threshold). It
    must still not be reported as ``supported``, because "which object is this
    about" has no answer yet: a confident attribute of an unknown object is not
    a usable conclusion.
    """
    ambiguous = make_summary(
        association_entropy=float(THRESHOLDS["ambiguity_entropy"]) + 0.3,
        truth_beliefs=make_belief(probability=0.95),
    )

    result = decide_object_truth(make_state(ambiguous), "obj_1", "category")

    assert result.status == "Uncertain"
    assert result.confidence == pytest.approx(0.95)  # value carried, not endorsed
    assert any("association entropy" in reason for reason in result.reasons)
    assert_well_formed(result)


def test_top_association_below_identity_threshold_is_ambiguous_identity():
    """Mean association under ``identity_threshold`` also blocks the attribute.

    ``association_support`` is split thinly across the supporting observations,
    so no observation is confidently this object even though entropy is low.
    """
    identity_threshold = float(THRESHOLDS["identity_threshold"])
    weak = make_summary(
        association_support=2 * (identity_threshold - 0.2),  # mean over 2 observations
        association_entropy=0.1,
        truth_beliefs=make_belief(probability=0.95),
    )

    result = decide_object_truth(make_state(weak), "obj_1", "category")

    assert result.status == "Uncertain"
    assert result.evidence_refs["mean_association"] < identity_threshold
    assert any("identity threshold" in reason for reason in result.reasons)
    assert_well_formed(result)


# ----------------------------------------------------------------------
# 4/5/6. truth strength bands
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "probability, expected",
    [
        (float(THRESHOLDS["truth_threshold"]), "Confirmed"),
        (float(THRESHOLDS["truth_threshold"]) + 0.2, "Confirmed"),
        (float(THRESHOLDS["qualified_truth_threshold"]), "Weak Support"),
        (float(THRESHOLDS["truth_threshold"]) - 0.01, "Weak Support"),
        (float(THRESHOLDS["qualified_truth_threshold"]) - 0.01, "Weak Support"),
        (0.0, "Weak Support"),
    ],
)
def test_truth_probability_bands(probability, expected):
    """Both band edges are inclusive lower bounds, per the documented order."""
    summary = make_summary(truth_beliefs=make_belief(probability=probability))

    result = decide_object_truth(make_state(summary), "obj_1", "category")

    assert result.status == expected
    assert result.value == "ship"
    assert result.confidence == pytest.approx(probability)
    assert result.attribute == "category"
    assert_well_formed(result)


def test_time_key_filter_selects_the_requested_axis():
    """A ``time_key`` narrows the belief search rather than mixing epochs."""
    beliefs = {}
    beliefs.update(make_belief(value="ship", probability=0.9, time_key="t0"))
    beliefs.update(make_belief(value="aircraft", probability=0.85, time_key="t1"))
    summary = make_summary(truth_beliefs=beliefs)

    t0 = decide_object_truth(make_state(summary), "obj_1", "category", time_key="t0")
    t1 = decide_object_truth(make_state(summary), "obj_1", "category", time_key="t1")

    assert (t0.value, t1.value) == ("ship", "aircraft")
    assert t0.evidence_refs["truth_axis"] == "category@t0"
    assert t1.evidence_refs["truth_axis"] == "category@t1"


# ----------------------------------------------------------------------
# decide_best_object
# ----------------------------------------------------------------------
def test_decide_best_object_without_hypotheses_abstains():
    """No object hypotheses at all: abstain, do not crash or invent an answer."""
    result = decide_best_object(EvidenceState(), attribute="category")

    assert result.status == "Uncertain"
    assert result.object_id is None
    assert result.value is None
    assert result.confidence == 0.0
    assert "no object hypotheses" in " ".join(result.reasons)
    assert_well_formed(result)


def test_decide_best_object_picks_the_highest_support_object():
    """Ranking is by association support, not by declaration order."""
    weak = make_summary(
        object_id="obj_weak",
        association_support=0.9,
        supporting_observations=("m1",),
        truth_beliefs=make_belief(value="tank", probability=0.95),
    )
    strong = make_summary(
        object_id="obj_strong",
        association_support=1.9,
        supporting_observations=("m2", "m3"),
        truth_beliefs=make_belief(value="ship", probability=0.85),
    )

    result = decide_best_object(make_state(weak, strong), attribute="category")

    assert result.object_id == "obj_strong"
    assert result.value == "ship"
    assert result.status == "Confirmed"
    assert_well_formed(result)


# ----------------------------------------------------------------------
# coverage_ratio
# ----------------------------------------------------------------------
def test_coverage_ratio_endpoints_and_capping():
    """0 when nothing is covered, 1.0 when the expected cells are all present."""
    nothing = make_summary(modality_count=0, time_count=0)
    assert coverage_ratio(nothing, 2) == 0.0

    # observed = modality_count + max(time_count - 1, 0) = 2 + 0 = 2 of 2.
    both_modalities = make_summary(modality_count=2, time_count=1)
    assert coverage_ratio(both_modalities, 2) == 1.0

    # One modality across two epochs also fills 2 cells.
    two_epochs = make_summary(modality_count=1, time_count=2)
    assert coverage_ratio(two_epochs, 2) == 1.0

    # Partial coverage is proportional, never negative.
    assert coverage_ratio(make_summary(modality_count=1, time_count=1), 2) == 0.5

    # Excess observation is capped at 1.0 so surplus cannot offset another gap.
    assert coverage_ratio(make_summary(modality_count=4, time_count=3), 2) == 1.0

    # A non-positive expectation means coverage cannot be judged; treat as met
    # rather than dividing by zero.
    assert coverage_ratio(nothing, 0) == 1.0


# ----------------------------------------------------------------------
# acquisition
# ----------------------------------------------------------------------
def test_suggest_acquisitions_proposes_useful_requests_for_a_thin_ambiguous_state():
    """An ambiguous, single-modality object should get actionable next steps."""
    thin_ambiguous = ObjectEvidenceSummary(
        object_id="obj_1",
        association_support=1.2,
        supporting_observations=["m1", "m2"],
        truth_beliefs=make_belief(probability=0.35, normalized_entropy=0.95),
        coverage={
            "modalities": {"optical": 2},
            "time_keys": {"t0": 2},
            "modality_count": 1,
            "time_count": 1,
        },
        ambiguity={"association_entropy": 1.10},
    )

    requests = suggest_acquisitions(make_state(thin_ambiguous))
    summary = summarize_requests(requests)

    assert summary["count"] == len(requests) >= 3
    types = [item["type"] for item in summary["requests"]]
    # All three rules fired: ambiguous identity, undecided truth axis, and the
    # SAR cell that was never observed.
    assert "discriminative_attribute" in types
    assert "attribute_evidence" in types
    assert "missing_modality" in types
    assert "sar" in [item["modality"] for item in summary["requests"]]

    # Highest priority first, so a caller can take requests[0] and be right.
    priorities = [item["priority"] for item in summary["requests"]]
    assert priorities == sorted(priorities, reverse=True)
    assert summary["requests"][0]["type"] == "discriminative_attribute"

    # Truncated to the configured budget.
    assert len(requests) <= int(CONFIG["acquisition"]["max_requests"])


def test_suggest_acquisitions_stays_quiet_on_a_settled_state():
    """A confident, fully covered object needs nothing more.

    Without this the acquisition layer would always demand another observation
    and the agent could never stop.
    """
    settled = ObjectEvidenceSummary(
        object_id="obj_1",
        association_support=1.9,
        supporting_observations=["m1", "m2"],
        truth_beliefs=make_belief(probability=0.97, normalized_entropy=0.05),
        coverage={
            "modalities": {"optical": 1, "sar": 1},
            "time_keys": {"t0": 2},
            "modality_count": 2,
            "time_count": 1,
        },
        ambiguity={"association_entropy": 0.10},
    )

    summary = summarize_requests(suggest_acquisitions(make_state(settled)))

    assert summary["count"] == 0
    assert summary["requests"] == []


def test_summarize_requests_on_empty_input():
    assert summarize_requests([]) == {"count": 0, "requests": []}


def _thin_ambiguous_summary() -> ObjectEvidenceSummary:
    return ObjectEvidenceSummary(
        object_id="obj_1",
        association_support=1.2,
        supporting_observations=["m1", "m2"],
        truth_beliefs=make_belief(probability=0.35, normalized_entropy=0.95),
        coverage={
            "modalities": {"optical": 2},
            "time_keys": {"t0": 2},
            "modality_count": 1,
            "time_count": 1,
        },
        ambiguity={"association_entropy": 1.10},
    )


def test_suggest_acquisitions_filters_to_public_capabilities():
    capabilities = [
        {"request_type": "missing_modality", "modality": "sar", "cost": 1.0}
    ]

    summary = summarize_requests(
        suggest_acquisitions(
            make_state(_thin_ambiguous_summary()),
            capabilities=capabilities,
            budget=1.0,
        )
    )

    # Only the action the environment can actually execute survives; the
    # policy must not be told to acquire evidence no tool can produce.
    assert summary["count"] == 1
    assert summary["requests"][0]["type"] == "missing_modality"
    assert summary["requests"][0]["modality"] == "sar"


def test_suggest_acquisitions_drops_unaffordable_requests():
    capabilities = [
        {"request_type": "missing_modality", "modality": "sar", "cost": 2.0}
    ]

    summary = summarize_requests(
        suggest_acquisitions(
            make_state(_thin_ambiguous_summary()),
            capabilities=capabilities,
            budget=0.5,
        )
    )

    assert summary["count"] == 0


def test_suggest_acquisitions_drops_already_queried_requests():
    capabilities = [
        {"request_type": "missing_modality", "modality": "sar", "cost": 1.0}
    ]
    queried = {("missing_modality", "sar", None, None)}

    summary = summarize_requests(
        suggest_acquisitions(
            make_state(_thin_ambiguous_summary()),
            capabilities=capabilities,
            budget=1.0,
            queried=queried,
        )
    )

    assert summary["count"] == 0


def test_summarize_requests_keeps_time_axis_and_rationale():
    summary = summarize_requests(suggest_acquisitions(make_state(_thin_ambiguous_summary())))
    request = summary["requests"][0]

    assert "time_key" in request
    assert request["rationale"]


# ----------------------------------------------------------------------
# compact_summary
# ----------------------------------------------------------------------
def test_compact_summary_respects_top_k_and_exposes_the_decision_inputs():
    """The digest must carry exactly what the decision layer reasons over.

    It is what reaches the LLM context, so every field the decision cites has to
    be present, and ``top_k`` has to actually bound the payload size.
    """
    objects = [
        make_summary(
            object_id=f"obj_{index}",
            association_support=1.9 - 0.1 * index,
            supporting_observations=(f"m{index}a", f"m{index}b"),
            contradictions=[
                {
                    "attribute": "category",
                    "time_key": "t0",
                    "competing_values": ["ship", "tank"],
                    "support": [0.8, 0.79],
                }
            ]
            if index == 0
            else [],
        )
        for index in range(4)
    ]
    state = make_state(*objects)

    digest = compact_summary(state, top_k=2)

    assert len(digest["top_objects"]) == 2
    # Ranked by association support, so the truncation keeps the best candidates.
    assert [item["object_id"] for item in digest["top_objects"]] == ["obj_0", "obj_1"]

    top = digest["top_objects"][0]
    for key in (
        "association_support",
        "supporting_observations",
        "association_entropy",
        "truth",
        "contradictions",
    ):
        assert key in top, key
    # Observation ids are collapsed to a count: the digest is a size bound, not
    # a copy of the store.
    assert top["supporting_observations"] == 2
    assert top["contradictions"] == 1
    assert top["truth"]["category@t0"] == {"value": "ship", "probability": 0.8}
    assert digest["observation_count"] == state.observation_count
    assert digest["mean_null_mass"] == pytest.approx(0.05)

    # top_k is floored at 1 so a zero or negative request cannot blank the digest.
    assert len(compact_summary(state, top_k=0)["top_objects"]) == 1


def test_compact_summary_of_an_empty_state_is_safe():
    digest = compact_summary(EvidenceState(), top_k=3)

    assert digest["top_objects"] == []
    assert digest["mean_null_mass"] == 0.0


# ----------------------------------------------------------------------
# end-to-end sanity against a real store
# ----------------------------------------------------------------------
def test_real_store_agrees_with_the_hand_built_supported_branch():
    """One consistent optical+SAR pair really does reach ``supported``.

    The branch tests above use synthetic states; this anchors them to the actual
    inference output so a threshold rename or unit change cannot pass unnoticed.
    """
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill",
        {
            "error_code": 0,
            "matched_pairs": [
                {
                    "rgb_bbox": [100, 100, 140, 140],
                    "sar_bbox": [101, 100, 141, 141],
                    "rgb_label": "ship",
                    "sar_label": "ship",
                    "rgb_score": 0.92,
                    "sar_score": 0.88,
                    "match_score": 0.85,
                    "iou": 0.80,
                }
            ],
        },
        {"optical_image": "/tmp/optical.png", "sar_image": "/tmp/sar.png"},
        round_id=0,
        api_args={"rgb_image": "/tmp/optical.png", "sar_image": "/tmp/sar.png"},
    )

    evidence_state = store.build_evidence_state()
    result = decide_best_object(evidence_state, attribute="category")

    assert result.status == "Confirmed"
    assert result.value == "ship"
    assert result.confidence >= float(THRESHOLDS["truth_threshold"])
    # The pair was split into two observations, so both modalities are covered.
    assert result.evidence_refs["coverage_ratio"] == 1.0
    assert len(result.evidence_refs["supporting_observations"]) == 2
    assert math.isfinite(result.evidence_refs["mean_null_mass"])
    assert_well_formed(result)
