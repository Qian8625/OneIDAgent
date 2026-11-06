"""Provenance: replay fingerprints, derivation groups, and weight budgets.

Two amplification failures are pinned down here:

1. **Replay amplification** — re-running the same tool with the same normalized
   inputs must fingerprint identically and contribute weight once.
2. **Correlated-evidence double counting** — one image -> one detector -> one crop
   -> three attribute claims is *one* piece of evidence.  The claims share a
   single weight budget, so three of them cannot outvote three independent
   observations.
"""
from __future__ import annotations

import pytest

from DS_VI.models.shared_identity.provenance import (
    ProvenanceGraph,
    assign_group_ids,
    compute_evidence_fingerprint,
    compute_group_weights,
    deterministic_restatement_ids,
    normalize_tool_inputs,
    register_problem_provenance,
    restatements_from_observations,
    verify_group_budget,
)
from DS_VI.models.shared_identity.schema import (
    Claim,
    InferenceProblem,
    Modality,
    PrimitiveObservation,
    SpatialSupport,
    TimeScope,
)


# ----------------------------------------------------------------------
# builders
# ----------------------------------------------------------------------
def make_observation(
    observation_id: str,
    input_root: str,
    bbox=(10.0, 20.0, 50.0, 80.0),
    modality: str = "optical",
    time_key: str = "t0",
) -> PrimitiveObservation:
    observation = PrimitiveObservation(
        observation_id=observation_id,
        modality=Modality.coerce(modality),
        source_tool="SyntheticDetector",
        time_scope=TimeScope(key=time_key),
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        detection_confidence=0.9,
        input_roots=[input_root],
    )
    observation.validate()
    return observation


def make_claim(
    claim_id: str,
    observation_id: str,
    attribute: str,
    value,
    time_key: str = "t0",
) -> Claim:
    claim = Claim(
        claim_id=claim_id,
        observation_id=observation_id,
        attribute=attribute,
        value=value,
        time_scope=TimeScope(key=time_key),
        source_tool="SyntheticDetector",
        source_channel="optical",
    )
    claim.validate()
    return claim


# ----------------------------------------------------------------------
# fingerprinting
# ----------------------------------------------------------------------
def test_fingerprint_is_deterministic_for_identical_tool_and_inputs():
    inputs = {"image": "/data/scene.png", "prompt": "ship", "threshold": 0.35}

    first = compute_evidence_fingerprint("ObjectDetection", inputs)
    second = compute_evidence_fingerprint("ObjectDetection", dict(inputs))

    # Replay detection is *only* possible if an identical request hashes
    # identically; a nondeterministic fingerprint would silently let every
    # re-run count as fresh evidence.
    assert first == second


def test_fingerprint_differs_when_inputs_differ():
    base = compute_evidence_fingerprint("ObjectDetection", {"image": "/data/a.png"})
    other_image = compute_evidence_fingerprint("ObjectDetection", {"image": "/data/b.png"})
    other_value = compute_evidence_fingerprint(
        "ObjectDetection", {"image": "/data/a.png", "threshold": 0.5}
    )

    # A different input is genuinely new evidence and must not be suppressed as
    # a replay, so the fingerprint has to separate these cases.
    assert base != other_image
    assert base != other_value


def test_fingerprint_differs_across_tools_with_the_same_inputs():
    inputs = {"image": "/data/a.png"}

    # Two different tools looking at one image are two independent measurements,
    # not a replay of each other.
    assert compute_evidence_fingerprint("ObjectDetection", inputs) != (
        compute_evidence_fingerprint("SARDetection", inputs)
    )


def test_fingerprint_differs_when_input_roots_differ():
    inputs = {"prompt": "ship"}

    assert compute_evidence_fingerprint(
        "ObjectDetection", inputs, input_roots=["/data/a.png"]
    ) != compute_evidence_fingerprint(
        "ObjectDetection", inputs, input_roots=["/data/b.png"]
    )


def test_fingerprint_ignores_input_root_ordering():
    roots_forward = compute_evidence_fingerprint(
        "CrossModalInterpret", {}, input_roots=["/opt.png", "/sar.png"]
    )
    roots_reversed = compute_evidence_fingerprint(
        "CrossModalInterpret", {}, input_roots=["/sar.png", "/opt.png"]
    )

    # Root order is an artifact of dict iteration in the caller, not evidence.
    assert roots_forward == roots_reversed


# ----------------------------------------------------------------------
# input normalization
# ----------------------------------------------------------------------
def test_normalize_tool_inputs_makes_key_order_irrelevant():
    forward = {"image": "/data/a.png", "prompt": "ship", "threshold": 0.35}
    reordered = {"threshold": 0.35, "prompt": "ship", "image": "/data/a.png"}

    assert normalize_tool_inputs(forward) == normalize_tool_inputs(reordered)
    # The fingerprint must inherit the order-insensitivity: two callers building
    # the same request with differently ordered kwargs are one piece of evidence.
    assert compute_evidence_fingerprint("ObjectDetection", forward) == (
        compute_evidence_fingerprint("ObjectDetection", reordered)
    )


def test_normalize_tool_inputs_is_recursive_for_nested_mappings():
    forward = {"outer": {"b": 1, "a": {"z": 3, "y": 4}}}
    reordered = {"outer": {"a": {"y": 4, "z": 3}, "b": 1}}

    assert normalize_tool_inputs(forward) == normalize_tool_inputs(reordered)


def test_normalize_tool_inputs_drops_volatile_keys():
    first = {"image": "/data/a.png", "request_id": "req-1", "timestamp": 111.0}
    second = {"image": "/data/a.png", "request_id": "req-2", "timestamp": 222.0}

    # request_id / timestamp change on every call.  If they were fingerprinted,
    # no replay would ever be detected and the posterior could be inflated by
    # simply calling the same tool again.
    assert compute_evidence_fingerprint("ObjectDetection", first) == (
        compute_evidence_fingerprint("ObjectDetection", second)
    )


def test_normalize_tool_inputs_preserves_list_order():
    # Lists are positional (e.g. a bbox), so reordering them *is* a different
    # request and must not be canonicalized away.
    assert normalize_tool_inputs([1, 2, 3]) == [1, 2, 3]
    assert normalize_tool_inputs([3, 2, 1]) != normalize_tool_inputs([1, 2, 3])


def test_normalize_tool_inputs_rounds_floats_to_stable_precision():
    # Float noise far below any threshold the tools use should not defeat replay
    # detection.
    assert normalize_tool_inputs(0.350000001) == normalize_tool_inputs(0.35)
    assert normalize_tool_inputs(0.35) != normalize_tool_inputs(0.36)


# ----------------------------------------------------------------------
# duplicate registration
# ----------------------------------------------------------------------
def test_register_detects_exact_duplicate_node():
    graph = ProvenanceGraph()

    original = graph.register("node_a", "observation", "ObjectDetection", "fp_identical")
    replay = graph.register("node_b", "observation", "ObjectDetection", "fp_identical")

    assert original.duplicate_of is None
    assert graph.is_duplicate("node_a") is False
    # The second registration of an identical fingerprint is a replay: it points
    # back at the original owner so its claims can be zero-weighted.
    assert replay.duplicate_of == "node_a"
    assert graph.is_duplicate("node_b") is True


def test_register_does_not_mark_distinct_fingerprints_as_duplicates():
    graph = ProvenanceGraph()

    graph.register("node_a", "observation", "ObjectDetection", "fp_a")
    graph.register("node_b", "observation", "ObjectDetection", "fp_b")

    assert graph.is_duplicate("node_a") is False
    assert graph.is_duplicate("node_b") is False


def test_re_registering_the_same_node_id_is_not_a_duplicate():
    graph = ProvenanceGraph()

    graph.register("node_a", "observation", "ObjectDetection", "fp_a")
    graph.register("node_a", "observation", "ObjectDetection", "fp_a")

    # A node owning its own fingerprint is idempotent re-ingestion of the same
    # node, not a second independent piece of evidence to discount.
    assert graph.is_duplicate("node_a") is False
    assert graph.summarize()["duplicate_count"] == 0


def test_dedup_can_be_disabled_by_config():
    graph = ProvenanceGraph({"provenance": {"dedup_exact_replay": False}})

    graph.register("node_a", "observation", "T", "fp_identical")
    replay = graph.register("node_b", "observation", "T", "fp_identical")

    assert replay.duplicate_of is None


# ----------------------------------------------------------------------
# root tracing
# ----------------------------------------------------------------------
def test_roots_of_traces_a_derivation_chain_back_to_the_input_root():
    graph = ProvenanceGraph()
    graph.register("img", "input", "ImageLoader", "fp_img", input_roots=["/data/scene.png"])
    graph.register("det", "observation", "ObjectDetection", "fp_det", parents=["img"])
    graph.register("crop", "observation", "CropTool", "fp_crop", parents=["det"])
    graph.register("attr", "claim", "AttributeTool", "fp_attr", parents=["crop"])

    # Only the image node declares the root; the claim four hops downstream must
    # still resolve to it, because that is how correlated evidence is detected.
    assert graph.roots_of("attr") == ["/data/scene.png"]
    assert graph.roots_of("crop") == ["/data/scene.png"]


def test_roots_of_merges_multiple_roots_without_duplicating_them():
    graph = ProvenanceGraph()
    graph.register("opt", "input", "Loader", "fp_opt", input_roots=["/opt.png"])
    graph.register("sar", "input", "Loader", "fp_sar", input_roots=["/sar.png"])
    graph.register("match", "observation", "CrossModalInterpret", "fp_m", parents=["opt", "sar"])
    # A second path back to the same optical root must not double-list it.
    graph.register("claim", "claim", "AttributeTool", "fp_c", parents=["match", "opt"])

    assert sorted(graph.roots_of("claim")) == ["/opt.png", "/sar.png"]
    assert len(graph.roots_of("claim")) == 2


def test_roots_of_returns_empty_for_unknown_node():
    assert ProvenanceGraph().roots_of("does_not_exist") == []


def test_roots_of_terminates_on_a_cycle():
    graph = ProvenanceGraph()
    # A malformed graph (parents pointing at each other) must not hang the
    # ingest path; the _seen set guards the recursion.
    graph.register("a", "claim", "T", "fp_a", parents=["b"], input_roots=["root_a"])
    graph.register("b", "claim", "T", "fp_b", parents=["a"], input_roots=["root_b"])

    roots = graph.roots_of("a")

    assert sorted(roots) == ["root_a", "root_b"]
    # Each root appears once despite the cycle re-entering both nodes.
    assert len(roots) == len(set(roots))


def test_roots_of_terminates_on_a_self_loop():
    graph = ProvenanceGraph()
    graph.register("s", "claim", "T", "fp_s", parents=["s"], input_roots=["root_s"])

    assert graph.roots_of("s") == ["root_s"]


# ----------------------------------------------------------------------
# group assignment
# ----------------------------------------------------------------------
def test_assign_group_ids_groups_claims_from_one_derivation_chain_together():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_cat", "obs_shared", "category", "ship"),
        make_claim("clm_sub", "obs_shared", "subtype", "cargo"),
        make_claim("clm_head", "obs_shared", "heading", "north"),
    ]

    groups = assign_group_ids(claims, {"obs_shared": observation})

    # One image -> one detection -> three attribute claims is one piece of
    # evidence.  Sharing a group is what later forces them to share one budget.
    assert len(set(groups.values())) == 1
    assert all(claim.provenance_group_id == groups[claim.claim_id] for claim in claims)


def test_assign_group_ids_separates_claims_from_different_input_roots():
    observation_a = make_observation("obs_a", "/data/root_a.png")
    observation_b = make_observation("obs_b", "/data/root_b.png", bbox=(60, 20, 100, 80))
    claims = [
        make_claim("clm_a", "obs_a", "category", "ship"),
        make_claim("clm_b", "obs_b", "category", "ship"),
    ]

    groups = assign_group_ids(claims, {"obs_a": observation_a, "obs_b": observation_b})

    # Independent acquisitions are independent evidence; collapsing them into
    # one group would suppress genuine corroboration.
    assert groups["clm_a"] != groups["clm_b"]


def test_assign_group_ids_respects_an_explicit_adapter_group():
    observation = make_observation("obs_x", "/data/scene.png")
    claim = make_claim("clm_x", "obs_x", "category", "ship")
    claim.provenance_group_id = "grp_adapter_supplied"

    groups = assign_group_ids([claim], {"obs_x": observation})

    # An adapter that knows two observations are correlated (e.g. a crop of a
    # detection) can pre-declare the group, and that must win.
    assert groups["clm_x"] == "grp_adapter_supplied"
    assert claim.provenance_group_id == "grp_adapter_supplied"


def test_default_grouping_is_connected_components_over_shared_roots():
    """Shared raw-input roots connect claims under the default A.19 grouping.

    Two detections in one image are one connected component, so they share one
    budget.  The legacy per-observation mode remains available for the paired
    ablation and is selected explicitly.
    """
    root = "/data/scene.png"
    observation_a = make_observation("obs_1", root, bbox=(10, 20, 50, 80))
    observation_b = make_observation("obs_2", root, bbox=(60, 20, 100, 80))
    observations = {"obs_1": observation_a, "obs_2": observation_b}
    claims = [
        make_claim("clm_1", "obs_1", "category", "ship"),
        make_claim("clm_2", "obs_2", "category", "ship"),
    ]

    connected = assign_group_ids(claims, observations)
    assert connected["clm_1"] == connected["clm_2"]
    assert connected["clm_1"].startswith("grp_cc_")

    legacy_claims = [
        make_claim("clm_1", "obs_1", "category", "ship"),
        make_claim("clm_2", "obs_2", "category", "ship"),
    ]
    legacy = assign_group_ids(legacy_claims, observations, grouping_mode="observation")
    assert legacy["clm_1"] != legacy["clm_2"]


def test_derivation_parents_connect_observations_with_distinct_roots():
    parent = make_observation("obs_parent", "/data/parent.png")
    child = make_observation("obs_child", "/data/crop.png")
    child.derivation_parents = ["obs_parent"]
    claims = [
        make_claim("clm_parent", "obs_parent", "category", "ship"),
        make_claim("clm_child", "obs_child", "subtype", "cargo"),
    ]

    groups = assign_group_ids(claims, {"obs_parent": parent, "obs_child": child})

    assert groups["clm_parent"] == groups["clm_child"]


def test_deterministic_restatements_are_explicit_never_inferred_from_values():
    original = make_claim("clm_1", "obs_1", "category", "ship")
    restated = make_claim("clm_2", "obs_1", "category", "ship")
    restated.context["deterministic_restatement_of"] = "clm_1"

    assert deterministic_restatement_ids([original, restated]) == {"clm_2"}
    # Equal values alone are legitimate duplicate detections, never a replay.
    twin = make_claim("clm_3", "obs_1", "category", "ship")
    assert deterministic_restatement_ids([original, twin]) == set()

    weights = compute_group_weights(
        [original, restated], None, 1.0, restatements={"clm_2"}
    )
    assert weights["clm_1"] == pytest.approx(1.0)
    assert weights["clm_2"] == 0.0


def test_observation_level_deterministic_transform_marks_all_of_its_claims():
    observation = make_observation("obs_crop", "/data/crop.png")
    observation.context["derivation_kind"] = "deterministic_transform"
    claims = [
        make_claim("clm_cat", "obs_crop", "category", "ship"),
        make_claim("clm_sub", "obs_crop", "subtype", "cargo"),
    ]

    removed = restatements_from_observations(claims, {"obs_crop": observation})

    assert removed == {"clm_cat", "clm_sub"}


# ----------------------------------------------------------------------
# group weights
# ----------------------------------------------------------------------
def test_group_weight_total_never_exceeds_the_budget():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_cat", "obs_shared", "category", "ship"),
        make_claim("clm_sub", "obs_shared", "subtype", "cargo"),
        make_claim("clm_head", "obs_shared", "heading", "north"),
        make_claim("clm_mot", "obs_shared", "motion", "static"),
    ]
    assign_group_ids(claims, {"obs_shared": observation})

    weights = compute_group_weights(claims, None, 1.0)

    total = sum(weights[claim.claim_id] for claim in claims)
    # Four claims off one detection contribute at most one unit of evidence:
    # N correlated claims cannot vote N times.
    assert total == pytest.approx(1.0)
    assert all(weights[claim.claim_id] == pytest.approx(0.25) for claim in claims)


def test_more_correlated_claims_do_not_buy_more_total_weight():
    observation = make_observation("obs_shared", "/data/scene.png")
    small = [make_claim("clm_1", "obs_shared", "category", "ship")]
    large = [
        make_claim(f"clm_{i}", "obs_shared", "category", f"value_{i}") for i in range(8)
    ]
    assign_group_ids(small, {"obs_shared": observation})
    assign_group_ids(large, {"obs_shared": observation})

    small_total = sum(compute_group_weights(small, None, 1.0).values())
    large_total = sum(compute_group_weights(large, None, 1.0).values())

    # A tool that emits eight fields instead of one gains no extra influence.
    assert small_total == pytest.approx(large_total)
    assert large_total == pytest.approx(1.0)


def test_independent_input_roots_each_get_their_own_budget():
    observation_a = make_observation("obs_a", "/data/root_a.png")
    observation_b = make_observation("obs_b", "/data/root_b.png", bbox=(60, 20, 100, 80))
    claims = [
        make_claim("clm_a1", "obs_a", "category", "ship"),
        make_claim("clm_a2", "obs_a", "subtype", "cargo"),
        make_claim("clm_b1", "obs_b", "category", "ship"),
    ]
    assign_group_ids(claims, {"obs_a": observation_a, "obs_b": observation_b})

    weights = compute_group_weights(claims, None, 1.0)

    group_a = weights["clm_a1"] + weights["clm_a2"]
    group_b = weights["clm_b1"]
    # Deduplication must not become evidence suppression: two independent
    # acquisitions carry two full budgets, i.e. corroboration still counts.
    assert group_a == pytest.approx(1.0)
    assert group_b == pytest.approx(1.0)
    assert sum(weights.values()) == pytest.approx(2.0)


def test_duplicate_claims_get_zero_weight_and_do_not_dilute_the_original():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_original", "obs_shared", "category", "ship"),
        make_claim("clm_replay_1", "obs_shared", "subtype", "cargo"),
        make_claim("clm_replay_2", "obs_shared", "heading", "north"),
    ]
    assign_group_ids(claims, {"obs_shared": observation})

    graph = ProvenanceGraph()
    graph.register("clm_original", "claim", "ObjectDetection", "fp_same")
    graph.register("clm_replay_1", "claim", "ObjectDetection", "fp_same")
    graph.register("clm_replay_2", "claim", "ObjectDetection", "fp_same")

    weights = compute_group_weights(claims, graph)

    assert weights["clm_replay_1"] == 0.0
    assert weights["clm_replay_2"] == 0.0
    # Replays are excluded from the group size too, so the original keeps its
    # full weight instead of being diluted to 1/3 by its own echoes.
    assert weights["clm_original"] == pytest.approx(1.0)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_group_of_only_duplicates_contributes_nothing():
    make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_owner", "obs_other", "category", "ship"),
        make_claim("clm_replay", "obs_shared", "category", "ship"),
    ]
    # Put the replay in its own group so the group has no effective member.
    claims[0].provenance_group_id = "grp_owner"
    claims[1].provenance_group_id = "grp_replay_only"

    graph = ProvenanceGraph()
    graph.register("clm_owner", "claim", "T", "fp_same")
    graph.register("clm_replay", "claim", "T", "fp_same")

    weights = compute_group_weights([claims[1]], graph)

    # A group made entirely of replays adds zero, never a default share.
    assert weights["clm_replay"] == 0.0
    assert sum(weights.values()) == pytest.approx(0.0)


def test_graph_budget_overrides_the_positional_budget_argument():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_1", "obs_shared", "category", "ship"),
        make_claim("clm_2", "obs_shared", "subtype", "cargo"),
    ]
    assign_group_ids(claims, {"obs_shared": observation})
    graph = ProvenanceGraph({"provenance": {"group_budget": 0.5}})

    weights = compute_group_weights(claims, graph, group_budget=1.0)

    # The configured budget is authoritative; a caller default cannot widen it.
    assert sum(weights.values()) == pytest.approx(0.5)


def test_claims_without_a_group_fall_back_to_their_observation():
    # No assign_group_ids call: the weighting must still bound per observation
    # rather than treating ungrouped claims as independent.
    claims = [
        make_claim("clm_1", "obs_shared", "category", "ship"),
        make_claim("clm_2", "obs_shared", "subtype", "cargo"),
    ]

    weights = compute_group_weights(claims, None, 1.0)

    assert sum(weights.values()) == pytest.approx(1.0)


# ----------------------------------------------------------------------
# budget verification
# ----------------------------------------------------------------------
def test_verify_group_budget_accepts_a_compliant_weighting():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_1", "obs_shared", "category", "ship"),
        make_claim("clm_2", "obs_shared", "subtype", "cargo"),
    ]
    assign_group_ids(claims, {"obs_shared": observation})
    weights = compute_group_weights(claims, None, 1.0)

    report = verify_group_budget(claims, weights, 1.0)

    # compute_group_weights output must be self-consistent with the verifier,
    # otherwise the invariant is unenforceable in production.
    assert report["ok"] is True
    assert report["violations"] == {}
    assert report["group_sums"][claims[0].provenance_group_id] == pytest.approx(1.0)


def test_verify_group_budget_rejects_an_over_budget_weighting():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_1", "obs_shared", "category", "ship"),
        make_claim("clm_2", "obs_shared", "subtype", "cargo"),
    ]
    assign_group_ids(claims, {"obs_shared": observation})
    group_id = claims[0].provenance_group_id

    # Each correlated claim given a full unit: exactly the double-counting bug.
    report = verify_group_budget(claims, {"clm_1": 1.0, "clm_2": 1.0}, 1.0)

    assert report["ok"] is False
    assert group_id in report["violations"]
    assert report["violations"][group_id] == pytest.approx(2.0)


def test_verify_group_budget_tolerates_floating_point_noise():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_1", "obs_shared", "category", "ship"),
        make_claim("clm_2", "obs_shared", "subtype", "cargo"),
        make_claim("clm_3", "obs_shared", "heading", "north"),
    ]
    assign_group_ids(claims, {"obs_shared": observation})
    weights = compute_group_weights(claims, None, 1.0)

    # 3 * (1/3) is not exactly 1.0 in binary floating point; the verifier must
    # not report a violation for rounding alone.
    assert verify_group_budget(claims, weights, 1.0)["ok"] is True


def test_verify_group_budget_counts_missing_weights_as_zero():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [make_claim("clm_1", "obs_shared", "category", "ship")]
    assign_group_ids(claims, {"obs_shared": observation})

    report = verify_group_budget(claims, {}, 1.0)

    assert report["ok"] is True
    assert report["group_sums"][claims[0].provenance_group_id] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------
def test_summarize_reports_counts_duplicates_and_budget():
    graph = ProvenanceGraph()
    graph.register("obs_1", "observation", "ObjectDetection", "fp_a")
    graph.register("obs_2", "observation", "ObjectDetection", "fp_b")
    graph.register("obs_3", "observation", "ObjectDetection", "fp_a")

    summary = graph.summarize()

    assert summary["node_count"] == 3
    # One of the three nodes replays fp_a, so exactly two fingerprints are real
    # evidence.  These two numbers are the auditable replay record.
    assert summary["duplicate_count"] == 1
    assert summary["duplicate_nodes"] == ["obs_3"]
    assert summary["distinct_fingerprints"] == 2
    assert summary["group_budget"] == pytest.approx(1.0)


def test_summarize_reflects_a_configured_group_budget():
    graph = ProvenanceGraph({"provenance": {"group_budget": 0.75}})

    assert graph.summarize()["group_budget"] == pytest.approx(0.75)


def test_summarize_on_an_empty_graph():
    summary = ProvenanceGraph().summarize()

    assert summary["node_count"] == 0
    assert summary["duplicate_count"] == 0
    assert summary["distinct_fingerprints"] == 0


# ----------------------------------------------------------------------
# evaluator-side registration
# ----------------------------------------------------------------------
def test_register_problem_provenance_marks_declared_replays():
    """A declared re-issue is marked, and its claims get zero weight.

    Without this registration the evaluator's ProvenanceGraph stays empty and
    the provenance variant is structurally identical to the baseline.
    """
    observation_a = make_observation("obs_a", "/data/scene.png")
    observation_b = make_observation("obs_b", "/data/scene.png")
    claims = [
        make_claim("clm_a", "obs_a", "category", "ship"),
        make_claim("clm_b", "obs_b", "category", "ship"),
    ]
    problem = InferenceProblem(
        observations=[observation_a, observation_b], claims=claims
    )
    graph = ProvenanceGraph()

    report = register_problem_provenance(
        problem, graph, replays={"obs_b": "obs_a"}
    )

    assert report["duplicate_nodes"] == 1
    assert graph.is_duplicate("obs_b") is True
    assert graph.is_duplicate("obs_a") is False
    assert graph.nodes["obs_b"].duplicate_of == "obs_a"

    assign_group_ids(claims, {"obs_a": observation_a, "obs_b": observation_b})
    weights = compute_group_weights(claims, graph)

    assert weights["clm_b"] == 0.0
    # The replay is excluded from the group size too, so the original keeps its
    # full budget instead of being diluted by its own echo.
    assert weights["clm_a"] == pytest.approx(1.0)


def test_duplicate_detections_are_not_silently_treated_as_replays():
    """Two identical detections inside one call are not a re-issued call.

    A bbox-level fingerprint cannot tell them apart, and EvidenceStore
    fingerprints the call arguments instead.  Deduplicating here would suppress
    legitimate detections, so auto-detection is opt-in only.
    """
    observation_a = make_observation("obs_a", "/data/scene.png")
    observation_b = make_observation("obs_b", "/data/scene.png")
    claims = [
        make_claim("clm_a", "obs_a", "category", "ship"),
        make_claim("clm_b", "obs_b", "category", "ship"),
    ]
    problem = InferenceProblem(
        observations=[observation_a, observation_b], claims=claims
    )
    graph = ProvenanceGraph()

    report = register_problem_provenance(problem, graph)

    assert report["duplicate_nodes"] == 0
    assign_group_ids(claims, {"obs_a": observation_a, "obs_b": observation_b})
    weighted = compute_group_weights(claims, graph)
    plain = compute_group_weights(claims, None)
    assert weighted == plain


def test_budget_disabled_uses_unit_weights_without_disabling_replay_dedup():
    observation = make_observation("obs_shared", "/data/scene.png")
    claims = [
        make_claim("clm_a", "obs_shared", "category", "ship"),
        make_claim("clm_b", "obs_shared", "subtype", "cargo"),
    ]
    assign_group_ids(claims, {observation.observation_id: observation})

    weights = compute_group_weights(claims, None, 1.0, budget_enabled=False)

    assert weights == {"clm_a": 1.0, "clm_b": 1.0}


def test_register_problem_provenance_auto_detect_is_opt_in():
    """Callers with real call-level fingerprints can opt into auto-detection."""
    observation_a = make_observation("obs_a", "/data/scene.png")
    observation_b = make_observation("obs_b", "/data/scene.png")
    claims = [
        make_claim("clm_a", "obs_a", "category", "ship"),
        make_claim("clm_b", "obs_b", "category", "ship"),
    ]
    problem = InferenceProblem(
        observations=[observation_a, observation_b], claims=claims
    )
    graph = ProvenanceGraph()

    report = register_problem_provenance(problem, graph, auto_detect_replays=True)

    assert report["duplicate_nodes"] == 1
    assert graph.nodes["obs_b"].duplicate_of == "obs_a"
    assign_group_ids(claims, {"obs_a": observation_a, "obs_b": observation_b})
    assert compute_group_weights(claims, graph)["clm_b"] == 0.0
