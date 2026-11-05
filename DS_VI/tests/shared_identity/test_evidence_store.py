"""EvidenceStore: adapter ingest, replay dedup, coverage, and round trips.

The store replaces the old ``latest_structured`` single-shot state.  These tests
pin the properties that make accumulation safe:

- a legacy fused ``matched_pair`` is split back into two observations, so the
  greedy matcher never pre-commits an identity;
- a failed or empty tool response contributes nothing;
- an exact replay leaves the posterior bit-identical and cannot inflate counts
  or coverage;
- a JSON round trip preserves enough state that replays are still detected.
"""
from __future__ import annotations

import copy
import json

import pytest

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.evidence_store import EvidenceStore
from DS_VI.models.shared_identity.schema import (
    Modality,
    ObjectHypothesis,
    RelationType,
    SpatialSupport,
)
from DS_VI.models.shared_identity.serialization import store_from_dict, store_to_dict

# ----------------------------------------------------------------------
# fixtures / builders
# ----------------------------------------------------------------------
OPTICAL_ARGS = {"image": "/data/scene_a.png", "pixel_size_m": 1.0}

CROSS_MODAL_ARGS = {
    "rgb_image": "/data/optical.png",
    "sar_image": "/data/sar.png",
    "pixel_size_m": 1.5,
}

TEMPORAL_ARGS = {
    "before_image": "/data/t0.png",
    "after_image": "/data/t1.png",
}


def optical_response(label: str = "ship", bbox=(10, 20, 50, 80), score: float = 0.9):
    return {
        "error_code": 0,
        "detections": [{"bbox": list(bbox), "label": label, "score": score}],
    }


def matched_pair_response():
    """One legacy fused RGB/SAR match, as the cross-modal worker emits it."""
    return {
        "error_code": 0,
        "matched_pairs": [
            {
                "rgb_bbox": [10, 20, 50, 80],
                "rgb_label": "ship",
                "rgb_score": 0.9,
                "sar_bbox": [12, 22, 52, 82],
                "sar_label": "ship",
                "sar_score": 0.8,
                "match_score": 0.77,
            }
        ],
    }


def change_detection_response():
    return {
        "error_code": 0,
        "relations": [
            {
                "relation": "stayed",
                "before_bbox": [10, 20, 50, 80],
                "after_bbox": [12, 22, 52, 82],
                "confidence": 0.8,
            }
        ],
    }


def legacy_store() -> EvidenceStore:
    """Single-image clustering tests explicitly select the supported legacy mode."""
    return EvidenceStore(load_config(overrides={"association": {
        "mode": "legacy_global", "strict_scope": False, "forbid_same_image": False,
    }}))


def seeded_store(legacy=False) -> EvidenceStore:
    store = legacy_store() if legacy else EvidenceStore()
    store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args=OPTICAL_ARGS, round_id=0
    )
    return store


# ----------------------------------------------------------------------
# matched-pair decomposition
# ----------------------------------------------------------------------
def test_matched_pair_is_split_into_two_observations_plus_a_relation():
    store = EvidenceStore()

    new_ids = store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS, round_id=0
    )

    # The fused pair must never be stored as one pre-committed match: it becomes
    # two independent identity units so joint inference, not the greedy matcher,
    # decides whether they share an object.
    assert len(new_ids) == 2
    assert len(store.observations) == 2

    modalities = sorted(o.modality.value for o in store.observations.values())
    assert modalities == [Modality.OPTICAL.value, Modality.SAR.value]

    assert len(store.relations) == 1
    relation = next(iter(store.relations.values()))
    # The matcher's geometric agreement survives only as a soft prior.
    assert relation.relation_type is RelationType.SAME_SCENE
    assert relation.strength == pytest.approx(0.77)


def test_split_pair_observations_carry_their_own_modality_specific_roots():
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS
    )

    by_modality = {o.modality: o for o in store.observations.values()}

    # Separate input roots are what let provenance treat optical and SAR as
    # independent evidence rather than one correlated group.
    assert by_modality[Modality.OPTICAL].input_roots == ["/data/optical.png"]
    assert by_modality[Modality.SAR].input_roots == ["/data/sar.png"]


def test_same_scene_relation_links_the_two_split_observations():
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS
    )
    relation = next(iter(store.relations.values()))
    by_modality = {o.modality: o.observation_id for o in store.observations.values()}

    assert relation.source_observation_id == by_modality[Modality.OPTICAL]
    assert relation.target_observation_id == by_modality[Modality.SAR]
    assert relation.context["origin"] == "legacy_matched_pair"


def test_split_pair_produces_one_claim_per_constituent_observation():
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS
    )

    # Each claim hangs off exactly one observation; there is no fused claim that
    # would implicitly assert the two views are the same object.
    for observation_id in store.observations:
        claims = store.claims_for_observation(observation_id)
        assert len(claims) == 1
        assert claims[0].value == "ship"


# ----------------------------------------------------------------------
# failed / empty responses
# ----------------------------------------------------------------------
def test_failed_response_adds_no_observations_and_no_claims():
    store = EvidenceStore()

    result = store.ingest_tool_response(
        "ObjectDetection",
        {"error_code": 1, "detections": [{"bbox": [10, 20, 50, 80], "label": "ship"}]},
        api_args=OPTICAL_ARGS,
    )

    # A crashed tool often still returns a payload.  Trusting it would inject
    # unvalidated detections into the posterior.
    assert result == []
    assert store.observations == {}
    assert store.claims == {}
    assert store.coverage == {}
    assert store.ingest_log[-1]["skipped"] == "error_code=1"


def test_failed_response_does_not_consume_a_replay_fingerprint():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ObjectDetection", {"error_code": 1, "detections": []}, api_args=OPTICAL_ARGS
    )

    # A later successful retry of the same request must be ingested, not
    # discarded as a replay of the failure.
    new_ids = store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args=OPTICAL_ARGS
    )

    assert len(new_ids) == 1
    assert store.duplicate_ingests == 0


def test_error_status_response_is_rejected_like_an_error_code():
    store = EvidenceStore()

    result = store.ingest_tool_response(
        "ObjectDetection",
        {"error_code": 0, "status": "invalid_request", "detections": [{"bbox": [1, 2, 30, 40]}]},
        api_args=OPTICAL_ARGS,
    )

    assert result == []
    assert store.observations == {}


def test_empty_successful_response_adds_nothing_and_does_not_crash():
    store = EvidenceStore()

    result = store.ingest_tool_response(
        "ObjectDetection", {"error_code": 0, "detections": []}, api_args=OPTICAL_ARGS
    )

    assert result == []
    assert store.observations == {}
    assert store.claims == {}
    assert store.ingest_log[-1]["skipped"] == "no detections"


def test_unknown_tool_is_skipped_without_raising():
    store = EvidenceStore()

    result = store.ingest_tool_response("NoSuchTool", {"error_code": 0}, api_args={})

    assert result == []
    assert store.ingest_log[-1]["skipped"] == "no adapter"


# ----------------------------------------------------------------------
# exact replay
# ----------------------------------------------------------------------
def test_exact_replay_leaves_the_posterior_identical():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ObjectDetection",
        {
            "error_code": 0,
            "detections": [
                {"bbox": [10, 20, 50, 80], "label": "ship", "score": 0.9},
                {"bbox": [100, 120, 150, 180], "label": "tank", "score": 0.7},
            ],
        },
        api_args=OPTICAL_ARGS,
        round_id=0,
    )
    before = copy.deepcopy(store.run_inference().to_dict())

    store.ingest_tool_response(
        "ObjectDetection",
        {
            "error_code": 0,
            "detections": [
                {"bbox": [10, 20, 50, 80], "label": "ship", "score": 0.9},
                {"bbox": [100, 120, 150, 180], "label": "tank", "score": 0.7},
            ],
        },
        api_args=OPTICAL_ARGS,
        round_id=1,
    )
    after = store.run_inference().to_dict()

    # The core replay guarantee: calling the same tool with the same args again
    # is not new information, so no belief may move.
    assert after["association"] == before["association"]
    assert after["truth"] == before["truth"]


def test_exact_replay_increments_duplicate_ingests():
    store = seeded_store()

    store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args=OPTICAL_ARGS, round_id=1
    )

    # The replay is recorded rather than silently dropped, so the audit trail
    # shows how much of the evidence stream was redundant.
    assert store.duplicate_ingests == 1
    assert store.ingest_log[-1]["replay"] is True


def test_exact_replay_does_not_inflate_observation_or_claim_counts():
    store = seeded_store()
    observations_before = dict(store.observations)
    claims_before = dict(store.claims)

    new_ids = store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args=OPTICAL_ARGS, round_id=1
    )

    # Deterministic ids mean the replayed observation collides with the original
    # instead of becoming a second row in the association matrix.
    assert new_ids == []
    assert set(store.observations) == set(observations_before)
    assert set(store.claims) == set(claims_before)


def test_exact_replay_does_not_inflate_coverage():
    store = seeded_store()
    coverage_before = dict(store.coverage)

    store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args=OPTICAL_ARGS, round_id=1
    )

    # Coverage drives acquisition decisions and abstention.  If a replay raised
    # it, the agent would believe it had corroboration it never obtained.
    assert store.coverage == coverage_before
    assert store.coverage == {"optical|t0": 1}


def test_replay_is_detected_across_different_round_ids():
    store = seeded_store()

    store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args=OPTICAL_ARGS, round_id=7
    )

    # round_id is volatile bookkeeping; it must not make an identical call look
    # like fresh evidence in a later round.
    assert store.duplicate_ingests == 1


def test_replay_is_detected_when_api_arg_order_differs():
    store = seeded_store()
    reordered = {"pixel_size_m": 1.0, "image": "/data/scene_a.png"}

    store.ingest_tool_response("ObjectDetection", optical_response(), api_args=reordered)

    assert store.duplicate_ingests == 1
    assert len(store.observations) == 1


def test_replay_of_a_nondeterministic_tool_does_not_amplify_the_posterior():
    """An identical request answered with jittered output must not double count.

    Same tool, same api_args, but the detector returns a box shifted by 1px.
    Deterministic replays are caught by stable_id collision alone; this case is
    the one that needs the replay flag to reach ProvenanceGraph.register, so that
    the jittered claims carry zero weight instead of inflating belief.
    """
    args = {"image": "/data/a.png"}
    first = optical_response(label="ship", bbox=(10, 20, 50, 80))
    jittered = optical_response(label="ship", bbox=(11, 21, 51, 81))

    store = EvidenceStore()
    store.ingest_tool_response("ObjectDetection", first, api_args=args, round_id=0)
    before = copy.deepcopy(store.run_inference().to_dict())

    store.ingest_tool_response("ObjectDetection", jittered, api_args=args, round_id=1)
    assert store.ingest_log[-1]["replay"] is True

    # The replay is recognized, so its claims should carry zero provenance
    # weight and leave every belief where it was.
    assert store.provenance.summarize()["duplicate_count"] > 0
    after = store.run_inference().to_dict()
    assert after["truth"] == before["truth"]


def test_different_api_args_are_not_treated_as_a_replay():
    store = seeded_store()

    new_ids = store.ingest_tool_response(
        "ObjectDetection",
        optical_response(),
        api_args={"image": "/data/scene_b.png", "pixel_size_m": 1.0},
        round_id=1,
    )

    # A second image is genuinely independent evidence.  Dedup must not become
    # blanket suppression of anything that looks similar.
    assert len(new_ids) == 1
    assert store.duplicate_ingests == 0
    assert len(store.observations) == 2
    assert store.coverage == {"optical|t0": 2}


def test_different_detections_from_the_same_image_add_observations():
    store = seeded_store()

    new_ids = store.ingest_tool_response(
        "ObjectDetection",
        optical_response(label="tank", bbox=(200, 200, 240, 240), score=0.6),
        api_args={"image": "/data/scene_a.png", "pixel_size_m": 1.0, "prompt": "tank"},
    )

    assert len(new_ids) == 1
    assert len(store.observations) == 2


# ----------------------------------------------------------------------
# coverage
# ----------------------------------------------------------------------
def test_coverage_is_keyed_by_modality_and_time():
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS
    )

    # One cell per (modality|time_key): this is the grid the acquisition layer
    # reads to decide what evidence is still missing.
    assert store.coverage == {"optical|t0": 1, "sar|t0": 1}


def test_denodetv2_ingests_as_sar_not_optical():
    store = EvidenceStore()
    new_ids = store.ingest_tool_response(
        "DenoDetV2",
        optical_response(),
        api_args={"sar_image": "/data/sar.png", "pixel_size_m": 1.0},
        round_id=0,
    )

    # DenoDetV2 is the SAR detector. Registering it as optical would satisfy the
    # optical coverage cell without any optical observation and let acquisition
    # skip a modality it has never seen.
    assert len(new_ids) == 1
    assert store.observations[new_ids[0]].modality == Modality.SAR
    assert store.coverage == {"sar|t0": 1}


def test_coverage_separates_time_keys():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ChangeDetection", change_detection_response(), api_args=TEMPORAL_ARGS
    )

    assert store.coverage == {"temporal|t0": 1, "temporal|t1": 1}


def test_coverage_accumulates_within_one_cell():
    store = seeded_store()
    store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args={"image": "/data/scene_b.png"}
    )

    assert store.coverage == {"optical|t0": 2}


def test_recompute_coverage_is_derived_from_the_deduplicated_observation_set():
    store = seeded_store()
    store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args=OPTICAL_ARGS, round_id=1
    )

    # Recomputing rather than incrementing is what makes coverage replay-proof:
    # it can only ever equal the number of distinct observations.
    assert store._recompute_coverage() == {"optical|t0": 1}
    assert sum(store.coverage.values()) == len(store.observations)


# ----------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------
def test_stats_reports_every_accumulated_quantity():
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS
    )
    store.run_inference()

    stats = store.stats()

    assert stats["observations"] == 2
    assert stats["claims"] == 2
    # Seeding clustered the optical and SAR boxes into one shared hypothesis, so
    # both observations compete for the same object plus NULL.
    assert stats["objects"] == 1
    assert stats["candidate_edges"] == 4
    assert stats["relations"] == 1
    assert stats["coverage"] == {"optical|t0": 1, "sar|t0": 1}
    assert stats["duplicate_ingests"] == 0
    assert stats["inference_ran"] is True
    # The provenance block is the auditable replay record carried in stats.
    assert set(stats["provenance"]) >= {
        "node_count",
        "duplicate_count",
        "distinct_fingerprints",
        "group_budget",
    }


def test_stats_on_a_fresh_store_reports_zeros_and_no_inference():
    stats = EvidenceStore().stats()

    assert stats["observations"] == 0
    assert stats["claims"] == 0
    assert stats["coverage"] == {}
    assert stats["duplicate_ingests"] == 0
    assert stats["inference_ran"] is False


def test_stats_duplicate_ingests_tracks_replays():
    store = seeded_store()
    store.ingest_tool_response("ObjectDetection", optical_response(), api_args=OPTICAL_ARGS)
    store.ingest_tool_response("ObjectDetection", optical_response(), api_args=OPTICAL_ARGS)

    assert store.stats()["duplicate_ingests"] == 2
    # Two replays still leave exactly one observation.
    assert store.stats()["observations"] == 1


# ----------------------------------------------------------------------
# JSON round trip
# ----------------------------------------------------------------------
def test_round_trip_preserves_observations_claims_objects_and_relations():
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS
    )
    store.run_inference()

    # Force real JSON so the check covers serializability, not just dict copying.
    payload = json.loads(json.dumps(store_to_dict(store)))
    restored = store_from_dict(payload, EvidenceStore)

    assert set(restored.observations) == set(store.observations)
    assert set(restored.claims) == set(store.claims)
    assert set(restored.objects) == set(store.objects)
    assert set(restored.relations) == set(store.relations)
    assert restored.coverage == store.coverage


def test_round_trip_preserves_observation_and_relation_content():
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS
    )
    restored = store_from_dict(json.loads(json.dumps(store_to_dict(store))), EvidenceStore)

    for observation_id, original in store.observations.items():
        rebuilt = restored.observations[observation_id]
        assert rebuilt.modality is original.modality
        assert rebuilt.input_roots == original.input_roots
        assert rebuilt.time_scope.key == original.time_scope.key
        assert rebuilt.spatial_support.bbox == original.spatial_support.bbox

    for relation_id, original in store.relations.items():
        rebuilt = restored.relations[relation_id]
        assert rebuilt.relation_type is original.relation_type
        assert rebuilt.strength == pytest.approx(original.strength)


def test_round_trip_preserves_call_fingerprints():
    store = seeded_store()

    restored = store_from_dict(json.loads(json.dumps(store_to_dict(store))), EvidenceStore)

    # Without the fingerprints, restoring a checkpoint would reset replay memory
    # and a resumed run could re-ingest its own history as new evidence.
    assert restored._call_fingerprints == store._call_fingerprints


def test_replay_is_still_detected_after_a_round_trip():
    store = seeded_store()
    restored = store_from_dict(json.loads(json.dumps(store_to_dict(store))), EvidenceStore)

    new_ids = restored.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args=OPTICAL_ARGS, round_id=5
    )

    assert restored.duplicate_ingests == store.duplicate_ingests + 1
    assert new_ids == []
    assert len(restored.observations) == len(store.observations)
    assert restored.coverage == store.coverage


def test_round_trip_preserves_provenance_duplicate_bookkeeping():
    store = seeded_store()
    store.ingest_tool_response("ObjectDetection", optical_response(), api_args=OPTICAL_ARGS)
    restored = store_from_dict(json.loads(json.dumps(store_to_dict(store))), EvidenceStore)

    assert restored.duplicate_ingests == store.duplicate_ingests == 1
    assert restored.provenance.summarize() == store.provenance.summarize()


def test_round_trip_yields_an_identical_posterior():
    store = seeded_store()
    before = copy.deepcopy(store.run_inference().to_dict())

    restored = store_from_dict(json.loads(json.dumps(store_to_dict(store))), EvidenceStore)
    after = restored.run_inference(seed_objects=False).to_dict()

    assert after["association"] == before["association"]
    assert after["truth"] == before["truth"]


def test_round_trip_rejects_an_unsupported_schema_version():
    payload = store_to_dict(seeded_store())
    payload["schema_version"] = "0.1"

    # Silent acceptance of an old checkpoint would mean silently wrong state.
    with pytest.raises(ValueError):
        store_from_dict(payload, EvidenceStore)


def test_store_to_dict_and_from_dict_are_reachable_from_the_instance():
    store = seeded_store()

    restored = EvidenceStore.from_dict(store.to_dict())

    assert set(restored.observations) == set(store.observations)


# ----------------------------------------------------------------------
# evidence state
# ----------------------------------------------------------------------
def test_build_evidence_state_reports_per_object_summaries():
    store = seeded_store(legacy=True)

    state = store.build_evidence_state()

    assert state.observation_count == 1
    assert state.claim_count == 1
    assert state.coverage == {"optical|t0": 1}
    assert state.objects

    summary = next(iter(state.objects.values()))
    assert summary.association_support > 0.0
    assert summary.supporting_observations == list(store.observations)
    # Association entropy is the ambiguity signal the decision layer needs in
    # order to abstain instead of guessing an identity.
    assert "association_entropy" in summary.ambiguity
    assert summary.truth_beliefs["category@t0"]["value"] == "ship"
    assert 0.0 <= summary.truth_beliefs["category@t0"]["probability"] <= 1.0
    assert summary.contradictions == []


def test_build_evidence_state_runs_inference_when_needed():
    store = seeded_store()
    assert store.inference_state is None

    store.build_evidence_state()

    assert store.inference_state is not None
    assert store.stats()["inference_ran"] is True


def test_later_acquisition_invalidates_the_cached_inference():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ObjectDetection", optical_response(), api_args={"image": "/data/a.png"}
    )
    first = store.build_evidence_state()
    assert first.observation_count == 1

    store.ingest_tool_response(
        "ObjectDetection",
        optical_response(label="ship", bbox=(300, 300, 340, 340)),
        api_args={"image": "/data/b.png"},
    )
    second = store.build_evidence_state()

    # A cached posterior must not be served after new evidence arrives, or the
    # decision layer would reason about a state that predates the acquisition.
    assert second.observation_count == 2
    assert store._inference_revision == store._revision
    assert set(store.inference_state.association) == set(store.observations)


def test_later_acquisition_can_discover_a_new_object_hypothesis():
    store = legacy_store()
    store.ingest_tool_response(
        "ObjectDetection",
        optical_response(bbox=(10, 10, 50, 50)),
        api_args={"image": "/data/a.png"},
    )
    store.build_evidence_state()
    objects_before = set(store.objects)

    store.ingest_tool_response(
        "ObjectDetection",
        optical_response(bbox=(300, 300, 340, 340)),
        api_args={"image": "/data/b.png"},
    )
    store.build_evidence_state()

    # Seeding only when the object set was empty would hide the second object
    # forever, so the acquisition layer could never propose evidence for it.
    assert len(store.objects) == 2
    assert objects_before.issubset(set(store.objects))


def test_adding_a_view_does_not_rename_or_duplicate_an_object():
    store = legacy_store()
    store.ingest_tool_response(
        "ObjectDetection",
        optical_response(bbox=(10, 10, 50, 50)),
        api_args={"image": "/data/a.png"},
    )
    store.build_evidence_state()
    objects_before = set(store.objects)

    store.ingest_tool_response(
        "ObjectDetection",
        optical_response(bbox=(12, 12, 52, 52)),
        api_args={"image": "/data/b.png"},
    )
    store.build_evidence_state()

    # The seeded id hashes the cluster mean, which moves when a view is added;
    # stable matching must keep the existing hypothesis instead of minting one.
    assert set(store.objects) == objects_before


def test_build_evidence_state_reports_contradictions_between_disagreeing_views():
    store = legacy_store()
    # Two near-identical boxes from separate images with incompatible labels:
    # both associate to the same object, so the category axis is contested.
    store.ingest_tool_response(
        "ObjectDetection",
        optical_response(label="ship", bbox=(10, 20, 50, 80)),
        api_args={"image": "/data/a.png"},
    )
    store.ingest_tool_response(
        "ObjectDetection",
        optical_response(label="aircraft", bbox=(11, 21, 51, 81)),
        api_args={"image": "/data/b.png"},
    )

    state = store.build_evidence_state()
    summary = max(state.objects.values(), key=lambda item: item.association_support)

    # A contradiction must surface as a diagnostic rather than being resolved by
    # arbitrary tie-breaking.
    assert summary.contradictions
    contradiction = summary.contradictions[0]
    assert contradiction["attribute"] == "category"
    assert sorted(contradiction["competing_values"]) == ["aircraft", "ship"]
    assert summary.ambiguity["max_truth_entropy"] > 0.0


def test_build_evidence_state_reports_null_mass_per_observation():
    store = seeded_store()

    state = store.build_evidence_state()

    # Null mass is how the store expresses "this observation may belong to no
    # tracked object", which is required for honest abstention.
    assert set(state.null_mass) == set(store.observations)
    assert all(0.0 <= value <= 1.0 for value in state.null_mass.values())


def test_build_evidence_state_on_an_empty_store_does_not_crash():
    state = EvidenceStore().build_evidence_state()

    assert state.observation_count == 0
    assert state.objects == {}


def test_build_evidence_state_records_provenance_roots_per_object():
    store = seeded_store(legacy=True)

    summary = next(iter(store.build_evidence_state().objects.values()))

    assert summary.provenance_summary["input_roots"] == ["/data/scene_a.png"]
    assert summary.provenance_summary["tools"] == ["ObjectDetection"]
    assert summary.provenance_summary["duplicate_count"] == 0


# ----------------------------------------------------------------------
# temporal: relations without asserted identity
# ----------------------------------------------------------------------
def test_change_detection_creates_a_temporal_relation():
    store = EvidenceStore()

    new_ids = store.ingest_tool_response(
        "ChangeDetection", change_detection_response(), api_args=TEMPORAL_ARGS, round_id=0
    )

    assert len(new_ids) == 2
    assert len(store.relations) == 1
    relation = next(iter(store.relations.values()))
    assert relation.relation_type is RelationType.STAYED
    assert relation.strength == pytest.approx(0.8)
    # The adapter documents the constraint it is honouring: a 'stayed' verdict is
    # a prior, not an identity edge.
    assert "does not assert identity" in relation.context["note"]


def test_temporal_before_and_after_remain_separate_observations():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ChangeDetection", change_detection_response(), api_args=TEMPORAL_ARGS
    )

    by_time = {o.time_scope.key: o for o in store.observations.values()}

    # t0 and t1 are distinct identity units with distinct input roots.  Fusing
    # them here would hard-code the change detector's verdict as truth.
    assert set(by_time) == {"t0", "t1"}
    assert by_time["t0"].observation_id != by_time["t1"].observation_id
    assert by_time["t0"].input_roots == ["/data/t0.png"]
    assert by_time["t1"].input_roots == ["/data/t1.png"]


def test_temporal_identity_is_resolved_by_inference_not_asserted():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ChangeDetection", change_detection_response(), api_args=TEMPORAL_ARGS
    )
    state = store.run_inference()

    by_time = {o.time_scope.key: o.observation_id for o in store.observations.values()}
    row_t0 = state.association_row(by_time["t0"])
    row_t1 = state.association_row(by_time["t1"])

    # Each observation gets its own association row with its own probabilities.
    # A deterministic identity link would instead show up as certainty (1.0) or
    # as a single shared row.
    assert row_t0 and row_t1
    assert max(row_t0.values()) < 1.0
    assert max(row_t1.values()) < 1.0
    # Residual null mass means the link stays revisable.
    assert row_t0.get("__NULL__", 0.0) > 0.0
    assert row_t1.get("__NULL__", 0.0) > 0.0


def test_temporal_relation_does_not_create_an_object_hypothesis_by_itself():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ChangeDetection", change_detection_response(), api_args=TEMPORAL_ARGS
    )

    # Ingest is purely evidence accumulation; object columns appear only when
    # inference is explicitly built.
    assert store.objects == {}
    assert store.candidate_edges == []

    store.build_problem()
    assert store.objects


def test_appeared_record_yields_a_relation_with_no_before_observation():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ChangeDetection",
        {
            "error_code": 0,
            "relations": [
                {"relation": "appeared", "after_bbox": [200, 200, 240, 240], "confidence": 0.6}
            ],
        },
        api_args=TEMPORAL_ARGS,
    )

    relation = next(iter(store.relations.values()))
    assert relation.relation_type is RelationType.APPEARED
    # 'appeared' has no t0 counterpart, so the source side is legitimately absent
    # rather than fabricated.
    assert relation.source_observation_id is None
    assert relation.target_observation_id in store.observations


def test_temporal_claims_are_bound_to_the_after_observation_only():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ChangeDetection", change_detection_response(), api_args=TEMPORAL_ARGS
    )
    by_time = {o.time_scope.key: o.observation_id for o in store.observations.values()}

    # presence/motion describe the state at t1; attaching them to t0 as well
    # would double count the same verdict.
    assert store.claims_for_observation(by_time["t0"]) == []
    after_claims = store.claims_for_observation(by_time["t1"])
    assert {c.attribute for c in after_claims} == {"presence", "motion"}
    assert all(c.time_scope.key == "t1" for c in after_claims)


def test_temporal_replay_does_not_duplicate_relations_or_coverage():
    store = EvidenceStore()
    store.ingest_tool_response(
        "ChangeDetection", change_detection_response(), api_args=TEMPORAL_ARGS, round_id=0
    )
    relations_before = dict(store.relations)
    coverage_before = dict(store.coverage)

    store.ingest_tool_response(
        "ChangeDetection", change_detection_response(), api_args=TEMPORAL_ARGS, round_id=1
    )

    assert store.duplicate_ingests == 1
    assert set(store.relations) == set(relations_before)
    assert store.coverage == coverage_before


# ----------------------------------------------------------------------
# object layer / problem building
# ----------------------------------------------------------------------
def test_register_object_is_preserved_by_seeding():
    store = seeded_store()
    store.register_object(
        ObjectHypothesis(
            object_id="obj_manual",
            spatial_support=SpatialSupport(bbox=[10, 20, 50, 80]),
            capacity=1,
        )
    )

    store.seed_object_hypotheses()

    # Seeding is a bootstrap for candidate columns; it must never clobber an
    # externally supplied hypothesis.
    assert "obj_manual" in store.objects
    assert store.objects["obj_manual"].capacity == 1


def test_build_problem_validates_and_carries_capacities():
    store = EvidenceStore()
    store.ingest_tool_response(
        "CrossModalMatchSkill", matched_pair_response(), api_args=CROSS_MODAL_ARGS
    )

    problem = store.build_problem()
    problem.validate()

    assert len(problem.observations) == 2
    assert len(problem.relations) == 1
    assert problem.candidate_edges
    # Paper Eq.(29) has no per-object column cap, so the store carries none
    # unless a hypothesis declares one explicitly.
    assert set(problem.capacities) <= set(store.objects)
    assert problem.capacities == {}


def test_rebuild_candidate_graph_is_idempotent_for_unchanged_state():
    store = seeded_store()
    store.seed_object_hypotheses()

    first = store.rebuild_candidate_graph()
    second = store.rebuild_candidate_graph()

    assert len(first) == len(second)
    assert [(e.observation_id, e.object_id) for e in first] == [
        (e.observation_id, e.object_id) for e in second
    ]
