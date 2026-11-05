"""Structural and numeric invariants of the shared-identity evidence schema.

The load-bearing invariant under test: identity is resolved once per
*observation*, never per claim.  A Claim therefore has no ``object_id`` and no
association vector, and every claim from one observation reads the same
association row.
"""
from __future__ import annotations

import pytest

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.schema import (
    NULL_OBJECT_ID,
    CandidateEdge,
    Claim,
    ClaimAttribute,
    InferenceProblem,
    Modality,
    ObjectHypothesis,
    PrimitiveObservation,
    SchemaValidationError,
    SpatialSupport,
    TimeScope,
    fingerprint,
    stable_id,
)


# ----------------------------------------------------------------------
# builders (SpatialSupport is a frozen dataclass; build it, never mutate it)
# ----------------------------------------------------------------------
def _observation(
    observation_id: str = "m1",
    bbox=(10.0, 20.0, 50.0, 80.0),
    time_key: str = "t0",
    detection_confidence=0.9,
) -> PrimitiveObservation:
    return PrimitiveObservation(
        observation_id=observation_id,
        modality=Modality.coerce("optical"),
        source_tool="SyntheticSensor",
        time_scope=TimeScope(key=time_key),
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        detection_confidence=detection_confidence,
        input_roots=["root_a"],
    )


def _claim(
    claim_id: str = "c1",
    observation_id: str = "m1",
    attribute: str = "category",
    value="ship",
    time_key: str = "t0",
    context=None,
) -> Claim:
    return Claim(
        claim_id=claim_id,
        observation_id=observation_id,
        attribute=attribute,
        value=value,
        time_scope=TimeScope(key=time_key),
        source_tool="SyntheticSensor",
        source_channel="optical",
        context=dict(context or {}),
    )


def _object(object_id: str = "k1", bbox=(10.0, 20.0, 50.0, 80.0)) -> ObjectHypothesis:
    return ObjectHypothesis(
        object_id=object_id,
        spatial_support=SpatialSupport(bbox=[float(v) for v in bbox]),
        attribute_hints={"category": "ship"},
    )


# ----------------------------------------------------------------------
# Claim carries no identity
# ----------------------------------------------------------------------
def test_claim_has_no_object_id_attribute():
    # The whole point of the schema: a claim cannot name an object, so no code
    # path can accidentally resolve identity per-claim instead of per-observation.
    claim = _claim()
    assert not hasattr(claim, "object_id")
    assert not hasattr(claim, "association")
    assert not hasattr(claim, "association_vector")
    assert not hasattr(claim, "assigned_object")


@pytest.mark.parametrize("forbidden", Claim.FORBIDDEN_FIELDS)
def test_claim_validate_rejects_every_forbidden_context_key(forbidden):
    # context is the only untyped escape hatch on Claim, so validate() must
    # police it. Otherwise **kwargs plumbing could smuggle identity back in.
    claim = _claim(context={forbidden: "k1"})
    with pytest.raises(SchemaValidationError) as excinfo:
        claim.validate()
    assert forbidden in str(excinfo.value)


def test_claim_forbidden_fields_covers_the_known_identity_spellings():
    assert Claim.FORBIDDEN_FIELDS == (
        "object_id",
        "association",
        "association_vector",
        "assigned_object",
    )


def test_claim_validate_allows_benign_context_keys():
    # Guard the negative: validate() must not reject arbitrary context, only
    # the identity-bearing keys.
    claim = _claim(context={"detector_version": "v3", "object_id_hint_unused": 1})
    claim.validate()
    assert claim.context["detector_version"] == "v3"


# ----------------------------------------------------------------------
# attribute coercion
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "alias,expected",
    [
        ("class", "category"),
        ("label", "category"),
        ("type", "category"),
        ("Category", "category"),
        ("subclass", "subtype"),
        ("fine_grained", "subtype"),
        ("orientation", "orientation"),
        ("direction", "heading"),
        ("exists", "presence"),
        ("existence", "presence"),
        ("moved", "motion"),
        ("movement", "motion"),
    ],
)
def test_claim_validate_coerces_attribute_aliases_in_place(alias, expected):
    # Truth variables are indexed by (attribute, time_key). If two tools spell
    # the same axis differently, their evidence would land in different truth
    # cells and never combine.
    claim = _claim(attribute=alias)
    assert claim.attribute == alias
    claim.validate()
    assert claim.attribute == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("Fine-Grained", "subtype"), ("fine grained", "subtype"), ("  class  ", "category")],
)
def test_claim_attribute_coerce_normalizes_separators_and_whitespace(raw, expected):
    assert ClaimAttribute.coerce(raw) == expected


def test_claim_attribute_coerce_passes_through_unknown_axes_normalized():
    # Unknown axes are deliberately allowed so a new tool can contribute
    # evidence without a schema change; they just get a stable spelling.
    assert ClaimAttribute.coerce("Hull-Material") == "hull_material"


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_claim_attribute_coerce_rejects_empty(empty):
    with pytest.raises(SchemaValidationError):
        ClaimAttribute.coerce(empty)


def test_registered_axes_cover_every_value_domain():
    # The config declares a value domain per attribute axis; every one of them
    # must also be a registered ClaimAttribute, otherwise the axis only works
    # through the (deliberate) unknown-axis pass-through and drifts silently.
    domains = set((load_config().get("value_domains") or {}).keys())
    registered = {axis.value for axis in ClaimAttribute}
    assert domains <= registered, f"unregistered axes: {sorted(domains - registered)}"
    assert {"category", "subtype", "size_bin", "state", "heading", "presence", "motion"} <= registered


def test_claim_truth_key_is_attribute_and_time_key():
    # truth_key is the address of Q[k, a, t, v]; it must reflect the *coerced*
    # attribute so aliases collapse onto one truth cell.
    claim = _claim(attribute="class", time_key="t7")
    claim.validate()
    assert claim.truth_key == ("category", "t7")


def test_claim_validate_rejects_empty_value_and_out_of_range_confidence():
    with pytest.raises(SchemaValidationError):
        _claim(value="").validate()
    with pytest.raises(SchemaValidationError):
        _claim(value=None).validate()

    claim = _claim()
    claim.raw_confidence = 1.5
    with pytest.raises(SchemaValidationError):
        claim.validate()


# ----------------------------------------------------------------------
# PrimitiveObservation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01, 2.0, -5.0])
def test_observation_validate_rejects_detection_confidence_out_of_unit_range(bad_confidence):
    # detection_confidence feeds the physical prior as a feature. Out-of-range
    # values would silently distort every diagnostic score downstream.
    observation = _observation(detection_confidence=bad_confidence)
    with pytest.raises(SchemaValidationError) as excinfo:
        observation.validate()
    assert "detection_confidence" in str(excinfo.value)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0, None])
def test_observation_validate_accepts_unit_range_and_none_confidence(confidence):
    # None must stay legal: not every observation source reports a score.
    _observation(detection_confidence=confidence).validate()


def test_observation_validate_rejects_empty_id_and_non_modality_enum():
    with pytest.raises(SchemaValidationError):
        _observation(observation_id="  ").validate()

    observation = _observation()
    observation.modality = "optical"  # a raw string, not Modality.OPTICAL
    with pytest.raises(SchemaValidationError):
        observation.validate()


def test_observation_validate_rejects_degenerate_spatial_support():
    observation = _observation()
    observation.spatial_support = SpatialSupport(bbox=[50.0, 80.0, 10.0, 20.0])
    with pytest.raises(SchemaValidationError):
        observation.validate()


def test_modality_coerce_maps_aliases_and_defaults_to_unknown():
    assert Modality.coerce("rgb") is Modality.OPTICAL
    assert Modality.coerce("radar") is Modality.SAR
    assert Modality.coerce("change") is Modality.TEMPORAL
    assert Modality.coerce("crossmodal") is Modality.CROSS_MODAL
    assert Modality.coerce("lidar") is Modality.UNKNOWN
    assert Modality.coerce(None) is Modality.UNKNOWN


# ----------------------------------------------------------------------
# ObjectHypothesis
# ----------------------------------------------------------------------
def test_object_hypothesis_validate_rejects_reserved_null_object_id():
    # __NULL__ is the "belongs to no tracked object" sink. If a real hypothesis
    # could take that id, NULL mass and real mass would be indistinguishable.
    assert NULL_OBJECT_ID == "__NULL__"
    with pytest.raises(SchemaValidationError) as excinfo:
        _object(object_id=NULL_OBJECT_ID).validate()
    assert NULL_OBJECT_ID in str(excinfo.value)


@pytest.mark.parametrize("bad_capacity", [0, -1, -10])
def test_object_hypothesis_validate_rejects_capacity_below_one(bad_capacity):
    # Capacity bounds how many observations may claim one object. Zero or
    # negative capacity makes the association projection infeasible.
    obj = _object()
    obj.capacity = bad_capacity
    with pytest.raises(SchemaValidationError) as excinfo:
        obj.validate()
    assert "capacity" in str(excinfo.value)


@pytest.mark.parametrize("capacity", [1, 2, 99, None])
def test_object_hypothesis_validate_accepts_capacity_at_or_above_one(capacity):
    obj = _object()
    obj.capacity = capacity
    obj.validate()


def test_object_hypothesis_validate_rejects_out_of_range_confidence():
    obj = _object()
    obj.confidence = 1.2
    with pytest.raises(SchemaValidationError):
        obj.validate()


# ----------------------------------------------------------------------
# InferenceProblem cross-references
# ----------------------------------------------------------------------
def test_inference_problem_validate_rejects_claim_on_unknown_observation():
    # A dangling claim has no association row to read, so inference would have
    # to invent identity for it. Fail loudly at the boundary instead.
    problem = InferenceProblem(
        observations=[_observation("m1")],
        claims=[_claim("c1", observation_id="m_missing")],
        objects=[_object("k1")],
    )
    with pytest.raises(SchemaValidationError) as excinfo:
        problem.validate()
    assert "unknown observation" in str(excinfo.value)
    assert "m_missing" in str(excinfo.value)


def test_inference_problem_validate_rejects_duplicate_observation_id():
    # Duplicate ids would collapse two identity units into one row, silently
    # double-counting the same evidence.
    problem = InferenceProblem(
        observations=[_observation("m1"), _observation("m1", bbox=(0.0, 0.0, 5.0, 5.0))],
        claims=[],
        objects=[],
    )
    with pytest.raises(SchemaValidationError) as excinfo:
        problem.validate()
    assert "Duplicate observation_id" in str(excinfo.value)


def test_inference_problem_validate_rejects_duplicate_claim_id():
    problem = InferenceProblem(
        observations=[_observation("m1")],
        claims=[_claim("c1"), _claim("c1", value="tank")],
        objects=[],
    )
    with pytest.raises(SchemaValidationError) as excinfo:
        problem.validate()
    assert "Duplicate claim_id" in str(excinfo.value)


def test_inference_problem_validate_rejects_duplicate_object_id():
    problem = InferenceProblem(
        observations=[],
        claims=[],
        objects=[_object("k1"), _object("k1", bbox=(0.0, 0.0, 5.0, 5.0))],
    )
    with pytest.raises(SchemaValidationError) as excinfo:
        problem.validate()
    assert "Duplicate object_id" in str(excinfo.value)


def test_inference_problem_validate_rejects_edge_to_unknown_object_but_allows_null():
    problem = InferenceProblem(
        observations=[_observation("m1")],
        objects=[_object("k1")],
        candidate_edges=[CandidateEdge(observation_id="m1", object_id="k_ghost")],
    )
    with pytest.raises(SchemaValidationError) as excinfo:
        problem.validate()
    assert "unknown object" in str(excinfo.value)

    # NULL is exempt from the object registry by design.
    InferenceProblem(
        observations=[_observation("m1")],
        objects=[_object("k1")],
        candidate_edges=[CandidateEdge(observation_id="m1", object_id=NULL_OBJECT_ID)],
    ).validate()


def test_inference_problem_validate_accepts_a_consistent_problem():
    problem = InferenceProblem(
        observations=[_observation("m1")],
        claims=[_claim("c1", "m1", attribute="class")],
        objects=[_object("k1")],
        candidate_edges=[
            CandidateEdge(observation_id="m1", object_id="k1", prior_probability=0.85),
            CandidateEdge(observation_id="m1", object_id=NULL_OBJECT_ID, prior_probability=0.15),
        ],
    )
    problem.validate()
    # validate() coerces claim attributes as a side effect.
    assert problem.claims[0].attribute == "category"


# ----------------------------------------------------------------------
# the identity unit is the observation
# ----------------------------------------------------------------------
def test_three_claims_on_one_observation_group_under_a_single_key():
    # This is the shared-identity invariant expressed as data: three claims,
    # one identity unit, therefore one key and one association row.
    observation = _observation("m_shared")
    claims = [
        _claim("c1", "m_shared", attribute="category", value="ship"),
        _claim("c2", "m_shared", attribute="subtype", value="cargo"),
        _claim("c3", "m_shared", attribute="heading", value="north"),
    ]
    problem = InferenceProblem(observations=[observation], claims=claims, objects=[_object("k1")])
    problem.validate()

    grouped = problem.claims_by_observation()
    assert list(grouped.keys()) == ["m_shared"]
    assert len(grouped["m_shared"]) == 3
    assert [c.claim_id for c in grouped["m_shared"]] == ["c1", "c2", "c3"]
    # Distinct truth cells, one shared identity.
    assert {c.truth_key for c in grouped["m_shared"]} == {
        ("category", "t0"),
        ("subtype", "t0"),
        ("heading", "t0"),
    }


def test_claims_by_observation_keeps_observations_with_no_claims():
    # Rows must exist even with zero claims; otherwise a silent observation
    # would vanish from the association matrix.
    problem = InferenceProblem(
        observations=[_observation("m1"), _observation("m2", bbox=(0.0, 0.0, 5.0, 5.0))],
        claims=[_claim("c1", "m1")],
    )
    grouped = problem.claims_by_observation()
    assert set(grouped) == {"m1", "m2"}
    assert grouped["m2"] == []


def test_object_ids_appends_null_last_and_skips_inactive_objects():
    inactive = _object("k_off")
    inactive.active = False
    problem = InferenceProblem(objects=[_object("k1"), inactive])
    assert problem.object_ids() == ["k1", NULL_OBJECT_ID]
    assert problem.object_ids(include_null=False) == ["k1"]


# ----------------------------------------------------------------------
# determinism of ids
# ----------------------------------------------------------------------
def test_stable_id_is_deterministic_and_input_sensitive():
    # Replaying the same tool call on the same input must reproduce the same
    # observation and claim ids, or dedup/replay detection breaks.
    payload = {"tool": "detector", "bbox": [1, 2, 3, 4]}
    assert stable_id("obs", payload) == stable_id("obs", payload)
    assert stable_id("obs", payload) != stable_id("obs", {"tool": "detector", "bbox": [1, 2, 3, 5]})


def test_stable_id_ignores_dict_key_order_but_honors_prefix():
    # Key order is an artifact of construction, not of content.
    assert stable_id("obs", {"a": 1, "b": 2}) == stable_id("obs", {"b": 2, "a": 1})
    assert stable_id("obs", {"a": 1}) != stable_id("claim", {"a": 1})
    assert stable_id("obs", {"a": 1}).startswith("obs_")


def test_fingerprint_is_deterministic_input_sensitive_and_fixed_width():
    payload = {"tool": "detector", "value": "ship"}
    assert fingerprint(payload) == fingerprint(payload)
    assert fingerprint(payload) != fingerprint({"tool": "detector", "value": "tank"})
    assert len(fingerprint(payload)) == 24
    # Same content, different key order -> same fingerprint, so replay of an
    # identical tool call is detected regardless of serialization order.
    assert fingerprint({"x": 1, "y": 2}) == fingerprint({"y": 2, "x": 1})


def test_stable_id_and_fingerprint_are_distinct_digests_of_the_same_payload():
    payload = {"a": 1}
    assert stable_id("obs", payload).split("_", 1)[1] != fingerprint(payload)
