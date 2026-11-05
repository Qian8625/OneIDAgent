"""Robustness controls for the shared-identity A/Q loop.

Four knobs are exercised here, all neutral by default so that enabling one is
always an explicit experimental choice:

``reliability.shrinkage``
    Pulls every reliability toward 0.5 before the clamp.
``reliability.outlier_mixture``
    Mixes a uniform outlier component into every claim likelihood.
``inference.evidence_cap``
    Clips a single claim's contribution to one association cell.
``inference.feedback_damping``
    Scales the whole truth-derived evidence term; 0.0 is the no-feedback path.

The load-bearing test is
:func:`test_one_corrupted_claim_drags_the_shared_identity_row`.  It reproduces
the failure mode behind the n=96 out-of-fold result: a single corrupted
``subtype`` claim moves the *entire* observation row off the geometrically
correct object, because shared identity accumulates all claims of an observation
into one row.  Tests that only assert "the knob changed a number" would pass for
a model that never had the defect, so the fixture asserts the defect first.
"""
from __future__ import annotations

import math

import pytest

from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.inference import run_joint_inference
from DS_VI.models.shared_identity.likelihood import claim_log_likelihood, truth_log_posterior
from DS_VI.models.shared_identity.reliability import ReliabilityModel
from DS_VI.models.shared_identity.schema import InferenceProblem, SchemaValidationError
from DS_VI.tests.shared_identity.test_inference_truth import (
    make_claim,
    make_object,
    make_observation,
)

# The observation sits closest to object A; B is a plausible but worse candidate.
OBJECT_A_BBOX = [10, 20, 50, 80]
OBJECT_B_BBOX = [14, 24, 54, 84]
OBSERVATION_BBOX = [11, 21, 51, 81]


def capture_scene(overrides=None, subtype_value: str = "warship"):
    """One observation, three claims, two candidate objects.

    Geometry prefers A.  Object priors say A is a ``cargo_ship`` and B is a
    ``warship``, so a corrupted ``subtype`` claim of ``warship`` is evidence for
    the wrong object.  The other two claims (``category``, ``size_bin``) are
    uninformative between A and B: only the corrupted one discriminates.
    """
    cfg = load_config(overrides=overrides)
    observations = [make_observation("m", "optical", OBSERVATION_BBOX)]
    claims = [
        make_claim("c_subtype", "m", "subtype", subtype_value),
        make_claim("c_category", "m", "category", "ship"),
        make_claim("c_size", "m", "size_bin", "large"),
    ]
    objects = [
        make_object("A", "ship", OBJECT_A_BBOX),
        make_object("B", "ship", OBJECT_B_BBOX),
    ]
    problem = InferenceProblem(
        observations=observations,
        claims=claims,
        objects=objects,
        candidate_edges=build_candidate_graph(observations, objects, cfg),
        truth_priors={
            ("A", "subtype", "t0"): {"cargo_ship": 60.0},
            ("B", "subtype", "t0"): {"warship": 60.0},
        },
    )
    return run_joint_inference(problem, cfg)


def merged(*sections):
    """Merge override fragments without letting a later one drop a section."""
    result: dict = {}
    for fragment in sections:
        for key, values in (fragment or {}).items():
            result[key] = {**result.get(key, {}), **values}
    return result


CONFIDENT_SUBTYPE = {"reliability": {"by_modality_attribute": {"optical": {"subtype": 0.97}}}}


# ----------------------------------------------------------------------
# neutral defaults
# ----------------------------------------------------------------------
def test_shipped_defaults_are_neutral():
    """The knobs ship in their no-op setting, so earlier runs stay reproducible."""
    cfg = load_config()
    assert cfg["reliability"]["shrinkage"] == 1.0
    assert cfg["reliability"]["outlier_mixture"] == 0.0
    assert cfg["inference"]["evidence_cap"] is None
    assert cfg["inference"]["feedback_damping"] == 1.0


def test_zero_mixture_reproduces_the_unmodified_likelihood():
    """``e = 0`` must be bit-identical to the pure symmetric noise model."""
    for reliability in (0.35, 0.7, 0.97):
        for observed, candidate in (("a", "a"), ("a", "b")):
            baseline = claim_log_likelihood(observed, candidate, reliability, 5)
            assert (
                claim_log_likelihood(
                    observed, candidate, reliability, 5, outlier_mixture=0.0
                )
                == baseline
            )


def test_explicit_neutral_overrides_match_the_default_run():
    """Writing the neutral values out explicitly must not change any number."""
    default = capture_scene()
    explicit = capture_scene(
        {
            "reliability": {"shrinkage": 1.0, "outlier_mixture": 0.0},
            "inference": {"evidence_cap": None, "feedback_damping": 1.0},
        }
    )
    assert explicit.association == default.association
    assert explicit.truth == default.truth
    assert explicit.objective_history == default.objective_history
    assert explicit.diagnostics["clipped_claim_contributions"] == 0


# ----------------------------------------------------------------------
# the failure mode these controls exist for
# ----------------------------------------------------------------------
def test_one_corrupted_claim_drags_the_shared_identity_row():
    """Assert the defect, then assert which controls contain it.

    Shared identity means the corrupted ``subtype`` claim does not merely
    mislabel the object: it moves the whole row, so ``category`` and ``size_bin``
    are attributed to the wrong object too.  This is the mechanism behind the
    untied-beats-shared result on multi-attribute primitives.
    """
    baseline = capture_scene(CONFIDENT_SUBTYPE)
    prior = baseline.diagnostics["physical_prior"]["m"]
    assert prior["A"] > prior["B"], "fixture must keep geometry in favour of A"

    row = baseline.association_row("m")
    assert row["B"] > row["A"], (
        "expected one corrupted claim to capture the shared identity row; "
        "if this fails the fixture no longer reproduces the failure mode"
    )

    # Clipping the per-claim contribution, or switching the feedback off, hands
    # the row back to geometry.
    for label, extra in (
        ("evidence_cap", {"inference": {"evidence_cap": 0.05}}),
        ("feedback_damping", {"inference": {"feedback_damping": 0.0}}),
    ):
        repaired = capture_scene(merged(CONFIDENT_SUBTYPE, extra)).association_row("m")
        assert repaired["A"] > repaired["B"], f"{label} did not restore the correct object"

    # Shrinkage and the outlier mixture reduce the damage without fully undoing
    # it: they bound one claim's influence, they do not remove the claim. Report
    # that honestly rather than picking a strength that happens to flip the sign.
    for label, extra in (
        ("shrinkage", {"reliability": {"shrinkage": 0.25}}),
        ("outlier_mixture", {"reliability": {"outlier_mixture": 0.2}}),
    ):
        softened = capture_scene(merged(CONFIDENT_SUBTYPE, extra)).association_row("m")
        assert softened["A"] > row["A"], f"{label} did not recover any mass for A"
        assert softened["B"] < row["B"], f"{label} did not reduce the wrong object"


def test_outlier_mixture_bounds_single_claim_certainty():
    """One claim must not reach its reliability as a posterior probability."""
    domain = ["a", "b", "c", "d", "e", "f"]
    plain = truth_log_posterior(domain, [("a", 1.0, 0.97)])
    assert math.isclose(max(plain.values()), 0.97, abs_tol=1e-9)

    previous = max(plain.values())
    for mixture in (0.05, 0.2, 0.5):
        bounded = truth_log_posterior(domain, [("a", 1.0, 0.97)], None, 1e-12, mixture)
        assert max(bounded.values()) < previous
        previous = max(bounded.values())


def test_outlier_mixture_keeps_the_agreeing_majority():
    """Bounding certainty must not invert the evidence: two agree, one does not."""
    domain = ["a", "b", "c", "d", "e", "f"]
    claims = [("a", 1.0, 0.97), ("b", 1.0, 0.97), ("b", 1.0, 0.97)]
    for mixture in (0.0, 0.2, 0.5):
        posterior = truth_log_posterior(domain, claims, None, 1e-12, mixture)
        assert max(posterior, key=posterior.get) == "b"


# ----------------------------------------------------------------------
# shrinkage
# ----------------------------------------------------------------------
def test_shrinkage_pulls_reliability_toward_one_half():
    model = ReliabilityModel(load_config(overrides={"reliability": {"shrinkage": 0.5}}))
    # optical|category ships at 0.88 -> 0.5 + 0.5 * 0.38
    assert math.isclose(model.reliability_for("optical", "category"), 0.69)

    # A cell below 0.5 is pulled up, not down: shrinkage removes information, it
    # never flips the direction of the evidence.
    low = ReliabilityModel(
        load_config(
            overrides={
                "reliability": {
                    "shrinkage": 0.5,
                    "by_modality_attribute": {"sar": {"probe": 0.40}},
                }
            }
        )
    )
    assert math.isclose(low.reliability_for("sar", "probe"), 0.45)


def test_shrinkage_applies_to_the_calibrated_table_too(tmp_path):
    """The point of shrinkage is to temper *fitted* tables, not just the shipped one."""
    table = tmp_path / "fitted.yaml"
    table.write_text(
        "reliability:\n  by_modality_attribute:\n    optical:\n      subtype: 0.96\n",
        encoding="utf-8",
    )
    overrides = {
        "reliability": {
            "shrinkage": 0.5,
            "contextual": {"enabled": True, "config_path": str(table)},
        }
    }
    model = ReliabilityModel(load_config(overrides=overrides))
    assert math.isclose(model.reliability_for("optical", "subtype"), 0.73)
    assert model.describe()["source"] == "calibrated_shrunk"


def test_full_shrinkage_makes_every_cell_uninformative():
    model = ReliabilityModel(load_config(overrides={"reliability": {"shrinkage": 0.0}}))
    assert math.isclose(model.reliability_for("optical", "category"), 0.5)
    assert math.isclose(model.reliability_for("sar", "subtype"), 0.5)


def test_shrinkage_is_reported_and_bounded():
    described = ReliabilityModel(
        load_config(overrides={"reliability": {"shrinkage": 0.25}})
    ).describe()
    assert described["shrinkage"] == 0.25
    assert described["source"] == "fixed_config_shrunk"
    for invalid in (1.5, -0.1):
        with pytest.raises(SchemaValidationError):
            ReliabilityModel(load_config(overrides={"reliability": {"shrinkage": invalid}}))


def test_outlier_mixture_is_validated_and_reported():
    model = ReliabilityModel(
        load_config(overrides={"reliability": {"outlier_mixture": 0.05}})
    )
    assert model.outlier_mixture == 0.05
    assert model.describe()["outlier_mixture"] == 0.05
    for invalid in (1.0, -0.01):
        with pytest.raises(SchemaValidationError):
            ReliabilityModel(
                load_config(overrides={"reliability": {"outlier_mixture": invalid}})
            )


# ----------------------------------------------------------------------
# evidence cap and feedback damping
# ----------------------------------------------------------------------
def test_evidence_cap_clips_and_counts_contributions():
    capped = capture_scene({"inference": {"evidence_cap": 0.01}})
    assert capped.diagnostics["evidence_cap"] == 0.01
    assert capped.diagnostics["clipped_claim_contributions"] > 0

    uncapped = capture_scene()
    assert uncapped.diagnostics["evidence_cap"] is None
    assert uncapped.diagnostics["clipped_claim_contributions"] == 0


def test_zero_feedback_damping_leaves_association_at_the_physical_prior():
    """``feedback_damping = 0`` means A is the projected physical prior alone."""
    damped = capture_scene({"inference": {"feedback_damping": 0.0}})
    assert damped.diagnostics["feedback_damping"] == 0.0

    prior = damped.diagnostics["physical_prior"]["m"]
    row = damped.association_row("m")
    for object_id, probability in row.items():
        assert math.isclose(probability, float(prior.get(object_id, 0.0)), abs_tol=1e-9)


def test_damping_interpolates_between_no_feedback_and_full_feedback():
    """Wrong-object mass must grow monotonically with the feedback strength."""
    masses = []
    for damping in (0.0, 0.25, 0.5, 1.0):
        state = capture_scene(
            merged(CONFIDENT_SUBTYPE, {"inference": {"feedback_damping": damping}})
        )
        masses.append(state.association_row("m")["B"])
    assert masses == sorted(masses), masses
    assert masses[0] < masses[-1]


def test_negative_feedback_damping_is_rejected():
    with pytest.raises(SchemaValidationError):
        capture_scene({"inference": {"feedback_damping": -0.5}})


# ----------------------------------------------------------------------
# value domains
# ----------------------------------------------------------------------
def test_configured_domains_cover_the_annotation_vocabulary():
    """Gold values must be representable, otherwise strict NLL is undefined.

    The vocabulary is the annotator label space declared in
    ``scripts/data/build_annotation_workbook.py``, not values read off the
    evaluation gold.
    """
    domains = load_config()["value_domains"]
    for value in ("road_bridge", "dock", "small_boat", "cargo_ship", "bus", "pier"):
        assert value in domains["subtype"], value
    for value in ("moored", "parked", "static", "moving", "unknown"):
        assert value in domains["state"], value
    for value in ("small", "medium", "large"):
        assert value in domains["size_bin"], value
