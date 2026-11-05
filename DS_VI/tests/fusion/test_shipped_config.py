"""Shipped configuration: the single-version baseline.

There is one shipped file (``runtime_best.yaml``).  These tests freeze the values
that matter and lock the two properties the pipeline relies on:

* the settings are resolved per family, not per file;
* the built-in defaults (used when PyYAML is missing) agree with the file for
  everything except the operating points, which are deliberately not mirrored.
"""
from __future__ import annotations

from DS_VI.models.cascade.vi import (
    BEST_CONFIG,
    resolve_cascade_settings,
    shipped_cascade_settings,
)
from DS_VI.models.shared_identity.config import load_config

OVERRIDES = {"provenance": {"grouping_mode": "observation"}}


def test_the_shipped_file_is_the_baseline():
    assert BEST_CONFIG.endswith("runtime_best.yaml")
    assert not BEST_CONFIG.endswith("_v"), "no versioned configuration may remain"


def test_settings_are_resolved_per_family():
    match = resolve_cascade_settings(BEST_CONFIG, "match", OVERRIDES)
    time = resolve_cascade_settings(BEST_CONFIG, "time", OVERRIDES)

    # Ambiguity rejection is tuned per family; time uses mutual construction, so
    # its ratio is inert and keeps the historical value.
    assert match["extra_margin"] == 0.30
    assert time["extra_margin"] == 0.20

    # The attribute gate is on for time only: match's SAR view annotates no
    # colour and adjacent targets share every other attribute.
    assert match["extra_consistency"] == ()
    assert time["extra_consistency"] == ("subtype", "color", "size_bin", "orientation")


def test_extra_mode_supports_per_family_mappings():
    assert resolve_cascade_settings(BEST_CONFIG, "match", OVERRIDES)["extra_mode"] == "pruned"
    assert resolve_cascade_settings(BEST_CONFIG, "time", OVERRIDES)["extra_mode"] == "mutual"


def test_shipped_channel_and_output_policy():
    for family in ("match", "time"):
        settings = resolve_cascade_settings(BEST_CONFIG, family, OVERRIDES)
        assert settings["extra_links"] == 2
        assert settings["extra_prior_floor"] == 0.2
        assert settings["extra_displacement_gate"] == 2.0
        assert settings["consistency_mode"] == "weighted"
        assert settings["consistency_tau"] == 0.0
        assert settings["consistency_min_weight"] == 0.5
        # Blend: keep every D-S pair and re-rank, instead of a second hard filter.
        assert settings["policy"] == "blend"
        assert settings["blend_weight"] == 0.5


def test_ds_acceptance_point_is_per_family():
    assert resolve_cascade_settings(BEST_CONFIG, "match", OVERRIDES)["accept_betp"] == 0.50
    assert resolve_cascade_settings(BEST_CONFIG, "time", OVERRIDES)["accept_betp"] == 0.50
    for family in ("match", "time"):
        # Single-side BetP only spans ~0.34-0.66, so a raised pair threshold must
        # not leak into the single-side decision.
        assert resolve_cascade_settings(BEST_CONFIG, family, OVERRIDES)["single_accept_betp"] == 0.30


def test_evidence_side_of_the_shipped_config():
    config = load_config(BEST_CONFIG)
    inference = config["inference"]
    assert inference["evidence_cap"] is None          # the old cap collapsed A to the prior
    assert inference["eta"] == 1.5
    assert inference["feedback_damping"] == 1.0   # plain S/eta, no damping
    # Paper C.5: different epochs define different facts.
    assert list(inference["time_invariant_attributes"]) == []
    assert inference["capacity_mode"] == "block_object_column"
    assert inference["default_capacity"] is None
    assert inference["block_capacity"] == 1.0
    assert config["reliability"]["outlier_mixture"] == 0.0   # paper has no outlier term


def test_shipped_config_knows_the_reissued_attributes():
    config = load_config(BEST_CONFIG)
    assert "color" in config["value_domains"]
    assert len(config["value_domains"]["color"]) == 10
    assert config["reliability"]["by_modality_attribute"]["optical"]["color"] > 0.5
    # ``state`` is kept only so a pre-2026-11 workbook still has a domain.
    assert "state" in config["value_domains"]


def test_fallback_mirrors_the_shipped_cascade():
    # Without PyYAML the loader starts from the built-in defaults; they must agree
    # with the shipped file, otherwise a PyYAML-less run silently differs.
    from DS_VI.models.shared_identity.config import _FALLBACK_CONFIG

    shipped = shipped_cascade_settings(load_config(BEST_CONFIG), "time")
    fallback = shipped_cascade_settings(_FALLBACK_CONFIG, "time")
    # The operating points are deliberately not mirrored into the built-in
    # defaults: they are choices, and mirroring them would make an explicit
    # ``vi_config`` inherit the shipped point instead of its own file's values.
    for key in ("accept_betp", "single_accept_betp", "extra_displacement_gate"):
        shipped.pop(key)
        fallback.pop(key)
    assert shipped == fallback
