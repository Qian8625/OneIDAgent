"""Unit tests for the D-S evidence rules (numpy only)."""

from __future__ import annotations

import math

import pytest


from DS_VI.models.ds_fusion.ds_rules import (
    VACUOUS,
    PCR6_fusion,
    RDC_fusion,
    adaptive_fusion,
    betp,
    dempster_combine,
    jousselme_distance,
    make_bpa,
    make_decision,
    normalize_bpa,
    validate_bpa,
)


def test_make_bpa_is_valid_and_clamped():
    mass = make_bpa(1.7, reliability=2.0)
    assert validate_bpa(mass)
    assert mass["S"] == pytest.approx(1.0)
    assert mass["B"] == pytest.approx(0.0)
    assert mass["SB"] == pytest.approx(0.0)

    bpa = make_bpa(0.6, reliability=0.8)
    assert bpa["S"] == pytest.approx(0.48)
    assert bpa["B"] == pytest.approx(0.32)
    assert bpa["SB"] == pytest.approx(0.2)


def test_make_bpa_support_blend():
    bpa = make_bpa(0.0, reliability=1.0, support=1.0, support_weight=0.5)
    assert bpa["S"] == pytest.approx(0.5)
    assert bpa["B"] == pytest.approx(0.5)


def test_validate_bpa_rejects_bad_masses():
    with pytest.raises(ValueError):
        validate_bpa({"S": 0.7, "B": 0.7, "SB": 0.0})
    with pytest.raises(ValueError):
        validate_bpa({"S": -0.1, "B": 0.6, "SB": 0.5})
    with pytest.raises(ValueError):
        validate_bpa({"S": 0.5, "B": 0.5, "SB": 0.0, "X": 0.0})


def test_normalize_zero_mass_is_vacuous():
    assert normalize_bpa({"S": 0.0, "B": 0.0, "SB": 0.0}) == VACUOUS


def test_jousselme_distance_properties():
    m1 = make_bpa(0.9, 0.8)
    m2 = make_bpa(0.1, 0.8)
    assert jousselme_distance(m1, m1) == pytest.approx(0.0, abs=1e-12)
    assert jousselme_distance(m1, m2) == pytest.approx(jousselme_distance(m2, m1))
    assert 0.0 <= jousselme_distance(m1, m2) <= 1.0
    assert jousselme_distance(m1, VACUOUS) > 0.0


def test_dempster_zadeh_conflict():
    m1 = {"S": 0.99, "B": 0.01, "SB": 0.0}
    m2 = {"S": 0.01, "B": 0.99, "SB": 0.0}
    mass, conflict = dempster_combine(m1, m2)
    assert conflict == pytest.approx(0.9802, abs=1e-9)
    assert mass is not None
    assert mass["S"] == pytest.approx(0.5, abs=1e-9)
    assert mass["B"] == pytest.approx(0.5, abs=1e-9)


def test_pcr6_zadeh_balanced():
    m1 = {"S": 0.99, "B": 0.01, "SB": 0.0}
    m2 = {"S": 0.01, "B": 0.99, "SB": 0.0}
    mass = PCR6_fusion(m1, m2)
    assert validate_bpa(mass, tol=1e-9)
    assert mass["S"] == pytest.approx(0.5, abs=1e-9)
    assert mass["B"] == pytest.approx(0.5, abs=1e-9)


def test_pcr6_without_conflict_equals_conjunctive():
    m1 = make_bpa(0.8, 1.0)
    mass = PCR6_fusion(m1, dict(VACUOUS))
    assert mass["S"] == pytest.approx(m1["S"])
    assert mass["B"] == pytest.approx(m1["B"])


def test_rdc_agreement_discounting():
    m1 = make_bpa(0.9, 0.8)
    m2 = make_bpa(0.1, 0.8)
    mass, info = RDC_fusion(m1, m2, with_info=True)
    assert validate_bpa(mass)
    assert 0.0 <= info["agreement"] <= 1.0
    assert info["rule"] == "rdc"


def test_adaptive_selects_rule_by_distance():
    identical = make_bpa(0.7, 0.8)
    mass, info = adaptive_fusion(identical, identical, theta=0.5, with_info=True)
    assert info["rule"] == "rdc"
    opposite = adaptive_fusion({"S": 1.0, "B": 0.0, "SB": 0.0}, {"S": 0.0, "B": 1.0, "SB": 0.0}, theta=0.5, with_info=True)
    assert opposite[1]["rule"] == "pcr6"


def test_betp_and_decision():
    mass = {"S": 0.6, "B": 0.2, "SB": 0.2}
    pignistic = betp(mass)
    assert pignistic["S"] == pytest.approx(0.7)
    assert pignistic["B"] == pytest.approx(0.3)
    decision = make_decision(mass, accept_betp=0.5)
    assert decision["decision"] == "ship"
    decision = make_decision({"S": 0.2, "B": 0.6, "SB": 0.2}, accept_betp=0.5)
    assert decision["decision"] == "background"
    decision = make_decision({"S": 0.5, "B": 0.5, "SB": 0.0}, accept_betp=0.5, abstain_band=0.05)
    assert decision["decision"] == "abstain"


def test_all_masses_finite_after_operations():
    m1 = make_bpa(0.31, 0.8, support=0.2)
    m2 = make_bpa(0.32, 0.7, support=0.2)
    for rule in ("rdc", "pcr6", "adaptive"):
        mass = adaptive_fusion(m1, m2, theta=0.5, rule=rule)
        assert validate_bpa(mass)
        assert all(math.isfinite(value) for value in mass.values())
