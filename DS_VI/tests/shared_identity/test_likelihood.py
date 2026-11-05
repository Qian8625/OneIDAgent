"""Noise-model guards: the symmetric likelihood and the below-chance sign."""
from __future__ import annotations

import math

import pytest

from DS_VI.models.shared_identity.likelihood import (
    claim_log_likelihood,
    expected_log_likelihood,
)


def test_reliability_below_chance_reverses_direction():
    # A 3-value domain has chance 1/3. Below chance the channel is anti-correlated
    # (paper Eq. 7): agreeing with the reported value must *lower* its odds.
    size = 3
    logged_same = claim_log_likelihood("horizontal", "horizontal", 0.24, size)
    logged_other = claim_log_likelihood("horizontal", "vertical", 0.24, size)
    assert logged_same < logged_other
    assert logged_other == pytest.approx(math.log(0.76 / 2), abs=1e-9)


def test_expected_likelihood_is_anti_informative_below_chance():
    truth = {"horizontal": 1 / 3, "vertical": 1 / 3, "diagonal": 1 / 3}
    value = expected_log_likelihood("vertical", truth, 0.2, 3)
    # E_Q[log P(y|v)] falls below the uniform baseline log(1/3): the report is
    # evidence against every candidate, which is what the sign flip means.
    assert value < math.log(1 / 3)


def test_reliability_above_chance_stays_informative():
    # SAR orientation measured 0.63 on a 3-value domain: keep the signal.
    logged_same = claim_log_likelihood("horizontal", "horizontal", 0.63, 3)
    logged_other = claim_log_likelihood("horizontal", "vertical", 0.63, 3)
    assert logged_same > logged_other


def test_single_value_domain_carries_no_information():
    assert claim_log_likelihood("a", "b", 0.9, 1) == 0.0
