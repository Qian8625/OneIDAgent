"""Contextual reliability calibration must be a load-bearing switch.

Enabling ``reliability.contextual`` replaces the shipped per-(modality,
attribute) table with one fitted from labelled agreement.  Enabling it without a
readable table is a hard error: a silent fallback would make the calibrated
variant indistinguishable from the baseline, which is exactly the dead-switch
failure this test file guards against.
"""
from __future__ import annotations

import pytest

from DS_VI.models.shared_identity.config import load_config
from DS_VI.models.shared_identity.reliability import ReliabilityModel, load_contextual_table
from DS_VI.models.shared_identity.schema import SchemaValidationError


def enable_contextual(tmp_path, body: str) -> dict:
    path = tmp_path / "reliability.yaml"
    path.write_text(body, encoding="utf-8")
    return load_config(
        overrides={
            "reliability": {
                "contextual": {"enabled": True, "config_path": str(path)},
            }
        }
    )


def test_calibrated_cells_override_the_shipped_table(tmp_path):
    model = ReliabilityModel(
        enable_contextual(
            tmp_path,
            "reliability:\n  by_modality_attribute:\n    optical:\n      category: 0.42\n",
        )
    )

    assert model.reliability_for("optical", "category") == pytest.approx(0.42)
    # A cell the calibration did not cover keeps the shipped value.
    assert model.reliability_for("optical", "subtype") == pytest.approx(0.72)
    described = model.describe()
    assert described["source"] == "calibrated"
    assert described["contextual_cells"] == 1


def test_bare_table_without_the_reliability_wrapper_is_accepted(tmp_path):
    model = ReliabilityModel(
        enable_contextual(
            tmp_path,
            "by_modality_attribute:\n  sar:\n    category: 0.9\n",
        )
    )

    assert model.reliability_for("sar", "category") == pytest.approx(0.9)
    assert model.reliability_for("sar", "subtype") == pytest.approx(0.60)


def test_enabled_without_a_path_raises():
    cfg = load_config(overrides={"reliability": {"contextual": {"enabled": True}}})

    with pytest.raises(SchemaValidationError, match="config_path"):
        ReliabilityModel(cfg)


def test_missing_file_raises(tmp_path):
    cfg = load_config(
        overrides={
            "reliability": {
                "contextual": {
                    "enabled": True,
                    "config_path": str(tmp_path / "absent.yaml"),
                }
            }
        }
    )

    with pytest.raises(SchemaValidationError, match="not found"):
        ReliabilityModel(cfg)


def test_empty_table_raises(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("reliability:\n  by_modality_attribute: {}\n", encoding="utf-8")

    with pytest.raises(SchemaValidationError, match="no 'by_modality_attribute'"):
        load_contextual_table(path)


def test_non_numeric_cells_raise(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "by_modality_attribute:\n  optical:\n    category: high\n", encoding="utf-8"
    )

    with pytest.raises(SchemaValidationError, match="not numeric"):
        load_contextual_table(path)


def test_calibrated_values_are_clamped(tmp_path):
    model = ReliabilityModel(
        enable_contextual(
            tmp_path,
            "by_modality_attribute:\n  optical:\n    category: 1.5\n",
        )
    )

    assert model.reliability_for("optical", "category") == pytest.approx(model.max)
