"""Per-pair breakdown must reproduce the aggregate metrics exactly."""
from __future__ import annotations

from DS_VI.experiments.metrics.property_metrics import evaluate_family
from DS_VI.tests.fusion.test_property_metrics import build_fixture


def test_per_pair_counts_match_the_aggregates(tmp_path):
    tool, truth = build_fixture(tmp_path)
    metrics = evaluate_family(tool, truth, "time", per_pair=True)
    pairs = metrics["per_pair"]["pairs"]

    aggregate = metrics["G5_identity"]["matched_by_shape"]
    assert sum(row["gt_double"] for row in pairs.values()) == aggregate["double_view"]["double"] + aggregate["double_view"]["single"] + aggregate["double_view"]["missed"]
    assert sum(row["double_confirmed"] for row in pairs.values()) == aggregate["double_view"]["double"]
    assert sum(row["over_merged"] for row in pairs.values()) == aggregate["single_side"]["double"]

    g4 = metrics["G4_end_to_end"]["category"]
    correct = sum(row["g4_category_correct"] for row in pairs.values())
    total = sum(row["g4_category_total"] for row in pairs.values())
    assert total == g4["support"]
    assert correct == round(g4["accuracy"] * g4["support"])


def test_per_pair_is_absent_by_default(tmp_path):
    tool, truth = build_fixture(tmp_path)
    metrics = evaluate_family(tool, truth, "time")
    assert "per_pair" not in metrics
