"""Unit tests for :mod:`DS_VI.experiments.metrics.property_metrics`.

The fixtures are tiny hand-built workbooks so every number below can be checked
by reading this file: the protocol under test is the *join*, not the data.
"""
from __future__ import annotations

import json

from openpyxl import Workbook

from DS_VI.experiments.metrics.property_metrics import (
    evaluate_family,
    match_image,
    object_change_label,
    write_report,
)

#: 2026-11 annotation release: ``color`` replaces ``state`` on the instances sheet.
INSTANCE_HEADER = [
    "instance_id", "pair_id", "view", "image_file", "image_width", "image_height",
    "x1", "y1", "x2", "y2", "category", "subtype", "color", "orientation",
    "is_real_object", "visibility", "size_bin", "detection_score",
]
IDENTITY_HEADER = [
    "object_id", "pair_id", "instance_a", "instance_b", "category", "subtype",
    "association_score",
]
SAMPLE_HEADER = ["pair_id", "image_a", "image_b", "identity_complete", "note"]


def _instance(instance_id, view, box, category="ship", subtype=None, color=None,
              orientation=None, size_bin=None, score=None):
    x1, y1, x2, y2 = box
    return [instance_id, instance_id.split(":")[0], view, f"{view}.png", 1024, 1024,
            x1, y1, x2, y2, category, subtype, color, orientation, True, "clear",
            size_bin, score]


def _identity(object_id, instance_a, instance_b, category="ship", subtype=None):
    return [object_id, object_id.split(":")[0], instance_a, instance_b, category, subtype, None]


def write_book(path, instances, identities, samples):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "instances"
    sheet.append(INSTANCE_HEADER)
    for row in instances:
        sheet.append(row)
    identity_sheet = workbook.create_sheet("identities")
    identity_sheet.append(IDENTITY_HEADER)
    for row in identities:
        identity_sheet.append(row)
    sample_sheet = workbook.create_sheet("samples")
    sample_sheet.append(SAMPLE_HEADER)
    for row in samples:
        sample_sheet.append(row)
    workbook.save(path)
    return path


def build_fixture(tmp_path):
    """One pair, two views: one duplicate box, one false positive, two misses."""
    truth = write_book(
        tmp_path / "truth.xlsx",
        instances=[
            # a1 + b1 are one persisted object, a2 only exists at t0 (disappeared).
            _instance("p:time1:0001", "time1", (0, 0, 10, 10), "ship", "cargo_ship", "white", "vertical", "large"),
            _instance("p:time1:0002", "time1", (100, 100, 120, 120), "tank", "storage_tank", "gray", "horizontal", "small"),
            _instance("p:time2:0001", "time2", (0, 0, 10, 10), "ship", "cargo_ship", "white", "vertical", "large"),
            _instance("p:time2:0002", "time2", (500, 500, 510, 510), "aircraft", "civilian_airliner", "blue", "diagonal", "medium"),
        ],
        identities=[
            _identity("p:obj:001", "p:time1:0001", "p:time2:0001", "ship", "cargo_ship"),
            _identity("p:obj:002", "p:time1:0002", None, "tank", "storage_tank"),
            _identity("p:obj:003", None, "p:time2:0002", "aircraft", "civilian_airliner"),
        ],
        samples=[["p", "a.png", "b.png", True, None]],
    )
    tool = write_book(
        tmp_path / "tool.xlsx",
        instances=[
            # matches truth a1; wrong subtype, no orientation prediction
            _instance("p:time1:0000", "time1", (0, 0, 10, 10), "ship", "small_boat", "white", None, "large", 0.9),
            # duplicate of the same truth box -> not a false positive
            _instance("p:time1:0001", "time1", (0, 0, 10, 11), "ship", "cargo_ship", "white", "vertical", "large", 0.8),
            # false positive
            _instance("p:time1:0002", "time1", (300, 300, 320, 320), "vehicle", "car", "red", "horizontal", "small", 0.3),
            _instance("p:time2:0000", "time2", (0, 0, 10, 10), "ship", "cargo_ship", "gray", "vertical", "large", 0.7),
        ],
        identities=[
            # predicted as one cross-view object; the second truth view is missed
            _identity("p:pred:001", "p:time1:0000", "p:time2:0000", "ship", "small_boat"),
            _identity("p:pred:002", "p:time1:0002", None, "vehicle", "car"),
        ],
        samples=[["p", "a.png", "b.png", True, None]],
    )
    return tool, truth


def test_duplicate_is_not_a_false_positive():
    matches = match_image(
        [
            {"instance_id": "t1", "x1": 0, "y1": 0, "x2": 10, "y2": 10},
            {"instance_id": "t2", "x1": 0, "y1": 0, "x2": 10, "y2": 11},
        ],
        [{"instance_id": "g1", "x1": 0, "y1": 0, "x2": 10, "y2": 10}],
    )
    assert len(matches["matches"]) == 1
    assert [row["tool"]["instance_id"] for row in matches["duplicates"]] == ["t2"]
    assert matches["unmatched_tool"] == []
    assert matches["unmatched_truth"] == []


def test_change_label_derivation():
    instances = {
        "a": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "b": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "far": {"x1": 100, "y1": 100, "x2": 110, "y2": 110},
    }
    assert object_change_label({"instance_a": "a", "instance_b": "b"}, instances) == "stayed"
    assert object_change_label({"instance_a": "a", "instance_b": "far"}, instances) == "moved"
    assert object_change_label({"instance_a": "a"}, instances) == "disappeared"
    assert object_change_label({"instance_b": "a"}, instances) == "appeared"


def test_family_metrics(tmp_path):
    tool, truth = build_fixture(tmp_path)
    metrics = evaluate_family(tool, truth, "time")

    # time1 : tool 0000 -> truth 0001 (IoU 1.0), tool 0001 is a duplicate,
    #         tool 0002 is a false positive, truth 0002 is missed.
    # time2 : tool 0000 -> truth 0001 (IoU 1.0), truth 0002 is missed.
    detection = metrics["G1_detection"]["micro"]
    assert detection["tp"] == 2
    assert detection["fp"] == 1
    assert detection["fn"] == 2
    assert metrics["G1_detection"]["duplicates"] == 1

    attributes = metrics["G2_attributes"]
    assert attributes["category"]["support"] == 2
    assert attributes["category"]["accuracy"] == 1.0
    # subtype: one of the two matched pairs disagrees (small_boat vs cargo_ship).
    assert attributes["subtype"]["support"] == 2
    assert attributes["subtype"]["accuracy"] == 0.5
    # orientation: one matched pair predicted nothing -> fill 0.5, the other is right.
    assert attributes["orientation"]["fill_rate"] == 0.5
    assert attributes["orientation"]["accuracy"] == 0.5

    identity = metrics["G5_identity"]
    # truth objects: 1 double-view (matched), 2 single-side (both missed).
    assert identity["objects"]["gt"] == 3
    assert identity["objects"]["tp"] == 1
    assert identity["objects"]["fp"] == 1  # the spurious vehicle prediction
    assert identity["objects"]["fn"] == 2
    assert identity["matched_by_shape"]["double_view"] == {"double": 1, "single": 0, "missed": 0}
    assert identity["matched_by_shape"]["single_side"] == {"double": 0, "single": 0, "missed": 2}
    assert identity["gt_double_view"]["association_recall"] == 1.0

    end_to_end = metrics["G4_end_to_end"]
    assert end_to_end["category"]["support"] == 3
    # only the cross-view object is identified; the two single-side truths are missed.
    assert end_to_end["category"]["accuracy"] == round(1 / 3, 6)
    assert end_to_end["subtype"]["accuracy"] == 0.0
    assert end_to_end["size_bin"]["accuracy"] == round(1 / 3, 6)

    change = metrics["G6_change"]["four_class"]
    assert change["gt_support"] == 3          # stayed + disappeared + appeared
    assert change["correct"] == 1             # only the persisted object is right
    assert change["missed"] == 2
    assert change["spurious"] == 1            # the vehicle prediction has no truth
    assert change["accuracy"] == 0.25         # 1 / (3 GT + 1 spurious)


def test_report_writes_json_and_workbook(tmp_path):
    tool, truth = build_fixture(tmp_path)
    metrics = evaluate_family(tool, truth, "time")
    paths = write_report(metrics, tmp_path / "out")
    assert paths["json"].is_file()
    assert paths["xlsx"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["G1_detection"]["micro"]["tp"] == 2


#: An ``identities`` sheet from the 2026-11 release: the change relation is
#: annotated instead of left to geometry.
ANNOTATED_IDENTITY_HEADER = IDENTITY_HEADER + ["temporal_relation"]


def write_annotated_book(path, instances, identities, samples):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "instances"
    sheet.append(INSTANCE_HEADER)
    for row in instances:
        sheet.append(row)
    identity_sheet = workbook.create_sheet("identities")
    identity_sheet.append(ANNOTATED_IDENTITY_HEADER)
    for row in identities:
        identity_sheet.append(row)
    sample_sheet = workbook.create_sheet("samples")
    sample_sheet.append(SAMPLE_HEADER)
    for row in samples:
        sample_sheet.append(row)
    workbook.save(path)
    return path


def _annotated_identity(object_id, instance_a, instance_b, relation, category="ship"):
    return [object_id, object_id.split(":")[0], instance_a, instance_b, category, None,
            None, relation]


def test_annotated_change_label_overrides_geometry():
    from DS_VI.experiments.metrics.property_metrics import annotated_change_label

    assert annotated_change_label({"temporal_relation": "persisted"}) == "stayed"
    assert annotated_change_label({"temporal_relation": "persisted_moved"}) == "moved"
    assert annotated_change_label({"temporal_relation": "appeared"}) == "appeared"
    assert annotated_change_label({"temporal_relation": "disappeared"}) == "disappeared"
    assert annotated_change_label({"temporal_relation": None}) is None


def test_change_labels_prefer_the_annotation(tmp_path):
    """A far-apart pair annotated ``persisted`` scores as stayed, not moved."""
    instances = [
        _instance("p:time1:0001", "time1", (0, 0, 10, 10), "ship", "cargo_ship", "white", "vertical", "large"),
        _instance("p:time2:0001", "time2", (500, 500, 510, 510), "ship", "cargo_ship", "white", "vertical", "large"),
    ]
    identities = [_annotated_identity("p:obj:001", "p:time1:0001", "p:time2:0001", "persisted")]
    truth = write_annotated_book(tmp_path / "truth.xlsx", instances, identities,
                                 [["p", "a.png", "b.png", True, None]])
    tool = write_annotated_book(tmp_path / "tool.xlsx", instances, identities,
                                [["p", "a.png", "b.png", True, None]])

    annotated = evaluate_family(tool, truth, "time")
    derived = evaluate_family(tool, truth, "time", truth_change_from_annotation=False)
    # The boxes are 700 px apart, so the *prediction* is "moved" either way; only
    # the ground truth changes: the annotation says stayed, the derivation says
    # moved.  The cell therefore proves which protocol labelled the GT.
    assert annotated["G6_change"]["four_class"]["confusion"]["stayed"]["moved"] == 1
    assert derived["G6_change"]["four_class"]["confusion"]["moved"]["moved"] == 1


def test_color_is_scored_and_state_is_absent(tmp_path):
    tool, truth = build_fixture(tmp_path)
    metrics = evaluate_family(tool, truth, "time")
    assert "color" in metrics["G2_attributes"]
    assert "state" not in metrics["G2_attributes"]
    assert "color" in metrics["G4_end_to_end"]
    assert "state" not in metrics["G4_end_to_end"]
    # matched pair predicted white/gray vs truth white/white -> half right
    assert metrics["G2_attributes"]["color"]["support"] == 2
    assert metrics["G2_attributes"]["color"]["accuracy"] == 0.5
