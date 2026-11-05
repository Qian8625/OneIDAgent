"""Attribute-injection mapping: workbook rows -> VI claim payloads."""
from __future__ import annotations

from openpyxl import Workbook

from DS_VI.experiments.property_attributes import (
    DEFAULT_ATTRIBUTES,
    attribute_names,
    claims_from_items,
    claims_from_table,
    load_attribute_table,
    observation_ids,
)

#: 2026-11 annotation release: ``color`` replaces ``state`` on the instances sheet.
HEADER = [
    "instance_id", "pair_id", "view", "image_file", "image_width", "image_height",
    "x1", "y1", "x2", "y2", "category", "subtype", "color", "orientation",
    "is_real_object", "visibility", "size_bin", "detection_score",
]


def write_workbook(path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "instances"
    sheet.append(HEADER)
    rows = [
        # pair p, rgb view: index 0 has all four attributes, index 1 has none
        ["p:rgb:0000", "p", "rgb", "a.png", 1024, 1024, 0, 0, 10, 10,
         "ship", "cargo_ship", "white", "horizontal", True, "clear", "large", 0.9],
        ["p:rgb:0001", "p", "rgb", "a.png", 1024, 1024, 20, 20, 30, 30,
         "tank", None, None, None, True, "clear", None, 0.8],
        # pair p, sar view: index 0 with a different subtype
        ["p:sar:0000", "p", "sar", "b.png", 1024, 1024, 0, 0, 10, 10,
         "ship", "fishing_boat", "gray", "vertical", True, "poor", "large", 0.7],
    ]
    for row in rows:
        sheet.append(row)
    for name in ("identities", "samples"):
        other = workbook.create_sheet(name)
        other.append(["placeholder"])
    workbook.save(path)
    return path


def test_load_attribute_table_keys_by_pair_view_index(tmp_path):
    path = write_workbook(tmp_path / "tool.xlsx")
    table = load_attribute_table(path, "match")
    assert set(table) == {"p"}
    assert table["p"]["a"][0] == {
        "subtype": "cargo_ship", "color": "white", "orientation": "horizontal",
        "size_bin": "large",
    }
    # A row whose attribute cells are all empty contributes no entry at all.
    assert 1 not in table["p"]["a"]
    assert table["p"]["b"][0]["subtype"] == "fishing_boat"


def test_claims_from_table_maps_to_observation_ids(tmp_path):
    path = write_workbook(tmp_path / "tool.xlsx")
    table = load_attribute_table(path, "match")
    by_view_index = {("a", 0): "obs_a0", ("b", 0): "obs_b0"}
    payloads = claims_from_table(table, "p", by_view_index)
    assert set(payloads) == {"obs_a0", "obs_b0"}
    attributes = {item["attribute"] for item in payloads["obs_a0"]}
    assert attributes == set(DEFAULT_ATTRIBUTES)
    assert payloads["obs_a0"][0]["source_channel"] == "a"


def test_claims_from_items_ignores_unmapped_indices():
    items = [
        {"view": "a", "index": 0, "score": 0.9,
         "attributes": {"subtype": "cargo_ship", "color": None}},
        {"view": "a", "index": 7, "score": 0.3, "attributes": {"subtype": "warship"}},
    ]
    payloads = claims_from_items(items, {("a", 0): "obs_a0"})
    assert set(payloads) == {"obs_a0"}
    assert [item["attribute"] for item in payloads["obs_a0"]] == ["subtype"]


def test_attribute_names_always_contains_category():
    assert attribute_names() == ("category", "subtype", "color", "size_bin", "orientation")
    assert attribute_names(("subtype", "category")) == ("category", "subtype")


def test_observation_ids_follow_deterministic_builder():
    # Two identical calls must produce the same ids: the injection key has to be
    # stable across the trunk entry point and the comparison arms.
    pair = {"base_id": "p", "family": "match", "images": {"a": "a.png", "b": "b.png"}}
    detections = {
        "a": [{"bbox": [0, 0, 10, 10], "label": "ship", "score": 0.9, "index": 0}],
        "b": [{"bbox": [0, 0, 10, 10], "label": "ship", "score": 0.8, "index": 0}],
    }
    config = {"families": {"match": {"views": {"a": {"tool_suffix": "_RGB", "image_suffix": "_RGB",
                                                   "modality": "rgb", "time_key": "t0"},
                                              "b": {"tool_suffix": "_SAR", "image_suffix": "_SAR",
                                                    "modality": "sar", "time_key": "t0"}}}}}
    first = observation_ids(pair, detections, config, "match")
    second = observation_ids(pair, detections, config, "match")
    assert first == second
    assert set(first) == {("a", 0), ("b", 0)}
