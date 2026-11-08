"""Controlled property replay: live detections, workbook attributes only.

The contract under test is the one this mode now promises:

* boxes, labels and scores come from the live detector payloads, never from the
  workbook;
* the workbook contributes the requested target's attributes and nothing else;
* attributes bind to live boxes by the frozen snapshot index (``view`` +
  ``instance_id`` suffix), with the frozen bbox used only to audit the binding.
"""
import pytest
from openpyxl import Workbook, load_workbook
from PIL import Image
from tool_server.tool_workers.workbook_replay import ReplayError, load_property_workbook
from tool_server.tool_workers.online_workers.CrossModalInterpret_worker import OneIDCrossModalEngine

PAIRS = {"task4": "task4_000001", "task5": "task5_000001"}

#: Workbook row per (view, raw detector index).  Index 2 is another class, so it
#: is not an attribute of the target and must not reach the attributes map.
WORKBOOK_ROWS = (
    (0, (10, 10, 40, 40), "ship", "cargo_ship", "gray"),
    (1, (50, 50, 90, 90), "ship", "fishing_boat", "white"),
    (2, (100, 100, 120, 120), "tank", "storage_tank", "green"),
)


@pytest.fixture
def replay_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ONEID_ENABLE_WORKBOOK_REPLAY", "1")
    monkeypatch.setenv("ONEID_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("ONEID_REPLAY_CONFIG", raising=False)
    monkeypatch.delenv("ONEID_FORCE_MODE", raising=False)
    for task, views in (("task4", ("rgb", "sar")), ("task5", ("time1", "time2"))):
        wb = Workbook()
        ws = wb.active
        ws.title = "instances"
        ws.append(["instance_id", "pair_id", "view", "image_file", "image_width", "image_height", "x1", "y1", "x2", "y2", "category", "detection_score", "subtype", "color", "size_bin", "orientation", "visibility", "is_real_object"])
        pair = PAIRS[task]
        for view in views:
            name = pair + "_" + view + ".png"
            Image.new("RGB", (128, 128)).save(tmp_path / name)
            for index, box, label, subtype, color in WORKBOOK_ROWS:
                ws.append([f"{pair}:{view}:{index:04d}", pair, view, name, 128, 128, *box, label, 0.95,
                           subtype, color, "small", "horizontal", "clear", False])
        samples = wb.create_sheet("samples")
        samples.append(["pair_id", "image_a", "image_b"])
        samples.append([pair, pair + "_" + views[0] + ".png", pair + "_" + views[1] + ".png"])
        wb.create_sheet("identities").append(["DO_NOT_READ", "truth_wrong"])
        wb.save(tmp_path / ("tool_" + task + "_property.xlsx"))
    return tmp_path


def _ship(box, score=0.9):
    return {"bbox": list(box), "label": "ship", "score": score}


def _tank(box, score=0.7):
    return {"bbox": list(box), "label": "tank", "score": score}


def _live():
    """The raw detector order the workbook froze: two ships, then another class."""
    return [_ship((10, 10, 40, 40), 0.9), _ship((50, 50, 90, 90), 0.8), _tank((100, 100, 120, 120))]


def _args(dataset="task4", family="match", **extra):
    args = dict(source_mode="workbook_replay", dataset_id=dataset, family=family,
                pair_id=PAIRS[dataset], target="ship",
                rgb_detections=_live(), sar_detections=_live())
    args.update(extra)
    return args


def test_loader_returns_target_attributes_only(replay_root):
    data = load_property_workbook("task4", "match", PAIRS["task4"], "ship")
    assert data["provenance"]["attribute_only"] is True
    assert data["provenance"]["detection_source"] == "live_detectors"
    assert data["provenance"]["sheets_read"] == ["instances", "samples"]
    assert len(data["provenance"]["sha256"]) == 64
    assert "detections" not in data and "input_counts" not in data
    # Only the requested target is kept, keyed by the frozen raw detector index.
    for view in ("a", "b"):
        assert sorted(data["attributes"][view]) == [0, 1]
        row = data["attributes"][view][0]
        assert row["category"] == "ship"
        assert row["attributes"] == {"subtype": "cargo_ship", "color": "gray", "size_bin": "small", "orientation": "horizontal"}
        assert row["bbox"] == [10.0, 10.0, 40.0, 40.0]
        assert "is_real_object" not in row
    assert data == load_property_workbook("task4", "match", PAIRS["task4"], "ship")


@pytest.mark.parametrize("dataset,family", [("task4", "match"), ("task5", "time")])
def test_real_pipeline_both_families(replay_root, dataset, family):
    result = OneIDCrossModalEngine().generate(_args(dataset, family, output_mode="full"))
    assert result["error_code"] == 0, result
    diagnostics = result["replay_diagnostics"]
    assert diagnostics["detection_source"] == "live_detectors"
    assert diagnostics["attribute_source"] == "property_workbook"
    assert diagnostics["attribute_binding"]["bound"] == 4
    assert diagnostics["attribute_binding"]["unbound_live"] == 0
    assert diagnostics["attribute_binding"]["unbound_workbook"] == 0
    assert diagnostics["attribute_binding"]["index_space_matches"] is True
    assert diagnostics["attribute_claims"] == 16
    assert result["structured"]["capability"]["time_family"] == (family == "time")
    assert not diagnostics["theta_enabled"]
    assert result["objects"]
    assert all(c["source_tool"] == "property_workbook" for claims in result["attribute_claims"].values() for c in claims)
    beliefs = result["vi_objects"][0]["attribute_beliefs"]
    assert any(k.startswith("color@") for k in beliefs)
    if family == "time":
        assert any(k.endswith("@t0") for k in beliefs)
        assert any(k.endswith("@t1") for k in beliefs)


def test_detections_and_scores_are_live_not_workbook(replay_root):
    live = [_ship((20, 20, 45, 45), 0.30), _ship((55, 55, 95, 95), 0.25)]
    result = OneIDCrossModalEngine().generate(_args(output_mode="full", rgb_detections=live, sar_detections=live))
    assert result["error_code"] == 0, result
    boxes = [obj["view_a_bbox"] for obj in result["objects"] if obj.get("view_a_bbox")]
    assert [20.0, 20.0, 45.0, 45.0] in boxes
    assert all(box != [10.0, 10.0, 40.0, 40.0] for box in boxes)
    # The frozen bbox survives only as an audit trail of the binding.
    binding = result["replay_diagnostics"]["attribute_binding"]
    assert 0.0 < binding["min_iou"] < 1.0


def test_missing_or_targetless_live_detections_fail(replay_root):
    engine = OneIDCrossModalEngine()
    base = dict(source_mode="workbook_replay", dataset_id="task4", family="match", pair_id=PAIRS["task4"], target="ship")
    for extra in ({"rgb_detections": [], "sar_detections": _live()},
                  {"rgb_detections": _live(), "sar_detections": []},
                  {"rgb_detections": [], "sar_detections": []}):
        result = engine.generate(dict(base, **extra))
        assert result["error_code"] != 0
        assert "live" in result["text"]
    tank = [{"bbox": [1, 1, 5, 5], "label": "tank", "score": 0.9}]
    result = engine.generate(dict(base, rgb_detections=tank, sar_detections=tank))
    assert result["error_code"] != 0
    assert "no ship" in result["text"]


def test_live_box_without_workbook_row_keeps_no_attributes(replay_root):
    live = [_ship((10, 10, 40, 40)), _ship((50, 50, 90, 90)), _ship((60, 60, 70, 70))]
    result = OneIDCrossModalEngine().generate(_args(output_mode="full", rgb_detections=live, sar_detections=live))
    assert result["error_code"] == 0, result
    binding = result["replay_diagnostics"]["attribute_binding"]
    assert binding["unbound_live"] == 2  # one per view; index 2 is a tank row
    assert binding["bound"] == 4
    assert result["replay_diagnostics"]["attribute_claims"] == 16


def test_no_index_binding_at_all_fails(replay_root):
    # Only the non-target classes sit on the snapshot's indices; the live target
    # box lands where the workbook has no target row, so nothing can bind.
    live = [_tank((100, 100, 120, 120)), _tank((60, 60, 70, 70)), _ship((10, 10, 40, 40))]
    result = OneIDCrossModalEngine().generate(_args(rgb_detections=live, sar_detections=live))
    assert result["error_code"] != 0
    assert "no live target box matched" in result["text"]


def test_index_space_audit_reports_length_mismatch(replay_root):
    # A shorter live payload cannot be compared positionally with the snapshot;
    # the run still succeeds, but the audit says so instead of trusting it.
    one = [_ship((10, 10, 40, 40))]
    result = OneIDCrossModalEngine().generate(_args(output_mode="full", rgb_detections=one, sar_detections=one))
    assert result["error_code"] == 0, result
    binding = result["replay_diagnostics"]["attribute_binding"]
    assert binding["snapshot_rows"] == {"a": 3, "b": 3}
    assert binding["live_payload"] == {"a": 1, "b": 1}
    assert binding["index_space_matches"] is False
    assert binding["bound"] == 2


def test_non_target_rows_are_not_read_or_validated(replay_root):
    path = replay_root / "tool_task4_property.xlsx"
    wb = load_workbook(path)
    ws = wb["instances"]
    ws.cell(4, 11, "car")          # third instance row: another class entirely
    ws.cell(4, 14, "chartreuse")   # and an out-of-domain attribute value
    wb.save(path)
    data = load_property_workbook("task4", "match", PAIRS["task4"], "ship")
    assert sorted(data["attributes"]["a"]) == [0, 1]
    assert data["instance_counts"] == {"a": 3, "b": 3}
    # The very same garbage on a row that *is* read still fails the run.
    wb = load_workbook(path)
    wb["instances"].cell(2, 14, "chartreuse")
    wb.save(path)
    result = OneIDCrossModalEngine().generate(_args())
    assert result["error_code"] != 0
    assert "Invalid attribute value" in result["text"]


def test_drifted_snapshot_fails_explicitly(replay_root):
    far = [_ship((80, 80, 120, 120))]
    result = OneIDCrossModalEngine().generate(_args(rgb_detections=far, sar_detections=far))
    assert result["error_code"] != 0
    assert "binding" in result["text"]


def test_disabled_and_invalid_requests(replay_root, monkeypatch):
    engine = OneIDCrossModalEngine()
    for extra in ({"ds_config_path": "/tmp/no"}, {"family": "time"}, {"mode": "legacy_hard"},
                  {"pair_id": "../../escape"}, {"workbook_path": "/tmp/truth.xlsx"}):
        result = engine.generate(_args(**extra))
        assert result["error_code"] != 0
        assert result["mode"] != "legacy_hard"
    monkeypatch.delenv("ONEID_ENABLE_WORKBOOK_REPLAY")
    assert engine.generate(_args())["error_code"] != 0


def test_path_and_duplicate_validation(replay_root, monkeypatch):
    import json
    whitelist = replay_root / "whitelist.json"
    whitelist.write_text(json.dumps({"datasets": {"task4": "../tool_task4_property.xlsx"}}))
    monkeypatch.setenv("ONEID_REPLAY_CONFIG", str(whitelist))
    with pytest.raises(ReplayError):
        load_property_workbook("task4", "match", PAIRS["task4"], "ship")


def test_duplicate_instance_index_is_rejected(replay_root):
    path = replay_root / "tool_task4_property.xlsx"
    wb = load_workbook(path)
    ws = wb["instances"]
    row = list(next(ws.iter_rows(min_row=2, max_row=2, values_only=True)))
    ws.append(row)  # same instance_id -> duplicate
    wb.save(path)
    with pytest.raises(ReplayError, match="Duplicate"):
        load_property_workbook("task4", "match", PAIRS["task4"], "ship")


def test_empty_attributes_do_not_become_claims(replay_root):
    path = replay_root / "tool_task4_property.xlsx"
    wb = load_workbook(path)
    ws = wb["instances"]
    row = list(next(ws.iter_rows(min_row=2, max_row=2, values_only=True)))
    row[12:16] = [None] * 4
    row[0] = f"{PAIRS['task4']}:rgb:0007"
    ws.append(row)
    wb.save(path)
    data = load_property_workbook("task4", "match", PAIRS["task4"], "ship")
    assert sorted(data["attributes"]["a"]) == [0, 1, 7]
    assert data["attributes"]["a"][7]["attributes"] == {}


def test_malformed_attribute_value_fails(replay_root):
    path = replay_root / "tool_task4_property.xlsx"
    wb = load_workbook(path)
    wb["instances"].cell(2, 14, "chartreuse")  # color column of the first row
    wb.save(path)
    result = OneIDCrossModalEngine().generate(_args())
    assert result["error_code"] != 0
    assert "Invalid attribute value" in result["text"]


def test_replay_cap_is_explicit(replay_root, monkeypatch):
    from tool_server.tool_workers.online_workers import ds_vi_adapter
    monkeypatch.setattr(ds_vi_adapter, "MAX_OBSERVATIONS", 1)
    result = OneIDCrossModalEngine().generate(_args())
    assert result["error_code"] != 0
    assert "cap" in result["text"]


def test_inference_failure_is_not_legacy(replay_root, monkeypatch):
    import DS_VI.models.cascade.vi as vi

    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(vi, "run_vi_pair", fail)
    result = OneIDCrossModalEngine().generate(_args())
    assert result["error_code"] != 0
    assert result["status"] == "replay_error"
    assert result["mode"] != "legacy_hard"


def test_vi_only_replay_preserves_attributes_and_time(replay_root):
    result = OneIDCrossModalEngine().generate(_args("task5", "time", mode="vi_only", output_mode="full"))
    assert result["error_code"] == 0, result
    assert result["replay_diagnostics"]["attribute_claims"] == 16
    assert result["capability"]["time_family"]


def test_runtime_image_binding_is_checked(replay_root):
    result = OneIDCrossModalEngine().generate(_args(rgb_image=str(replay_root / "wrong.png")))
    assert result["error_code"] != 0


def test_time_family_prefers_its_own_image_keys(replay_root):
    # App wiring binds rgb_image/sar_image for every mode; a time request must
    # still validate against its declared time1/time2 pair.
    real = str(replay_root / (PAIRS["task5"] + "_time1.png"))
    args = _args("task5", "time", time1_image=real, rgb_image=str(replay_root / "wrong.png"))
    assert OneIDCrossModalEngine().generate(args)["error_code"] == 0
    wrong = _args("task5", "time", time1_image=str(replay_root / "wrong.png"))
    assert OneIDCrossModalEngine().generate(wrong)["error_code"] != 0


def test_reordered_rows_and_repeated_calls_keep_same_evidence(replay_root):
    engine = OneIDCrossModalEngine()
    first = engine.generate(_args(output_mode="full"))
    assert first["error_code"] == 0
    assert first == engine.generate(_args(output_mode="full"))
    path = replay_root / "tool_task4_property.xlsx"
    wb = load_workbook(path)
    ws = wb["instances"]
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    ws.delete_rows(2, len(rows))
    for row in reversed(rows):
        ws.append(row)
    wb.save(path)
    after = engine.generate(_args(output_mode="full"))
    assert first["attribute_claims"] == after["attribute_claims"]
    assert first["vi_objects"] == after["vi_objects"]
