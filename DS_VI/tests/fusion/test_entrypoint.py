"""Check the relocated real-detection entry point without external data."""
import json

from DS_VI.experiments.run_detection import main


def test_entrypoint_runs_without_truth(tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    images = tmp_path / "images"
    tools.mkdir()
    images.mkdir()
    for view in ("RGB", "SAR"):
        (tools / f"sample_{view}.json").write_text(json.dumps({"detections": [
            {"bbox": [10, 10, 60, 60], "label": "ship", "score": 0.95},
        ]}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"families": {"match": {
        "tool_dir": str(tools), "image_dir": str(images),
        "truth_dir": str(tmp_path / "truth-must-not-be-read"),
    }}}), encoding="utf-8")
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["run_detection", "--config", str(config),
                                     "--output-dir", str(output)])
    assert main() == 0
    row = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert row["pair_id"] == "sample"
    assert row["objects"]
    assert row["objects"][0]["object_id"] == "sample:pred:0"
    assert row["objects"][0]["label"] == "ship"
