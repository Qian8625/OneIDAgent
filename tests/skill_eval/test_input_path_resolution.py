from pathlib import Path

import pytest

from tool_server.tf_eval.agent_runtime.input_paths import InputPathResolutionError, resolve_input_path, resolve_registry_paths


def test_resolve_input_path_from_dataset_root(tmp_path: Path):
    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    image = dataset_root / "images_test" / "sample.png"
    image.parent.mkdir()
    image.write_text("x", encoding="utf-8")

    resolved = resolve_input_path("images_test/sample.png", dataset_root=dataset_root)
    assert resolved == str(image.resolve())


def test_resolve_input_path_failure_lists_candidates(tmp_path: Path):
    with pytest.raises(InputPathResolutionError) as excinfo:
        resolve_input_path("missing.png", dataset_root=tmp_path)
    assert "missing.png" in str(excinfo.value)


def test_resolve_registry_paths(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    image = root / "images_test" / "sample.png"
    image.parent.mkdir()
    image.write_text("x", encoding="utf-8")

    registry = resolve_registry_paths({"primary_image": "images_test/sample.png"}, dataset_root=root)
    assert registry["primary_image"] == str(image.resolve())
