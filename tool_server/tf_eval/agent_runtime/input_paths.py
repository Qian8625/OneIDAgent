from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tool_server.utils.paths import PROJECT_ROOT

#: 数据集根目录。仓库不承载数据集，默认指向外部数据目录；
#: 用 ``ONEID_DATA_ROOT`` 覆盖。仅当候选全部不存在时才会走到这里。
DEFAULT_DATA_ROOT = Path(os.environ.get("ONEID_DATA_ROOT")
                         or (Path(__file__).resolve().parents[3] / "data"))


IMAGE_KEYS = (
    "primary_image",
    "optical_image",
    "sar_image",
    "time1_image",
    "time2_image",
)


class InputPathResolutionError(ValueError):
    def __init__(self, value: Any, candidates: Iterable[Path]):
        self.value = value
        self.candidates = [str(item) for item in candidates]
        joined = "\n".join(f"- {item}" for item in self.candidates)
        super().__init__(f"Input path not found: {value}\nTried:\n{joined}")


def _path_or_none(value: Any) -> Optional[Path]:
    if value in [None, ""]:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def _append_candidate(candidates: List[Path], value: Path) -> None:
    try:
        candidate = value.resolve(strict=False)
    except Exception:
        candidate = value
    if candidate not in candidates:
        candidates.append(candidate)


def _env_roots() -> List[Path]:
    roots: List[Path] = []
    for name in ("ONEID_IMAGE_ROOT", "RS_DATA_ROOT", "ONEID_DATA_ROOT"):
        value = os.environ.get(name)
        if value:
            roots.append(Path(value).expanduser())
    if DEFAULT_DATA_ROOT not in roots:
        roots.append(DEFAULT_DATA_ROOT)
    return roots


def resolve_input_path(
    value: Any,
    dataset_file: Any = None,
    image_root: Any = None,
    dataset_root: Any = None,
    project_root: Any = None,
) -> str:
    path = _path_or_none(value)
    if path is None:
        return "" if value is None else str(value)

    roots: List[Path] = []
    if dataset_file:
        roots.append(Path(dataset_file).expanduser().resolve().parent)
    if image_root:
        roots.append(Path(image_root).expanduser())
    if dataset_root:
        roots.append(Path(dataset_root).expanduser())
    roots.extend(_env_roots())
    roots.append(Path(project_root).expanduser() if project_root else PROJECT_ROOT)

    candidates: List[Path] = []
    if path.is_absolute():
        _append_candidate(candidates, path)
    else:
        for root in roots:
            _append_candidate(candidates, root / path)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    raise InputPathResolutionError(value, candidates)


def resolve_sample_images(
    sample: Dict[str, Any],
    dataset_file: Any = None,
    image_root: Any = None,
    dataset_root: Any = None,
    project_root: Any = None,
) -> List[str]:
    images = sample.get("images") or []
    if not isinstance(images, list):
        images = [images]
    resolved = []
    for image in images:
        if image:
            resolved.append(resolve_input_path(image, dataset_file, image_root, dataset_root, project_root))
    return resolved


def resolve_registry_paths(
    registry: Dict[str, Any],
    dataset_file: Any = None,
    image_root: Any = None,
    dataset_root: Any = None,
    project_root: Any = None,
) -> Dict[str, Any]:
    resolved = dict(registry or {})
    for key in IMAGE_KEYS:
        if resolved.get(key):
            resolved[key] = resolve_input_path(resolved[key], dataset_file, image_root, dataset_root, project_root)
    return resolved


def preflight_validate_samples(
    samples: Iterable[Dict[str, Any]],
    dataset_file: Any = None,
    image_root: Any = None,
    dataset_root: Any = None,
    project_root: Any = None,
) -> None:
    errors = []
    for index, sample in enumerate(samples):
        sample_id = sample.get("sample_id") or sample.get("idx") or index
        try:
            resolve_sample_images(sample, dataset_file, image_root, dataset_root, project_root)
            explicit = sample.get("input_registry") if isinstance(sample.get("input_registry"), dict) else {}
            resolve_registry_paths(explicit, dataset_file, image_root, dataset_root, project_root)
        except InputPathResolutionError as exc:
            errors.append({"sample_id": sample_id, "value": str(exc.value), "candidates": exc.candidates})
    if errors:
        preview = "\n".join(
            f"sample={item['sample_id']} value={item['value']} candidates={item['candidates'][:4]}"
            for item in errors[:10]
        )
        raise InputPathResolutionError(
            f"{len(errors)} sample input path(s) failed preflight. First failures:\n{preview}",
            [],
        )
