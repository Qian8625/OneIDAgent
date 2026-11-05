"""
Manifest-driven skill plugin loading utilities.

A skill is owned by one plugin directory:

    skills/plugins/<skill_id>/
      manifest.yaml
      skill.md
      worker.py
      normalizer.py
      test_case.yaml

The manifest is the single source of truth for routing, inputs, runtime worker
configuration, documentation, and optional normalizer module metadata.
"""
from __future__ import annotations

from copy import deepcopy
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml


SKILLS_DIR = Path(__file__).resolve().parent
DEFAULT_PLUGIN_DIR = SKILLS_DIR / "plugins"


class SkillManifestError(RuntimeError):
    """Raised when skill manifests cannot be loaded or normalized."""


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SkillManifestError(f"Manifest must be a mapping: {path}")
    return data


def resolve_manifest_path(manifest: Mapping[str, Any], value: Optional[str]) -> Optional[str]:
    """Resolve a manifest path relative to the plugin directory."""
    if not value:
        return None

    raw = Path(str(value)).expanduser()
    if raw.is_absolute():
        return str(raw)

    plugin_dir = Path(str(manifest.get("_plugin_dir") or SKILLS_DIR))
    return str((plugin_dir / raw).resolve())


def import_python_file(path: str | Path, module_name: Optional[str] = None):
    """Import a Python file by path and return the loaded module."""
    py_path = Path(path)
    if not py_path.exists():
        raise SkillManifestError(f"Python module file does not exist: {py_path}")

    final_module_name = module_name or f"skill_plugin_{py_path.parent.name}_{py_path.stem}"
    spec = importlib_util.spec_from_file_location(final_module_name, py_path)
    if spec is None or spec.loader is None:
        raise SkillManifestError(f"Unable to build import spec for {py_path}")

    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_skill_manifests(plugin_dir: Path | str = DEFAULT_PLUGIN_DIR) -> Dict[str, Dict[str, Any]]:
    """Load all enabled per-skill manifests keyed by canonical skill name."""
    root = Path(plugin_dir)
    if not root.exists():
        return {}

    manifests: Dict[str, Dict[str, Any]] = {}
    for manifest_path in sorted(root.glob("*/manifest.yaml")):
        manifest = _read_yaml(manifest_path)
        if manifest.get("enabled", True) is False:
            continue

        name = manifest.get("name")
        if not name or not isinstance(name, str):
            raise SkillManifestError(f"Missing string field 'name' in {manifest_path}")
        if name in manifests:
            raise SkillManifestError(f"Duplicate skill manifest name '{name}' at {manifest_path}")

        normalized = deepcopy(manifest)
        normalized.setdefault("type", "skill")
        normalized["_manifest_path"] = str(manifest_path.resolve())
        normalized["_plugin_dir"] = str(manifest_path.parent.resolve())
        manifests[name] = normalized

    return manifests


def import_skill_normalizers(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    """Import plugin-local normalizer modules declared by manifests."""
    for skill_name, manifest in manifests.items():
        normalizer = dict(manifest.get("normalizer") or {})
        module_path = normalizer.get("module")
        if not module_path:
            continue
        resolved = resolve_manifest_path(manifest, module_path)
        if not resolved:
            continue
        import_python_file(resolved, module_name=f"skill_plugin_{skill_name}_normalizer")


def build_skill_registry(
    manifests: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Build runtime skill registry data from plugin manifests."""
    registry: Dict[str, Dict[str, Any]] = {}

    for skill_name, manifest in manifests.items():
        executor = dict(manifest.get("executor") or {})
        doc = dict(manifest.get("doc") or {})
        inputs = dict(manifest.get("inputs") or {})
        normalizer = dict(manifest.get("normalizer") or {})

        md_path = resolve_manifest_path(manifest, doc.get("path"))
        script_path = resolve_manifest_path(manifest, executor.get("script"))
        test_case_path = resolve_manifest_path(manifest, "test_case.yaml")

        registry[skill_name] = {
            "executor_model": executor.get("model", skill_name),
            "md_path": md_path,
            "normalizer": normalizer.get("name"),
            "required_inputs": _as_list(inputs.get("required")),
            "optional_inputs": _as_list(inputs.get("optional")),
            "defaults": dict(inputs.get("defaults") or {}),
            "allow_extra_args": inputs.get("allow_extra_args", True),
            "timeout_sec": int(executor.get("timeout_sec", 240)),
            "input_bindings": dict(inputs.get("bindings") or {}),
            "executor_script": script_path,
            "test_case_path": test_case_path,
            "executor": executor,
            "routing": dict(manifest.get("routing") or {}),
            "manifest": dict(manifest),
            "manifest_path": manifest.get("_manifest_path"),
            "plugin_dir": manifest.get("_plugin_dir"),
        }

    return registry


# Backward-compatible alias for older imports inside this package.
build_legacy_skill_registry = build_skill_registry


def build_router_rules(
    manifests: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    return {
        name: dict(manifest.get("routing") or {})
        for name, manifest in manifests.items()
    }


def build_skill_input_bindings(
    manifests: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, str]]:
    bindings: Dict[str, Dict[str, str]] = {}
    for name, manifest in manifests.items():
        inputs = dict(manifest.get("inputs") or {})
        bindings[name] = dict(inputs.get("bindings") or {})
    return bindings


def build_skill_timeout_map(
    manifests: Mapping[str, Mapping[str, Any]]
) -> Dict[str, int]:
    timeouts: Dict[str, int] = {}
    for name, manifest in manifests.items():
        executor = dict(manifest.get("executor") or {})
        timeouts[name] = int(executor.get("timeout_sec", 240))
    return timeouts


def get_skill_manifest(
    manifests: Mapping[str, Mapping[str, Any]], skill_name: str
) -> Dict[str, Any]:
    return dict(manifests.get(skill_name) or {})


def get_skill_router_rules(
    manifests: Mapping[str, Mapping[str, Any]], skill_name: str
) -> Dict[str, Any]:
    manifest = manifests.get(skill_name) or {}
    return dict(manifest.get("routing") or {})


def get_skill_input_bindings(
    manifests: Mapping[str, Mapping[str, Any]], skill_name: str
) -> Dict[str, str]:
    manifest = manifests.get(skill_name) or {}
    inputs = dict(manifest.get("inputs") or {})
    return dict(inputs.get("bindings") or {})


def iter_skill_worker_configs(
    manifests: Mapping[str, Mapping[str, Any]]
) -> Iterable[Dict[str, Any]]:
    def sort_key(item):
        skill_name, manifest = item
        executor = dict(manifest.get("executor") or {})
        port = executor.get("port")
        return (port is None, port if port is not None else 0, skill_name)

    for skill_name, manifest in sorted(manifests.items(), key=sort_key):
        executor = dict(manifest.get("executor") or {})
        yield {
            "skill_name": skill_name,
            "worker_name": executor.get("worker_name", executor.get("model", skill_name)),
            "job_name": executor.get("job_name", f"{skill_name.lower()}_skill"),
            "calculate_type": executor.get("calculate_type", "control"),
            "script_addr": resolve_manifest_path(manifest, executor.get("script")),
            "port": executor.get("port"),
            "host": executor.get("host", "0.0.0.0"),
            "conda_env": executor.get("conda_env"),
            "cuda_visible_devices": executor.get("cuda_visible_devices"),
        }


__all__ = [
    "SKILLS_DIR",
    "DEFAULT_PLUGIN_DIR",
    "SkillManifestError",
    "resolve_manifest_path",
    "import_python_file",
    "load_skill_manifests",
    "import_skill_normalizers",
    "build_skill_registry",
    "build_legacy_skill_registry",
    "build_router_rules",
    "build_skill_input_bindings",
    "build_skill_timeout_map",
    "get_skill_manifest",
    "get_skill_router_rules",
    "get_skill_input_bindings",
    "iter_skill_worker_configs",
]
