"""
Validation helpers for manifest-driven skills.

This module is intentionally dependency-light so it can be used by future CI,
local scripts, or launch-config generators.  It validates only files under the
skills package and does not mutate repository state.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from tool_server.tool_workers.skills.manifest import (
    DEFAULT_PLUGIN_DIR,
    SkillManifestError,
    load_skill_manifests,
    resolve_manifest_path,
)
from tool_server.tool_workers.skills.registry import SKILL_REGISTRY


_REQUIRED_TOP_LEVEL_FIELDS = ("name", "executor", "doc", "inputs", "routing")
_REQUIRED_EXECUTOR_FIELDS = ("model", "script")
_REQUIRED_DOC_FIELDS = ("path",)
_REQUIRED_INPUT_FIELDS = ("required",)
_REQUIRED_ROUTING_FIELDS = ("keywords",)


class SkillValidationError(RuntimeError):
    """Raised when one or more skill manifests fail validation."""

    def __init__(self, errors: Iterable[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _is_non_empty_list(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def _path_exists(manifest: Mapping[str, Any], raw_path: Optional[str]) -> bool:
    resolved = resolve_manifest_path(manifest, raw_path)
    return bool(resolved and Path(resolved).exists())


def validate_manifest_shape(skill_name: str, manifest: Mapping[str, Any]) -> List[str]:
    """Validate required manifest fields and basic value types."""
    errors: List[str] = []
    prefix = f"[{skill_name}]"

    for field in _REQUIRED_TOP_LEVEL_FIELDS:
        if field not in manifest:
            errors.append(f"{prefix} missing top-level field: {field}")

    executor = manifest.get("executor") or {}
    if not isinstance(executor, Mapping):
        errors.append(f"{prefix} executor must be a mapping")
        executor = {}
    for field in _REQUIRED_EXECUTOR_FIELDS:
        if not executor.get(field):
            errors.append(f"{prefix} missing executor.{field}")

    doc = manifest.get("doc") or {}
    if not isinstance(doc, Mapping):
        errors.append(f"{prefix} doc must be a mapping")
        doc = {}
    for field in _REQUIRED_DOC_FIELDS:
        if not doc.get(field):
            errors.append(f"{prefix} missing doc.{field}")

    inputs = manifest.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        errors.append(f"{prefix} inputs must be a mapping")
        inputs = {}
    for field in _REQUIRED_INPUT_FIELDS:
        if field not in inputs:
            errors.append(f"{prefix} missing inputs.{field}")
    if "required" in inputs and not isinstance(inputs.get("required"), list):
        errors.append(f"{prefix} inputs.required must be a list")
    if "optional" in inputs and not isinstance(inputs.get("optional"), list):
        errors.append(f"{prefix} inputs.optional must be a list")
    if "defaults" in inputs and not isinstance(inputs.get("defaults"), Mapping):
        errors.append(f"{prefix} inputs.defaults must be a mapping")
    if "bindings" in inputs and not isinstance(inputs.get("bindings"), Mapping):
        errors.append(f"{prefix} inputs.bindings must be a mapping")

    routing = manifest.get("routing") or {}
    if not isinstance(routing, Mapping):
        errors.append(f"{prefix} routing must be a mapping")
        routing = {}
    for field in _REQUIRED_ROUTING_FIELDS:
        if not _is_non_empty_list(routing.get(field)):
            errors.append(f"{prefix} routing.{field} must be a non-empty list")

    return errors


def validate_manifest_paths(skill_name: str, manifest: Mapping[str, Any]) -> List[str]:
    """Validate file paths declared by a manifest."""
    errors: List[str] = []
    prefix = f"[{skill_name}]"

    executor = manifest.get("executor") or {}
    doc = manifest.get("doc") or {}
    normalizer = manifest.get("normalizer") or {}

    if executor.get("script") and not _path_exists(manifest, executor.get("script")):
        errors.append(f"{prefix} executor.script does not exist: {executor.get('script')}")

    if doc.get("path") and not _path_exists(manifest, doc.get("path")):
        errors.append(f"{prefix} doc.path does not exist: {doc.get('path')}")

    # normalizer.module is optional during the compatibility phase.  If supplied,
    # it must exist so plugin-based normalizers can be introduced safely.
    if normalizer.get("module") and not _path_exists(manifest, normalizer.get("module")):
        errors.append(f"{prefix} normalizer.module does not exist: {normalizer.get('module')}")

    return errors


def validate_unique_executor_fields(manifests: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Validate uniqueness of executor models and ports."""
    errors: List[str] = []
    models: Dict[str, str] = {}
    ports: Dict[int, str] = {}

    for skill_name, manifest in manifests.items():
        executor = manifest.get("executor") or {}
        model = executor.get("model") or skill_name
        if model in models:
            errors.append(
                f"[{skill_name}] executor.model duplicates [{models[model]}]: {model}"
            )
        models[model] = skill_name

        port = executor.get("port")
        if port in [None, ""]:
            continue
        try:
            port_int = int(port)
        except Exception:
            errors.append(f"[{skill_name}] executor.port must be an integer: {port}")
            continue
        if port_int in ports:
            errors.append(
                f"[{skill_name}] executor.port duplicates [{ports[port_int]}]: {port_int}"
            )
        ports[port_int] = skill_name

    return errors


def validate_registry_projection(manifests: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Validate that loaded manifests project into the legacy registry correctly."""
    errors: List[str] = []
    manifest_names = set(manifests.keys())
    registry_names = set(SKILL_REGISTRY.keys())

    missing_in_registry = sorted(manifest_names - registry_names)
    if missing_in_registry:
        errors.append(f"Manifests missing from SKILL_REGISTRY: {missing_in_registry}")

    orphan_registry = sorted(registry_names - manifest_names)
    if orphan_registry:
        errors.append(f"SKILL_REGISTRY contains entries without manifests: {orphan_registry}")

    for skill_name in sorted(manifest_names & registry_names):
        spec = SKILL_REGISTRY[skill_name]
        manifest = manifests[skill_name]
        executor = manifest.get("executor") or {}
        inputs = manifest.get("inputs") or {}

        if spec.get("executor_model") != executor.get("model", skill_name):
            errors.append(f"[{skill_name}] registry executor_model mismatch")

        if spec.get("required_inputs") != _as_list(inputs.get("required")):
            errors.append(f"[{skill_name}] registry required_inputs mismatch")

        if spec.get("optional_inputs") != _as_list(inputs.get("optional")):
            errors.append(f"[{skill_name}] registry optional_inputs mismatch")

    return errors


def _literal_string_keys(node: ast.AST) -> List[str]:
    if not isinstance(node, ast.Dict):
        return []
    keys = []
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.append(key.value)
    return keys


def _dict_value(node: ast.AST, key_name: str) -> Optional[ast.AST]:
    if not isinstance(node, ast.Dict):
        return None
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and key.value == key_name:
            return value
    return None


def _find_dict_value(node: ast.AST, key_name: str) -> Optional[ast.AST]:
    for child in ast.walk(node):
        value = _dict_value(child, key_name)
        if value is not None:
            return value
    return None


def _get_tool_instruction_property_keys(script_path: str) -> Optional[List[str]]:
    try:
        tree = ast.parse(Path(script_path).read_text(encoding="utf-8"))
    except Exception:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "get_tool_instruction":
            continue
        properties = _find_dict_value(node, "properties")
        if properties is not None:
            return _literal_string_keys(properties)
    return None


def validate_worker_instruction_projection(skill_name: str, manifest: Mapping[str, Any]) -> List[str]:
    """Validate worker get_tool_instruction exposes manifest-declared inputs."""
    errors: List[str] = []
    prefix = f"[{skill_name}]"

    executor = manifest.get("executor") or {}
    script_path = resolve_manifest_path(manifest, executor.get("script"))
    if not script_path or not Path(script_path).exists():
        return errors

    property_keys = _get_tool_instruction_property_keys(script_path)
    if property_keys is None:
        errors.append(f"{prefix} unable to parse get_tool_instruction properties from {executor.get('script')}")
        return errors

    properties = set(property_keys)
    inputs = manifest.get("inputs") or {}
    expected = set(_as_list(inputs.get("required")))
    expected.update(_as_list(inputs.get("optional")))
    expected.update((inputs.get("defaults") or {}).keys())

    missing = sorted(field for field in expected if field not in properties)
    for field in missing:
        errors.append(f"{prefix} get_tool_instruction missing manifest input: {field}")

    return errors


def validate_skill_manifests(
    plugin_dir: Path | str = DEFAULT_PLUGIN_DIR,
    raise_on_error: bool = False,
) -> Tuple[bool, List[str]]:
    """
    Validate all enabled skill manifests.

    Returns:
        (ok, errors)
    """
    errors: List[str] = []

    try:
        manifests = load_skill_manifests(plugin_dir)
    except SkillManifestError as exc:
        errors.append(str(exc))
        if raise_on_error:
            raise SkillValidationError(errors)
        return False, errors

    if not manifests:
        errors.append(f"No enabled skill manifests found under {plugin_dir}")

    for skill_name, manifest in manifests.items():
        if manifest.get("name") != skill_name:
            errors.append(f"[{skill_name}] manifest name/key mismatch")
        errors.extend(validate_manifest_shape(skill_name, manifest))
        errors.extend(validate_manifest_paths(skill_name, manifest))
        errors.extend(validate_worker_instruction_projection(skill_name, manifest))

    errors.extend(validate_unique_executor_fields(manifests))
    errors.extend(validate_registry_projection(manifests))

    ok = len(errors) == 0
    if errors and raise_on_error:
        raise SkillValidationError(errors)
    return ok, errors


def format_validation_report(ok: bool, errors: Iterable[str]) -> str:
    errors = list(errors)
    if ok:
        return "Skill manifest validation passed."
    return "Skill manifest validation failed:\n" + "\n".join(f"- {err}" for err in errors)


__all__ = [
    "SkillValidationError",
    "validate_skill_manifests",
    "format_validation_report",
]
