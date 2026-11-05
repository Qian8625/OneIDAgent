"""
Manifest-backed skill registry.

Each skill is owned by one plugin directory:
    tool_server/tool_workers/skills/plugins/<skill_id>/
      manifest.yaml
      skill.md
      worker.py
      normalizer.py
      test_case.yaml

The manifest is the single source of truth. Runtime registries below are derived
from manifests at import time.
"""
from tool_server.tool_workers.skills.manifest import (
    build_skill_registry,
    get_skill_input_bindings,
    get_skill_manifest,
    get_skill_router_rules,
    import_skill_normalizers,
    load_skill_manifests,
)


SKILL_MANIFESTS = load_skill_manifests()
import_skill_normalizers(SKILL_MANIFESTS)

SKILL_REGISTRY = build_skill_registry(SKILL_MANIFESTS)
ROUTER_RULES = {
    name: dict(spec.get("routing") or {})
    for name, spec in SKILL_REGISTRY.items()
}
SKILL_INPUT_BINDINGS = {
    name: dict(spec.get("input_bindings") or {})
    for name, spec in SKILL_REGISTRY.items()
}
SKILL_TIMEOUTS = {
    name: int(spec.get("timeout_sec", 240))
    for name, spec in SKILL_REGISTRY.items()
}


__all__ = [
    "SKILL_MANIFESTS",
    "SKILL_REGISTRY",
    "ROUTER_RULES",
    "SKILL_INPUT_BINDINGS",
    "SKILL_TIMEOUTS",
    "get_skill_manifest",
    "get_skill_router_rules",
    "get_skill_input_bindings",
]
