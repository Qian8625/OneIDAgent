"""D-S experiment configuration and project-relative paths."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[3]

def _resolve(path_value: str) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else ROOT / path

def load_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = ROOT / "DS_VI/configs" / "ds_default.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
    return config


def merge_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge ``overrides`` into a copy of ``config`` (two levels)."""
    merged: Dict[str, Any] = {
        key: (dict(value) if isinstance(value, Mapping) else value) for key, value in config.items()
    }
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            nested = dict(merged[key])
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, Mapping) and isinstance(nested.get(inner_key), Mapping):
                    nested[inner_key] = {**nested[inner_key], **inner_value}
                else:
                    nested[inner_key] = inner_value
            merged[key] = nested
        else:
            merged[key] = value
    return merged


__all__ = ["ROOT", "load_config", "merge_overrides"]
