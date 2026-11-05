"""Shared helpers for plugin-local skill normalizers."""
from __future__ import annotations

from typing import Any, Dict, List


def pick_first(params: Dict[str, Any], names: List[str], default=None):
    for name in names:
        value = params.get(name)
        if value not in [None, ""]:
            return value
    return default


def set_if_missing(params: Dict[str, Any], key: str, value: Any):
    if params.get(key) in [None, ""] and value not in [None, ""]:
        params[key] = value


def to_bool(value: Any, default: bool = False) -> bool:
    if value in [None, ""]:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "需要"}:
        return True
    if s in {"false", "0", "no", "n", "否", "不需要"}:
        return False
    return default


def to_float_or_none(value: Any):
    if value in [None, ""]:
        return None
    try:
        return float(value)
    except Exception:
        return None


def to_int_or_none(value: Any):
    if value in [None, ""]:
        return None
    try:
        return int(value)
    except Exception:
        return None


def normalize_target_text(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value if x not in [None, ""])
    if isinstance(value, dict):
        for key in ["target", "object", "name", "text", "class"]:
            if value.get(key):
                return str(value[key])
        return str(value)
    return str(value).strip() if value not in [None, ""] else ""
