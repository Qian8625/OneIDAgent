from __future__ import annotations

import json
from typing import Any, Dict, Optional


KEY_FIELDS = [
    "structured",
    "measurement",
    "bbox",
    "selected",
    "selected_detections",
    "detections",
    "objects_used",
    "match_summary",
    "dedup_detections",
    "matched_pairs",
    "rgb_only",
    "sar_only",
    "artifacts",
    "download_path",
]

def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def truncate_structured_value(
    value: Any,
    max_list_items: int = 10,
    max_depth: int = 4,
    max_string_chars: int = 500,
    max_dict_keys: int = 40,
    _depth: int = 0,
) -> Any:
    if _depth >= max_depth:
        if isinstance(value, dict):
            return {"__truncated__": True, "type": "dict", "keys": list(value.keys())[:max_dict_keys]}
        if isinstance(value, list):
            return {"__truncated__": True, "type": "list", "original_len": len(value)}

    if isinstance(value, dict):
        items = list(value.items())
        truncated = {
            str(key): truncate_structured_value(
                item,
                max_list_items=max_list_items,
                max_depth=max_depth,
                max_string_chars=max_string_chars,
                max_dict_keys=max_dict_keys,
                _depth=_depth + 1,
            )
            for key, item in items[:max_dict_keys]
        }
        if len(items) > max_dict_keys:
            truncated["__truncated_keys__"] = {
                "original_len": len(items),
                "kept": max_dict_keys,
            }
        return truncated

    if isinstance(value, list):
        truncated = [
            truncate_structured_value(
                item,
                max_list_items=max_list_items,
                max_depth=max_depth,
                max_string_chars=max_string_chars,
                max_dict_keys=max_dict_keys,
                _depth=_depth + 1,
            )
            for item in value[:max_list_items]
        ]
        if len(value) > max_list_items:
            truncated.append(
                {
                    "__truncated__": True,
                    "original_len": len(value),
                    "kept": max_list_items,
                }
            )
        return truncated

    if isinstance(value, tuple):
        return truncate_structured_value(
            list(value),
            max_list_items=max_list_items,
            max_depth=max_depth,
            max_string_chars=max_string_chars,
            max_dict_keys=max_dict_keys,
            _depth=_depth,
        )

    if isinstance(value, str) and len(value) > max_string_chars:
        return value[:max_string_chars] + f"... <truncated {len(value) - max_string_chars} chars>"

    return _json_safe(value)


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple, str)):
        return bool(value)
    return True


def extract_structured_result(
    tool_resp: Dict[str, Any],
    max_list_items: int = 10,
    max_depth: int = 4,
) -> Dict[str, Any]:
    if not isinstance(tool_resp, dict) or tool_resp.get("error_code") != 0:
        return {}

    result: Dict[str, Any] = {}

    structured = tool_resp.get("structured")
    if isinstance(structured, dict):
        result.update(structured)
    elif _has_content(structured):
        result["structured"] = structured

    for key in KEY_FIELDS:
        if key == "structured":
            continue
        value = tool_resp.get(key)
        if _has_content(value):
            result[key] = value

    if not result:
        return {}

    return truncate_structured_value(
        result,
        max_list_items=max_list_items,
        max_depth=max_depth,
    )


def extract_successful_structured_result(
    tool_resp: Dict[str, Any],
    max_list_items: int = 10,
    max_depth: int = 4,
) -> Dict[str, Any]:
    return extract_structured_result(tool_resp, max_list_items=max_list_items, max_depth=max_depth)


def _to_json_text(value: Dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_structured_observation(
    tool_resp: Dict[str, Any],
    max_chars: int = 5000,
) -> str:
    structured = extract_structured_result(tool_resp)
    if not structured:
        return ""

    json_text = _to_json_text(structured)
    if len(json_text) > max_chars:
        structured = extract_structured_result(tool_resp, max_list_items=3, max_depth=3)
        structured["__summary_truncated__"] = True
        json_text = _to_json_text(structured)

    if len(json_text) > max_chars:
        structured = {
            "__summary_truncated__": True,
            "available_keys": sorted(list(structured.keys())),
        }
        json_text = _to_json_text(structured)

    return "STRUCTURED_RESULT:\n```json\n" + json_text + "\n```"


def get_structured_result_for_registry(tool_resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(tool_resp, dict) or tool_resp.get("error_code") != 0:
        return None
    structured = tool_resp.get("structured")
    result = dict(structured) if isinstance(structured, dict) else {}
    if not isinstance(structured, dict) and _has_content(structured):
        result["structured"] = structured
    for key in KEY_FIELDS:
        if key != "structured" and _has_content(tool_resp.get(key)):
            result[key] = tool_resp[key]
    return result or None
