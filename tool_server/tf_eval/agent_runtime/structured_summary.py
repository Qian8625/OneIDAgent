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


#: 体积大、对"下一步决策"几乎没有增量信息的字段：摘要时整体丢弃。
_HEAVY_KEYS = {
    "detections", "dedup_detections", "matched_pairs", "objects_used", "objects",
    "object_hypotheses", "observation_context", "claims", "primitive_observations",
    "vi_diagnostics", "evidence", "replay_diagnostics", "artifacts", "snapshot",
}

#: 决策必需的字段：优先保留，且允许单独占额度。
_PRIORITY_KEYS = (
    "summary", "source_counts", "count", "dedup_count", "matched_count", "conclusion_type",
    "final_bbox", "mode", "degraded", "degrade_reason", "family", "pair_id", "target",
    "attribute_claims", "attributes", "selected", "selected_detections", "measurement", "bbox",
)


def _compact_attribute_claims(claims: Any, max_objects: int = 4, max_claims: int = 8) -> Any:
    """只留 attribute/value，便于模型直接抄；否则模型会答"无法确定"。"""
    if not isinstance(claims, dict):
        return claims
    out = {}
    for index, (key, value) in enumerate(claims.items()):
        if index >= max_objects:
            break
        if isinstance(value, list):
            out[key] = [
                {"attribute": item.get("attribute"), "value": item.get("value")}
                for item in value[:max_claims] if isinstance(item, dict)
            ]
        else:
            out[key] = value
    return out


_OBJECT_KEYS = {"label", "bbox", "score", "attributes", "subtype", "color", "size_bin",
                "orientation", "observation_id", "index", "view"}


def _slim_object(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    return {k: v for k, v in obj.items() if k in _OBJECT_KEYS}


def compact_structured(structured: Dict[str, Any]) -> Dict[str, Any]:
    """按"决策相关性"压缩大 payload，而不是退化成只有键名的骨架。

    之前的兜底是 {"__summary_truncated__": True, "available_keys": [...]}，
    模型据此只能回答"无法确定"，等于把工具输出丢了。
    """
    compact: Dict[str, Any] = {}
    for key in _PRIORITY_KEYS:
        if key not in structured:
            continue
        value = structured[key]
        compact[key] = _compact_attribute_claims(value) if key == "attribute_claims" else value
    # 只留"目标是谁、多大、什么属性"，否则模型无法判断"largest"指向哪个对象。
    objects = structured.get("objects")
    if isinstance(objects, list) and objects:
        compact["objects"] = [_slim_object(item) for item in objects[:5]]
    for key, value in structured.items():
        if key in compact or key in _HEAVY_KEYS or key.startswith("__"):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
    return compact


def build_structured_observation(
    tool_resp: Dict[str, Any],
    max_chars: int = 5000,
) -> str:
    structured = extract_structured_result(tool_resp)
    if not structured:
        return ""

    json_text = _to_json_text(structured)
    if len(json_text) > max_chars:
        # 先按"决策相关性"压缩，再做深度截断。顺序不能反：先截断会把 attribute_claims
        # 压成 {"__truncated__": ...} 骨架，模型就只能回答"无法确定"。
        structured = compact_structured(structured)
        structured["__summary_truncated__"] = True
        json_text = _to_json_text(structured)
    if len(json_text) > max_chars:
        structured = truncate_structured_value(structured, max_list_items=3, max_depth=3)
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
