from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")

BBOX_KEYS = ("bbox", "box", "rgb_bbox", "sar_bbox", "bounding_box")

POSITIVE_PHRASES = (
    "not inconsistent",
    "yes",
    "true",
    "consistent",
    "supported",
    "credible",
    "present",
    "exists",
    "matched",
    "match",
    "same",
    "correct",
    "positive",
    "一致",
    "支持",
    "可信",
    "存在",
    "匹配",
    "相同",
    "正确",
)
NEGATIVE_PHRASES = (
    "not supported",
    "not credible",
    "not consistent",
    "no match",
    "no",
    "false",
    "inconsistent",
    "unsupported",
    "absent",
    "missing",
    "unmatched",
    "different",
    "incorrect",
    "negative",
    "不一致",
    "不支持",
    "不可信",
    "不存在",
    "未匹配",
    "不同",
    "错误",
)


def normalize_answer(text: Any) -> str:
    text = "" if text is None else str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff.%-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_stringify(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(item) for item in value)
    return str(value)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).replace(",", ""))
    except Exception:
        return None


def extract_numbers(value: Any) -> List[float]:
    numbers: List[float] = []
    if value is None:
        return numbers
    if isinstance(value, bool):
        return numbers
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return [float(value)]
        return numbers
    if isinstance(value, dict):
        for item in value.values():
            numbers.extend(extract_numbers(item))
        return numbers
    if isinstance(value, (list, tuple)):
        for item in value:
            numbers.extend(extract_numbers(item))
        return numbers
    if isinstance(value, str):
        for match in NUMBER_RE.finditer(value):
            number = _to_float(match.group(0))
            if number is not None and math.isfinite(number):
                numbers.append(number)
    return numbers


def _unit_factor(unit: str) -> Optional[float]:
    normalized = unit.lower().replace("²", "2").replace("^", "")
    normalized = normalized.strip(" .")
    length = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "metre": 1.0,
        "metres": 1.0,
        "km": 1000.0,
        "kilometer": 1000.0,
        "kilometers": 1000.0,
        "cm": 0.01,
    }
    area = {
        "m2": 1.0,
        "sqm": 1.0,
        "sq m": 1.0,
        "square meter": 1.0,
        "square meters": 1.0,
        "km2": 1_000_000.0,
        "sq km": 1_000_000.0,
        "square kilometer": 1_000_000.0,
        "square kilometers": 1_000_000.0,
        "ha": 10_000.0,
        "hectare": 10_000.0,
        "hectares": 10_000.0,
    }
    return area.get(normalized) or length.get(normalized)


def _extract_quantities(text: Any) -> List[float]:
    if not isinstance(text, str):
        return extract_numbers(text)
    quantities: List[float] = []
    pattern = re.compile(
        r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
        r"(km\^?2|km²|m\^?2|m²|sq\.?\s*m|sq\.?\s*km|square\s+meters?|square\s+kilometers?|hectares?|ha|kilometers?|km|meters?|metres?|m|cm)\b",
        re.IGNORECASE,
    )
    for number_text, unit in pattern.findall(text):
        number = _to_float(number_text)
        factor = _unit_factor(unit)
        if number is not None and factor is not None:
            quantities.append(number * factor)
    return quantities or extract_numbers(text)


def _relative_close(gold: float, pred: float, tolerance: float) -> bool:
    if gold == 0:
        return abs(pred - gold) <= tolerance
    return abs(pred - gold) / abs(gold) <= tolerance


def score_numeric(gold: Any, pred: Any, tolerance: float = 0.1) -> Optional[float]:
    gold_numbers = _extract_quantities(gold)
    pred_numbers = _extract_quantities(pred)
    if not gold_numbers:
        return None
    if not pred_numbers:
        return 0.0

    matched = 0
    near = 0
    for gold_number in gold_numbers:
        best_error = min(
            (abs(pred_number - gold_number) / max(abs(gold_number), 1e-9))
            if gold_number != 0 else abs(pred_number - gold_number)
            for pred_number in pred_numbers
        )
        if best_error <= tolerance:
            matched += 1
        elif best_error <= tolerance * 2:
            near += 1
    return min(1.0, (matched + 0.5 * near) / len(gold_numbers))


def _coerce_bbox_list(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        coords = [_to_float(item) for item in value[:4]]
        if all(item is not None for item in coords):
            return [float(item) for item in coords if item is not None]
    return None


def extract_bbox(value: Any) -> Optional[List[float]]:
    direct = _coerce_bbox_list(value)
    if direct is not None:
        return direct
    if isinstance(value, dict):
        for key in BBOX_KEYS:
            if key in value:
                bbox = extract_bbox(value.get(key))
                if bbox is not None:
                    return bbox
        for item in value.values():
            bbox = extract_bbox(item)
            if bbox is not None:
                return bbox
    if isinstance(value, (list, tuple)):
        for item in value:
            bbox = extract_bbox(item)
            if bbox is not None:
                return bbox
    if isinstance(value, str):
        numbers = extract_numbers(value)
        if len(numbers) >= 4:
            return [float(item) for item in numbers[:4]]
    return None


def bbox_iou(box1: Any, box2: Any) -> float:
    b1 = _coerce_bbox_list(box1)
    b2 = _coerce_bbox_list(box2)
    if b1 is None or b2 is None:
        return 0.0
    x1 = max(min(b1[0], b1[2]), min(b2[0], b2[2]))
    y1 = max(min(b1[1], b1[3]), min(b2[1], b2[3]))
    x2 = min(max(b1[0], b1[2]), max(b2[0], b2[2]))
    y2 = min(max(b1[1], b1[3]), max(b2[1], b2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = abs((b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = abs((b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def score_bbox(gold: Any, pred: Any, iou_threshold: float = 0.5) -> Optional[float]:
    gold_bbox = extract_bbox(gold)
    pred_bbox = extract_bbox(pred)
    if gold_bbox is None:
        return None
    if pred_bbox is None:
        return 0.0
    iou = bbox_iou(gold_bbox, pred_bbox)
    if iou >= iou_threshold:
        return 1.0
    return round(max(0.0, iou / max(iou_threshold, 1e-9)) * 0.8, 6)


def _extract_integers(value: Any) -> List[int]:
    return [int(round(number)) for number in extract_numbers(value) if abs(number - round(number)) < 1e-6]


def score_count(gold: Any, pred: Any, tolerance: int = 0) -> Optional[float]:
    gold_counts = _extract_integers(gold)
    pred_counts = _extract_integers(pred)
    if not gold_counts:
        return None
    if not pred_counts:
        return 0.0
    gold_count = gold_counts[0]
    best_delta = min(abs(pred_count - gold_count) for pred_count in pred_counts)
    if best_delta == 0:
        return 1.0
    if best_delta <= int(tolerance):
        return 1.0
    if tolerance == 0 and best_delta == 1:
        return 0.5
    return 0.0


def _binary_label(value: Any) -> Optional[bool]:
    text = normalize_answer(_stringify(value))
    if not text:
        return None
    for phrase in NEGATIVE_PHRASES:
        if re.search(rf"(^|\s){re.escape(phrase)}($|\s)", text) or phrase in text:
            return False
    for phrase in POSITIVE_PHRASES:
        if re.search(rf"(^|\s){re.escape(phrase)}($|\s)", text) or phrase in text:
            return True
    return None


def score_binary_judgement(gold: Any, pred: Any) -> Optional[float]:
    gold_label = _binary_label(gold)
    pred_label = _binary_label(pred)
    if gold_label is None:
        return None
    if pred_label is None:
        return 0.0
    return 1.0 if gold_label == pred_label else 0.0


def _trace_has_artifact(trace: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(trace, dict):
        return False
    for item in trace.get("tool_responses", []):
        response = item.get("response") if isinstance(item, dict) else {}
        if not isinstance(response, dict):
            continue
        if response.get("download_path") or response.get("primary_image") or response.get("image"):
            return True
        if isinstance(response.get("artifacts"), list) and response.get("artifacts"):
            return True
    for call in trace.get("skill_calls", []):
        response = call.get("response") if isinstance(call, dict) else {}
        if isinstance(response, dict) and isinstance(response.get("artifacts"), list) and response.get("artifacts"):
            return True
    return False


def fallback_string_score(gold: Any, pred: Any) -> Optional[float]:
    gold_norm = normalize_answer(gold)
    pred_norm = normalize_answer(pred)
    if not gold_norm:
        return None
    if pred_norm == gold_norm:
        return 1.0
    if pred_norm and (pred_norm in gold_norm or gold_norm in pred_norm):
        return 0.5
    return 0.0


def _requires_visual(eval_meta: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(eval_meta, dict):
        return False
    task_family = str(eval_meta.get("task_family", ""))
    return bool(eval_meta.get("requires_visual_output")) or "visual" in task_family


def score_visual(gold: Any, pred: Any, trace: Optional[Dict[str, Any]], eval_meta: Optional[Dict[str, Any]]) -> Optional[float]:
    if not _requires_visual(eval_meta):
        return None
    has_artifact = _trace_has_artifact(trace)
    text_score = fallback_string_score(gold, pred)
    if not has_artifact:
        return 0.0
    if text_score is None:
        return 1.0
    return max(0.8, text_score)


def _infer_metric_type(gold: Any, pred: Any, eval_meta: Optional[Dict[str, Any]]) -> str:
    if isinstance(eval_meta, dict):
        metric_type = eval_meta.get("metric_type")
        if metric_type:
            return str(metric_type).lower()
        task_family = str(eval_meta.get("task_family", "")).lower()
        if eval_meta.get("requires_visual_output") or "visual" in task_family:
            return "visual"
        if "count" in task_family:
            return "count"
        if any(token in task_family for token in ["measure", "distance", "area", "gsd"]):
            return "numeric"
        if any(token in task_family for token in ["match", "consistency", "cross_modal_task"]):
            return "binary"
    if extract_bbox(gold) is not None and extract_bbox(pred) is not None:
        return "bbox"
    if _binary_label(gold) is not None:
        return "binary"
    gold_ints = _extract_integers(gold)
    pred_ints = _extract_integers(pred)
    if len(gold_ints) == 1 and len(pred_ints) >= 1:
        return "count"
    if extract_numbers(gold) and extract_numbers(pred):
        return "numeric"
    return "string"


def answer_score(
    gold: Any,
    pred: Any,
    eval_meta: Optional[Dict[str, Any]] = None,
    trace: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    try:
        if gold is None or normalize_answer(gold) == "":
            return None
        eval_meta = eval_meta if isinstance(eval_meta, dict) else {}
        metric_type = _infer_metric_type(gold, pred, eval_meta)
        numeric_tolerance = float(eval_meta.get("numeric_tolerance", 0.1))
        iou_threshold = float(eval_meta.get("iou_threshold", 0.5))
        count_tolerance = int(eval_meta.get("count_tolerance", 0))

        if metric_type in {"visual", "visual_output"}:
            return score_visual(gold, pred, trace, eval_meta)
        if metric_type in {"bbox", "box", "detection"}:
            return score_bbox(gold, pred, iou_threshold=iou_threshold)
        if metric_type in {"count", "counting"}:
            return score_count(gold, pred, tolerance=count_tolerance)
        if metric_type in {"binary", "yes_no", "yes/no", "consistency", "judgement", "judgment"}:
            return score_binary_judgement(gold, pred)
        if metric_type in {"numeric", "number", "distance", "area", "gsd", "measure", "measurement"}:
            return score_numeric(gold, pred, tolerance=numeric_tolerance)

        for scorer in (
            lambda: score_bbox(gold, pred, iou_threshold=iou_threshold),
            lambda: score_binary_judgement(gold, pred),
            lambda: score_numeric(gold, pred, tolerance=numeric_tolerance),
            lambda: fallback_string_score(gold, pred),
        ):
            score = scorer()
            if score is not None:
                return score
        return 0.0
    except Exception:
        return fallback_string_score(gold, pred) or 0.0
