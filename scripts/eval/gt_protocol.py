"""冻结的 GT 构造协议（RGB/SAR 匹配评测用）。

为什么要有这个文件：如果 GT 构造直接复用解释器的打分函数，评测口径就会随
算法一起漂移，历史数字也就无法比较。本模块把 GT 需要的
全部要素——标签别名、几何原语、打分权重、匹配贪心——冻结成显式常量与纯函数，
**不 import 任何 interpreter / worker / DS_VI 代码**。

改动本文件的任何常量都等于换评测口径，必须递增 ``GT_PROTOCOL_VERSION``。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: GT 口径版本号。写进 metrics.json 与报告首行；改动本文件任何常量都必须递增它。
GT_PROTOCOL_VERSION = "1.0"

#: 冻结的标签别名（与检测侧保持一致，但此处是独立副本）。
TARGET_ALIASES: Dict[str, str] = {
    "vehicles": "vehicle", "vehicle": "vehicle", "car": "vehicle", "cars": "vehicle",
    "small vehicle": "vehicle", "large vehicle": "vehicle",
    "tank": "tank", "storage tank": "tank", "oil tank": "tank", "oil storage tank": "tank",
    "ship": "ship", "ships": "ship", "boat": "ship", "boats": "ship",
    "aircraft": "aircraft", "airplane": "aircraft", "airplanes": "aircraft",
    "plane": "aircraft", "planes": "aircraft", "helicopter": "aircraft",
    "helicopters": "aircraft",
    "bridge": "bridge", "harbor": "harbor",
}

#: 冻结的匹配权重（与 legacy_hard 一致，逐位可复现）。
W_IOU = 0.50
W_CENTER = 0.30
W_AREA = 0.10
W_CONF = 0.10

#: 冻结的默认阈值。
MATCH_THRESHOLD = 0.45
DEFAULT_DETECTION_CONFIDENCE = 0.5

#: 类别集合（GT 只在这 6 类上构造）。
FINAL_TARGETS = ("aircraft", "bridge", "harbor", "ship", "tank", "vehicle")


def normalize_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def canonical_target(value: Any) -> str:
    label = normalize_label(value)
    return TARGET_ALIASES.get(label, label)


def bbox_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def bbox_center(box: Sequence[float]) -> Tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def bbox_diag(box: Sequence[float]) -> float:
    return max(1.0, math.hypot(max(0.0, float(box[2]) - float(box[0])),
                               max(0.0, float(box[3]) - float(box[1]))))


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ac, bc = bbox_center(a), bbox_center(b)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


def _center_score(distance_px: float, scale_px: float) -> float:
    scale = max(float(scale_px), 1.0)
    return max(0.0, min(1.0, 1.0 - float(distance_px) / scale))


def pair_score(left_bbox: Sequence[float], right_bbox: Sequence[float],
               left_score: float = DEFAULT_DETECTION_CONFIDENCE,
               right_score: float = DEFAULT_DETECTION_CONFIDENCE) -> Dict[str, float]:
    """几何 + 置信度的混合分（冻结公式）。"""
    iou = bbox_iou(left_bbox, right_bbox)
    distance = center_distance(left_bbox, right_bbox)
    scale = max(bbox_diag(left_bbox), bbox_diag(right_bbox), 1.0)
    center = _center_score(distance, scale)
    area_left, area_right = bbox_area(left_bbox), bbox_area(right_bbox)
    area_score = 0.0 if area_left <= 0 or area_right <= 0 else min(area_left, area_right) / max(area_left, area_right)
    confidence = (float(left_score) + float(right_score)) / 2.0
    score = W_IOU * iou + W_CENTER * center + W_AREA * area_score + W_CONF * confidence
    return {
        "match_score": round(score, 4),
        "iou": round(iou, 4),
        "center_distance": round(distance, 3),
        "center_score": round(center, 4),
        "area_score": round(area_score, 4),
        "confidence_score": round(confidence, 4),
    }


def match_items(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    mode: str = "hybrid",
    iou_threshold: float = 0.2,
    hybrid_threshold: float = MATCH_THRESHOLD,
    max_center_distance: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """GT 用的一对一贪心匹配（冻结算法）。

    ``mode='iou'`` 复现旧的口径；``mode='hybrid'`` 用 :func:`pair_score`。
    """
    candidates: List[Tuple[float, float, float, int, int, Dict[str, Any]]] = []
    for i, l_item in enumerate(left):
        for j, r_item in enumerate(right):
            iou = bbox_iou(l_item["bbox"], r_item["bbox"])
            dist = center_distance(l_item["bbox"], r_item["bbox"])
            if max_center_distance is not None and dist > float(max_center_distance):
                continue
            if mode == "iou":
                if iou < iou_threshold:
                    continue
                info: Dict[str, Any] = {
                    "iou": round(iou, 4),
                    "match_score": round(iou, 4),
                    "center_distance": round(dist, 3),
                    "match_type": "gt_iou_match",
                }
                primary, secondary = iou, iou
            elif mode == "hybrid":
                info = pair_score(
                    l_item["bbox"], r_item["bbox"],
                    l_item.get("score", DEFAULT_DETECTION_CONFIDENCE),
                    r_item.get("score", DEFAULT_DETECTION_CONFIDENCE),
                )
                if float(info.get("match_score", 0.0)) < float(hybrid_threshold):
                    continue
                primary = float(info["match_score"])
                secondary = float(info["iou"])
            else:
                raise ValueError(f"Unsupported GT match mode: {mode}")
            candidates.append((primary, secondary, -dist, i, j, info))

    candidates.sort(reverse=True)
    used_left, used_right = set(), set()
    matched: List[Dict[str, Any]] = []
    for _, _, _, i, j, info in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matched.append({
            "left_index": i, "right_index": j,
            "left": left[i], "right": right[j], **info,
        })
    left_only = [item for i, item in enumerate(left) if i not in used_left]
    right_only = [item for j, item in enumerate(right) if j not in used_right]
    return matched, left_only, right_only


__all__ = [
    "DEFAULT_DETECTION_CONFIDENCE",
    "FINAL_TARGETS",
    "GT_PROTOCOL_VERSION",
    "MATCH_THRESHOLD",
    "TARGET_ALIASES",
    "bbox_area",
    "bbox_center",
    "bbox_diag",
    "bbox_iou",
    "canonical_target",
    "center_distance",
    "match_items",
    "normalize_label",
    "pair_score",
]
