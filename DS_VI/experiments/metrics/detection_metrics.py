"""Detection AP/F1, RGB/SAR ground-truth pairing and reference inference.

Only helpers used by the current six-class experiments are retained.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


from DS_VI.models.cross_modal import (
    CrossModalInterpreter,
)

DEFAULT_CLASS_MAP = {
    "0": "aircraft",
    "1": "bridge",
    "2": "harbor",
    "3": "ship",
    "4": "tank",
    "5": "vehicle",
}


def normalize_detection_label(value: Any) -> str:
    label = CrossModalInterpreter.normalize_label(value)
    return CrossModalInterpreter.TARGET_ALIASES.get(label, label)

def split_pair_name(path: Path) -> Tuple[str, str]:
    stem = path.stem
    upper = stem.upper()
    if upper.endswith("_RGB"):
        return stem[:-4], "rgb"
    if upper.endswith("_SAR"):
        return stem[:-4], "sar"
    raise ValueError(f"Cannot infer modality from filename: {path.name}")

def bbox_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

def bbox_center(box: Sequence[float]) -> Tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)

def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union

def fuse_bbox(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [(float(x) + float(y)) / 2.0 for x, y in zip(a[:4], b[:4], strict=False)]

def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ac = bbox_center(a)
    bc = bbox_center(b)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])

def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    image_width: float,
    image_height: float,
) -> List[float]:
    x1 = (cx - w / 2.0) * image_width
    y1 = (cy - h / 2.0) * image_height
    x2 = (cx + w / 2.0) * image_width
    y2 = (cy + h / 2.0) * image_height
    return [x1, y1, x2, y2]

def safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den

def metric_key(value: float) -> str:
    """Return a stable integer-percent key such as 50 for IoU=0.5.

    round() avoids confusing keys from binary floating-point artifacts and from
    non-standard values such as 0.29 or 0.3.
    """
    return str(int(round(float(value) * 100)))

def load_truth_dir(
    truth_dir: Path,
    image_width: float,
    image_height: float,
    class_map: Dict[str, str],
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"rgb": [], "sar": []})
    for path in sorted(truth_dir.glob("*.txt")):
        base_id, modality = split_pair_name(path)
        objects: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"Malformed YOLO line in {path}:{line_no}: {raw!r}")

                class_id = parts[0]
                if class_id not in class_map:
                    raise ValueError(f"Unknown class id {class_id!r} in {path}:{line_no}")

                cx, cy, w, h = [float(x) for x in parts[1:5]]
                bbox = yolo_to_xyxy(cx, cy, w, h, image_width, image_height)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue

                label = normalize_detection_label(class_map[class_id])
                objects.append(
                    {
                        "image_id": f"{base_id}_{modality}",
                        "base_id": base_id,
                        "modality": modality,
                        "class_id": class_id,
                        "label": label,
                        "bbox": bbox,
                        "area": bbox_area(bbox),
                    }
                )
        data[base_id][modality] = objects
    return dict(data)

def hybrid_match_info(
    left_item: Dict[str, Any],
    right_item: Dict[str, Any],
    engine: CrossModalInterpreter,
) -> Dict[str, Any]:
    left_det = {
        "bbox": left_item["bbox"],
        "score": left_item.get("score", CrossModalInterpreter.DEFAULT_DETECTION_CONFIDENCE),
    }
    right_det = {
        "bbox": right_item["bbox"],
        "score": right_item.get("score", CrossModalInterpreter.DEFAULT_DETECTION_CONFIDENCE),
    }
    return engine.compute_pair_score(left_det, right_det)

def match_rgb_sar_items(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
    mode: str,
    iou_threshold: float,
    hybrid_threshold: float,
    max_center_distance: Optional[float],
    engine: CrossModalInterpreter,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One-to-one RGB/SAR matching for GT object construction.

    mode='iou' preserves the old protocol.
    mode='hybrid' aligns GT construction with CrossModalInterpreter matching.
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
                info = {
                    "iou": round(iou, 4),
                    "match_score": round(iou, 4),
                    "center_distance": round(dist, 3),
                    "match_type": "gt_iou_match",
                }
                primary = iou
                secondary = iou
            elif mode == "hybrid":
                info = hybrid_match_info(l_item, r_item, engine)
                if float(info.get("match_score", 0.0)) < hybrid_threshold:
                    continue
                primary = float(info.get("match_score", 0.0))
                secondary = float(info.get("iou", 0.0))
            else:
                raise ValueError(f"Unsupported GT match mode: {mode}")

            # Sort by best score, then IoU, then shorter center distance.
            candidates.append((primary, secondary, -dist, i, j, info))

    candidates.sort(reverse=True)
    used_left, used_right = set(), set()
    matched: List[Dict[str, Any]] = []

    for _, _, _, i, j, info in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matched.append(
            {
                "left_index": i,
                "right_index": j,
                "left": left[i],
                "right": right[j],
                **info,
            }
        )

    left_only = [item for i, item in enumerate(left) if i not in used_left]
    right_only = [item for j, item in enumerate(right) if j not in used_right]
    return matched, left_only, right_only

def conclusion_from_counts(rgb_count: int, sar_count: int, matched_count: int) -> str:
    if matched_count > 0:
        return "confirmed"
    if rgb_count > 0 and sar_count > 0:
        return "spatial_conflict"
    if rgb_count > 0:
        return "rgb_possible"
    if sar_count > 0:
        return "sar_possible"
    return "not_found"

def gt_for_target(
    pair: Dict[str, Any],
    target: str,
    gt_match_mode: str,
    gt_match_iou: float,
    gt_hybrid_threshold: float,
    max_center_distance: Optional[float],
    engine: CrossModalInterpreter,
) -> Dict[str, Any]:
    rgb = [x for x in pair["rgb_gt"] if normalize_detection_label(x.get("label")) == target]
    sar = [x for x in pair["sar_gt"] if normalize_detection_label(x.get("label")) == target]

    matched, rgb_only, sar_only = match_rgb_sar_items(
        rgb,
        sar,
        mode=gt_match_mode,
        iou_threshold=gt_match_iou,
        hybrid_threshold=gt_hybrid_threshold,
        max_center_distance=max_center_distance,
        engine=engine,
    )

    dedup: List[Dict[str, Any]] = []
    for idx, item in enumerate(matched):
        bbox = fuse_bbox(item["left"]["bbox"], item["right"]["bbox"])
        dedup.append(
            {
                "name": f"{target}_gt_matched_{idx}",
                "source": "matched",
                "label": target,
                "bbox": bbox,
                "area": bbox_area(bbox),
                "iou": item.get("iou"),
                "match_score": item.get("match_score"),
                "match_type": item.get("match_type", f"gt_{gt_match_mode}_match"),
                "rgb_bbox": item["left"]["bbox"],
                "sar_bbox": item["right"]["bbox"],
            }
        )

    for idx, item in enumerate(rgb_only):
        dedup.append(
            {
                "name": f"{target}_gt_rgb_only_{idx}",
                "source": "rgb_only",
                "label": target,
                "bbox": item["bbox"],
                "area": item["area"],
                "rgb_bbox": item["bbox"],
                "sar_bbox": None,
            }
        )

    for idx, item in enumerate(sar_only):
        dedup.append(
            {
                "name": f"{target}_gt_sar_only_{idx}",
                "source": "sar_only",
                "label": target,
                "bbox": item["bbox"],
                "area": item["area"],
                "rgb_bbox": None,
                "sar_bbox": item["bbox"],
            }
        )

    largest = sorted(dedup, key=lambda x: (float(x["area"]), str(x["name"])), reverse=True)[0] if dedup else None
    smallest = sorted(dedup, key=lambda x: (float(x["area"]), str(x["name"]))) [0] if dedup else None
    conclusion = conclusion_from_counts(len(rgb), len(sar), len(matched))

    return {
        "rgb_count": len(rgb),
        "sar_count": len(sar),
        "matched_count": len(matched),
        "dedup_count": len(dedup),
        "dedup_detections": dedup,
        "largest": largest,
        "smallest": smallest,
        "conclusion_type": conclusion,
        "confirmed": conclusion == "confirmed",
    }

def run_interpreter(
    engine: CrossModalInterpreter,
    pair: Dict[str, Any],
    target: str,
    task: str,
    match_threshold: float,
    pixel_size_m: float,
    category_match_mode: str,
    max_center_distance: Optional[float],
) -> Dict[str, Any]:
    payload = {
        "target": target,
        "task": task,
        "rgb_detections": pair["rgb_detections"],
        "sar_detections": pair["sar_detections"],
        "output_mode": "full",
        "match_threshold": match_threshold,
        "pixel_size_m": pixel_size_m,
        "category_match_mode": category_match_mode,
    }
    if max_center_distance is not None:
        payload["max_center_distance"] = max_center_distance
    return engine.generate(payload)

def compute_ap_for_class(
    gt_by_image: Dict[str, List[Dict[str, Any]]],
    pred_by_image: Dict[str, List[Dict[str, Any]]],
    cls: str,
    iou_threshold: float,
) -> Dict[str, Any]:
    gt_state: Dict[str, Dict[str, Any]] = {}
    n_gt = 0
    image_ids = sorted(set(gt_by_image) | set(pred_by_image))

    for image_id in image_ids:
        gts = gt_by_image.get(image_id, [])
        boxes = [g["bbox"] for g in gts if normalize_detection_label(g.get("label")) == cls]
        gt_state[image_id] = {"boxes": boxes, "matched": [False] * len(boxes)}
        n_gt += len(boxes)

    preds = []
    for image_id in image_ids:
        for p in pred_by_image.get(image_id, []):
            if normalize_detection_label(p.get("label")) == cls:
                preds.append({**p, "image_id": image_id})
    preds.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    tp, fp = [], []
    for pred in preds:
        state = gt_state.get(pred["image_id"], {"boxes": [], "matched": []})
        best_idx, best_score = -1, 0.0
        for idx, gt_box in enumerate(state["boxes"]):
            if state["matched"][idx]:
                continue
            iou = bbox_iou(pred["bbox"], gt_box)
            if iou > best_score:
                best_score = iou
                best_idx = idx

        if best_idx >= 0 and best_score >= iou_threshold:
            state["matched"][best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)

    if n_gt == 0:
        return {"ap": 0.0, "recall": 0.0, "tp": 0, "fp": sum(fp), "num_gt": 0, "num_pred": len(preds)}

    cum_tp, cum_fp = [], []
    total_tp, total_fp = 0, 0
    for t, f in zip(tp, fp, strict=False):
        total_tp += t
        total_fp += f
        cum_tp.append(total_tp)
        cum_fp.append(total_fp)

    recalls = [x / n_gt for x in cum_tp]
    precisions = [cum_tp[i] / max(cum_tp[i] + cum_fp[i], 1) for i in range(len(cum_tp))]

    # 101-point interpolated AP.
    ap = 0.0
    for r in [x / 100 for x in range(101)]:
        p_at_r = 0.0
        for recall, precision in zip(recalls, precisions, strict=False):
            if recall >= r:
                p_at_r = max(p_at_r, precision)
        ap += p_at_r / 101.0

    return {
        "ap": ap,
        "recall": recalls[-1] if recalls else 0.0,
        "tp": sum(tp),
        "fp": sum(fp),
        "num_gt": n_gt,
        "num_pred": len(preds),
    }

def compute_detection_metrics(
    gt_by_image: Dict[str, List[Dict[str, Any]]],
    pred_by_image: Dict[str, List[Dict[str, Any]]],
    targets: Sequence[str],
    iou_threshold: float,
) -> Dict[str, Any]:
    """Compute object-level detection metrics with matched aggregation protocols.

    ``mAP`` and ``Recall``/``Precision``/``F1`` do not share a weighting scheme
    by default: mAP is an unweighted macro average over classes, while
    Recall/Precision/F1 are micro (instance-pooled) aggregates.  On a class
    distribution like this evaluation set a macro mAP and a micro Recall can
    move in opposite directions for the same change, so a delta that compares
    them is not interpretable.

    Both families are therefore reported explicitly:

    * ``mAP_macro`` / ``Recall_macro`` / ``Precision_macro`` / ``F1_macro`` --
      unweighted mean over the classes that have ground truth.
    * ``mAP_gt_weighted`` -- AP averaged with per-class GT counts as weights,
      the AP counterpart of the micro Recall.
    * ``mAP`` / ``Recall`` / ``Precision`` / ``F1`` -- the historical keys.
      ``mAP`` stays macro (unchanged) and the rates stay micro, so existing
      reports remain comparable.

    ``mAP_gt_weighted`` and ``Recall`` are the pair to compare when asking
    whether a change helped overall; ``mAP_macro`` and ``Recall_macro`` are
    the pair that answers the class-average question.
    """
    per_class = {}
    ap_values = []
    ap_gt_weights = []
    macro_recalls = []
    macro_precisions = []
    totals = defaultdict(int)

    for target in targets:
        result = compute_ap_for_class(gt_by_image, pred_by_image, target, iou_threshold)
        per_class[target] = result
        if result["num_gt"] > 0:
            ap_values.append(result["ap"])
            ap_gt_weights.append(int(result["num_gt"]))
            macro_recalls.append(result["recall"])
            macro_precisions.append(safe_div(result["tp"], result["tp"] + result["fp"]))
        for key in ["tp", "fp", "num_gt", "num_pred"]:
            totals[key] += int(result[key])

    precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_div(totals["tp"], totals["num_gt"])
    f1 = safe_div(2 * precision * recall, precision + recall)
    macro_ap = safe_div(sum(ap_values), len(ap_values))
    gt_weighted_ap = safe_div(
        sum(ap * weight for ap, weight in zip(ap_values, ap_gt_weights, strict=False)),
        sum(ap_gt_weights),
    )
    macro_recall = safe_div(sum(macro_recalls), len(macro_recalls))
    macro_precision = safe_div(sum(macro_precisions), len(macro_precisions))
    macro_f1 = safe_div(2 * macro_precision * macro_recall, macro_precision + macro_recall)
    key = metric_key(iou_threshold)

    return {
        f"mAP@{key}": macro_ap,
        f"mAP_macro@{key}": macro_ap,
        f"mAP_gt_weighted@{key}": gt_weighted_ap,
        f"Recall@{key}": recall,
        f"Precision@{key}": precision,
        f"F1@{key}": f1,
        f"Recall_macro@{key}": macro_recall,
        f"Precision_macro@{key}": macro_precision,
        f"F1_macro@{key}": macro_f1,
        f"TP@{key}": totals["tp"],
        f"FP@{key}": totals["fp"],
        f"FN@{key}": max(0, totals["num_gt"] - totals["tp"]),
        f"GT@{key}": totals["num_gt"],
        f"Pred@{key}": totals["num_pred"],
        f"FP_per_image@{key}": safe_div(totals["fp"], max(len(set(gt_by_image) | set(pred_by_image)), 1)),
        f"Pred_GT_ratio@{key}": safe_div(totals["num_pred"], totals["num_gt"]),
        "per_class": per_class,
    }

def add_per_class_rates(per_class: Dict[str, Dict[str, Any]], iou_key: str = "50") -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for target, metrics in per_class.items():
        tp = int(metrics.get("tp", 0))
        fp = int(metrics.get("fp", 0))
        gt = int(metrics.get("num_gt", 0))
        pred = int(metrics.get("num_pred", 0))
        recall = float(metrics.get("recall", 0.0))
        precision = safe_div(tp, tp + fp)
        out[target] = {
            f"ap@{iou_key}": float(metrics.get("ap", 0.0)),
            f"recall@{iou_key}": recall,
            f"precision@{iou_key}": precision,
            f"f1@{iou_key}": safe_div(2 * precision * recall, precision + recall),
            f"tp@{iou_key}": tp,
            f"fp@{iou_key}": fp,
            f"fn@{iou_key}": max(0, gt - tp),
            f"gt@{iou_key}": gt,
            f"pred@{iou_key}": pred,
            f"pred_gt_ratio@{iou_key}": safe_div(pred, gt),
        }
    return out
