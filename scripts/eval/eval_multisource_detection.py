#!/usr/bin/env python3
"""Evaluate RGB/SAR cross-modal interpretation on paired detection outputs.

This is a corrected evaluator for proving whether the cross-modal
interpretation module is useful when real tool detections and detection-box
ground truth are available.

Main fixes compared with the original version:
  1. GT object construction defaults to an independent IoU protocol, with an
     optional hybrid protocol for sensitivity analysis.
  2. crosscheck_exists uses count-based conclusion semantics for evaluation,
     so spatial_conflict is not overwritten by a selected single-modality box.
  3. rgb_only, sar_only, cross_modal, and cross_modal_confirmed_only are compared
     against the same unified object-level GT, with modality-aware GT boxes.
  4. Missing RGB/SAR tool or truth files are strict by default.
  5. Distance tasks support correctness thresholds in addition to MAE.
  6. Class labels from the class map are canonicalized.
  7. Non-square images are supported through --image-width/--image-height.
  8. Task and target names are validated before evaluation.
  9. Largest-selection metrics use dynamic threshold names.
 10. Delta metrics are emitted only when both compared modes are present.
 11. Legacy dedup TP/FP/FN/GT/Pred aliases are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tool_server.tool_workers.online_workers.CrossModalInterpret_worker import (  # noqa: E402
    CrossModalInterpreter,
)
from scripts.eval import gt_protocol  # noqa: E402


DEFAULT_CLASS_MAP = {
    "0": "aircraft",
    "1": "bridge",
    "2": "harbor",
    "3": "ship",
    "4": "tank",
    "5": "vehicle",
}
DEFAULT_TARGETS = ["aircraft", "bridge", "harbor", "ship", "tank", "vehicle"]
DEFAULT_TASKS = [
    "crosscheck_exists",
    "joint_scene",
    "select_largest",
    "distance_largest_smallest",
]
# Tasks for which this evaluator has explicit metric branches.
# CrossModalInterpreter also supports select_highest_score, but there is no
# independent GT highest-score target in YOLO truth files, so accepting it here
# would create unevaluated samples with correct=True by default.
EVALUATED_TASKS = set(DEFAULT_TASKS)
DEFAULT_MODALITIES = ["rgb_only", "sar_only", "cross_modal", "cross_modal_confirmed_only"]


def normalize_detection_label(value: Any) -> str:
    """GT 侧标签归一化：走冻结协议，不再依赖 interpreter 实现（F9）。"""
    return gt_protocol.canonical_target(value)


def parse_csv(value: str, default: Sequence[str]) -> List[str]:
    if not value:
        return list(default)
    return [x.strip() for x in value.split(",") if x.strip()]


def unique_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def parse_class_map(value: Optional[str]) -> Dict[str, str]:
    if not value:
        return {k: normalize_detection_label(v) for k, v in DEFAULT_CLASS_MAP.items()}
    out: Dict[str, str] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Invalid class map entry: {item!r}. Expected id:label")
        key, label = item.split(":", 1)
        out[key.strip()] = normalize_detection_label(label.strip())
    return out


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


def bbox_diag(box: Sequence[float]) -> float:
    return max(1.0, math.hypot(max(0.0, float(box[2]) - float(box[0])), max(0.0, float(box[3]) - float(box[1]))))


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def fuse_bbox(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [(float(x) + float(y)) / 2.0 for x, y in zip(a[:4], b[:4])]


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


def load_tool_dir(tool_dir: Path) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for path in sorted(tool_dir.glob("*.json")):
        base_id, file_modality = split_pair_name(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        modality = str(obj.get("modality") or file_modality).strip().lower()
        if modality not in {"rgb", "sar"}:
            modality = file_modality

        base_id = str(obj.get("stem") or base_id)
        detections = obj.get("detections") or []
        if not isinstance(detections, list):
            detections = []

        data[base_id][modality] = {
            "path": str(path),
            "detections": detections,
            "metadata": {k: v for k, v in obj.items() if k != "detections"},
        }
    return dict(data)


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


def build_pairs(
    tool_data: Dict[str, Dict[str, Any]],
    truth_data: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_ids = sorted(set(tool_data) | set(truth_data))
    pairs: List[Dict[str, Any]] = []
    missing = {"tool_rgb": [], "tool_sar": [], "truth_rgb": [], "truth_sar": []}

    for base_id in base_ids:
        tool_pair = tool_data.get(base_id, {})
        truth_pair = truth_data.get(base_id, {})

        for modality in ["rgb", "sar"]:
            if modality not in tool_pair:
                missing[f"tool_{modality}"].append(base_id)
            if modality not in truth_pair:
                missing[f"truth_{modality}"].append(base_id)

        pairs.append(
            {
                "base_id": base_id,
                "rgb_detections": tool_pair.get("rgb", {}).get("detections", []),
                "sar_detections": tool_pair.get("sar", {}).get("detections", []),
                "rgb_gt": truth_pair.get("rgb", []),
                "sar_gt": truth_pair.get("sar", []),
            }
        )

    return pairs, {"num_pairs": len(pairs), "missing": missing}


def assert_no_missing_pairs(pair_summary: Dict[str, Any]) -> None:
    missing = pair_summary.get("missing", {})
    bad = {k: v for k, v in missing.items() if v}
    if not bad:
        return
    details = "; ".join(f"{k}={len(v)}" for k, v in sorted(bad.items()))
    examples = {
        k: v[:5]
        for k, v in sorted(bad.items())
    }
    raise FileNotFoundError(
        "Missing paired RGB/SAR tool or truth files. "
        f"Counts: {details}. First examples: {examples}. "
        "Use --allow-missing only for debugging."
    )


def hybrid_match_info(
    left_item: Dict[str, Any],
    right_item: Dict[str, Any],
    engine: Optional[CrossModalInterpreter] = None,
) -> Dict[str, Any]:
    """几何混合分：走冻结协议（``engine`` 参数仅为兼容旧调用保留）。"""
    return gt_protocol.pair_score(
        left_item["bbox"],
        right_item["bbox"],
        left_item.get("score", gt_protocol.DEFAULT_DETECTION_CONFIDENCE),
        right_item.get("score", gt_protocol.DEFAULT_DETECTION_CONFIDENCE),
    )


def match_rgb_sar_items(
    left: Sequence[Dict[str, Any]],
    right: Sequence[Dict[str, Any]],
    mode: str,
    iou_threshold: float,
    hybrid_threshold: float,
    max_center_distance: Optional[float],
    engine: Optional[CrossModalInterpreter] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """GT 一对一匹配：走冻结协议（``engine`` 仅为兼容旧调用保留）。"""
    return gt_protocol.match_items(
        left, right,
        mode=mode,
        iou_threshold=iou_threshold,
        hybrid_threshold=hybrid_threshold,
        max_center_distance=max_center_distance,
    )


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


def conclusion_from_summary_for_eval(summary: Dict[str, Any]) -> str:
    """Correct crosscheck semantics for evaluation.

    The worker's response may overwrite spatial_conflict when it selects a
    single-modality evidence box. For evaluation, the crosscheck conclusion
    should be derived from match/count evidence.
    """

    return conclusion_from_counts(
        rgb_count=int(summary.get("rgb_filtered_count", 0) or 0),
        sar_count=int(summary.get("sar_filtered_count", 0) or 0),
        matched_count=int(summary.get("matched_count", 0) or 0),
    )


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


def selected_from_response(resp: Dict[str, Any], task: str) -> Optional[Dict[str, Any]]:
    selected = resp.get("selected_targets") or {}
    if task == "crosscheck_exists":
        return selected.get("best_evidence")
    if task == "select_largest":
        return selected.get("largest")
    if task == "select_highest_score":
        return selected.get("highest_score")
    if task == "distance_largest_smallest":
        return selected.get("largest")
    reps = selected.get("representative_targets")
    if isinstance(reps, list) and reps:
        return reps[0]
    return None


def best_iou(box: Optional[Sequence[float]], candidates: Sequence[Dict[str, Any]]) -> float:
    if not box or not candidates:
        return 0.0
    return max(bbox_iou(box, item["bbox"]) for item in candidates)


def add_metric_bucket(store: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in store:
        store[key] = defaultdict(float)
    return store[key]


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
    for t, f in zip(tp, fp):
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
        for recall, precision in zip(recalls, precisions):
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
    per_class = {}
    ap_values = []
    totals = defaultdict(int)

    for target in targets:
        result = compute_ap_for_class(gt_by_image, pred_by_image, target, iou_threshold)
        per_class[target] = result
        if result["num_gt"] > 0:
            ap_values.append(result["ap"])
        for key in ["tp", "fp", "num_gt", "num_pred"]:
            totals[key] += int(result[key])

    precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_div(totals["tp"], totals["num_gt"])
    f1 = safe_div(2 * precision * recall, precision + recall)
    key = metric_key(iou_threshold)

    return {
        f"mAP@{key}": safe_div(sum(ap_values), len(ap_values)),
        f"Recall@{key}": recall,
        f"Precision@{key}": precision,
        f"F1@{key}": f1,
        f"TP@{key}": totals["tp"],
        f"FP@{key}": totals["fp"],
        f"FN@{key}": max(0, totals["num_gt"] - totals["tp"]),
        f"GT@{key}": totals["num_gt"],
        f"Pred@{key}": totals["num_pred"],
        f"FP_per_image@{key}": safe_div(totals["fp"], max(len(set(gt_by_image) | set(pred_by_image)), 1)),
        f"Pred_GT_ratio@{key}": safe_div(totals["num_pred"], totals["num_gt"]),
        "per_class": per_class,
    }


def normalize_prediction(det: Dict[str, Any], fallback_label: Optional[str] = None) -> Optional[Dict[str, Any]]:
    bbox = det.get("bbox") or det.get("rgb_bbox") or det.get("sar_bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None

    try:
        box = [float(x) for x in bbox[:4]]
    except Exception:
        return None

    if box[2] <= box[0] or box[3] <= box[1]:
        return None

    label = normalize_detection_label(det.get("label") or det.get("class_name") or fallback_label or "object")
    score = det.get("score")
    if score is None:
        score = det.get("match_score")

    try:
        score_value = float(score if score is not None else 1.0)
    except Exception:
        score_value = 1.0

    out = {"label": label, "bbox": box, "score": score_value}
    if det.get("source") is not None:
        out["source"] = str(det.get("source"))
    if det.get("name") is not None:
        out["name"] = str(det.get("name"))
    if det.get("match_score") is not None:
        try:
            out["match_score"] = float(det.get("match_score"))
        except Exception:
            pass
    return out


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


def project_object_gt_for_mode(
    item: Dict[str, Any],
    mode: str,
    object_gt_bbox_mode: str,
) -> Dict[str, Any]:
    """Project a unified object-level GT item into a mode-specific box space.

    The object identity set is shared by all modes, but matched RGB/SAR GT boxes
    can differ because of registration or detector-box conventions.  In
    modality_aware mode, rgb_only is evaluated against rgb_bbox when available,
    sar_only against sar_bbox when available, and cross-modal modes against the
    fused object bbox.
    """

    out = dict(item)
    bbox = item.get("bbox")
    if object_gt_bbox_mode == "modality_aware":
        if mode == "rgb_only" and item.get("rgb_bbox") is not None:
            bbox = item.get("rgb_bbox")
        elif mode == "sar_only" and item.get("sar_bbox") is not None:
            bbox = item.get("sar_bbox")
    elif object_gt_bbox_mode != "fused":
        raise ValueError(f"Unsupported object GT bbox mode: {object_gt_bbox_mode}")

    if bbox is not None:
        out["bbox"] = [float(x) for x in bbox[:4]]
        out["area"] = bbox_area(out["bbox"])
    return out


def build_unified_object_detection_sets(
    pairs: Sequence[Dict[str, Any]],
    object_gt_by_image: Dict[str, List[Dict[str, Any]]],
    cross_pred_by_image: Dict[str, List[Dict[str, Any]]],
    object_gt_bbox_mode: str,
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    """Build fair modality comparison sets.

    All modes share the same object identity set.  The default
    object_gt_bbox_mode='modality_aware' avoids unfairly penalizing raw RGB/SAR
    predictions against an averaged fused GT box when RGB/SAR boxes differ.
    """

    datasets = {
        "rgb_only": {"gt": defaultdict(list), "pred": defaultdict(list)},
        "sar_only": {"gt": defaultdict(list), "pred": defaultdict(list)},
        "cross_modal": {"gt": defaultdict(list), "pred": defaultdict(list)},
        "cross_modal_confirmed_only": {"gt": defaultdict(list), "pred": defaultdict(list)},
    }

    for base_id, gts in object_gt_by_image.items():
        for mode in datasets:
            for item in gts:
                datasets[mode]["gt"][base_id].append(project_object_gt_for_mode(item, mode, object_gt_bbox_mode))

    for pair in pairs:
        base_id = pair["base_id"]
        for det in pair["rgb_detections"]:
            pred = normalize_prediction(det)
            if pred:
                datasets["rgb_only"]["pred"][base_id].append(pred)
        for det in pair["sar_detections"]:
            pred = normalize_prediction(det)
            if pred:
                datasets["sar_only"]["pred"][base_id].append(pred)

    for base_id, items in cross_pred_by_image.items():
        for item in items:
            datasets["cross_modal"]["pred"][base_id].append(item)
            if item.get("source") == "matched":
                datasets["cross_modal_confirmed_only"]["pred"][base_id].append(item)

    return {
        mode: {"gt": dict(values["gt"]), "pred": dict(values["pred"])}
        for mode, values in datasets.items()
    }


def evaluate_modality_metrics(
    datasets: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    targets: Sequence[str],
    iou_threshold: float,
    modalities: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    mode_metrics: Dict[str, Any] = {}
    per_class_mode: Dict[str, Any] = {target: {} for target in targets}
    key = metric_key(iou_threshold)

    for mode in modalities:
        values = datasets[mode]
        metrics = compute_detection_metrics(values["gt"], values["pred"], targets, iou_threshold)
        mode_metrics[mode] = {k: v for k, v in metrics.items() if k != "per_class"}
        rates = add_per_class_rates(metrics["per_class"], key)
        for target in targets:
            per_class_mode.setdefault(target, {})[mode] = rates.get(target, {})

    for target, modes in per_class_mode.items():
        mode_scores = {
            mode: modes.get(mode, {}).get(f"ap@{key}", 0.0)
            for mode in modalities
            if mode in modes
        }
        recall_scores = {
            mode: modes.get(mode, {}).get(f"recall@{key}", 0.0)
            for mode in modalities
            if mode in modes
        }
        analysis = {
            "best_map_mode": max(mode_scores, key=mode_scores.get) if mode_scores else "",
            "best_recall_mode": max(recall_scores, key=recall_scores.get) if recall_scores else "",
        }
        if "cross_modal" in modes and "rgb_only" in modes:
            cross = modes["cross_modal"]
            rgb = modes["rgb_only"]
            analysis.update({
                "cross_vs_rgb_map_delta": cross.get(f"ap@{key}", 0.0) - rgb.get(f"ap@{key}", 0.0),
                "cross_vs_rgb_recall_delta": cross.get(f"recall@{key}", 0.0) - rgb.get(f"recall@{key}", 0.0),
                "cross_vs_rgb_precision_delta": cross.get(f"precision@{key}", 0.0) - rgb.get(f"precision@{key}", 0.0),
            })
        if "cross_modal" in modes and "sar_only" in modes:
            cross = modes["cross_modal"]
            sar = modes["sar_only"]
            analysis.update({
                "cross_vs_sar_map_delta": cross.get(f"ap@{key}", 0.0) - sar.get(f"ap@{key}", 0.0),
                "cross_vs_sar_recall_delta": cross.get(f"recall@{key}", 0.0) - sar.get(f"recall@{key}", 0.0),
                "cross_vs_sar_precision_delta": cross.get(f"precision@{key}", 0.0) - sar.get(f"precision@{key}", 0.0),
            })
        modes["analysis"] = analysis

    mode_metrics["deltas"] = build_modality_deltas(mode_metrics, key)
    return mode_metrics, per_class_mode


def build_modality_deltas(mode_metrics: Dict[str, Any], key: str = "50") -> Dict[str, Any]:
    def delta(a: str, b: str) -> Dict[str, float]:
        left = mode_metrics.get(a, {})
        right = mode_metrics.get(b, {})
        fp_left = float(left.get(f"FP@{key}", 0.0))
        fp_right = float(right.get(f"FP@{key}", 0.0))
        return {
            f"delta_mAP@{key}": float(left.get(f"mAP@{key}", 0.0)) - float(right.get(f"mAP@{key}", 0.0)),
            f"delta_Recall@{key}": float(left.get(f"Recall@{key}", 0.0)) - float(right.get(f"Recall@{key}", 0.0)),
            f"delta_Precision@{key}": float(left.get(f"Precision@{key}", 0.0)) - float(right.get(f"Precision@{key}", 0.0)),
            f"delta_F1@{key}": float(left.get(f"F1@{key}", 0.0)) - float(right.get(f"F1@{key}", 0.0)),
            f"fp_reduction_percent@{key}": safe_div(fp_right - fp_left, fp_right) * 100.0,
            f"pred_delta@{key}": float(left.get(f"Pred@{key}", 0.0)) - float(right.get(f"Pred@{key}", 0.0)),
        }

    out: Dict[str, Any] = {}

    def maybe_add(name: str, a: str, b: str) -> None:
        if a in mode_metrics and b in mode_metrics:
            out[name] = delta(a, b)

    maybe_add("cross_vs_rgb", "cross_modal", "rgb_only")
    maybe_add("cross_vs_sar", "cross_modal", "sar_only")
    maybe_add("rgb_vs_sar", "rgb_only", "sar_only")
    maybe_add("confirmed_only_vs_rgb", "cross_modal_confirmed_only", "rgb_only")
    maybe_add("confirmed_only_vs_sar", "cross_modal_confirmed_only", "sar_only")
    maybe_add("confirmed_only_vs_cross", "cross_modal_confirmed_only", "cross_modal")
    return out


def match_counts_for_target(
    gt_items: Sequence[Dict[str, Any]],
    pred_items: Sequence[Dict[str, Any]],
    target: str,
    iou_threshold: float,
) -> Dict[str, Any]:
    gt = [x for x in gt_items if normalize_detection_label(x.get("label")) == target]
    pred = [x for x in pred_items if normalize_detection_label(x.get("label")) == target]

    candidates: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            iou = bbox_iou(p["bbox"], g["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, pi, gi))

    candidates.sort(reverse=True)
    used_pred, used_gt = set(), set()
    for _, pi, gi in candidates:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)

    best_iou_value = 0.0
    for p in pred:
        for g in gt:
            best_iou_value = max(best_iou_value, bbox_iou(p["bbox"], g["bbox"]))

    tp = len(used_pred)
    fp = len(pred) - tp
    fn = len(gt) - len(used_gt)
    return {
        "gt_count": len(gt),
        "pred_count": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, len(gt)),
        "best_iou": best_iou_value,
    }


def collect_modality_error_cases(
    datasets: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]],
    targets: Sequence[str],
    iou_threshold: float,
    top_k: int,
    modalities: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for mode in modalities:
        values = datasets[mode]
        image_ids = sorted(set(values["gt"]) | set(values["pred"]))
        for image_id in image_ids:
            gt_items = values["gt"].get(image_id, [])
            pred_items = values["pred"].get(image_id, [])
            for target in targets:
                counts = match_counts_for_target(gt_items, pred_items, target, iou_threshold)
                if counts["fp"] == 0 and counts["fn"] == 0:
                    continue
                if counts["fp"] > 0 and counts["fn"] > 0:
                    error_type = "mixed_fp_fn"
                elif counts["fp"] > 0:
                    error_type = "false_positive"
                else:
                    error_type = "missed_gt"
                rows.append({"mode": mode, "image_id": image_id, "target": target, "error_type": error_type, **counts})

    rows.sort(key=lambda x: (x["fn"] + x["fp"], x["fn"], x["fp"], x["best_iou"]), reverse=True)
    return rows[:top_k] if top_k > 0 else rows


def finalize_bucket(bucket: Dict[str, float]) -> Dict[str, Any]:
    out = {k: int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v for k, v in bucket.items()}

    if "crosscheck_total" in bucket:
        confirmed_tp = bucket.get("confirmed_tp", 0.0)
        confirmed_fp = bucket.get("confirmed_fp", 0.0)
        confirmed_fn = bucket.get("confirmed_fn", 0.0)
        out["crosscheck_accuracy"] = safe_div(bucket.get("crosscheck_correct", 0.0), bucket["crosscheck_total"])
        out["confirmed_precision"] = safe_div(confirmed_tp, confirmed_tp + confirmed_fp)
        out["confirmed_recall"] = safe_div(confirmed_tp, confirmed_tp + confirmed_fn)
        precision = out["confirmed_precision"]
        recall = out["confirmed_recall"]
        out["confirmed_f1"] = safe_div(2 * precision * recall, precision + recall)

    if "joint_total" in bucket:
        out["dedup_count_exact_accuracy"] = safe_div(bucket.get("joint_count_exact", 0.0), bucket["joint_total"])
        out["dedup_count_mae"] = safe_div(bucket.get("joint_count_abs_error", 0.0), bucket["joint_total"])

    if "largest_eligible" in bucket:
        largest_acc = safe_div(bucket.get("largest_correct", 0.0), bucket["largest_eligible"])
        out["largest_selection_accuracy"] = largest_acc
        out["largest_mean_iou"] = safe_div(bucket.get("largest_iou_sum", 0.0), bucket["largest_eligible"])

    if "distance_eligible" in bucket:
        out["distance_mae"] = safe_div(bucket.get("distance_abs_error", 0.0), bucket["distance_eligible"])
        out["distance_relative_error"] = safe_div(bucket.get("distance_relative_error_sum", 0.0), bucket["distance_eligible"])

    if "distance_correctness_total" in bucket:
        out["distance_accuracy"] = safe_div(bucket.get("distance_correct", 0.0), bucket["distance_correctness_total"])

    return out


def distance_is_correct(
    abs_error: float,
    rel_error: float,
    abs_threshold: Optional[float],
    rel_threshold: Optional[float],
) -> bool:
    checks = []
    if abs_threshold is not None:
        checks.append(abs_error <= abs_threshold)
    if rel_threshold is not None:
        checks.append(rel_error <= rel_threshold)
    if not checks:
        return True
    return any(checks)


def evaluate(
    pairs: Sequence[Dict[str, Any]],
    targets: Sequence[str],
    tasks: Sequence[str],
    modalities: Sequence[str],
    gt_match_mode: str,
    gt_match_iou: float,
    gt_hybrid_threshold: float,
    selected_iou: float,
    interpreter_match_threshold: float,
    pixel_size_m: float,
    category_match_mode: str,
    max_center_distance: Optional[float],
    distance_abs_threshold: Optional[float],
    distance_rel_threshold: Optional[float],
    analysis_top_k: int,
    object_gt_bbox_mode: str,
    eval_iou: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    engine = CrossModalInterpreter()
    global_stats: Dict[str, float] = defaultdict(float)
    class_stats: Dict[str, Any] = {}
    per_sample: List[Dict[str, Any]] = []
    error_cases: List[Dict[str, Any]] = []
    pred_dedup_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    gt_dedup_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    joint_resp_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    gt_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for pair in pairs:
        for target in targets:
            gt_info = gt_for_target(
                pair=pair,
                target=target,
                gt_match_mode=gt_match_mode,
                gt_match_iou=gt_match_iou,
                gt_hybrid_threshold=gt_hybrid_threshold,
                max_center_distance=max_center_distance,
                engine=engine,
            )
            gt_cache[(pair["base_id"], target)] = gt_info
            gt_dedup_by_image[pair["base_id"]].extend(gt_info["dedup_detections"])

    def get_resp(pair: Dict[str, Any], target: str, task: str) -> Dict[str, Any]:
        cache_key = (pair["base_id"], target)
        if task == "joint_scene" and cache_key in joint_resp_cache:
            return joint_resp_cache[cache_key]
        resp = run_interpreter(
            engine,
            pair,
            target,
            task,
            interpreter_match_threshold,
            pixel_size_m,
            category_match_mode,
            max_center_distance,
        )
        if task == "joint_scene":
            joint_resp_cache[cache_key] = resp
        return resp

    for pair in pairs:
        for target in targets:
            gt_info = gt_cache[(pair["base_id"], target)]
            cstats = add_metric_bucket(class_stats, target)

            for task in tasks:
                resp = get_resp(pair, target, task)
                if resp.get("error_code", 0) != 0:
                    record = {
                        "base_id": pair["base_id"],
                        "target": target,
                        "task": task,
                        "error_type": "interpreter_error",
                        "response": resp,
                    }
                    per_sample.append(record)
                    error_cases.append(record)
                    continue

                raw_pred_conclusion = str(resp.get("conclusion_type") or "")
                summary = resp.get("summary") or {}

                # Corrected conclusion for crosscheck evaluation. For other tasks,
                # preserve the worker's response conclusion for record analysis.
                pred_conclusion = (
                    conclusion_from_summary_for_eval(summary)
                    if task == "crosscheck_exists"
                    else raw_pred_conclusion
                )

                pred_dedup = resp.get("dedup_detections") or []
                pred_count = int(summary.get("dedup_count", len(pred_dedup)) or 0)
                count_error = pred_count - int(gt_info["dedup_count"])
                selected = selected_from_response(resp, task)
                selected_box = selected.get("bbox") if isinstance(selected, dict) else resp.get("final_bbox")

                correct = True
                error_type = ""
                selected_iou_value = None
                distance_abs_error = None
                distance_rel_error = None

                if task == "crosscheck_exists":
                    global_stats["crosscheck_total"] += 1
                    cstats["crosscheck_total"] += 1

                    is_correct = pred_conclusion == gt_info["conclusion_type"]
                    if is_correct:
                        global_stats["crosscheck_correct"] += 1
                        cstats["crosscheck_correct"] += 1
                    else:
                        correct = False
                        error_type = "wrong_conclusion"

                    pred_confirmed = pred_conclusion == "confirmed"
                    gt_confirmed = bool(gt_info["confirmed"])
                    if pred_confirmed and gt_confirmed:
                        global_stats["confirmed_tp"] += 1
                        cstats["confirmed_tp"] += 1
                    elif pred_confirmed and not gt_confirmed:
                        global_stats["confirmed_fp"] += 1
                        cstats["confirmed_fp"] += 1
                    elif (not pred_confirmed) and gt_confirmed:
                        global_stats["confirmed_fn"] += 1
                        cstats["confirmed_fn"] += 1

                elif task == "joint_scene":
                    global_stats["joint_total"] += 1
                    cstats["joint_total"] += 1
                    abs_error = abs(count_error)
                    global_stats["joint_count_abs_error"] += abs_error
                    cstats["joint_count_abs_error"] += abs_error
                    if count_error == 0:
                        global_stats["joint_count_exact"] += 1
                        cstats["joint_count_exact"] += 1
                    else:
                        correct = False
                        error_type = "wrong_dedup_count"

                elif task == "select_largest":
                    if gt_info["largest"]:
                        global_stats["largest_eligible"] += 1
                        cstats["largest_eligible"] += 1
                        selected_iou_value = best_iou(selected_box, [gt_info["largest"]])
                        global_stats["largest_iou_sum"] += selected_iou_value
                        cstats["largest_iou_sum"] += selected_iou_value

                        if selected_iou_value >= selected_iou:
                            global_stats["largest_correct"] += 1
                            cstats["largest_correct"] += 1
                        else:
                            correct = False
                            error_type = "wrong_largest_bbox"
                    elif selected_box:
                        global_stats["largest_false_selection_on_empty_gt"] += 1
                        cstats["largest_false_selection_on_empty_gt"] += 1
                        correct = False
                        error_type = "selected_on_empty_gt"

                elif task == "distance_largest_smallest":
                    if gt_info["largest"] and gt_info["smallest"] and gt_info["largest"]["name"] != gt_info["smallest"]["name"]:
                        global_stats["distance_eligible"] += 1
                        cstats["distance_eligible"] += 1
                        gt_distance = center_distance(gt_info["largest"]["bbox"], gt_info["smallest"]["bbox"]) * pixel_size_m
                        pred_distance = resp.get("distance_m")

                        if pred_distance is None:
                            correct = False
                            error_type = "missing_distance"
                            if distance_abs_threshold is not None or distance_rel_threshold is not None:
                                global_stats["distance_correctness_total"] += 1
                                cstats["distance_correctness_total"] += 1
                        else:
                            distance_abs_error = abs(float(pred_distance) - gt_distance)
                            distance_rel_error = safe_div(distance_abs_error, max(gt_distance, 1e-9))
                            global_stats["distance_abs_error"] += distance_abs_error
                            cstats["distance_abs_error"] += distance_abs_error
                            global_stats["distance_relative_error_sum"] += distance_rel_error
                            cstats["distance_relative_error_sum"] += distance_rel_error

                            if distance_abs_threshold is not None or distance_rel_threshold is not None:
                                global_stats["distance_correctness_total"] += 1
                                cstats["distance_correctness_total"] += 1
                                if distance_is_correct(
                                    distance_abs_error,
                                    distance_rel_error,
                                    distance_abs_threshold,
                                    distance_rel_threshold,
                                ):
                                    global_stats["distance_correct"] += 1
                                    cstats["distance_correct"] += 1
                                else:
                                    correct = False
                                    error_type = "large_distance_error"

                record = {
                    "base_id": pair["base_id"],
                    "target": target,
                    "task": task,
                    "gt_rgb_count": gt_info["rgb_count"],
                    "gt_sar_count": gt_info["sar_count"],
                    "gt_matched_count": gt_info["matched_count"],
                    "gt_dedup_count": gt_info["dedup_count"],
                    "gt_conclusion_type": gt_info["conclusion_type"],
                    "raw_pred_conclusion_type": raw_pred_conclusion,
                    "pred_conclusion_type": pred_conclusion,
                    "pred_matched_count": int(summary.get("matched_count", 0) or 0),
                    "pred_dedup_count": pred_count,
                    "count_error": count_error,
                    "selected_bbox": selected_box,
                    "gt_largest_bbox": gt_info["largest"].get("bbox") if gt_info["largest"] else None,
                    "selected_iou": selected_iou_value,
                    "distance_abs_error": distance_abs_error,
                    "distance_rel_error": distance_rel_error,
                    "is_correct": correct,
                    "error_type": error_type,
                }
                per_sample.append(record)
                if not correct:
                    error_cases.append(record)

    # Build cross-modal predictions exactly once from joint_scene outputs.
    for pair in pairs:
        for target in targets:
            resp = get_resp(pair, target, "joint_scene")
            if resp.get("error_code", 0) != 0:
                continue
            for det in resp.get("dedup_detections") or []:
                pred = normalize_prediction(det, fallback_label=target)
                if pred:
                    pred_dedup_by_image[pair["base_id"]].append(pred)

    eval_key = metric_key(eval_iou)
    object_metrics = compute_detection_metrics(gt_dedup_by_image, pred_dedup_by_image, targets, eval_iou)
    modality_datasets = build_unified_object_detection_sets(
        pairs,
        gt_dedup_by_image,
        pred_dedup_by_image,
        object_gt_bbox_mode=object_gt_bbox_mode,
    )
    modality_metrics, per_class_modality = evaluate_modality_metrics(modality_datasets, targets, eval_iou, modalities)
    modality_error_cases = collect_modality_error_cases(modality_datasets, targets, eval_iou, analysis_top_k, modalities)

    metrics = finalize_bucket(global_stats)
    metrics.update(
        {
            "num_pairs": len(pairs),
            "num_targets": len(targets),
            "num_tasks": len(tasks),
            "num_task_samples": len(per_sample),
            "eval_iou": eval_iou,
            "selected_iou_threshold": selected_iou,
            "largest_selection_iou_threshold": selected_iou,
            f"object_mAP@{eval_key}": object_metrics[f"mAP@{eval_key}"],
            f"object_Recall@{eval_key}": object_metrics[f"Recall@{eval_key}"],
            f"object_Precision@{eval_key}": object_metrics[f"Precision@{eval_key}"],
            f"object_F1@{eval_key}": object_metrics[f"F1@{eval_key}"],
            f"object_TP@{eval_key}": object_metrics[f"TP@{eval_key}"],
            f"object_FP@{eval_key}": object_metrics[f"FP@{eval_key}"],
            f"object_FN@{eval_key}": object_metrics[f"FN@{eval_key}"],
            f"object_GT@{eval_key}": object_metrics[f"GT@{eval_key}"],
            f"object_Pred@{eval_key}": object_metrics[f"Pred@{eval_key}"],
            f"dedup_mAP@{eval_key}": object_metrics[f"mAP@{eval_key}"],
            f"dedup_Recall@{eval_key}": object_metrics[f"Recall@{eval_key}"],
            f"dedup_Precision@{eval_key}": object_metrics[f"Precision@{eval_key}"],
            f"dedup_F1@{eval_key}": object_metrics[f"F1@{eval_key}"],
            # Backward-compatible aliases used by earlier reports/templates.
            f"dedup_TP@{eval_key}": object_metrics[f"TP@{eval_key}"],
            f"dedup_FP@{eval_key}": object_metrics[f"FP@{eval_key}"],
            f"dedup_FN@{eval_key}": object_metrics[f"FN@{eval_key}"],
            f"dedup_GT@{eval_key}": object_metrics[f"GT@{eval_key}"],
            f"dedup_Pred@{eval_key}": object_metrics[f"Pred@{eval_key}"],
        }
    )

    largest_key = metric_key(selected_iou)
    if "largest_selection_accuracy" in metrics:
        metrics[f"largest_selection_acc@{largest_key}"] = metrics["largest_selection_accuracy"]

    per_class = {target: finalize_bucket(dict(stats)) for target, stats in class_stats.items()}
    for values in per_class.values():
        if "largest_selection_accuracy" in values:
            values[f"largest_selection_acc@{largest_key}"] = values["largest_selection_accuracy"]

    for target, det in object_metrics["per_class"].items():
        per_class.setdefault(target, {})[f"object_ap@{eval_key}"] = det["ap"]
        per_class[target][f"object_recall@{eval_key}"] = det["recall"]
        per_class[target][f"object_precision@{eval_key}"] = safe_div(det["tp"], det["tp"] + det["fp"])
        per_class[target][f"object_f1@{eval_key}"] = safe_div(
            2 * per_class[target][f"object_precision@{eval_key}"] * det["recall"],
            per_class[target][f"object_precision@{eval_key}"] + det["recall"],
        )
        per_class[target][f"object_gt@{eval_key}"] = det["num_gt"]
        per_class[target][f"object_pred@{eval_key}"] = det["num_pred"]
        per_class[target][f"dedup_ap@{eval_key}"] = det["ap"]
        per_class[target][f"dedup_recall@{eval_key}"] = det["recall"]
        per_class[target][f"dedup_precision@{eval_key}"] = per_class[target][f"object_precision@{eval_key}"]
        per_class[target][f"dedup_f1@{eval_key}"] = per_class[target][f"object_f1@{eval_key}"]
        per_class[target][f"dedup_gt@{eval_key}"] = det["num_gt"]
        per_class[target][f"dedup_pred@{eval_key}"] = det["num_pred"]
        per_class[target][f"dedup_tp@{eval_key}"] = det["tp"]
        per_class[target][f"dedup_fp@{eval_key}"] = det["fp"]
        per_class[target][f"dedup_fn@{eval_key}"] = max(0, det["num_gt"] - det["tp"])

    return metrics, per_class, modality_metrics, per_class_modality, per_sample, error_cases, modality_error_cases


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_summary(path: Path, metrics: Dict[str, Any], per_class: Dict[str, Any], config: Dict[str, Any]) -> None:
    eval_key = metric_key(float(config.get("eval_iou", 0.5)))
    selected_key = metric_key(float(config.get("selected_iou", 0.5)))
    main_keys = [
        "num_pairs",
        "num_task_samples",
        "crosscheck_accuracy",
        "confirmed_precision",
        "confirmed_recall",
        "confirmed_f1",
        "dedup_count_exact_accuracy",
        "dedup_count_mae",
        f"object_mAP@{eval_key}",
        f"object_Recall@{eval_key}",
        f"object_Precision@{eval_key}",
        f"object_F1@{eval_key}",
        "largest_selection_accuracy",
        f"largest_selection_acc@{selected_key}",
        "largest_mean_iou",
        "distance_mae",
        "distance_relative_error",
        "distance_accuracy",
    ]
    class_keys = [
        "crosscheck_accuracy",
        "confirmed_precision",
        "confirmed_recall",
        "confirmed_f1",
        "dedup_count_exact_accuracy",
        "dedup_count_mae",
        f"object_ap@{eval_key}",
        f"object_recall@{eval_key}",
        f"object_precision@{eval_key}",
        f"object_f1@{eval_key}",
        "largest_selection_accuracy",
        f"largest_selection_acc@{selected_key}",
        "distance_mae",
        "distance_accuracy",
    ]
    lines = [
        "# Cross-Modal Interpretation Evaluation",
        "",
        "## Configuration",
        "",
        f"- tool_dir: `{config['tool_dir']}`",
        f"- truth_dir: `{config['truth_dir']}`",
        f"- image_size: `{config['image_size']}`",
        f"- image_width: `{config.get('image_width', config['image_size'])}`",
        f"- image_height: `{config.get('image_height', config['image_size'])}`",
        f"- targets: `{', '.join(config['targets'])}`",
        f"- tasks: `{', '.join(config['tasks'])}`",
        f"- gt_match_mode: `{config['gt_match_mode']}`",
        f"- gt_match_iou: `{config['gt_match_iou']}`",
        f"- gt_hybrid_threshold: `{config['gt_hybrid_threshold']}`",
        f"- object_gt_bbox_mode: `{config.get('object_gt_bbox_mode', 'modality_aware')}`",
        f"- eval_iou: `{config.get('eval_iou', 0.5)}`",
        f"- interpreter_match_threshold: `{config['interpreter_match_threshold']}`",
        f"- distance_abs_threshold: `{config.get('distance_abs_threshold')}`",
        f"- distance_rel_threshold: `{config.get('distance_rel_threshold')}`",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in main_keys:
        if key in metrics:
            lines.append(f"| {key} | {fmt(metrics[key])} |")

    lines.extend(["", "## Per-Class Metrics", "", "| Class | " + " | ".join(class_keys) + " |"])
    lines.append("| --- | " + " | ".join(["---:" for _ in class_keys]) + " |")
    for target in sorted(per_class):
        row = [target] + [fmt(per_class[target].get(key, "")) for key in class_keys]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines = [
        f"- gt_protocol_version: `{metrics.get('gt_protocol_version')}`",
        f"- predict_mode: `{metrics.get('predict_mode')}`",
        f"- run_id: `{metrics.get('run_id')}`",
        "",
    ] + lines
    path.write_text("\n".join(lines), encoding="utf-8")

def write_modality_analysis(
    path: Path,
    modality_metrics: Dict[str, Any],
    per_class_modality: Dict[str, Any],
    modality_error_cases: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    modes = [m for m in config.get("modalities", DEFAULT_MODALITIES) if m in modality_metrics]
    key = metric_key(float(config.get("eval_iou", 0.5)))
    metric_keys = [f"mAP@{key}", f"Recall@{key}", f"Precision@{key}", f"F1@{key}", f"TP@{key}", f"FP@{key}", f"FN@{key}", f"GT@{key}", f"Pred@{key}", f"FP_per_image@{key}", f"Pred_GT_ratio@{key}"]
    class_metric_keys = [f"ap@{key}", f"recall@{key}", f"precision@{key}", f"f1@{key}", f"gt@{key}", f"pred@{key}", f"fp@{key}", f"fn@{key}"]

    lines = [
        "# Unified Object-Level Multi-Modal Detection and Interpretation Analysis",
        "",
        "All modes are evaluated against the same object-level GT identity set. In modality_aware mode, RGB/SAR-only predictions use modality-specific GT boxes when available.",
        "",
        "## Overall Modality Comparison",
        "",
        "| Mode | " + " | ".join(metric_keys) + " |",
        "| --- | " + " | ".join(["---:" for _ in metric_keys]) + " |",
    ]

    for mode in modes:
        row = [mode] + [fmt(modality_metrics.get(mode, {}).get(metric_key, "")) for metric_key in metric_keys]
        lines.append("| " + " | ".join(row) + " |")

    deltas = modality_metrics.get("deltas", {})
    if deltas:
        delta_keys = [f"delta_mAP@{key}", f"delta_Recall@{key}", f"delta_Precision@{key}", f"delta_F1@{key}", f"fp_reduction_percent@{key}", f"pred_delta@{key}"]
        lines.extend(["", "## Delta Analysis", "", "| Comparison | " + " | ".join(delta_keys) + " |"])
        lines.append("| --- | " + " | ".join(["---:" for _ in delta_keys]) + " |")
        for name, values in deltas.items():
            row = [name] + [fmt(values.get(delta_key, "")) for delta_key in delta_keys]
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(["", "## Per-Class Modality Comparison"])
    for target in sorted(per_class_modality):
        lines.extend(["", f"### {target}", "", "| Mode | " + " | ".join(class_metric_keys) + " |"])
        lines.append("| --- | " + " | ".join(["---:" for _ in class_metric_keys]) + " |")
        for mode in modes:
            values = per_class_modality[target].get(mode, {})
            row = [mode] + [fmt(values.get(class_metric_key, "")) for class_metric_key in class_metric_keys]
            lines.append("| " + " | ".join(row) + " |")

        analysis = per_class_modality[target].get("analysis", {})
        if analysis:
            lines.extend(
                [
                    "",
                    f"Best mAP mode: `{analysis.get('best_map_mode', '')}`.",
                    f"Best recall mode: `{analysis.get('best_recall_mode', '')}`.",
                    f"Cross vs RGB mAP delta: `{fmt(analysis.get('cross_vs_rgb_map_delta', 0.0))}`.",
                    f"Cross vs SAR mAP delta: `{fmt(analysis.get('cross_vs_sar_map_delta', 0.0))}`.",
                ]
            )

    lines.extend(["", "## Error Profile", ""])
    if modality_error_cases:
        lines.extend(
            [
                "| Mode | Image | Target | Error | GT | Pred | TP | FP | FN | Best IoU |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in modality_error_cases[: min(30, len(modality_error_cases))]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("mode", "")),
                        str(row.get("image_id", "")),
                        str(row.get("target", "")),
                        str(row.get("error_type", "")),
                        fmt(row.get("gt_count", "")),
                        fmt(row.get("pred_count", "")),
                        fmt(row.get("tp", "")),
                        fmt(row.get("fp", "")),
                        fmt(row.get("fn", "")),
                        fmt(row.get("best_iou", "")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No modality-level error cases found.")

    lines.extend(["", "## Key Findings", ""])
    if modes:
        best_map_mode = max(modes, key=lambda m: modality_metrics.get(m, {}).get(f"mAP@{key}", 0.0))
        best_recall_mode = max(modes, key=lambda m: modality_metrics.get(m, {}).get(f"Recall@{key}", 0.0))
        best_precision_mode = max(modes, key=lambda m: modality_metrics.get(m, {}).get(f"Precision@{key}", 0.0))
        lines.append(f"- Best overall mAP@{key}: `{best_map_mode}` ({fmt(modality_metrics[best_map_mode].get(f'mAP@{key}', 0.0))}).")
        lines.append(f"- Best overall Recall@{key}: `{best_recall_mode}` ({fmt(modality_metrics[best_recall_mode].get(f'Recall@{key}', 0.0))}).")
        lines.append(f"- Best overall Precision@{key}: `{best_precision_mode}` ({fmt(modality_metrics[best_precision_mode].get(f'Precision@{key}', 0.0))}).")
        if "cross_modal" in modes and "rgb_only" in modes:
            delta = deltas.get("cross_vs_rgb", {})
            lines.append(f"- Cross-modal vs RGB-only mAP@{key} delta: `{fmt(delta.get(f'delta_mAP@{key}', 0.0))}`.")
            lines.append(f"- Cross-modal vs RGB-only FP reduction: `{fmt(delta.get(f'fp_reduction_percent@{key}', 0.0))}%`.")
        if "cross_modal_confirmed_only" in modes and "cross_modal" in modes:
            delta = deltas.get("confirmed_only_vs_cross", {})
            lines.append(f"- Confirmed-only vs full cross-modal FP reduction: `{fmt(delta.get(f'fp_reduction_percent@{key}', 0.0))}%`.")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RGB/SAR CrossModalInterpret outputs against paired YOLO GT.")
    parser.add_argument("--tool-dir", type=Path, default=Path(os.environ.get("ONEID_DATA_ROOT") or (REPO_ROOT / "data")) / "match_images/0_tool_json")
    parser.add_argument("--truth-dir", type=Path, default=Path(os.environ.get("ONEID_DATA_ROOT") or (REPO_ROOT / "data")) / "match_images/0_truth_json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/demo_check/multisource_interpret_eval")
    parser.add_argument("--image-size", type=float, default=1024.0, help="Fallback square image size used when --image-width/--image-height are not set.")
    parser.add_argument("--image-width", type=float, default=None, help="Image width in pixels for YOLO-normalized GT conversion.")
    parser.add_argument("--image-height", type=float, default=None, help="Image height in pixels for YOLO-normalized GT conversion.")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--modalities", default=",".join(DEFAULT_MODALITIES), help="Comma-separated modes: rgb_only,sar_only,cross_modal,cross_modal_confirmed_only")
    parser.add_argument("--analysis-top-k", type=int, default=100, help="Maximum modality error cases to write and rank.")
    parser.add_argument("--class-map", default=None, help="Comma-separated id:label entries. Default: 0:aircraft,1:bridge,2:harbor,3:ship,4:tank,5:vehicle")

    parser.add_argument("--gt-match-mode", choices=["iou", "hybrid"], default="iou", help="How to pair RGB/SAR GT boxes into object-level GT. Default iou is more independent; hybrid is recommended only for sensitivity analysis.")
    parser.add_argument("--gt-match-iou", type=float, default=0.5, help="IoU threshold used when --gt-match-mode=iou.")
    parser.add_argument("--gt-hybrid-threshold", type=float, default=None, help="Hybrid GT matching threshold. Defaults to --interpreter-match-threshold.")
    parser.add_argument("--selected-iou", type=float, default=0.5)
    parser.add_argument("--eval-iou", type=float, default=0.5, help="IoU threshold for object-level detection metrics.")
    parser.add_argument("--object-gt-bbox-mode", choices=["modality_aware", "fused"], default="modality_aware", help="Use mode-specific GT boxes for rgb_only/sar_only when available, or always use fused object boxes.")
    parser.add_argument("--interpreter-match-threshold", type=float, default=CrossModalInterpreter.MATCH_THRESHOLD)
    parser.add_argument("--max-center-distance", type=float, default=None, help="Optional maximum center distance in pixels for RGB/SAR matching.")
    parser.add_argument("--pixel-size-m", type=float, default=1.0)
    parser.add_argument("--category-match-mode", choices=["exact", "substring"], default="exact")

    parser.add_argument("--distance-abs-threshold", type=float, default=None, help="Absolute distance-error threshold in meters. If unset, only relative threshold is used.")
    parser.add_argument("--distance-rel-threshold", type=float, default=0.20, help="Relative distance-error threshold. Set negative to disable.")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing RGB/SAR tool or truth files. By default missing pairs raise an error.")

    args = parser.parse_args()

    if not args.tool_dir.exists():
        raise FileNotFoundError(f"tool dir does not exist: {args.tool_dir}")
    if not args.truth_dir.exists():
        raise FileNotFoundError(f"truth dir does not exist: {args.truth_dir}")

    targets = unique_preserve_order([normalize_detection_label(x) for x in parse_csv(args.targets, DEFAULT_TARGETS)])
    if not targets:
        raise ValueError("No targets specified.")
    supported_targets = {normalize_detection_label(x) for x in CrossModalInterpreter.FINAL_TARGETS}
    unsupported_targets = sorted(set(targets) - supported_targets)
    if unsupported_targets:
        raise ValueError(
            f"Unsupported targets: {unsupported_targets}. Supported: {sorted(supported_targets)}"
        )

    task_engine = CrossModalInterpreter()
    raw_tasks = parse_csv(args.tasks, DEFAULT_TASKS)
    tasks = []
    for raw_task in raw_tasks:
        normalized = CrossModalInterpreter.normalize_label(raw_task)
        underscore = normalized.replace(" ", "_")
        if underscore not in CrossModalInterpreter.TASK_ALIASES and normalized not in CrossModalInterpreter.TASK_ALIASES:
            raise ValueError(
                f"Unsupported task alias: {raw_task!r}. Supported evaluated tasks: {sorted(EVALUATED_TASKS)}"
            )
        canonical_task = task_engine.canonical_task(raw_task)
        if canonical_task not in EVALUATED_TASKS:
            raise ValueError(
                f"Task {canonical_task!r} is supported by CrossModalInterpreter but not by this evaluator. "
                f"Supported evaluated tasks: {sorted(EVALUATED_TASKS)}"
            )
        tasks.append(canonical_task)
    tasks = unique_preserve_order(tasks)
    if not tasks:
        raise ValueError("No evaluated tasks specified.")

    modalities = unique_preserve_order(parse_csv(args.modalities, DEFAULT_MODALITIES))
    unsupported_modalities = sorted(set(modalities) - set(DEFAULT_MODALITIES))
    if unsupported_modalities:
        raise ValueError(f"Unsupported modalities: {unsupported_modalities}. Supported: {DEFAULT_MODALITIES}")

    unsupported_tasks = sorted(set(tasks) - EVALUATED_TASKS)
    if unsupported_tasks:
        raise ValueError(f"Unsupported evaluated tasks: {unsupported_tasks}. Supported: {sorted(EVALUATED_TASKS)}")

    image_width = args.image_width if args.image_width is not None else args.image_size
    image_height = args.image_height if args.image_height is not None else args.image_size
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: width={image_width}, height={image_height}")

    class_map = parse_class_map(args.class_map)
    gt_hybrid_threshold = args.gt_hybrid_threshold
    if gt_hybrid_threshold is None:
        gt_hybrid_threshold = args.interpreter_match_threshold

    distance_rel_threshold = args.distance_rel_threshold
    if distance_rel_threshold is not None and distance_rel_threshold < 0:
        distance_rel_threshold = None

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tool_data = load_tool_dir(args.tool_dir)
    truth_data = load_truth_dir(args.truth_dir, image_width, image_height, class_map)
    pairs, pair_summary = build_pairs(tool_data, truth_data)

    if not args.allow_missing:
        assert_no_missing_pairs(pair_summary)

    metrics, per_class, modality_metrics, per_class_modality, per_sample, error_cases, modality_error_cases = evaluate(
        pairs=pairs,
        targets=targets,
        tasks=tasks,
        modalities=modalities,
        gt_match_mode=args.gt_match_mode,
        gt_match_iou=args.gt_match_iou,
        gt_hybrid_threshold=float(gt_hybrid_threshold),
        selected_iou=args.selected_iou,
        interpreter_match_threshold=args.interpreter_match_threshold,
        pixel_size_m=args.pixel_size_m,
        category_match_mode=args.category_match_mode,
        max_center_distance=args.max_center_distance,
        distance_abs_threshold=args.distance_abs_threshold,
        distance_rel_threshold=distance_rel_threshold,
        analysis_top_k=args.analysis_top_k,
        object_gt_bbox_mode=args.object_gt_bbox_mode,
        eval_iou=args.eval_iou,
    )

    config = {
        "tool_dir": str(args.tool_dir),
        "truth_dir": str(args.truth_dir),
        "output_dir": str(args.output_dir),
        "image_size": args.image_size,
        "image_width": image_width,
        "image_height": image_height,
        "targets": targets,
        "tasks": tasks,
        "modalities": modalities,
        "analysis_top_k": args.analysis_top_k,
        "class_map": class_map,
        "gt_match_mode": args.gt_match_mode,
        "gt_match_iou": args.gt_match_iou,
        "gt_hybrid_threshold": float(gt_hybrid_threshold),
        "selected_iou": args.selected_iou,
        "eval_iou": args.eval_iou,
        "object_gt_bbox_mode": args.object_gt_bbox_mode,
        "interpreter_match_threshold": args.interpreter_match_threshold,
        "max_center_distance": args.max_center_distance,
        "pixel_size_m": args.pixel_size_m,
        "category_match_mode": args.category_match_mode,
        "distance_abs_threshold": args.distance_abs_threshold,
        "distance_rel_threshold": distance_rel_threshold,
        "allow_missing": args.allow_missing,
        "pair_summary": pair_summary,
        "ap_protocol": f"101-point interpolated AP@{metric_key(args.eval_iou)}",
        "gt_protocol_version": gt_protocol.GT_PROTOCOL_VERSION,
        "predict_mode": "interpreter_response",
        "modality_protocol": "unified_object_level_gt",
    }

    metrics["gt_protocol_version"] = gt_protocol.GT_PROTOCOL_VERSION
    metrics["predict_mode"] = config.get("predict_mode", "interpreter_response")
    metrics["run_id"] = args.output_dir.name
    write_json(args.output_dir / "config.json", config)
    write_json(args.output_dir / "metrics_summary.json", metrics)
    write_json(args.output_dir / "per_class_metrics.json", per_class)
    write_json(args.output_dir / "modality_metrics.json", modality_metrics)
    write_json(args.output_dir / "per_class_modality_metrics.json", per_class_modality)
    write_jsonl(args.output_dir / "per_sample_results.jsonl", per_sample)
    write_jsonl(args.output_dir / "error_cases.jsonl", error_cases)
    write_jsonl(args.output_dir / "modality_error_cases.jsonl", modality_error_cases)
    write_markdown_summary(args.output_dir / "summary.md", metrics, per_class, config)
    write_modality_analysis(args.output_dir / "modality_analysis.md", modality_metrics, per_class_modality, modality_error_cases, config)

    print("Cross-modal interpretation evaluation complete")
    print(f"pairs: {metrics['num_pairs']}")
    for key in [
        "crosscheck_accuracy",
        "confirmed_precision",
        "confirmed_recall",
        "confirmed_f1",
        "dedup_count_exact_accuracy",
        "dedup_count_mae",
        f"object_mAP@{metric_key(args.eval_iou)}",
        f"object_Recall@{metric_key(args.eval_iou)}",
        f"object_Precision@{metric_key(args.eval_iou)}",
        f"object_F1@{metric_key(args.eval_iou)}",
        "largest_selection_accuracy",
        "distance_mae",
        "distance_accuracy",
    ]:
        if key in metrics:
            print(f"{key}: {fmt(metrics[key])}")

    eval_key = metric_key(args.eval_iou)
    print("unified object-level modality comparison:")
    for mode in modalities:
        values = modality_metrics.get(mode, {})
        print(
            f"  {mode}: mAP@{eval_key}={fmt(values.get(f'mAP@{eval_key}', 0.0))}, "
            f"Recall@{eval_key}={fmt(values.get(f'Recall@{eval_key}', 0.0))}, "
            f"Precision@{eval_key}={fmt(values.get(f'Precision@{eval_key}', 0.0))}, "
            f"F1@{eval_key}={fmt(values.get(f'F1@{eval_key}', 0.0))}"
        )
    print(f"output_dir: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
