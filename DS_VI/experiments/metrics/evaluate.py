#!/usr/bin/env python3
"""Dual-arm evaluation: match (6-class box level) and time (identity + change).

Match arm: object-level detection metrics over the unified RGB/SAR object GT
(per class, macro/micro, box level and matched-only).

Time arm: per-image sizes are read from the PNGs (the time set mixes 1024,
1000, 800x600, 1500 and larger frames). GT tracks are built per class by
pairing time1/time2 GT boxes by the configured center-distance radius,
then:

* identity matching: predicted cross-time pairs vs GT matched tracks
  (precision / recall / F1 per class and micro),
* change classification: stayed / moved / left / appeared confusion matrix,
  accuracy and macro-F1,
* presence(t0/t1) and motion(static/moving) accuracies derived from the change
  labels.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


from DS_VI.models.ds_fusion.pipeline import (
    CANONICAL_CLASSES,
    image_size,
    normalize_class,
)
from DS_VI.models.ds_fusion.config import _resolve, load_config

from DS_VI.experiments.metrics.detection_metrics import (
    DEFAULT_CLASS_MAP,
    CrossModalInterpreter,
    add_per_class_rates,
    bbox_iou,
    compute_detection_metrics,
    fuse_bbox,
    gt_for_target,
    load_truth_dir,
    safe_div,
)


ROOT = Path(__file__).resolve().parents[1]

MATCH_IMAGE_SIZE = 1024.0
CHANGE_CLASSES = ["stayed", "moved", "left", "appeared"]


def read_predictions(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["pair_id"])].extend(row.get("objects") or [])
    return dict(out)


def load_match_gt(config: Mapping[str, Any], base_ids: Sequence[str], targets: Sequence[str]):
    spec = dict(config["families"]["match"])
    truth = load_truth_dir(_resolve(spec["truth_dir"]), MATCH_IMAGE_SIZE, MATCH_IMAGE_SIZE, DEFAULT_CLASS_MAP)
    engine = CrossModalInterpreter()
    gt_by_image: Dict[str, List[Dict[str, Any]]] = {}
    for base in base_ids:
        truth_pair = truth.get(base, {})
        pair = {
            "base_id": base,
            "rgb_gt": truth_pair.get("rgb", []),
            "sar_gt": truth_pair.get("sar", []),
        }
        items: List[Dict[str, Any]] = []
        for target in targets:
            info = gt_for_target(pair, target, "iou", 0.5, 0.75, None, engine)
            for item in info["dedup_detections"]:
                items.append({"bbox": item["bbox"], "label": target, "source": item["source"]})
        gt_by_image[base] = items
    return gt_by_image


def _prediction_boxes(objects: Sequence[Mapping[str, Any]], matched_only: bool = False):
    boxes = []
    for obj in objects:
        if matched_only and obj.get("source") != "matched":
            continue
        bbox = obj.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        boxes.append({
            "bbox": [float(v) for v in bbox[:4]],
            "label": normalize_class(obj.get("label")),
            "score": float(obj.get("score") or 0.0),
        })
    return boxes


def match_metrics(
    gt_by_image: Mapping[str, List[Dict[str, Any]]],
    predictions: Mapping[str, List[Dict[str, Any]]],
    base_ids: Sequence[str],
    targets: Sequence[str],
) -> Dict[str, Any]:
    pred_by_image = {
        base: _prediction_boxes(predictions.get(base, []))
        for base in base_ids
    }
    pred_matched = {
        base: _prediction_boxes(predictions.get(base, []), matched_only=True)
        for base in base_ids
    }
    gt_matched = {
        base: [item for item in gt_by_image.get(base, []) if item.get("source") == "matched"]
        for base in base_ids
    }
    box_level = compute_detection_metrics(dict(gt_by_image), pred_by_image, targets, 0.5)
    matched_level = compute_detection_metrics(gt_matched, pred_matched, targets, 0.5)
    return {"box_level": box_level, "matched_level": matched_level}


def _load_yolo_boxes(path: Path, width: float, height: float) -> List[Dict[str, Any]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = parts[0]
        label = normalize_class(DEFAULT_CLASS_MAP.get(class_id))
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        boxes.append({
            "bbox": [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
            "label": label,
        })
    return boxes


def _track_geometry(bbox: Sequence[float]) -> Tuple[float, float, float]:
    from math import hypot

    center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
    center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
    diagonal = hypot(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
    return center_x, center_y, max(diagonal, 1.0)


def pair_time_boxes(
    a_boxes: Sequence[Mapping[str, Any]],
    b_boxes: Sequence[Mapping[str, Any]],
    radius_ratio: float,
    stayed_ratio: float,
) -> Tuple[List[Dict[str, Any]], List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """Greedy center-distance pairing so moved objects keep their identity.

    An IoU-only pairing collapses every matched track into ``stayed`` because
    boxes must overlap to pair; the radius pairing keeps the identity link for
    objects up to ``radius_ratio`` average diagonals apart and labels the move.
    """
    candidates = []
    for ai, a_item in enumerate(a_boxes):
        for bi, b_item in enumerate(b_boxes):
            ax, ay, adiag = _track_geometry(a_item["bbox"])
            bx, by, bdiag = _track_geometry(b_item["bbox"])
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            radius = float(radius_ratio) * (adiag + bdiag) / 2.0
            if distance <= radius:
                candidates.append((distance, ai, bi, distance / max((adiag + bdiag) / 2.0, 1.0)))
    candidates.sort(key=lambda item: item[0])
    used_a, used_b = set(), set()
    matched = []
    for _distance, ai, bi, ratio in candidates:
        if ai in used_a or bi in used_b:
            continue
        used_a.add(ai)
        used_b.add(bi)
        bbox_a = [float(v) for v in a_boxes[ai]["bbox"]]
        bbox_b = [float(v) for v in b_boxes[bi]["bbox"]]
        matched.append({
            "label": a_boxes[ai]["label"],
            "change": "stayed" if ratio <= float(stayed_ratio) else "moved",
            "bbox": fuse_bbox(bbox_a, bbox_b),
            "bbox_t0": bbox_a,
            "bbox_t1": bbox_b,
            "source": "matched",
        })
    a_only = [item for index, item in enumerate(a_boxes) if index not in used_a]
    b_only = [item for index, item in enumerate(b_boxes) if index not in used_b]
    return matched, a_only, b_only


def load_time_gt(config: Mapping[str, Any], base_ids: Sequence[str], targets: Sequence[str]):
    spec = dict(config["families"]["time"])
    truth_dir = _resolve(spec["truth_dir"])
    image_dir = _resolve(spec["image_dir"])
    stayed_ratio = float(dict(config.get("time") or {}).get("stayed_diag_ratio", 0.5))
    radius_ratio = float(spec.get("track_radius_diag_ratio", 1.0))
    gt_tracks: Dict[str, List[Dict[str, Any]]] = {}
    for base in base_ids:
        per_view: Dict[str, List[Dict[str, Any]]] = {}
        for view_key in ("a", "b"):
            view_spec = dict(spec["views"][view_key])
            image_path = image_dir / f"{base}{view_spec['image_suffix']}.png"
            width, height = image_size(str(image_path))
            truth_path = truth_dir / f"{base}{view_spec['tool_suffix']}.txt"
            per_view[view_key] = _load_yolo_boxes(truth_path, width, height)
        tracks: List[Dict[str, Any]] = []
        for target in targets:
            a_list = [item for item in per_view["a"] if item["label"] == target]
            b_list = [item for item in per_view["b"] if item["label"] == target]
            matched, a_only, b_only = pair_time_boxes(a_list, b_list, radius_ratio, stayed_ratio)
            tracks.extend(matched)
            for item in a_only:
                bbox = [float(v) for v in item["bbox"]]
                tracks.append({"label": target, "change": "left", "bbox": bbox, "bbox_t0": bbox,
                               "bbox_t1": None, "source": "view_a_only"})
            for item in b_only:
                bbox = [float(v) for v in item["bbox"]]
                tracks.append({"label": target, "change": "appeared", "bbox": bbox, "bbox_t0": None,
                               "bbox_t1": bbox, "source": "view_b_only"})
        gt_tracks[base] = tracks
    return gt_tracks


def _greedy_pairs(
    gt_items: Sequence[Mapping[str, Any]],
    pred_items: Sequence[Mapping[str, Any]],
    iou_threshold: float = 0.5,
    temporal: bool = False,
) -> List[Tuple[int, int, float]]:
    candidates = []
    for gi, gt in enumerate(gt_items):
        for pi, pred in enumerate(pred_items):
            if normalize_class(pred.get("label")) != normalize_class(gt.get("label")):
                continue
            if temporal:
                # Verify each endpoint; an average box can conceal a wrong link.
                gt_presence = tuple(bool(gt.get(k)) for k in ("bbox_t0", "bbox_t1"))
                pred_presence = tuple(bool(pred.get(k)) for k in ("bbox_t0", "bbox_t1"))
                if gt_presence != pred_presence or not any(gt_presence):
                    continue
                iou = min(
                    bbox_iou(gt[k], pred[k])
                    for k in ("bbox_t0", "bbox_t1") if gt.get(k)
                )
            else:
                iou = bbox_iou(gt["bbox"], pred["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    used_gt, used_pred, pairs = set(), set(), []
    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((gi, pi, iou))
    return pairs


def time_metrics(
    gt_tracks: Mapping[str, List[Dict[str, Any]]],
    predictions: Mapping[str, List[Dict[str, Any]]],
    targets: Sequence[str],
) -> Dict[str, Any]:
    identity = {target: {"tp": 0, "fp": 0, "fn": 0} for target in targets}
    for base in sorted(set(gt_tracks) | set(predictions)):
        tracks = gt_tracks.get(base, [])
        preds = [
            obj for obj in predictions.get(base, [])
            if obj.get("bbox_t0") and obj.get("bbox_t1") and obj.get("source") == "matched"
        ]
        for target in targets:
            gt_pairs = [t for t in tracks if t["label"] == target and t["source"] == "matched"]
            pred_pairs = [o for o in preds if normalize_class(o.get("label")) == target]
            pairs = _greedy_pairs(gt_pairs, pred_pairs, temporal=True)
            identity[target]["tp"] += len(pairs)
            identity[target]["fp"] += len(pred_pairs) - len(pairs)
            identity[target]["fn"] += len(gt_pairs) - len(pairs)

    per_class = {}
    for target in targets:
        counts = identity[target]
        precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        per_class[target] = {
            "tp": counts["tp"], "fp": counts["fp"], "fn": counts["fn"],
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(safe_div(2 * precision * recall, precision + recall), 6),
        }
    total = {key: sum(identity[t][key] for t in targets) for key in ("tp", "fp", "fn")}
    micro_precision = safe_div(total["tp"], total["tp"] + total["fp"])
    micro_recall = safe_div(total["tp"], total["tp"] + total["fn"])
    micro = {
        **total,
        "precision": round(micro_precision, 6),
        "recall": round(micro_recall, 6),
        "f1": round(safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall), 6),
    }
    macro_f1 = sum(per_class[t]["f1"] for t in targets) / len(targets)

    confusion: Dict[str, Counter] = {gt_label: Counter() for gt_label in CHANGE_CLASSES}
    confusion["__spurious__"] = Counter()
    matched_support = 0
    gt_support = 0
    prediction_support = 0
    presence_t0 = [0, 0]
    presence_t1 = [0, 0]
    motion = [0, 0]
    for base in sorted(set(gt_tracks) | set(predictions)):
        tracks = gt_tracks.get(base, [])
        preds = predictions.get(base, [])
        for target in targets:
            gt_items = [t for t in tracks if t["label"] == target]
            pred_items = [o for o in preds if normalize_class(o.get("label")) == target]
            pairs = _greedy_pairs(gt_items, pred_items, temporal=True)
            used_gt = {gi for gi, _, _ in pairs}
            used_pred = {pi for _, pi, _ in pairs}
            gt_support += len(gt_items)
            prediction_support += len(pred_items)
            matched_support += len(pairs)
            for gi, gt_item in enumerate(gt_items):
                if gi not in used_gt:
                    confusion[str(gt_item["change"])]["__missed__"] += 1
            for pi, pred_item in enumerate(pred_items):
                if pi not in used_pred:
                    confusion["__spurious__"][str(pred_item.get("change", "__invalid__"))] += 1
            for gi, pi, _iou in pairs:
                gt_label = str(gt_items[gi].get("change"))
                pred_label = str(pred_items[pi].get("change"))
                if gt_label in confusion:
                    confusion[gt_label][pred_label] += 1
                gt_t0 = gt_items[gi].get("bbox_t0") is not None
                gt_t1 = gt_items[gi].get("bbox_t1") is not None
                pred_t0 = pred_items[pi].get("bbox_t0") is not None
                pred_t1 = pred_items[pi].get("bbox_t1") is not None
                presence_t0[0] += int(gt_t0 == pred_t0)
                presence_t1[0] += int(gt_t1 == pred_t1)
                presence_t0[1] += 1
                presence_t1[1] += 1
                if gt_t0 and gt_t1:
                    motion[0] += int((gt_label == "moved") == (pred_label == "moved"))
                    motion[1] += 1

    change_total = sum(sum(row.values()) for row in confusion.values())
    change_correct = sum(confusion[label][label] for label in CHANGE_CLASSES)
    change_per_class = {}
    f1_values = []
    for label in CHANGE_CLASSES:
        tp = confusion[label][label]
        fp = sum(row[label] for other, row in confusion.items() if other != label)
        fn = sum(value for other, value in confusion[label].items() if other != label)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        change_per_class[label] = {"tp": tp, "fp": fp, "fn": fn,
                                   "precision": round(precision, 6), "recall": round(recall, 6),
                                   "f1": round(f1, 6)}
        f1_values.append(f1)
    return {
        "identity": {
            "per_class": per_class,
            "micro": micro,
            "macro_f1": round(macro_f1, 6),
        },
        "change": {
            "protocol": "endpoint_iou50_same_presence_e2e_v2",
            "accuracy_definition": "correct / (GT events + unmatched predictions)",
            "accuracy": round(safe_div(change_correct, change_total), 6),
            "conditional_accuracy": round(safe_div(change_correct, matched_support), 6),
            "micro_precision": round(safe_div(change_correct, prediction_support), 6),
            "micro_recall": round(safe_div(change_correct, gt_support), 6),
            "micro_f1": round(safe_div(2 * change_correct, prediction_support + gt_support), 6),
            "macro_f1": round(sum(f1_values) / len(f1_values), 6),
            "support": change_total,
            "gt_support": gt_support,
            "prediction_support": prediction_support,
            "matched_support": matched_support,
            "missed": gt_support - matched_support,
            "spurious": prediction_support - matched_support,
            "per_class": change_per_class,
            "confusion": {label: dict(row) for label, row in confusion.items()},
        },
        "presence": {
            "t0_accuracy": round(safe_div(presence_t0[0], presence_t0[1]), 6),
            "t1_accuracy": round(safe_div(presence_t1[0], presence_t1[1]), 6),
            "support": presence_t0[1],
        },
        "motion": {
            "accuracy": round(safe_div(motion[0], motion[1]), 6),
            "support": motion[1],
        },
    }


def write_match_tables(
    metrics: Mapping[str, Any],
    targets: Sequence[str],
    output_dir: Path,
    prefix: str = "",
) -> None:
    for level in ("box_level", "matched_level"):
        rows = []
        per_class = (metrics[level] or {}).get("per_class") or {}
        rates = add_per_class_rates(per_class, "50")
        for target in targets:
            item = rates.get(target) or {}
            rows.append([
                target,
                item.get("ap@50", 0.0),
                item.get("precision@50", 0.0),
                item.get("recall@50", 0.0),
                item.get("f1@50", 0.0),
                item.get("tp@50", 0),
                item.get("fp@50", 0),
                item.get("gt@50", 0),
                item.get("pred@50", 0),
            ])
        name = f"per_class_{prefix}_{level}.csv" if prefix else f"per_class_{level}.csv"
        with (output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class", "AP@50", "Precision", "Recall", "F1", "TP", "FP", "GT", "Pred"])
            writer.writerows(rows)


def summarize_match(metrics: Mapping[str, Any]) -> str:
    lines = []
    for level in ("box_level", "matched_level"):
        item = metrics[level] or {}
        lines.append(
            f"- {level}: mAP@50={item.get('mAP@50', 0):.4f}, Recall={item.get('Recall@50', 0):.4f}, "
            f"Precision={item.get('Precision@50', 0):.4f}, F1={item.get('F1@50', 0):.4f}, "
            f"TP={item.get('TP@50', 0)}, FP={item.get('FP@50', 0)}, Pred={item.get('Pred@50', 0)}"
        )
    return "\n".join(lines)


def summarize_time(metrics: Mapping[str, Any]) -> str:
    identity = metrics["identity"]
    change = metrics["change"]
    return (
        f"- identity: micro P={identity['micro']['precision']:.4f}, R={identity['micro']['recall']:.4f}, "
        f"F1={identity['micro']['f1']:.4f}, macro-F1={identity['macro_f1']:.4f}\n"
        f"- change: accuracy={change['accuracy']:.4f}, macro-F1={change['macro_f1']:.4f}, support={change['support']}\n"
        f"- presence: t0={metrics['presence']['t0_accuracy']:.4f}, t1={metrics['presence']['t1_accuracy']:.4f}; "
        f"motion={metrics['motion']['accuracy']:.4f} (support={metrics['motion']['support']})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate dual-arm predictions")
    parser.add_argument("--family", required=True, choices=["match", "time"])
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default=",".join(CANONICAL_CLASSES))
    parser.add_argument("--variant", default="")
    args = parser.parse_args()

    config = load_config()
    targets = [normalize_class(value) for value in args.targets.split(",") if value.strip()]
    predictions = read_predictions(args.predictions)
    base_ids = sorted(predictions)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.family == "match":
        gt = load_match_gt(config, base_ids, targets)
        metrics = match_metrics(gt, predictions, base_ids, targets)
        write_match_tables(metrics, targets, args.output)
    else:
        gt = load_time_gt(config, base_ids, targets)
        metrics = time_metrics(gt, predictions, targets)
        with (args.output / "change_confusion.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            columns = CHANGE_CLASSES + ["__missed__"]
            writer.writerow(["gt\\pred"] + columns)
            for label in CHANGE_CLASSES + ["__spurious__"]:
                writer.writerow([label] + [metrics["change"]["confusion"][label].get(other, 0) for other in columns])

    with (args.output / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"variant": args.variant, "family": args.family, "targets": targets, "metrics": metrics},
                  handle, ensure_ascii=False, indent=2)
    text = summarize_match(metrics) if args.family == "match" else summarize_time(metrics)
    print(f"[{args.family}] {args.variant}\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
