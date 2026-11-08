"""Object-level metric table: Precision / F1 / Recall / PairBrier / TruthAcc / IDAcc / IDECE.

Where the numbers come from
---------------------------
* **input to the engine** = ``tool_task{4,5}_property.xlsx`` (detections + the
  attributes the tools reported);
* **the engine** = D-S pairing -> shared-identity VI -> cascade, run per fold;
* **the labels** = ``truth_task{4,5}_property.xlsx`` (independent annotation),
  joined to predictions only *after* inference.

Every metric below is computed on **held-out** folds, so a refit arm's number is
an out-of-fold estimate.

Definitions (all object level, per family, pooled over folds)
-------------------------------------------------------------
``TP`` / ``FP`` / ``FN``
    A predicted object is TP when its member detections resolve to the same
    annotated object as its GT counterpart.  Member resolution goes through the
    instance bridge (tool instance -> truth instance by IoU>=0.5); two objects
    match when their member sets overlap (Jaccard > 0, the repo's default), the
    greedy pass takes the largest overlap first -- the same rule as
    ``property_metrics.identity_metrics``.

``Precision`` = TP/(TP+FP), ``Recall`` = TP/(TP+FN), ``F1`` = harmonic mean.

``PairBrier`` (lower is better)
    Brier score over **cross-view pair claims**, i.e. predicted objects that
    assert two members.  ``y = 1`` when the object resolves to an annotated
    double-view object, ``y = 0`` otherwise (including unmatched predictions).
    ``p`` is the engine's association probability for that object
    (``vi_association``, falling back to the cascade ``score``).

``TruthAcc``
    Of the predicted objects that resolve to an annotated object, the share
    whose **VI-decided category** equals the annotated category.  (The
    ``G4/|GT|`` variant, where a missed object counts as wrong, is reported
    separately so the two readings cannot be confused.)

``IDAcc``
    Cross-view identity accuracy: the repo's ``association_accuracy`` --
    F1 of "confirmed double-view pairs" (recall over GT double-view objects,
    precision against over-merges).

``IDECE`` (lower is better)
    Expected calibration error of the same pair claims used by ``PairBrier``:
    10 equal-width bins, ``sum_b (n_b/N) * |acc_b - conf_b|``.  It answers
    "when the engine says 0.8, is it right 80% of the time?".
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import data as exp_data

CONFIDENCE_FIELDS = ("vi_association", "score", "betp", "vi_p_class")
BINS = 10


def _norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _group(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("pair_id")), str(row.get("view")))].append(row)
    return grouped


def load_truth(family: str) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """``{pair_id: [gt objects]}`` and ``{truth instance id: row}``."""
    from DS_VI.experiments.metrics.property_metrics import read_book

    book = read_book(exp_data.truth_workbook(family))
    instances = {str(row.get("instance_id")): row
                 for row in book.get("instances") or [] if row.get("instance_id")}
    objects: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in book.get("identities") or []:
        members = {str(row.get(key)) for key in ("instance_a", "instance_b")
                   if row.get(key) not in (None, "")}
        objects[str(row.get("pair_id"))].append({
            "object_id": row.get("object_id"),
            "members": members,
            "category": _norm(row.get("category")),
            "double": len(members) == 2,
        })
    return objects, instances


def build_bridge(family: str) -> Dict[str, str]:
    """``tool instance id -> truth instance id`` by per-view IoU greedy match."""
    from DS_VI.experiments.metrics.property_metrics import match_image, read_book

    tool_book = read_book(exp_data.tool_workbook(family))
    truth_book = read_book(exp_data.truth_workbook(family))
    tool_images = _group(tool_book.get("instances") or [])
    truth_images = _group(truth_book.get("instances") or [])
    bridge: Dict[str, str] = {}
    for key in sorted(tool_images):
        record = match_image(tool_images[key], truth_images.get(key) or [], 0.5)
        for match in record.get("matches") or []:
            tool_id = str(match["tool"].get("instance_id"))
            truth_id = str(match["truth"].get("instance_id"))
            if tool_id and truth_id:
                bridge[tool_id] = truth_id
    return bridge


def _confidence(obj: Mapping[str, Any], field: str) -> Optional[float]:
    for name in (field, *CONFIDENCE_FIELDS):
        value = obj.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, min(1.0, float(value)))
    return None


def member_ids(obj: Mapping[str, Any], pair_id: str, family: str) -> List[str]:
    """Rebuild the tool instance ids of one predicted object.

    ``predictions.jsonl`` drops the cascade's ``members`` list, so the ids come
    back from the echoed member indices and the family's view tags.
    """
    views = exp_data.family_views(family)
    ids: List[str] = []
    for side in ("a", "b"):
        index = obj.get(f"view_{side}_index")
        if index is None:
            continue
        ids.append(f"{pair_id}:{views[side]}:{int(index):04d}")
    return ids


def match_pair(
    predictions: Sequence[Mapping[str, Any]],
    gt_objects: Sequence[Mapping[str, Any]],
    bridge: Mapping[str, str],
    pair_id: str,
    family: str,
) -> List[Tuple[Mapping[str, Any], Optional[Mapping[str, Any]]]]:
    """Greedy object matching, largest member overlap first."""
    candidates: List[Tuple[float, int, int]] = []
    for index, obj in enumerate(predictions):
        truth_ids = set()
        for member in member_ids(obj, pair_id, family):
            mapped = bridge.get(str(member))
            if mapped:
                truth_ids.add(mapped)
        for gt_index, gt in enumerate(gt_objects):
            overlap = len(truth_ids & gt["members"])
            if overlap:
                union = len(truth_ids | gt["members"])
                candidates.append((overlap / union if union else 0.0, index, gt_index))
    candidates.sort(reverse=True)
    used_pred, used_gt = set(), set()
    assignment: Dict[int, int] = {}
    for _score, pred_index, gt_index in candidates:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        assignment[pred_index] = gt_index
    return [(predictions[index], gt_objects[assignment[index]] if index in assignment else None)
            for index in range(len(predictions))]


def score_fold(
    family: str,
    predictions: Mapping[str, List[Dict[str, Any]]],
    gt_objects: Mapping[str, List[Dict[str, Any]]],
    bridge: Mapping[str, str],
    confidence_field: str = "vi_association",
) -> Dict[str, Any]:
    tp = fp = fn = 0
    truth_correct = truth_scored = 0
    pair_p: List[float] = []
    pair_y: List[int] = []
    category_correct_all = 0
    gt_total = 0

    for pair_id, objects in predictions.items():
        truth = gt_objects.get(pair_id) or []
        gt_total += len(truth)
        matched = match_pair(objects, truth, bridge, pair_id, family)
        hit = {id(gt) for _obj, gt in matched if gt is not None}
        tp += len(hit)
        fp += len(matched) - len(hit)
        fn += len(truth) - len(hit)

        for obj, gt in matched:
            decided = _decided_category(obj)
            if gt is not None:
                truth_scored += 1
                if decided is not None and decided == gt["category"]:
                    truth_correct += 1
            if len(member_ids(obj, pair_id, family)) >= 2:
                probability = _confidence(obj, confidence_field)
                if probability is None:
                    continue
                wrong = gt is None or not gt.get("double")
                pair_p.append(probability)
                pair_y.append(0 if wrong else 1)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    brier = (sum((p - y) ** 2 for p, y in zip(pair_p, pair_y)) / len(pair_p)) if pair_p else None
    return {
        "_pair_p": pair_p, "_pair_y": pair_y,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "pair_count": len(pair_p),
        "pair_brier": brier,
        "idece": expected_calibration_error(pair_p, pair_y),
        "truth_acc": (truth_correct / truth_scored) if truth_scored else None,
        "truth_acc_g4": (truth_correct / gt_total) if gt_total else None,
        "truth_scored": truth_scored,
        "gt_total": gt_total,
    }


def _decided_category(obj: Mapping[str, Any]) -> Optional[str]:
    beliefs = obj.get("attribute_beliefs") or {}
    best = None
    for belief in beliefs.values():
        if str(belief.get("attribute")) != "category":
            continue
        if belief.get("value") in (None, ""):
            continue
        if best is None or float(belief.get("probability") or 0.0) > float(best.get("probability") or 0.0):
            best = belief
    if best is not None:
        return _norm(best.get("value"))
    return _norm(obj.get("label"))


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int],
                               bins: int = BINS) -> Optional[float]:
    if not probabilities:
        return None
    counts = [0] * bins
    conf_sum = [0.0] * bins
    correct = [0] * bins
    for probability, label in zip(probabilities, labels):
        index = min(bins - 1, max(0, int(probability * bins)))
        counts[index] += 1
        conf_sum[index] += probability
        correct[index] += int(label)
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        if not counts[index]:
            continue
        confidence = conf_sum[index] / counts[index]
        accuracy = correct[index] / counts[index]
        error += (counts[index] / total) * abs(accuracy - confidence)
    return error


def association_block(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """Repo-aligned identity accuracy (``G5_identity.composite``)."""
    if not metrics_path.is_file():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    composite = ((payload.get("G5_identity") or {}).get("composite") or {})
    return {
        "association_accuracy": composite.get("association_accuracy"),
        "detection_accuracy": composite.get("detection_accuracy"),
        "confirmed": composite.get("confirmed"),
        "over_merged": composite.get("over_merged"),
        "gt_double_view": composite.get("gt_double_view"),
    }


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    values = [float(v) for v in values if v is not None]
    k = len(values)
    if not k:
        return {"n": 0}
    mean = sum(values) / k
    variance = sum((v - mean) ** 2 for v in values) / (k - 1) if k > 1 else 0.0
    sd = math.sqrt(variance)
    se = sd / math.sqrt(k) if k else 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}.get(k - 1, 1.96)
    return {"n": k, "mean": round(mean, 6), "sd": round(sd, 6),
            "ci95": [round(mean - t * se, 6), round(mean + t * se, 6)],
            "values": [round(v, 6) for v in values]}


def run_arm(arm_dir: Path, confidence_field: str) -> Dict[str, Any]:
    fold_results = json.loads((arm_dir / "fold_results.json").read_text(encoding="utf-8"))
    out: Dict[str, Any] = {"arm": fold_results.get("arm"), "families": {}}
    for family in fold_results.get("families") or []:
        gt_objects, _ = load_truth(family)
        bridge = build_bridge(family)
        per_fold: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        pooled = {"tp": 0, "fp": 0, "fn": 0, "truth_correct": 0, "truth_scored": 0,
                  "pair_p": [], "pair_y": []}
        for fold in fold_results["fold_results"]:
            index = int(fold["fold"])
            pred_path = arm_dir / f"fold_{index}" / family / "engine" / "predictions.jsonl"
            if not pred_path.is_file():
                continue
            predictions = _read_predictions(pred_path)
            scored = score_fold(family, predictions, gt_objects, bridge, confidence_field)
            association = association_block(arm_dir / f"fold_{index}" / family / "metrics.json") or {}
            scored["association_accuracy"] = association.get("association_accuracy")
            pooled["tp"] += scored["tp"]
            pooled["fp"] += scored["fp"]
            pooled["fn"] += scored["fn"]
            pooled["truth_correct"] += round((scored["truth_acc"] or 0.0) * scored["truth_scored"])
            pooled["truth_scored"] += scored["truth_scored"]
            pooled["pair_p"].extend(scored.pop("_pair_p", []))
            pooled["pair_y"].extend(scored.pop("_pair_y", []))
            for key in ("precision", "recall", "f1", "pair_brier", "idece",
                        "truth_acc", "truth_acc_g4", "association_accuracy", "pair_count"):
                if scored.get(key) is not None:
                    per_fold[key].append(scored[key])
        block: Dict[str, Any] = {"metrics": {key: summarize(values) for key, values in per_fold.items()}}
        precision = pooled["tp"] / (pooled["tp"] + pooled["fp"]) if (pooled["tp"] + pooled["fp"]) else 0.0
        recall = pooled["tp"] / (pooled["tp"] + pooled["fn"]) if (pooled["tp"] + pooled["fn"]) else 0.0
        block["pooled"] = {
            "tp": pooled["tp"], "fp": pooled["fp"], "fn": pooled["fn"],
            "pair_count": len(pooled["pair_p"]),
            "pair_brier": (sum((p - y) ** 2 for p, y in zip(pooled["pair_p"], pooled["pair_y"]))
                           / len(pooled["pair_p"])) if pooled["pair_p"] else None,
            "idece": expected_calibration_error(pooled["pair_p"], pooled["pair_y"]),
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1": round(2 * precision * recall / (precision + recall), 6) if (precision + recall) else 0.0,
        }
        out["families"][family] = block
    return out


def _read_predictions(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[str(record["pair_id"])] = list(record.get("objects") or [])
    return out


ORDER = ("precision", "f1", "recall", "pair_brier", "truth_acc", "association_accuracy", "idece")
LABEL = {"precision": "Precision", "f1": "F1", "recall": "Recall",
         "pair_brier": "PairBrier↓", "truth_acc": "TruthAcc.",
         "association_accuracy": "IDAcc.", "idece": "IDECE↓"}


def render(results: Mapping[str, Mapping[str, Any]], families: Sequence[str]) -> str:
    arms = list(results)
    lines = ["# exp_3 — 指标表（五折交叉拟合，折间均值 [95% t-CI]）", ""]
    for family in families:
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| 指标 | " + " | ".join(f"`{arm}`" for arm in arms) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in arms) + " |")
        for key in ORDER:
            cells = []
            for arm in arms:
                entry = ((results[arm].get("families") or {}).get(family) or {}).get("metrics", {}).get(key)
                if not entry:
                    cells.append("-")
                    continue
                low, high = entry["ci95"]
                cells.append(f"{entry['mean']:.4f} [{low:.4f}, {high:.4f}]")
            lines.append(f"| {LABEL[key]} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("Pooled（跨折合并计数）:")
        lines.append("")
        lines.append("| arm | TP / FP / FN | P | R | F1 | PairBrier | IDECE |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for arm in arms:
            pooled = ((results[arm].get("families") or {}).get(family) or {}).get("pooled") or {}
            brier = pooled.get("pair_brier")
            ece = pooled.get("idece")
            lines.append(
                f"| `{arm}` | {pooled.get('tp')} / {pooled.get('fp')} / {pooled.get('fn')} | "
                f"{pooled.get('precision', 0):.4f} | {pooled.get('recall', 0):.4f} | {pooled.get('f1', 0):.4f} | "
                f"{'-' if brier is None else f'{brier:.4f}'} | "
                f"{'-' if ece is None else f'{ece:.4f}'} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory of <arm>/fold_results.json")
    parser.add_argument("--arms", default="refit,shipped,noctx")
    parser.add_argument("--confidence", choices=CONFIDENCE_FIELDS, default="vi_p_class",
                        help="probability scored by PairBrier/IDECE; vi_p_class is the "
                             "VI object posterior, vi_association the pairing mass, "
                             "betp the D-S stage (theta-independent control)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = Path(args.root)
    results: Dict[str, Any] = {}
    for arm in [item.strip() for item in args.arms.split(",") if item.strip()]:
        arm_dir = root / arm
        if (arm_dir / "fold_results.json").is_file():
            results[arm] = run_arm(arm_dir, args.confidence)
    if not results:
        raise SystemExit(f"no fold_results.json under {root}")
    families = sorted({family for arm in results.values() for family in arm["families"]})
    text = render(results, families)
    out_dir = Path(args.out) if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metric_table.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metric_table.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
