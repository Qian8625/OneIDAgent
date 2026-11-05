"""Property-level metrics computed from the four annotation workbooks.

Two workbooks per task family (``match`` = task4, ``time`` = task5):

* ``truth_*_property.xlsx`` — human ground truth.
* ``tool_*_property.xlsx``  — one pipeline snapshot over the same images.

Each workbook has three sheets:

``instances``  one detection box: ``instance_id``, ``pair_id``, ``view``,
               ``x1..y2``, attribute columns (``category``, ``subtype``,
               ``color``, ``orientation``, ``size_bin``; older sheets carry
               ``state`` instead of ``color``), ``is_real_object``,
               ``detection_score``.
``identities`` one resolved object: ``object_id``, ``pair_id``,
               ``instance_a``/``instance_b``, ``category``, ``subtype`` and
               (task5 only) ``temporal_relation``.
``samples``    one image pair: ``pair_id``, ``identity_complete``, ``note``.

Metric groups (protocol documented in ``docs`` of each function):

* ``G1`` detection, instance level: IoU matching, P/R/F1, AP@50, per class/view.
* ``G2`` attributes, instance level: accuracy, macro-F1, confusion, fill rate,
  per view. The scored axes are discovered from the workbook columns
  (``category``, ``subtype``, ``color``/``state``, ``orientation``,
  ``size_bin``): the 2026-11 annotation release replaced ``state`` with
  ``color``, and either spelling is scored when present.
* ``G4`` end-to-end attributes, object level: a missed object is wrong.
* ``G5`` identity, object level: predicted objects vs truth objects, matched
  through the shared ``instance_id`` members; double-view / single-side split.
* ``G6`` change, task5 only: persisted / appeared / disappeared, plus the
  four-class variant that splits ``persisted`` into ``stayed`` / ``moved``. The
  truth side reads the annotated ``temporal_relation`` column (``persisted_moved``
  is the moved case); predictions derive the label from box displacement.

Protocol decisions baked in:

* ``tool`` instance ids are 0-based and follow the raw tool-JSON detection
  order; ``truth`` ids are independent, so instances are joined by
  ``(pair_id, view)`` + IoU only — never by id.
* Duplicate boxes are reported separately and do **not** count as FP: a truth
  box may be claimed once; further tool boxes overlapping it are duplicates.
* Missing predictions are never dropped silently: fill rate is reported next to
  accuracy, and end-to-end accuracy counts misses as wrong.
* ``visibility`` is acquisition metadata consumed by the contextual reliability
  model, not a predicted attribute, so it is not scored here.
* Change accuracy follows the evaluator convention
  ``correct / (GT events + spurious predictions)``; ``__missed__`` and
  ``__spurious__`` are explicit buckets.

Only ``openpyxl`` is required: the module avoids pandas so the project keeps its
current dependency set.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from openpyxl import load_workbook

from DS_VI.models.cascade.status import STATUS_TO_CHANGE

ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = Path("H:/1_RSTIBenchMark")

#: family -> (tool workbook, truth workbook, (view_a, view_b)) under DATASET_ROOT
FAMILY_FILES: Dict[str, Tuple[str, str, Tuple[str, str]]] = {
    "match": ("tool_task4_property.xlsx", "truth_task4_property.xlsx", ("rgb", "sar")),
    "time": ("tool_task5_property.xlsx", "truth_task5_property.xlsx", ("time1", "time2")),
}

#: Candidate instance-level attribute axes, in report order.  Which of them are
#: scored is decided per workbook by :func:`present_attributes`: the 2026-11
#: annotation release replaced ``state`` with ``color``, and hardcoding either
#: name would silently report ``support = 0`` for the other instead of failing.
INSTANCE_ATTRIBUTE_CANDIDATES: Tuple[str, ...] = (
    "category",
    "subtype",
    "color",
    "state",
    "orientation",
    "size_bin",
)

#: Candidate object-level axes (orientation is view-specific, so not pooled).
OBJECT_ATTRIBUTE_CANDIDATES: Tuple[str, ...] = (
    "category",
    "subtype",
    "color",
    "state",
    "size_bin",
)

#: Backwards-compatible aliases for callers that just want the full candidate set.
INSTANCE_ATTRIBUTES: Tuple[str, ...] = INSTANCE_ATTRIBUTE_CANDIDATES
OBJECT_ATTRIBUTES: Tuple[str, ...] = OBJECT_ATTRIBUTE_CANDIDATES

#: Annotated change vocabulary (``identities.temporal_relation``) -> the four
#: classes G6 scores.  ``persisted`` means "present at both times and not moving";
#: ``persisted_moved`` is its moved counterpart.  Older spellings are accepted so
#: a workbook from either annotation round scores.
CHANGE_ANNOTATION: Dict[str, str] = {
    "persisted": "stayed",
    "persisted_moved": "moved",
    "stayed": "stayed",
    "moved": "moved",
    "appeared": "appeared",
    "new": "appeared",
    "disappeared": "disappeared",
    "left": "disappeared",
    "gone": "disappeared",
}

MISSING = "__missing__"
MISSED = "__missed__"
SPURIOUS = "__spurious__"
IOU_THRESHOLD = 0.5
#: ``persisted`` objects whose relative displacement exceeds this ratio count as
#: ``moved``; identical to ``time.stayed_diag_ratio`` in ``ds_default.json``.
STAYED_RATIO = 0.5
CHANGE_CLASSES = ("stayed", "moved", "appeared", "disappeared")


# --------------------------------------------------------------------- io

def read_sheet(path: Path, sheet: str) -> List[Dict[str, Any]]:
    """Read one sheet into plain dicts (first row is the header)."""
    workbook = load_workbook(Path(path), read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet]
        rows = worksheet.iter_rows(values_only=True)
        header = [str(value) if value is not None else "" for value in next(rows)]
        records: List[Dict[str, Any]] = []
        for row in rows:
            if row is None or all(value is None or value == "" for value in row):
                continue
            records.append({name: value for name, value in zip(header, row) if name})
        return records
    finally:
        workbook.close()


def read_book(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Read ``instances``/``identities``/``samples`` from one workbook."""
    workbook = load_workbook(Path(path), read_only=True, data_only=True)
    try:
        names = list(workbook.sheetnames)
    finally:
        workbook.close()
    return {sheet: read_sheet(path, sheet) for sheet in ("instances", "identities", "samples") if sheet in names}


# ----------------------------------------------------------------- helpers

def bbox_of(record: Mapping[str, Any]) -> Optional[List[float]]:
    try:
        box = [float(record["x1"]), float(record["y1"]), float(record["x2"]), float(record["y2"])]
    except (KeyError, TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0.0 else 0.0


def normalized(value: Any) -> Optional[str]:
    """Lower-case/strip an attribute value; ``None`` when the cell is empty."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def present_attributes(
    records: Sequence[Mapping[str, Any]],
    candidates: Sequence[str] = INSTANCE_ATTRIBUTE_CANDIDATES,
) -> Tuple[str, ...]:
    """Candidate axes that actually exist as columns in ``records``.

    Scoring is driven by the workbook, not by a hardcoded vocabulary, so a data
    update that renames or adds an axis is reported (or omitted) instead of
    silently producing ``support = 0``.
    """
    columns = set()
    for row in records:
        columns.update(str(key) for key in row)
    return tuple(attribute for attribute in candidates if attribute in columns)


def reported_change_label(row: Mapping[str, Any]) -> Optional[str]:
    """Change label a *prediction* row already carries.

    Arms now write a ``status`` column from :mod:`DS_VI.models.cascade.status`
    (``static``/``moved``/``time1-only``/``time2-only``). Reading it keeps the
    reported status and the scored label the same quantity; a workbook without
    the column (the historical runs) falls back to geometry.
    """
    status = normalized(row.get("status"))
    if status is not None and status in STATUS_TO_CHANGE:
        return STATUS_TO_CHANGE[status]
    return None


def annotated_change_label(row: Mapping[str, Any]) -> Optional[str]:
    """Change label a resolved object carries, from ``temporal_relation``.

    The time-truth workbook annotates the relation itself (``persisted`` /
    ``persisted_moved`` / ``appeared`` / ``disappeared``). Reading it is more
    faithful than re-deriving the label from box displacement: on the 2026-11
    release 9.5% of annotated rows disagree with the geometric derivation.
    Predictions do not carry the column, so they keep the derivation.
    """
    return CHANGE_ANNOTATION.get(normalized(row.get("temporal_relation")) or "")


def composite_association(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """HOTA-style detection / association split and their geometric mean.

    ``detection_accuracy`` is the object-level Jaccard of the identity matching,
    ``TP / (TP + FP + FN)`` — HOTA's DetA (Luiten et al., IJCV 2020).  It charges
    a merge for the GT object it can no longer claim, and charges a missed
    detection, so "never merge" and "always merge" both cost something.

    ``association_accuracy`` is the cross-view identity F1 (precision over
    confirmed-vs-over-merged pairs, recall over GT double-view objects).  This
    task has no track IDs, so the identity half plays the role HOTA gives AssA.

    ``hota`` is ``sqrt(DetA * AssA)``.  It exists because the two halves pull in
    opposite directions: a headline that only counts coverage rewards a method
    for staying silent, and a headline that only counts association rewards it
    for merging everything.
    """
    objects = identity["objects"]
    tp, fp, fn = int(objects["tp"]), int(objects["fp"]), int(objects["fn"])
    denominator = tp + fp + fn
    detection = (tp / denominator) if denominator else 0.0
    shape = identity["gt_double_view"]
    gt_double = int(shape["total"])
    confirmed = int(identity["matched_by_shape"]["double_view"]["double"])
    over = int(identity["matched_by_shape"]["single_side"]["double"])
    precision = (confirmed / (confirmed + over)) if (confirmed + over) else 0.0
    recall = (confirmed / gt_double) if gt_double else 0.0
    association = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    product = detection * association
    return {
        "detection_accuracy": round(detection, 6),
        "association_accuracy": round(association, 6),
        "hota": round(math.sqrt(product), 6) if product > 0 else 0.0,
        "confirmed": confirmed,
        "over_merged": over,
        "gt_double_view": gt_double,
    }


def merge_rates(counter: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive precision/recall/F1 from a ``tp/fp/fn`` mapping."""
    tp, fp, fn = int(counter.get("tp", 0)), int(counter.get("fp", 0)), int(counter.get("fn", 0))
    precision, recall = rate(tp, tp + fp), rate(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(rate(2 * precision * recall, precision + recall), 6),
        "gt": tp + fn,
        "pred": tp + fp,
    }


def confusion_block(pairs: Sequence[Tuple[Optional[str], Optional[str]]]) -> Dict[str, Any]:
    """Confusion counts plus accuracy / macro-F1 / fill rate for one attribute.

    Pairs whose ground truth is empty are skipped (there is nothing to be right
    about). A ``None`` prediction is scored as a wrong value and reported again
    through ``fill_rate``.
    """
    scored = [(pred, gold) for pred, gold in pairs if gold is not None]
    total = len(scored)
    correct = sum(1 for pred, gold in scored if pred == gold)
    filled = sum(1 for pred, _ in scored if pred is not None)

    cells: Dict[str, Dict[str, int]] = {}
    for pred, gold in scored:
        row = cells.setdefault(gold, {})
        key = pred or MISSING
        row[key] = row.get(key, 0) + 1

    per_class: Dict[str, Dict[str, Any]] = {}
    f1_values: List[float] = []
    for klass in sorted(cells):
        tp = cells[klass].get(klass, 0)
        fn = sum(value for key, value in cells[klass].items() if key != klass)
        fp = sum(row.get(klass, 0) for gold, row in cells.items() if gold != klass)
        precision, recall = rate(tp, tp + fp), rate(tp, tp + fn)
        f1 = rate(2 * precision * recall, precision + recall)
        per_class[klass] = {
            "support": tp + fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        f1_values.append(f1)

    return {
        "support": total,
        "correct": correct,
        "accuracy": round(rate(correct, total), 6),
        "accuracy_on_filled": round(rate(correct, filled), 6),
        "fill_rate": round(rate(filled, total), 6),
        "missing_predictions": total - filled,
        "macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else 0.0,
        "classes": {klass: dict(sorted(row.items())) for klass, row in sorted(cells.items())},
        "per_class": per_class,
    }


def average_precision(predictions: Sequence[Tuple[str, float, bool]], num_gt: int) -> float:
    """101-point interpolated AP@50 for one class."""
    if num_gt <= 0:
        return 0.0
    recalls: List[float] = []
    precisions: List[float] = []
    tp = fp = 0
    for _image, _score, is_tp in predictions:
        if is_tp:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / float(num_gt))
        precisions.append(tp / float(tp + fp))
    total = 0.0
    for step in range(101):
        target = step / 100.0
        best = 0.0
        for recall, precision in zip(recalls, precisions):
            if recall >= target and precision > best:
                best = precision
        total += best
    return total / 101.0


# ------------------------------------------------------------- G1 detection

def match_image(
    tool_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
    iou_threshold: float = IOU_THRESHOLD,
) -> Dict[str, List[Dict[str, Any]]]:
    """Greedy highest-IoU matching for one image (one pair + one view).

    Returns ``matches`` (each with both records), ``duplicates`` (extra tool
    boxes overlapping an already-claimed truth box), ``unmatched_tool`` (FP) and
    ``unmatched_truth`` (FN).
    """
    tool = [(row, box) for row in tool_rows if (box := bbox_of(row)) is not None]
    truth = [(row, box) for row in truth_rows if (box := bbox_of(row)) is not None]

    candidates: List[Tuple[float, int, int]] = []
    for ti, (_row, tool_box) in enumerate(tool):
        for gi, (_truth_row, truth_box) in enumerate(truth):
            value = bbox_iou(tool_box, truth_box)
            if value >= iou_threshold:
                candidates.append((value, ti, gi))
    candidates.sort(reverse=True)

    used_tool: set = set()
    used_truth: set = set()
    matches: List[Dict[str, Any]] = []
    duplicate_candidates: List[Tuple[float, int]] = []
    for value, ti, gi in candidates:
        if gi in used_truth:
            if ti not in used_tool:
                duplicate_candidates.append((value, ti))
            continue
        if ti in used_tool:
            continue
        used_tool.add(ti)
        used_truth.add(gi)
        matches.append({"iou": value, "tool": tool[ti][0], "truth": truth[gi][0]})

    duplicates: List[Dict[str, Any]] = []
    seen: set = set()
    for value, ti in duplicate_candidates:
        instance_id = tool[ti][0].get("instance_id")
        if instance_id in seen:
            continue
        seen.add(instance_id)
        duplicates.append({"tool": tool[ti][0], "iou": value})

    unmatched_tool = [
        tool[ti][0] for ti in range(len(tool))
        if ti not in used_tool and tool[ti][0].get("instance_id") not in seen
    ]
    unmatched_truth = [truth[gi][0] for gi in range(len(truth)) if gi not in used_truth]
    return {
        "matches": matches,
        "duplicates": duplicates,
        "unmatched_tool": unmatched_tool,
        "unmatched_truth": unmatched_truth,
    }


def detection_metrics(
    image_records: Mapping[Tuple[str, str], Mapping[str, Any]],
    truth_class_counts: Mapping[str, int],
) -> Dict[str, Any]:
    """Aggregate G1 counts into micro/macro rates, per class AP@50 and per view."""
    totals = {"tp": 0, "fp": 0, "fn": 0}
    per_class: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_view: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    predictions: Dict[str, List[Tuple[str, float, bool]]] = defaultdict(list)
    duplicates = 0

    for (pair_id, view), record in image_records.items():
        image_key = f"{pair_id}|{view}"
        for match in record["matches"]:
            gold = normalized(match["truth"].get("category")) or "unknown"
            totals["tp"] += 1
            per_class[gold]["tp"] += 1
            per_view[view]["tp"] += 1
            predictions[gold].append((image_key, float(match["tool"].get("detection_score") or 0.0), True))
        for row in record["unmatched_tool"]:
            label = normalized(row.get("category")) or "unknown"
            totals["fp"] += 1
            per_class[label]["fp"] += 1
            per_view[view]["fp"] += 1
            predictions[label].append((image_key, float(row.get("detection_score") or 0.0), False))
        for row in record["unmatched_truth"]:
            label = normalized(row.get("category")) or "unknown"
            totals["fn"] += 1
            per_class[label]["fn"] += 1
            per_view[view]["fn"] += 1
        duplicates += len(record["duplicates"])

    per_class_block: Dict[str, Dict[str, Any]] = {}
    ap_values: List[float] = []
    ap_weights: List[int] = []
    for klass in sorted(set(per_class) | set(truth_class_counts)):
        block = merge_rates(per_class.get(klass, {"tp": 0, "fp": 0, "fn": 0}))
        ordered = sorted(predictions.get(klass, []), key=lambda item: -item[1])
        block["ap50"] = round(average_precision(ordered, block["gt"]), 6)
        per_class_block[klass] = block
        if block["gt"] > 0:
            ap_values.append(block["ap50"])
            ap_weights.append(block["gt"])

    micro = merge_rates(totals)
    macro_precision = rate(sum(item["precision"] for item in per_class_block.values()), len(per_class_block))
    macro_recall = rate(sum(item["recall"] for item in per_class_block.values()), len(per_class_block))
    return {
        "micro": micro,
        "macro": {
            "precision": round(macro_precision, 6),
            "recall": round(macro_recall, 6),
            "f1": round(rate(2 * macro_precision * macro_recall, macro_precision + macro_recall), 6),
        },
        "map50": round(sum(ap_values) / len(ap_values), 6) if ap_values else 0.0,
        "map50_gt_weighted": round(
            rate(sum(ap * weight for ap, weight in zip(ap_values, ap_weights)), sum(ap_weights)), 6
        ) if ap_weights else 0.0,
        "duplicates": duplicates,
        "duplicate_rate": round(rate(duplicates, micro["tp"] + duplicates), 6),
        "per_class": per_class_block,
        "per_view": {view: merge_rates(counter) for view, counter in sorted(per_view.items())},
    }


# ------------------------------------------------------------- G2 attributes

def attribute_metrics(
    matches: Sequence[Mapping[str, Any]],
    attributes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Attribute accuracy/confusion on matched instance pairs, overall and per view.

    ``attributes=None`` scores every candidate axis present in the workbook.
    """
    if attributes is None:
        attributes = present_attributes(
            [match["tool"] for match in matches] + [match["truth"] for match in matches]
        )
    block: Dict[str, Any] = {}
    for attribute in attributes:
        overall = [(normalized(match["tool"].get(attribute)), normalized(match["truth"].get(attribute)))
                   for match in matches]
        entry = confusion_block(overall)
        by_view: Dict[str, Any] = {}
        for view in sorted({str(match.get("view")) for match in matches}):
            subset = [
                (normalized(match["tool"].get(attribute)), normalized(match["truth"].get(attribute)))
                for match in matches if str(match.get("view")) == view
            ]
            by_view[view] = confusion_block(subset)
        entry["per_view"] = by_view
        block[attribute] = entry
    return block


# ------------------------------------------------------- object-level helpers

def instance_index(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``instance_id -> record`` for one instances sheet."""
    return {str(row.get("instance_id")): row for row in rows if row.get("instance_id")}


def object_members(row: Mapping[str, Any]) -> Dict[str, str]:
    """``{'a': instance_id, 'b': instance_id}`` for one identities row."""
    members: Dict[str, str] = {}
    for key, column in (("a", "instance_a"), ("b", "instance_b")):
        value = row.get(column)
        if value not in (None, ""):
            members[key] = str(value)
    return members


def object_attributes(
    row: Mapping[str, Any],
    instances: Mapping[str, Dict[str, Any]],
    attributes: Sequence[str] = OBJECT_ATTRIBUTE_CANDIDATES,
) -> Dict[str, Optional[str]]:
    """Object-level attributes: identity columns first, then the first member.

    ``category``/``subtype`` live on the identities sheet; ``color``,
    ``state`` and ``size_bin`` only exist per instance, so the first non-empty
    member value is used — the same rule on prediction and truth sides.
    """
    out: Dict[str, Optional[str]] = {}
    for attribute in attributes:
        value = normalized(row.get(attribute))
        if value is None:
            for member in object_members(row).values():
                record = instances.get(member) or {}
                value = normalized(record.get(attribute))
                if value is not None:
                    break
        out[attribute] = value
    return out


def object_change_label(
    row: Mapping[str, Any],
    instances: Mapping[str, Dict[str, Any]],
    stayed_ratio: float = STAYED_RATIO,
) -> Optional[str]:
    """Change label derived from the member boxes.

    ``only a`` -> ``disappeared``, ``only b`` -> ``appeared``, both members ->
    ``moved``/``stayed`` by relative centre displacement. ``None`` when a member
    box is unusable.
    """
    members = object_members(row)
    if not members:
        return None
    if "a" not in members:
        return "appeared"
    if "b" not in members:
        return "disappeared"
    box_a = bbox_of(instances.get(members["a"]) or {})
    box_b = bbox_of(instances.get(members["b"]) or {})
    if box_a is None or box_b is None:
        return None
    center_a = ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0)
    center_b = ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0)
    distance = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    diagonal_a = math.hypot(box_a[2] - box_a[0], box_a[3] - box_a[1])
    diagonal_b = math.hypot(box_b[2] - box_b[0], box_b[3] - box_b[1])
    scale = max((diagonal_a + diagonal_b) / 2.0, 1.0)
    return "moved" if distance / scale > float(stayed_ratio) else "stayed"


def change_confusion(
    pairs: Sequence[Tuple[Optional[str], Optional[str]]],
    classes: Sequence[str] = CHANGE_CLASSES,
) -> Dict[str, Any]:
    """Confusion over change labels with explicit missed/spurious buckets.

    Accuracy follows the evaluator convention
    ``correct / (GT events + spurious predictions)``.
    """
    cells: Dict[str, Counter] = {label: Counter() for label in classes}
    cells[SPURIOUS] = Counter()
    gt_support = pred_support = matched = correct = 0
    for predicted, gold in pairs:
        if gold is not None:
            gt_support += 1
        if predicted is not None:
            pred_support += 1
        if gold is None:
            if predicted is not None:
                cells[SPURIOUS][predicted] += 1
            continue
        if predicted is None:
            cells[gold][MISSED] += 1
            continue
        cells[gold][predicted] += 1
        matched += 1
        if predicted == gold:
            correct += 1

    per_class: Dict[str, Dict[str, Any]] = {}
    f1_values: List[float] = []
    for label in classes:
        tp = cells[label].get(label, 0)
        fp = sum(row.get(label, 0) for other, row in cells.items() if other != label)
        fn = sum(value for other, value in cells[label].items() if other != label)
        precision, recall = rate(tp, tp + fp), rate(tp, tp + fn)
        f1 = rate(2 * precision * recall, precision + recall)
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        f1_values.append(f1)

    total = gt_support + pred_support - matched
    return {
        "support": total,
        "gt_support": gt_support,
        "prediction_support": pred_support,
        "matched": matched,
        "missed": gt_support - matched,
        "spurious": pred_support - matched,
        "correct": correct,
        "accuracy": round(rate(correct, total), 6),
        "conditional_accuracy": round(rate(correct, matched), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        "confusion": {label: dict(sorted(row.items())) for label, row in cells.items()},
        "per_class": per_class,
    }


# ------------------------------------------------- G5 identity + G4 and G6

def identity_metrics(
    tool_objects: Sequence[Mapping[str, Any]],
    truth_objects: Sequence[Mapping[str, Any]],
    bridge: Mapping[str, str],
    tool_instances: Mapping[str, Dict[str, Any]],
    truth_instances: Mapping[str, Dict[str, Any]],
    family: str,
    per_pair: Optional[Dict[str, Any]] = None,
    jaccard_floor: float = 0.0,
    attributes: Optional[Sequence[str]] = None,
    truth_change_from_annotation: bool = True,
) -> Dict[str, Any]:
    """Object-level identity matching, G4 end-to-end attributes and G6 change.

    A predicted object resolves to truth instances through ``bridge`` (the
    instance-level matches). It matches a truth object when the member-set
    Jaccard is at least ``jaccard_floor``; matching is greedy per image pair,
    highest overlap first.

    ``jaccard_floor`` defaults to ``0.0``, which is the historical behaviour used
    by every published number: any overlap counts, so a predicted double-view
    object whose second member is a different GT object (Jaccard 1/3) or an
    unmatched detection (Jaccard 1/2) still resolves a GT double-view object.
    Pass ``0.5`` to apply the floor documented here, which drops the 1/3 cases.

    Two additional counters are reported next to the rates so the strict reading
    is available without changing any existing number:

    ``predicted_exact``
        predicted double-view objects whose member set is exactly the GT pair
        (Jaccard 1.0) — a genuinely recovered cross-view identity.
    ``association_recall_exact``
        ``predicted_exact / GT double-view objects``.

    ``attributes=None`` scores every candidate object axis present in the
    workbooks. ``truth_change_from_annotation`` reads the truth side's
    ``temporal_relation`` column (the annotation) instead of deriving the change
    label from box displacement; the prediction side has no such column and is
    always derived.
    """
    if attributes is None:
        attributes = present_attributes(
            list(tool_instances.values()) + list(truth_instances.values()),
            OBJECT_ATTRIBUTE_CANDIDATES,
        )
    predicted_by_pair: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in tool_objects:
        predicted_by_pair[str(row.get("pair_id"))].append(row)
    truth_by_pair: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in truth_objects:
        truth_by_pair[str(row.get("pair_id"))].append(row)

    totals = {"tp": 0, "fp": 0, "fn": 0}
    # GT bucket x predicted shape. ``double``/``single`` are predicted shapes,
    # ``missed`` means no prediction was matched to that truth object.
    matrix: Dict[str, Dict[str, int]] = {
        "double_view": {"double": 0, "single": 0, "missed": 0},
        "single_side": {"double": 0, "single": 0, "missed": 0},
    }
    exact_double = 0
    change_pairs: List[Tuple[Optional[str], Optional[str]]] = []
    end_to_end: Dict[str, List[Tuple[Optional[str], Optional[str]]]] = {
        attribute: [] for attribute in attributes
    }
    coverage_end_to_end: Dict[str, List[Tuple[Optional[str], Optional[str]]]] = {
        attribute: [] for attribute in attributes
    }
    coverage_totals = {"tp": 0, "fp": 0, "fn": 0}
    hota_pair_counts: Dict[str, Any] = {}

    pair_rows: Dict[str, Dict[str, Any]] = {}

    for pair_id in sorted(set(predicted_by_pair) | set(truth_by_pair)):
        predicted = predicted_by_pair.get(pair_id, [])
        truth = truth_by_pair.get(pair_id, [])

        resolved: List[set] = []
        for row in predicted:
            members = object_members(row)
            resolved.append({bridge[member] for member in members.values() if member in bridge})

        candidates: List[Tuple[float, int, int]] = []
        for pi, pred_set in enumerate(resolved):
            if not pred_set:
                continue
            for gi, truth_row in enumerate(truth):
                truth_set = set(object_members(truth_row).values())
                if not truth_set:
                    continue
                inter = len(pred_set & truth_set)
                if not inter:
                    continue
                jaccard = inter / len(pred_set | truth_set)
                if jaccard < float(jaccard_floor):
                    continue
                candidates.append((jaccard, pi, gi))
        candidates.sort(reverse=True)

        chosen: List[Tuple[int, int]] = []
        used_pred: set = set()
        used_truth: set = set()
        for _score, pi, gi in candidates:
            if pi in used_pred or gi in used_truth:
                continue
            used_pred.add(pi)
            used_truth.add(gi)
            chosen.append((pi, gi))
        mapping = dict(chosen)
        score_of = {pi: score for score, pi, gi in candidates if mapping.get(pi) == gi}

        # Coverage-aware (part-aware) claims: a fused prediction whose members
        # belong to two different GT objects is allowed to claim both, one per
        # member.  This is the panoptic-per-part view of a merge, and it is
        # reported *next to* the official one-to-one numbers, never instead of
        # them: it isolates how much of a coverage loss is protocol rather than
        # missing evidence.
        gt_members = {
            gi: set(object_members(truth_row).values())
            for gi, truth_row in enumerate(truth)
        }
        claimed_gt = set(mapping.values())   # mapping is predicted -> GT
        part_owner: Dict[int, int] = {}
        for gi, members in gt_members.items():
            if gi in claimed_gt or not members:
                continue
            best_pi, best_overlap = None, 0
            for pi, pred_set in enumerate(resolved):
                overlap = len(pred_set & members)
                if overlap > best_overlap:
                    best_pi, best_overlap = pi, overlap
            if best_pi is not None:
                part_owner[gi] = best_pi
        coverage_tp = len(used_truth) + len(part_owner)
        coverage_owners = set(used_pred) | set(part_owner.values())
        coverage_fp = len(predicted) - len(coverage_owners)

        # Official HOTA (TrackEval definition): per-pair per-threshold counters,
        # aggregated over the dataset before any ratio is taken.
        from DS_VI.experiments.metrics.hota import pair_hota

        hota_pair_counts[pair_id] = pair_hota(
            predicted, truth, tool_instances, truth_instances
        )

        coverage_totals["tp"] += coverage_tp
        coverage_totals["fp"] += coverage_fp
        coverage_totals["fn"] += len(truth) - coverage_tp

        for gi, truth_row in enumerate(truth):
            bucket = "double_view" if len(object_members(truth_row)) == 2 else "single_side"
            if gi in used_truth:
                totals["tp"] += 1
                owner = next(pi for pi, target in mapping.items() if target == gi)
                shape = "double" if len(object_members(predicted[owner])) == 2 else "single"
                matrix[bucket][shape] += 1
                if bucket == "double_view" and shape == "double" and score_of.get(owner, 0.0) >= 1.0:
                    exact_double += 1
            else:
                totals["fn"] += 1
                matrix[bucket]["missed"] += 1

        for pi, pred_row in enumerate(predicted):
            if pi in used_pred:
                continue
            totals["fp"] += 1

        if per_pair is not None:
            row = pair_rows.setdefault(pair_id, {
                "gt_double": 0, "double_confirmed": 0, "over_merged": 0,
                "single_true_pred": 0, "gt_single": 0,
                "det_tp": 0, "det_fp": 0, "det_fn": 0,
            })
            row["det_tp"] = len(used_truth)
            row["det_fp"] = len(predicted) - len(used_pred)
            row["det_fn"] = len(truth) - len(used_truth)
            row["hota_counts"] = hota_pair_counts[pair_id]
            row["cov_tp"] = coverage_tp
            row["cov_fp"] = coverage_fp
            row["cov_fn"] = len(truth) - coverage_tp
            for gi, truth_row in enumerate(truth):
                bucket = "double_view" if len(object_members(truth_row)) == 2 else "single_side"
                owner = next((pi for pi, target in mapping.items() if target == gi), None)
                shape = "double" if owner is not None and len(object_members(predicted[owner])) == 2 else "single"
                if bucket == "double_view":
                    row["gt_double"] += 1
                    if owner is not None and shape == "double":
                        row["double_confirmed"] += 1
                else:
                    row["gt_single"] += 1
                    if owner is not None and shape == "single":
                        row["single_true_pred"] += 1
                    elif owner is not None and shape == "double":
                        row["over_merged"] += 1

        if family == "time":
            for gi, truth_row in enumerate(truth):
                gold = (
                    annotated_change_label(truth_row) if truth_change_from_annotation else None
                ) or object_change_label(truth_row, truth_instances)
                owner = next((pi for pi, target in mapping.items() if target == gi), None)
                predicted_label = None
                if owner is not None:
                    predicted_label = (
                        reported_change_label(predicted[owner])
                        or object_change_label(predicted[owner], tool_instances)
                    )
                change_pairs.append((predicted_label, gold))
            for pi, pred_row in enumerate(predicted):
                if pi in mapping:
                    continue
                change_pairs.append((
                    reported_change_label(pred_row)
                    or object_change_label(pred_row, tool_instances),
                    None,
                ))

        predicted_attrs_cov = {
            pi: object_attributes(row, tool_instances, attributes)
            for pi, row in enumerate(predicted)
        }
        for gi, truth_row in enumerate(truth):
            # ``mapping`` is pred -> GT, so the owner has to be looked up by value.
            owner = next((pi for pi, target in mapping.items() if target == gi), None)
            if owner is None:
                owner = part_owner.get(gi)
            gold = object_attributes(truth_row, truth_instances, attributes)
            for attribute in attributes:
                coverage_end_to_end[attribute].append(
                    (predicted_attrs_cov[owner][attribute] if owner is not None else None,
                     gold[attribute])
                )

        attribute_start = {attribute: len(end_to_end[attribute]) for attribute in attributes}
        predicted_attributes = {
            pi: object_attributes(row, tool_instances, attributes) for pi, row in enumerate(predicted)
        }
        for gi, truth_row in enumerate(truth):
            gold = object_attributes(truth_row, truth_instances, attributes)
            owner = next((pi for pi, target in mapping.items() if target == gi), None)
            for attribute in attributes:
                predicted_value = predicted_attributes[owner][attribute] if owner is not None else None
                end_to_end[attribute].append((predicted_value, gold[attribute]))

        if per_pair is not None:
            row = pair_rows.setdefault(pair_id, {})
            for attribute in attributes:
                values = end_to_end[attribute][attribute_start[attribute]:]
                row[f"g4_{attribute}_correct"] = sum(
                    1 for predicted_value, gold in values
                    if predicted_value is not None and gold is not None and predicted_value == gold)
                row[f"g4_{attribute}_total"] = len(values)

    if per_pair is not None:
        change_rows = change_pairs if family == "time" else []
        per_pair["pairs"] = pair_rows
        per_pair["change"] = {
            "pairs": len(change_rows),
            "correct": sum(1 for predicted_label, gold in change_rows
                           if predicted_label is not None and gold is not None
                           and predicted_label == gold),
            "total": len(change_rows),
        }

    def bucket_block(counts: Mapping[str, int]) -> Dict[str, Any]:
        # ``exact`` is an extra counter, not a shape bucket: keep it out of the totals.
        total = sum(counts.get(key, 0) for key in ("double", "single", "missed"))
        predicted = counts["double"] + counts["single"]
        return {
            "total": total,
            "predicted_double_view": counts["double"],
            "predicted_single_side": counts["single"],
            "missed": counts["missed"],
            "detection_recall": round(rate(predicted, total), 6),
            "association_recall": round(rate(counts["double"], total), 6),
            "over_merged": round(rate(counts["double"], predicted), 6),
            "predicted_exact": counts.get("exact", 0),
            "association_recall_exact": round(rate(counts.get("exact", 0), total), 6),
        }

    double_counts = dict(matrix["double_view"])
    double_counts["exact"] = exact_double
    identity = {
        "objects": merge_rates(totals),
        "matched_by_shape": {bucket: dict(counts) for bucket, counts in matrix.items()},
        "gt_double_view": bucket_block(double_counts),
        "gt_single_side": bucket_block(matrix["single_side"]),
        "view_pattern_errors": matrix["double_view"]["single"] + matrix["single_side"]["double"],
        "view_pattern_error_rate": round(rate(
            matrix["double_view"]["single"] + matrix["single_side"]["double"], totals["tp"]
        ), 6),
        "jaccard_floor": float(jaccard_floor),
    }
    identity["composite"] = composite_association(identity)
    from DS_VI.experiments.metrics.hota import aggregate as hota_aggregate

    identity["hota_official"] = hota_aggregate(
        [hota_pair_counts[key] for key in sorted(hota_pair_counts)]
    )
    coverage_denominator = (
        coverage_totals["tp"] + coverage_totals["fp"] + coverage_totals["fn"]
    )
    coverage_detection = (
        coverage_totals["tp"] / coverage_denominator if coverage_denominator else 0.0
    )
    coverage_association = identity["composite"]["association_accuracy"]
    coverage_product = coverage_detection * coverage_association
    identity["coverage_aware"] = {
        "tp": coverage_totals["tp"],
        "fp": coverage_totals["fp"],
        "fn": coverage_totals["fn"],
        "detection_accuracy": round(coverage_detection, 6),
        "hota": round(math.sqrt(coverage_product), 6) if coverage_product > 0 else 0.0,
        "end_to_end": {
            attribute: confusion_block(values)
            for attribute, values in coverage_end_to_end.items()
        },
    }

    change = None
    if family == "time":
        collapsed = [
            (None if predicted is None else
             ("persisted" if predicted in ("stayed", "moved") else predicted),
             None if gold is None else
             ("persisted" if gold in ("stayed", "moved") else gold))
            for predicted, gold in change_pairs
        ]
        change = {
            "three_class": change_confusion(collapsed, ("persisted", "appeared", "disappeared")),
            "four_class": change_confusion(change_pairs),
        }

    return {
        "identity": identity,
        "change": change,
        "end_to_end": {attribute: confusion_block(values) for attribute, values in end_to_end.items()},
    }


# ------------------------------------------------------------------ evaluate

def evaluate_family(
    tool_path: Path,
    truth_path: Path,
    family: str,
    iou_threshold: float = IOU_THRESHOLD,
    per_pair: bool = False,
    jaccard_floor: float = 0.0,
    truth_change_from_annotation: bool = True,
) -> Dict[str, Any]:
    """Run G1/G2/G4/G5/G6 for one family.

    ``jaccard_floor`` is forwarded to :func:`identity_metrics`; ``0.0`` keeps the
    historical object matching (any overlap), ``0.5`` applies the floor the
    docstring documents. Attribute axes are discovered from the workbooks, so a
    reissued annotation sheet is scored as-is instead of reporting an absent axis
    with ``support = 0``.
    """
    tool = read_book(Path(tool_path))
    truth = read_book(Path(truth_path))

    tool_images: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in tool["instances"]:
        tool_images[(str(row.get("pair_id")), str(row.get("view")))].append(row)
    truth_images: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in truth["instances"]:
        truth_images[(str(row.get("pair_id")), str(row.get("view")))].append(row)

    tool_pairs = {str(row.get("pair_id")) for row in tool["instances"]}
    truth_pairs = {str(row.get("pair_id")) for row in truth["instances"]}
    shared_pairs = tool_pairs & truth_pairs

    image_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
    matches: List[Dict[str, Any]] = []
    bridge: Dict[str, str] = {}
    for key in sorted(set(tool_images) | set(truth_images)):
        pair_id, view = key
        if pair_id not in shared_pairs:
            continue
        record = match_image(tool_images.get(key, []), truth_images.get(key, []), iou_threshold)
        image_records[key] = record
        for match in record["matches"]:
            match["pair_id"] = pair_id
            match["view"] = view
            matches.append(match)
            bridge[str(match["tool"].get("instance_id"))] = str(match["truth"].get("instance_id"))

    truth_class_counts: Dict[str, int] = Counter(
        normalized(row.get("category")) or "unknown"
        for row in truth["instances"]
        if str(row.get("pair_id")) in shared_pairs
    )

    detection = detection_metrics(image_records, truth_class_counts)
    instance_axes = present_attributes(
        list(tool["instances"]) + list(truth["instances"]), INSTANCE_ATTRIBUTE_CANDIDATES
    )
    object_axes = present_attributes(
        list(tool["instances"])
        + list(truth["instances"])
        + list(tool.get("identities", []))
        + list(truth.get("identities", [])),
        OBJECT_ATTRIBUTE_CANDIDATES,
    )
    attributes = attribute_metrics(matches, instance_axes)
    per_pair_block: Dict[str, Any] = {}
    object_block = identity_metrics(
        [row for row in tool.get("identities", []) if str(row.get("pair_id")) in shared_pairs],
        [row for row in truth.get("identities", []) if str(row.get("pair_id")) in shared_pairs],
        bridge,
        instance_index(tool["instances"]),
        instance_index(truth["instances"]),
        family,
        per_pair=per_pair_block if per_pair else None,
        jaccard_floor=float(jaccard_floor),
        attributes=object_axes,
        truth_change_from_annotation=truth_change_from_annotation,
    )

    samples_block: Dict[str, Any] = {}
    for tag, book in (("truth", truth), ("tool", tool)):
        rows = book.get("samples") or []
        samples_block[tag] = {
            "samples": len(rows),
            "identity_incomplete": [
                str(row.get("pair_id")) for row in rows
                if str(row.get("identity_complete")).strip().lower() not in ("true", "1")
            ],
            "notes": [
                {"pair_id": row.get("pair_id"), "note": row.get("note")}
                for row in rows if row.get("note")
            ],
        }

    return {
        "family": family,
        "scope": {
            "tool_workbook": str(tool_path),
            "truth_workbook": str(truth_path),
            "iou_threshold": float(iou_threshold),
            "jaccard_floor": float(jaccard_floor),
            "duplicate_protocol": "reported separately, not counted as FP",
            "tool_pairs": len(tool_pairs),
            "truth_pairs": len(truth_pairs),
            "shared_pairs": len(shared_pairs),
            "tool_only_pairs": sorted(tool_pairs - truth_pairs),
            "truth_only_pairs": sorted(truth_pairs - tool_pairs),
            "tool_instances": len(tool["instances"]),
            "truth_instances": len(truth["instances"]),
            "tool_objects": len(tool.get("identities", [])),
            "truth_objects": len(truth.get("identities", [])),
            "matched_instances": len(bridge),
        },
        "G1_detection": detection,
        "G2_attributes": attributes,
        "G5_identity": object_block["identity"],
        "G4_end_to_end": object_block["end_to_end"],
        "G6_change": object_block["change"],
        "samples": samples_block,
        **( {"per_pair": per_pair_block} if per_pair else {} ),
    }


# ------------------------------------------------------------------- report

def flatten(metrics: Mapping[str, Any]) -> List[Tuple[Any, ...]]:
    """Flat ``(group, metric, value)`` rows for the summary sheet and CSV."""
    rows: List[Tuple[Any, ...]] = []

    def push(group: str, metric: str, value: Any) -> None:
        rows.append((group, metric, value))

    for group in ("G1_detection", "G2_attributes", "G5_identity", "G4_end_to_end"):
        block = metrics.get(group)
        if not block:
            continue
        if group == "G2_attributes":
            for attribute, item in block.items():
                for key in ("accuracy", "accuracy_on_filled", "fill_rate", "macro_f1", "support"):
                    push(f"G2|{attribute}", key, item[key])
                for view, view_item in item["per_view"].items():
                    push(f"G2|{attribute}|{view}", "accuracy", view_item["accuracy"])
                    push(f"G2|{attribute}|{view}", "fill_rate", view_item["fill_rate"])
                    push(f"G2|{attribute}|{view}", "support", view_item["support"])
            continue
        if group == "G4_end_to_end":
            for attribute, item in block.items():
                for key in ("accuracy", "macro_f1", "support"):
                    push(f"G4|{attribute}", key, item[key])
            continue
        if group == "G5_identity":
            composite = block.get("composite") or {}
            for key in ("detection_accuracy", "association_accuracy", "hota"):
                if key in composite:
                    push("G5|composite", key, composite[key])
            official = block.get("hota_official") or {}
            for key in ("hota", "deta", "assa"):
                if key in official:
                    push("G5|hota_official", key, official[key])
            coverage = block.get("coverage_aware") or {}
            for key in ("tp", "fp", "fn", "detection_accuracy", "hota"):
                if key in coverage:
                    push("G5|coverage_aware", key, coverage[key])
            for attribute, item in (coverage.get("end_to_end") or {}).items():
                push(f"G4cov|{attribute}", "accuracy", item["accuracy"])
            for part, item in block.items():
                if part == "composite":
                    continue
                if isinstance(item, dict) and "f1" in item:
                    for key in ("precision", "recall", "f1", "tp", "fp", "fn"):
                        push(f"G5|{part}", key, item[key])
                elif isinstance(item, dict) and "detection_recall" in item:
                    for key in ("total", "predicted_double_view", "predicted_single_side",
                                "missed", "detection_recall", "association_recall", "over_merged",
                                "predicted_exact", "association_recall_exact"):
                        push(f"G5|{part}", key, item[key])
                elif isinstance(item, (int, float)):
                    push(f"G5|{part}", "value", item)
            continue
        for key, value in block.items():
            if isinstance(value, (int, float)):
                push(group, key, value)
            elif isinstance(value, dict) and "f1" in value:
                for part in ("precision", "recall", "f1", "tp", "fp", "fn"):
                    if part in value:
                        push(f"{group}|{key}", part, value[part])

    change = metrics.get("G6_change")
    if change:
        for protocol, item in change.items():
            for key in ("accuracy", "conditional_accuracy", "macro_f1", "support",
                        "gt_support", "prediction_support", "missed", "spurious"):
                push(f"G6|{protocol}", key, item[key])
    return rows


def write_report(metrics: Mapping[str, Any], output_dir: Path) -> Dict[str, Path]:
    """Write ``metrics.json``, ``summary.csv`` and a formatted workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "metric", "value"])
        writer.writerows(flatten(metrics))

    workbook = Workbook()
    bold = Font(bold=True)

    summary = workbook.active
    summary.title = "汇总"
    summary.append(["组", "指标", "值"])
    for cell in summary[1]:
        cell.font = bold
    for row in flatten(metrics):
        summary.append(list(row))
    summary.column_dimensions["A"].width = 34
    summary.column_dimensions["B"].width = 22
    summary.column_dimensions["C"].width = 16

    def add_sheet(title: str, rows: Sequence[Sequence[Any]], widths: Sequence[int]) -> None:
        sheet = workbook.create_sheet(title)
        sheet.append(list(rows[0]))
        for cell in sheet[1]:
            cell.font = bold
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for row in rows[1:]:
            sheet.append(list(row))
        for index, width in enumerate(widths, start=1):
            sheet.column_dimensions[get_column_letter(index)].width = width

    detection = metrics["G1_detection"]
    rows: List[Sequence[Any]] = [["类别", "TP", "FP", "FN", "P", "R", "F1", "AP@50", "GT"]]
    for klass, item in sorted(detection["per_class"].items()):
        rows.append([klass, item["tp"], item["fp"], item["fn"], item["precision"],
                     item["recall"], item["f1"], item["ap50"], item["gt"]])
    micro = detection["micro"]
    rows.append(["micro", micro["tp"], micro["fp"], micro["fn"], micro["precision"],
                 micro["recall"], micro["f1"], detection["map50"], micro["gt"]])
    add_sheet("G1_检测", rows, [14, 8, 8, 8, 10, 10, 10, 10, 8])

    rows = [["视图", "TP", "FP", "FN", "P", "R", "F1"]]
    for view, item in sorted(detection["per_view"].items()):
        rows.append([view, item["tp"], item["fp"], item["fn"], item["precision"],
                     item["recall"], item["f1"]])
    add_sheet("G1_按视图", rows, [12, 8, 8, 8, 10, 10, 10])

    rows = [["属性", "视图", "support", "accuracy", "accuracy_on_filled",
             "fill_rate", "macro_F1", "缺失预测"]]
    for attribute, item in metrics["G2_attributes"].items():
        rows.append([attribute, "全部", item["support"], item["accuracy"],
                     item["accuracy_on_filled"], item["fill_rate"], item["macro_f1"],
                     item["missing_predictions"]])
        for view, view_item in item["per_view"].items():
            rows.append([attribute, view, view_item["support"], view_item["accuracy"],
                         view_item["accuracy_on_filled"], view_item["fill_rate"],
                         view_item["macro_f1"], view_item["missing_predictions"]])
    add_sheet("G2_属性", rows, [14, 10, 10, 12, 18, 12, 12, 12])

    rows = [["属性", "GT 类别", "预测类别", "数量"]]
    for attribute, item in metrics["G2_attributes"].items():
        for gold, counts in item["classes"].items():
            for predicted, count in counts.items():
                rows.append([attribute, gold, predicted, count])
    add_sheet("G2_混淆", rows, [14, 22, 22, 10])

    rows = [["属性", "support", "accuracy", "macro_F1"]]
    for attribute, item in metrics["G4_end_to_end"].items():
        rows.append([attribute, item["support"], item["accuracy"], item["macro_f1"]])
    add_sheet("G4_端到端", rows, [14, 10, 12, 12])

    identity = metrics["G5_identity"]
    rows = [["层级", "TP", "FP", "FN", "P", "R", "F1", "GT", "Pred"]]
    objects = identity["objects"]
    rows.append(["全部对象", objects["tp"], objects["fp"], objects["fn"], objects["precision"],
                 objects["recall"], objects["f1"], objects["gt"], objects["pred"]])
    for tag, key in (("双视图对象", "gt_double_view"), ("单侧对象", "gt_single_side")):
        item = identity[key]
        rows.append([tag, item["predicted_double_view"], "-", item["missed"], "-",
                     item["detection_recall"], item["association_recall"], item["total"], "-"])
    add_sheet("G5_身份", rows, [16, 8, 8, 8, 10, 18, 18, 8, 8])

    rows = [["GT 形态", "预测双视图", "预测单侧", "漏检", "合计"]]
    for tag, key in (("双视图", "gt_double_view"), ("单侧", "gt_single_side")):
        item = identity[key]
        rows.append([tag, item["predicted_double_view"], item["predicted_single_side"],
                     item["missed"], item["total"]])
    add_sheet("G5_形态矩阵", rows, [14, 14, 12, 10, 10])

    change = metrics.get("G6_change")
    if change:
        rows = [["协议", "support", "accuracy", "conditional_accuracy", "macro_F1",
                 "GT 事件", "预测事件", "正确", "漏检", "误报"]]
        for protocol, item in change.items():
            rows.append([protocol, item["support"], item["accuracy"], item["conditional_accuracy"],
                         item["macro_f1"], item["gt_support"], item["prediction_support"],
                         item["correct"], item["missed"], item["spurious"]])
        add_sheet("G6_变化汇总", rows, [16, 10, 12, 18, 12, 10, 12, 10, 10, 10])

        rows = [["协议", "GT 类别", "预测类别", "数量"]]
        for protocol, item in change.items():
            for gold, counts in item["confusion"].items():
                for predicted, count in counts.items():
                    rows.append([protocol, gold, predicted, count])
        add_sheet("G6_变化混淆", rows, [14, 20, 20, 10])

    rows = [["项目", "值"]]
    for key, value in metrics["scope"].items():
        rows.append([key, json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value])
    for tag, item in metrics["samples"].items():
        rows.append([f"{tag}_samples", item["samples"]])
        rows.append([f"{tag}_identity_incomplete", json.dumps(item["identity_incomplete"], ensure_ascii=False)])
        rows.append([f"{tag}_notes", json.dumps(item["notes"], ensure_ascii=False)])
    add_sheet("口径与数据", rows, [32, 90])

    xlsx_path = output_dir / "property_metrics_report.xlsx"
    workbook.save(xlsx_path)
    return {"json": json_path, "csv": csv_path, "xlsx": xlsx_path}


def print_summary(metrics: Mapping[str, Any]) -> None:
    scope = metrics["scope"]
    print(f"[{metrics['family']}] pairs shared={scope['shared_pairs']} "
          f"(tool={scope['tool_pairs']}, truth={scope['truth_pairs']}) | "
          f"instances tool={scope['tool_instances']} truth={scope['truth_instances']} | "
          f"objects tool={scope['tool_objects']} truth={scope['truth_objects']}")
    detection = metrics["G1_detection"]
    micro = detection["micro"]
    print(f"G1 detection  P={micro['precision']:.4f} R={micro['recall']:.4f} "
          f"F1={micro['f1']:.4f} mAP@50={detection['map50']:.4f} | "
          f"duplicates={detection['duplicates']} rate={detection['duplicate_rate']:.4f}")
    print("G2 attributes (instance level)")
    for attribute, item in metrics["G2_attributes"].items():
        views = " ".join(f"{view}={view_item['accuracy']:.3f}"
                         for view, view_item in item["per_view"].items())
        print(f"  {attribute:12s} acc={item['accuracy']:.4f} acc(filled)={item['accuracy_on_filled']:.4f} "
              f"fill={item['fill_rate']:.4f} macroF1={item['macro_f1']:.4f} n={item['support']} | {views}")
    identity = metrics["G5_identity"]
    objects = identity["objects"]
    double = identity["gt_double_view"]
    single = identity["gt_single_side"]
    print(f"G5 identity   objects P={objects['precision']:.4f} R={objects['recall']:.4f} "
          f"F1={objects['f1']:.4f} | double-view: detect={double['detection_recall']:.4f} "
          f"associate={double['association_recall']:.4f} | single-side: detect={single['detection_recall']:.4f} "
          f"| pattern_errors={identity['view_pattern_errors']}")
    print("G4 end-to-end (object level)")
    for attribute, item in metrics["G4_end_to_end"].items():
        print(f"  {attribute:12s} acc={item['accuracy']:.4f} macroF1={item['macro_f1']:.4f} "
              f"n={item['support']}")
    change = metrics.get("G6_change")
    if change:
        for protocol, item in change.items():
            print(f"G6 {protocol:12s} acc={item['accuracy']:.4f} "
                  f"macroF1={item['macro_f1']:.4f} n={item['support']} "
                  f"(missed={item['missed']}, spurious={item['spurious']})")


def comparison_rows(metrics_by_family: Mapping[str, Mapping[str, Any]]) -> List[List[Any]]:
    """Metric rows side by side, one column per family."""
    families = sorted(metrics_by_family)
    rows: List[List[Any]] = [["指标"] + families]

    def add(label: str, values: Sequence[Any]) -> None:
        rows.append([label] + [round(value, 6) if isinstance(value, float) else value for value in values])

    add("共享图对", [metrics_by_family[f]["scope"]["shared_pairs"] for f in families])
    add("工具实例数", [metrics_by_family[f]["scope"]["tool_instances"] for f in families])
    add("真值实例数", [metrics_by_family[f]["scope"]["truth_instances"] for f in families])
    add("工具对象数", [metrics_by_family[f]["scope"]["tool_objects"] for f in families])
    add("真值对象数", [metrics_by_family[f]["scope"]["truth_objects"] for f in families])

    for key in ("precision", "recall", "f1"):
        add(f"G1 检测 {key}", [metrics_by_family[f]["G1_detection"]["micro"][key] for f in families])
    add("G1 mAP@50", [metrics_by_family[f]["G1_detection"]["map50"] for f in families])
    add("G1 重复框率", [metrics_by_family[f]["G1_detection"]["duplicate_rate"] for f in families])

    for attribute in sorted(set().union(*(metrics_by_family[f]["G2_attributes"] for f in families))):
        add(f"G2 {attribute} acc", [metrics_by_family[f]["G2_attributes"][attribute]["accuracy"] for f in families])
        add(f"G2 {attribute} fill", [metrics_by_family[f]["G2_attributes"][attribute]["fill_rate"] for f in families])
        add(f"G2 {attribute} macroF1", [metrics_by_family[f]["G2_attributes"][attribute]["macro_f1"] for f in families])

    for key in ("precision", "recall", "f1"):
        add(f"G5 对象 {key}", [metrics_by_family[f]["G5_identity"]["objects"][key] for f in families])
    add("G5 双视图检出率", [metrics_by_family[f]["G5_identity"]["gt_double_view"]["detection_recall"] for f in families])
    add("G5 双视图关联率", [metrics_by_family[f]["G5_identity"]["gt_double_view"]["association_recall"] for f in families])
    add("G5 单侧检出率", [metrics_by_family[f]["G5_identity"]["gt_single_side"]["detection_recall"] for f in families])
    add("G5 形态错误数", [metrics_by_family[f]["G5_identity"]["view_pattern_errors"] for f in families])

    for attribute in sorted(set().union(*(metrics_by_family[f]["G4_end_to_end"] for f in families))):
        add(f"G4 {attribute} acc", [metrics_by_family[f]["G4_end_to_end"][attribute]["accuracy"] for f in families])
        add(f"G4 {attribute} macroF1", [metrics_by_family[f]["G4_end_to_end"][attribute]["macro_f1"] for f in families])

    for protocol in ("three_class", "four_class"):
        add(f"G6 {protocol} acc", [
            metrics_by_family[f]["G6_change"][protocol]["accuracy"]
            if metrics_by_family[f].get("G6_change") else ""
            for f in families
        ])
        add(f"G6 {protocol} macroF1", [
            metrics_by_family[f]["G6_change"][protocol]["macro_f1"]
            if metrics_by_family[f].get("G6_change") else ""
            for f in families
        ])
    add("（G6 仅 time 族适用）", ["" for _ in families])
    return rows


def write_comparison(metrics_by_family: Mapping[str, Mapping[str, Any]], path: Path) -> Path:
    """Write the cross-task comparison workbook."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "对比"
    rows = comparison_rows(metrics_by_family)
    sheet.append(rows[0])
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in rows[1:]:
        sheet.append(row)
    sheet.column_dimensions[get_column_letter(1)].width = 28
    for index in range(2, 2 + len(metrics_by_family)):
        sheet.column_dimensions[get_column_letter(index)].width = 16

    for family, metrics in sorted(metrics_by_family.items()):
        report = workbook.create_sheet(f"{family}_明细")
        report.append(["组", "指标", "值"])
        for cell in report[1]:
            cell.font = Font(bold=True)
        for row in flatten(metrics):
            report.append(list(row))
        report.column_dimensions["A"].width = 34
        report.column_dimensions["B"].width = 24
        report.column_dimensions["C"].width = 18

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILY_FILES) + ["both"], default="both")
    parser.add_argument("--tool", type=Path, help="tool workbook (default: dataset root)")
    parser.add_argument("--truth", type=Path, help="truth workbook (default: dataset root)")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument(
        "--jaccard-floor", type=float, default=0.0,
        help="object-level member-set Jaccard floor (0.0 = historical 'any overlap', "
             "0.5 = the floor documented in identity_metrics)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--derive-change-labels", action="store_true",
        help="derive G6 truth labels from box displacement instead of reading the "
             "annotated identities.temporal_relation column (pre-2026-11 protocol)")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of a summary")
    args = parser.parse_args(argv)

    if args.tool or args.truth:
        if args.family == "both":
            parser.error("--tool/--truth need an explicit --family (match or time)")

    families = sorted(FAMILY_FILES) if args.family == "both" else [args.family]
    output_root = args.output_dir or (ROOT / "DS_VI/results/property_eval")

    metrics_by_family: Dict[str, Dict[str, Any]] = {}
    for family in families:
        tool_name, truth_name, _views = FAMILY_FILES[family]
        tool_path = args.tool or (args.dataset_root / tool_name)
        truth_path = args.truth or (args.dataset_root / truth_name)
        for path in (tool_path, truth_path):
            if not Path(path).is_file():
                parser.error(f"workbook not found: {path}")
        metrics = evaluate_family(
            Path(tool_path), Path(truth_path), family, args.iou,
            jaccard_floor=float(args.jaccard_floor),
            truth_change_from_annotation=not args.derive_change_labels)
        metrics_by_family[family] = metrics
        paths = write_report(metrics, Path(output_root) / family)
        if not args.json:
            print_summary(metrics)
            print(f"wrote {paths['json']}")
            print(f"wrote {paths['xlsx']}\n")

    if len(metrics_by_family) > 1:
        path = write_comparison(metrics_by_family, Path(output_root) / "property_metrics_report.xlsx")
        if not args.json:
            print(f"wrote {path}")

    if args.json:
        print(json.dumps(
            metrics_by_family[families[0]] if len(families) == 1 else metrics_by_family,
            ensure_ascii=False, indent=2,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
