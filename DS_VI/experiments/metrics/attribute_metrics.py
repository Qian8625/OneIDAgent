"""Attribute metrics for the cascade-v2 experiment (mechanism-replay attributes).

Two views are reported per attribute:

* conditional: among kept predictions whose member observations map to one
  ctx object, the top-1 attribute posterior vs the workbook gold value;
* end-to-end: over every gold ``object|attribute|time`` key, whether the
  object was identified and its attribute correct (misses count as wrong).

An optional ``object_filter`` restricts the gold keys to the objects covered by
the D-S matched candidates (isolating VI's role from the full GT set).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from DS_VI.models.cascade import observation_id_for


def _rate(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def attribute_metrics(
    ctx,
    family: str,
    kept_by_sample: Mapping[str, Sequence[Mapping[str, Any]]],
    attributes: Sequence[str],
    object_filter: Optional[Mapping[str, set]] = None,
) -> Dict[str, Any]:
    per_attr = {
        attribute: {
            "cond_total": 0,
            "cond_correct": 0,
            "gold_total": 0,
            "gold_correct": 0,
            "class_tp": Counter(),
            "class_fp": Counter(),
            "class_fn": Counter(),
        }
        for attribute in attributes
    }
    coverage = {
        "samples": 0,
        "objects": 0,
        "identity_ok": 0,
        "members_total": 0,
        "members_mapped": 0,
    }
    for sample_id, objects in kept_by_sample.items():
        if not ctx.has_sample(sample_id):
            continue
        coverage["samples"] += 1
        assignment = ctx.gt_assignment(sample_id)
        truth = ctx.gt_truth(sample_id)
        allowed = object_filter.get(sample_id) if object_filter else None
        predictions_by_key: Dict[tuple, List[tuple]] = {}
        for prediction in objects:
            coverage["objects"] += 1
            members: List[Optional[str]] = []
            for view, index_key in (("a", "view_a_index"), ("b", "view_b_index")):
                index = prediction.get(index_key)
                if index is None:
                    continue
                observation_id = observation_id_for(sample_id, family, view, index)
                coverage["members_total"] += 1
                mapped = assignment.get(observation_id)
                if mapped:
                    coverage["members_mapped"] += 1
                members.append(mapped)
            unique = {member for member in members if member and member != "__NULL__"}
            if len(unique) != 1 or not members or any(not member or member == "__NULL__" for member in members):
                continue
            object_id = next(iter(unique))
            if allowed is not None and object_id not in allowed:
                continue
            coverage["identity_ok"] += 1
            beliefs = prediction.get("attribute_beliefs") or {}
            for attribute in attributes:
                choices = [item for item in beliefs.values() if item.get("attribute") == attribute]
                if not choices:
                    continue
                for best in choices:
                    time_key = best.get("time_key")
                    gold = truth.get(f"{object_id}|{attribute}|{time_key}")
                    if gold in (None, ""):
                        continue
                    predicted = str(best.get("value"))
                    counters = per_attr[attribute]
                    counters["cond_total"] += 1
                    if predicted == gold:
                        counters["cond_correct"] += 1
                        counters["class_tp"][gold] += 1
                    else:
                        counters["class_fp"][predicted] += 1
                        counters["class_fn"][gold] += 1
                    predictions_by_key.setdefault((object_id, attribute, time_key), []).append(
                        (float(best.get("probability") or 0.0), predicted)
                    )
        for key, gold in truth.items():
            parts = str(key).split("|")
            if len(parts) != 3:
                continue
            object_id, attribute, time_key = parts
            if attribute not in per_attr:
                continue
            if allowed is not None and object_id not in allowed:
                continue
            counters = per_attr[attribute]
            counters["gold_total"] += 1
            candidates = predictions_by_key.get((object_id, attribute, time_key)) or []
            if not candidates:
                continue
            predicted = max(candidates, key=lambda item: item[0])[1]
            if predicted == gold:
                counters["gold_correct"] += 1

    summary: Dict[str, Any] = {}
    for attribute, counters in per_attr.items():
        classes = sorted(set(counters["class_tp"]) | set(counters["class_fp"]) | set(counters["class_fn"]))
        per_class: Dict[str, Dict[str, Any]] = {}
        f1_values = []
        for class_name in classes:
            tp = counters["class_tp"].get(class_name, 0)
            fp = counters["class_fp"].get(class_name, 0)
            fn = counters["class_fn"].get(class_name, 0)
            precision = _rate(tp, tp + fp)
            recall = _rate(tp, tp + fn)
            f1 = _rate(2 * precision * recall, precision + recall)
            per_class[class_name] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
            }
            f1_values.append(f1)
        summary[attribute] = {
            "conditional_accuracy": round(_rate(counters["cond_correct"], counters["cond_total"]), 6),
            "conditional_macro_f1": round(sum(f1_values) / len(f1_values), 6) if f1_values else 0.0,
            "conditional_support": counters["cond_total"],
            "end_to_end_accuracy": round(_rate(counters["gold_correct"], counters["gold_total"]), 6),
            "end_to_end_support": counters["gold_total"],
            "per_class": per_class,
        }
    return {"attributes": summary, "coverage": coverage}


def summarize(metrics: Mapping[str, Any]) -> str:
    lines = []
    for attribute, item in metrics["attributes"].items():
        lines.append(
            f"- {attribute}: cond acc={item['conditional_accuracy']:.4f} "
            f"macro-F1={item['conditional_macro_f1']:.4f} (n={item['conditional_support']}); "
            f"e2e acc={item['end_to_end_accuracy']:.4f} (n={item['end_to_end_support']})"
        )
    coverage = metrics["coverage"]
    lines.append(
        f"- coverage: objects={coverage['objects']} identity_ok={coverage['identity_ok']} "
        f"members={coverage['members_mapped']}/{coverage['members_total']}"
    )
    return "\n".join(lines)
