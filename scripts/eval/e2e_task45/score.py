"""字段级确定性评分 + 汇总。

为什么不用仓库现有的 ``sim_score``（LLM judge）：消融实验要在四个臂之间做差值，
judge 的抖动会直接变成"增益"。task4/task5 的答案恰好是结构化事实（4 个计数 + 4 个
属性），可以逐字段精确比对；judge 只作为 ``--judge`` 可选的复核，不进入主表。

口径：
* 计数：GT 里给出几个就比几个，全部相等才算 ``counts_ok``；
* 属性：GT 为 None（"no recorded ..."）时**不许编造**，预测也必须没有值才算命中；
* ``answer_ok`` = 所有可比较字段都命中（含计数与属性）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .data import ATTR_KEYS, parse_prediction

COUNT_KEYS = {
    "task4": ("matched", "rgb_only", "sar_only"),
    "task5": ("appeared", "disappeared", "static", "moved"),
}


def score_answer(gt: Dict[str, Any], final_answer: str, family: str) -> Dict[str, Any]:
    pred = parse_prediction(final_answer, family)
    compared = hits = 0
    counts_compared = counts_hits = 0
    detail: Dict[str, Any] = {}

    for key in COUNT_KEYS[family]:
        truth = gt["counts"].get(key)
        if truth is None:
            continue
        compared += 1
        counts_compared += 1
        got = pred["counts"].get(key)
        ok = got == truth
        hits += int(ok)
        counts_hits += int(ok)
        detail[f"count.{key}"] = {"gt": truth, "pred": got, "ok": ok}

    for key in ATTR_KEYS:
        truth = gt["attributes"].get(key)
        compared += 1
        got = pred["attributes"].get(key)
        ok = (got == truth) if truth is not None else (got is None)
        hits += int(ok)
        detail[f"attr.{key}"] = {"gt": truth, "pred": got, "ok": ok}

    return {
        "counts_ok": counts_compared > 0 and counts_hits == counts_compared,
        "counts_hits": counts_hits,
        "counts_total": counts_compared,
        "attrs_ok": all(detail[f"attr.{k}"]["ok"] for k in ATTR_KEYS),
        "attr_hits": sum(1 for k in ATTR_KEYS if detail[f"attr.{k}"]["ok"]),
        "attr_total": len(ATTR_KEYS),
        "answer_ok": compared > 0 and hits == compared,
        "field_hits": hits,
        "field_total": compared,
        "empty_answer": not (final_answer or "").strip(),
        "detail": detail,
        "pred": pred,
    }


def _pct(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def aggregate(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault((record["arm"], record["family"]), []).append(record)

    rows: List[Dict[str, Any]] = []
    for (arm, family), items in sorted(buckets.items()):
        scored = [item["score"] for item in items]
        n = len(items)
        rows.append({
            "arm": arm,
            "family": family,
            "n": n,
            "answer_acc": _pct(sum(1 for s in scored if s["answer_ok"]), n),
            "counts_acc": _pct(sum(1 for s in scored if s["counts_ok"]), n),
            "attrs_acc": _pct(sum(1 for s in scored if s["attrs_ok"]), n),
            "attr_field_acc": _pct(sum(s["attr_hits"] for s in scored), 4 * n),
            "empty_rate": _pct(sum(1 for s in scored if s["empty_answer"]), n),
            "avg_rounds": round(sum(item["trace"]["rounds"] for item in items) / n, 2),
            "oneid_call_rate": _pct(sum(1 for item in items if item.get("oneid_calls")), n),
            "tool_error_rate": _pct(
                sum(item["trace"]["tool_errors"] for item in items),
                max(1, sum(item["trace"]["raw_tool_calls"] for item in items)),
            ),
        })
    return rows


def write_outputs(records: List[Dict[str, Any]], rows: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["arm"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / "REPORT.md"
    md_path.write_text(render_markdown(rows, records), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "md": md_path}


def render_markdown(rows: List[Dict[str, Any]], records: Optional[List[Dict[str, Any]]] = None) -> str:
    by_key = {(row["arm"], row["family"]): row for row in rows}
    arms = []
    for row in rows:
        if row["arm"] not in arms:
            arms.append(row["arm"])

    lines = ["# task4/task5 消融结果", ""]
    for metric, title in (
        ("answer_acc", "整题全对率 (%)"),
        ("counts_acc", "计数全对率 (%)"),
        ("attr_field_acc", "属性字段准确率 (%)"),
        ("oneid_call_rate", "调用 OneID 的样本占比 (%)"),
    ):
        lines += [f"## {title}", "", "| arm | task4 | task5 |", "| --- | --- | --- |"]
        for arm in arms:
            cells = []
            for family in ("task4", "task5"):
                row = by_key.get((arm, family))
                cells.append(f"{row[metric]:.2f}" if row else "—")
            lines.append(f"| {arm} | {cells[0]} | {cells[1]} |")
        lines.append("")

    if records:
        lines += ["## 样本量", "", "| arm | task | n |", "| --- | --- | --- |"]
        for row in rows:
            lines.append(f"| {row['arm']} | {row['family']} | {row['n']} |")
        lines.append("")
    lines.append("> answer_ok 要求计数与属性逐字段全中；属性 GT 为 None 时预测也必须留空。")
    return "\n".join(lines) + "\n"
