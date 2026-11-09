"""汇总与统计：把 traces 变成 2x2 主表、记忆表、增益与检验。

只用标准库（可在无 GPU 的机器上跑）。核心量：

    G_E  = acc(det)   - acc(none)    证据（检测工具）带来的增量
    G_O  = acc(oneid) - acc(det)     工具不变、只多 OneID 的增量  <- 论文主量
    G_S  = acc(sft, c) - acc(base, c) SFT 在条件 c 上的增量
    DiD  = G_O(sft) - G_O(base)      SFT 是否更会用 OneID

检验：配对 McNemar 精确检验（二值指标）+ 按 pair 聚类的 bootstrap CI。
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILIES = ("task4", "task5")


def load(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files = sorted(path.rglob("*.jsonl"))
        else:
            files = [path]
        for file in files:
            with open(file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
    return records


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def cell_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("probe"):
            continue
        buckets[(record["cell"], record["family"])].append(record)

    rows: List[Dict[str, Any]] = []
    for (cell, family), items in sorted(buckets.items()):
        n = len(items)
        scores = [item["score"] for item in items]
        rows.append({
            "cell": cell, "family": family, "n": n,
            "counts_acc": round(100 * _mean(1.0 if s["counts_ok"] else 0.0 for s in scores), 2),
            "answer_acc": round(100 * _mean(1.0 if s["answer_ok"] else 0.0 for s in scores), 2),
            "attr_field_acc": round(100 * _mean(s["attr_hits"] / 4.0 for s in scores), 2),
            "empty_rate": round(100 * _mean(1.0 if s["empty_answer"] else 0.0 for s in scores), 2),
            "oneid_call_rate": round(100 * _mean(1.0 if item.get("oneid_calls") else 0.0 for item in items), 2),
            "tool_error_rate": round(100 * _mean(
                item["trace"]["tool_errors"] / max(1, item["trace"]["raw_tool_calls"]) for item in items), 2),
            "avg_rounds": round(_mean(item["trace"]["rounds"] for item in items), 2),
            "max_rounds_rate": round(100 * _mean(
                1.0 if "max_rounds" in item["trace"]["error_tags"] else 0.0 for item in items), 2),
        })
    return rows


def _by_sample(records: List[Dict[str, Any]], cell: str, family: Optional[str] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for record in records:
        if record.get("probe") or record["cell"] != cell:
            continue
        if family and record["family"] != family:
            continue
        out[record["sample_id"]] = 1.0 if record["score"]["answer_ok"] else 0.0
    return out


def paired(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, Any]:
    """b - a 的配对差，附 McNemar 精确检验。"""
    common = sorted(set(a) & set(b))
    if not common:
        return {"n": 0}
    only_a = sum(1 for k in common if a[k] and not b[k])
    only_b = sum(1 for k in common if b[k] and not a[k])
    return {
        "n": len(common),
        "a_rate": round(100 * _mean(a[k] for k in common), 2),
        "b_rate": round(100 * _mean(b[k] for k in common), 2),
        "diff": round(100 * _mean(b[k] - a[k] for k in common), 2),
        "mcnemar_p": mcnemar_exact(only_a, only_b),
        "discordant": [only_a, only_b],
    }


def mcnemar_exact(b: int, c: int) -> float:
    """双侧精确 McNemar：在 b+c 次不一致里做 Binomial(b+c, 0.5)。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 6)


def bootstrap_diff(a: Dict[str, float], b: Dict[str, float],
                   pairs: Dict[str, str], rounds: int = 10000, seed: int = 0) -> Optional[List[float]]:
    """按 pair 聚类的配对 bootstrap：先重采样 pair，再算 b-a 的均值。"""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    by_pair: Dict[str, List[str]] = defaultdict(list)
    for sid in common:
        by_pair[pairs.get(sid, sid)].append(sid)
    keys = sorted(by_pair)
    rng = random.Random(seed)
    deltas: List[float] = []
    for _ in range(rounds):
        picked = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        ids = [sid for key in picked for sid in by_pair[key]]
        deltas.append(100 * _mean(b[sid] - a[sid] for sid in ids))
    deltas.sort()
    return [round(deltas[int(0.025 * rounds)], 2), round(deltas[int(0.975 * rounds) - 1], 2)]


def memory_table(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not record.get("probe"):
            continue
        buckets[(record.get("probe_cell", record["cell"]), record["probe"])].append(record)
    rows = []
    for (cell, probe), items in sorted(buckets.items()):
        rows.append({
            "cell": cell, "probe": probe, "n": len(items),
            "answer_acc": round(100 * _mean(1.0 if i["score"]["answer_ok"] else 0.0 for i in items), 2),
            "counts_acc": round(100 * _mean(1.0 if i["score"]["counts_ok"] else 0.0 for i in items), 2),
            "tool_used": round(100 * _mean(1.0 if i["tool_sequence"] else 0.0 for i in items), 2),
        })
    return rows


def gains(records: List[Dict[str, Any]], pairs: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        for weights in ("base", "sft"):
            none = _by_sample(records, f"{weights}_none", family)
            det = _by_sample(records, f"{weights}_det", family)
            oneid = _by_sample(records, f"{weights}_oneid", family)
            if det:
                rows.append({"family": family, "weights": weights, "metric": "G_E(det-none)",
                             **paired(none, det),
                             "ci": bootstrap_diff(none, det, pairs)})
            if oneid:
                rows.append({"family": family, "weights": weights, "metric": "G_O(oneid-det)",
                             **paired(det, oneid),
                             "ci": bootstrap_diff(det, oneid, pairs)})
        a_det, b_det = _by_sample(records, "base_det", family), _by_sample(records, "sft_det", family)
        a_one, b_one = _by_sample(records, "base_oneid", family), _by_sample(records, "sft_oneid", family)
        if a_det and b_det:
            rows.append({"family": family, "weights": "base->sft", "metric": "G_S(det)",
                         **paired(a_det, b_det), "ci": bootstrap_diff(a_det, b_det, pairs)})
        if a_one and b_one:
            rows.append({"family": family, "weights": "base->sft", "metric": "G_S(oneid)",
                         **paired(a_one, b_one), "ci": bootstrap_diff(a_one, b_one, pairs)})
        # DiD：两条臂的 OneID 增益之差
        if a_det and a_one and b_det and b_one:
            did = {
                sid: (b_one.get(sid, 0.0) - b_det.get(sid, 0.0)) - (a_one.get(sid, 0.0) - a_det.get(sid, 0.0))
                for sid in set(a_det) & set(a_one) & set(b_det) & set(b_one)
            }
            zeros = {sid: 0.0 for sid in did}
            rows.append({"family": family, "weights": "DiD", "metric": "DiD(oneid gain)",
                         "n": len(did), "diff": round(100 * _mean(did.values()), 2),
                         "ci": bootstrap_diff(zeros, did, pairs),
                         "mcnemar_p": None, "discordant": None})
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                             for k, v in row.items()})


def report(rows: List[Dict[str, Any]], gains_rows: List[Dict[str, Any]],
           mem_rows: List[Dict[str, Any]]) -> str:
    look = {(r["cell"], r["family"]): r for r in rows}
    lines = ["# exp_1_1 · 2x2 主表", ""]
    lines += ["| 指标 | 臂 | task4 | task5 |", "| --- | --- | --- | --- |"]
    for metric, label in (("counts_acc", "计数全对率 (%)"), ("answer_acc", "整题全对率 (%)"),
                          ("attr_field_acc", "属性字段准确率 (%)")):
        for cell in ("base_det", "base_oneid", "sft_det", "sft_oneid"):
            cells = [f"{look[(cell, f)][metric]:.2f}" if (cell, f) in look else "—" for f in FAMILIES]
            lines.append(f"| {label} | {cell} | {cells[0]} | {cells[1]} |")
    lines.append("")

    lines += ["## 增益与检验", "", "| 任务 | 对象 | 量 | 差值(pp) | 95% CI | n | McNemar p |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in gains_rows:
        ci = row.get("ci") or ["—", "—"]
        p = row.get("mcnemar_p")
        lines.append(f"| {row['family']} | {row['weights']} | {row['metric']} | {row.get('diff', 0):+.2f} | "
                     f"[{ci[0]}, {ci[1]}] | {row.get('n', 0)} | {('%.4g' % p) if p is not None else '—'} |")
    lines.append("")

    if mem_rows:
        lines += ["## 记忆 / 敏感性探针", "", "| 探针 | 臂 | n | answer_acc | 有工具调用占比 |", "| --- | --- | --- | --- | --- |"]
        for row in mem_rows:
            lines.append(f"| {row['probe']} | {row['cell']} | {row['n']} | {row['answer_acc']:.2f} | {row['tool_used']:.2f} |")
        lines.append("")
        lines += ["> m1 闭卷 = 记忆下限；m3 假 id 高 = 按标识符回忆；m2 换图 / splice 高 = 没在看证据。"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="+", required=True, help="trace 目录或 jsonl")
    parser.add_argument("--probes", nargs="*", default=[], help="探针目录（可选）")
    parser.add_argument("--out", default="outputs/exp_1_1/report")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--rescore", action="store_true",
                        help="用当前 score.py 的口径重算每条记录的分数（含属性归一化修复）")
    args = parser.parse_args(argv)

    records = load([Path(p) for p in args.traces])
    if args.rescore:
        from scripts.eval.e2e_task45.data import parse_gt
        from scripts.eval.e2e_task45.score import score_answer

        for record in records:
            if record.get("probe"):
                continue
            gt = parse_gt(record.get("label") or "", record["family"])
            record["score"] = score_answer(gt, record.get("final_answer") or "", record["family"])
    probe_records = load([Path(p) for p in args.probes]) if args.probes else []
    pairs = {r["sample_id"]: r["pair_id"] for r in records}

    metrics = cell_metrics(records)
    gain_rows = gains(records, pairs) if any(r["condition"] == "none" for r in records) else []
    if not gain_rows:
        # 没有闭卷条件时，至少给 OneID 与 SFT 的配对对比
        gain_rows = []
        for family in FAMILIES:
            for left, right, label in (("base_det", "base_oneid", "G_O(base)"),
                                       ("sft_det", "sft_oneid", "G_O(sft)"),
                                       ("base_det", "sft_det", "G_S(det)"),
                                       ("base_oneid", "sft_oneid", "G_S(oneid)")):
                a, b = _by_sample(records, left, family), _by_sample(records, right, family)
                if a and b:
                    gain_rows.append({"family": family, "weights": label.split("(")[0], "metric": label,
                                      **paired(a, b), "ci": bootstrap_diff(a, b, pairs)})
    mem_rows = memory_table(probe_records) if probe_records else []

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(metrics, out / "metrics.csv")
    write_csv(gain_rows, out / "gains.csv")
    if mem_rows:
        write_csv(mem_rows, out / "probes.csv")
    (out / "REPORT.md").write_text(report(metrics, gain_rows, mem_rows), encoding="utf-8")
    print((out / "REPORT.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
