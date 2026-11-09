"""按仓库标准口径（task.py::evaluate_function）重算同一批 trace，并与当前口径对比。

标准口径（来自 tool_server/tf_eval/tasks/rsagent/task.py）：
  Macro      = 4 个工具类别（perception/operation/logic/gis）各自 F1 之后再宏平均
               F1 = 2PR/(P+R)，P/R 按"类别内多重集合"统计：correct=min(gt,pred) 逐工具求和
  ToolRec.   = all_tools_called_unique：去 Terminate 后的**唯一工具集合**完全一致的比例
  Attempt    = total_tool_calls_pred / n：只数成功派发的调用（我们的 tool_sequence）
  Blocked    = tool_call_error：工具返回 error_code != 0 的次数（标准里没有"策略拦截"概念）
  Steps      = avg_steps_taken：轮次
  Seq.       = all_tools_called_same_order：去 Terminate 后序列完全一致（与当前一致）
  Norm.Arg.  = 同位置同工具的参数归一化命中（当前同一逻辑，仅多了图像角色归一化）

E2E：标准用 LLM judge（answer_acc），离线无 key 不可复算；当前用字段合取。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATEGORY = {}
for name in ("BaseModel", "ImageDescription", "RegionAttributeDescription", "TextToBbox",
             "ChangeDetection", "ObjectDetection", "SmallObjectDetection", "DenoDetV2",
             "CrossModalInterpret", "SegmentObjectPixels", "CountGivenObject"):
    CATEGORY[name] = "perception"
for name in ("DrawBox", "AddText", "GoogleSearch"):
    CATEGORY[name] = "operation"
for name in ("Calculator", "Solver", "Plot"):
    CATEGORY[name] = "logic"
for name in ("GetAreaBoundary", "AddPoisLayer", "ComputeDistance", "GetBboxFromGeotiff",
             "DisplayOnGeotiff"):
    CATEGORY[name] = "gis"
CATS = ("perception", "operation", "logic", "gis")


def _seq(record: Dict[str, Any]) -> List[str]:
    return [n for n in (record.get("tool_sequence") or []) if n and n != "Terminate"]


def standard_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("probe"):
            continue
        buckets[(record["cell"], record["family"])].append(record)

    rows = []
    for (cell, family), items in sorted(buckets.items()):
        totals = Counter()
        preds = Counter()
        correct = Counter()
        unique_ok = 0.0
        attempts = 0.0
        errors = 0.0
        steps = 0.0
        for record in items:
            gt = [n for n in record.get("gold_tools") or [] if n != "Terminate"]
            if not gt:
                gt = [n for n in (record.get("gold_sequence") or []) if n != "Terminate"]
            pred = _seq(record)
            for name in gt:
                cat = CATEGORY.get(name)
                if cat:
                    totals[cat] += 1
            for name in pred:
                cat = CATEGORY.get(name)
                if cat:
                    preds[cat] += 1
            gt_bag, pred_bag = Counter(gt), Counter(pred)
            for name, count in gt_bag.items():
                cat = CATEGORY.get(name)
                if cat:
                    correct[cat] += min(count, pred_bag.get(name, 0))
            if gt and set(gt) == set(pred):
                unique_ok += 1
            attempts += len(pred)
            errors += record["trace"].get("tool_errors", 0)
            steps += record["trace"].get("rounds", 0)

        f1 = {}
        for cat in CATS:
            precision = correct[cat] / preds[cat] if preds[cat] else 0.0
            recall = correct[cat] / totals[cat] if totals[cat] else 0.0
            f1[cat] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        n = len(items)
        rows.append({
            "cell": cell, "family": family, "n": n,
            "Macro_std": round(100 * sum(f1.values()) / len(CATS), 2),
            "ToolRec_std": round(100 * unique_ok / n, 2),
            "Attempt_std": round(attempts / n, 2),
            "Blocked_std": round(errors / n, 2),
            "Steps_std": round(steps / n, 2),
            "detail": {cat: round(f1[cat] * 100, 2) for cat in CATS},
        })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--dataset", default="/home/ubuntu/00_Data/RSTI/test.jsonl")
    parser.add_argument("--out", default="outputs/exp_1_1/report_std")
    args = parser.parse_args(argv)

    from scripts.exp_1_1 import metrics, run as run_mod

    paths = []
    for item in args.traces:
        item = Path(item)
        paths.extend(sorted(item.glob("*.jsonl")) if item.is_dir() else [item])
    records = run_mod.load_records(paths)

    # 注入 gold 工具序列（标准口径需要 GT 序列；metrics 里已有同源函数）
    gold = metrics._gold_actions_from_dataset(Path(args.dataset))
    for record in records:
        actions = gold.get((record["pair_id"], record["target"]))
        if actions:
            record["gold_tools"] = [a["name"] for a in actions]

    std = standard_rows(records)
    cur = {(r["cell"], r["family"]): r for r in metrics.compute(records, gold)}
    table = []
    for row in std:
        now = cur.get((row["cell"], row["family"]), {})
        table.append({
            "cell": row["cell"], "family": row["family"], "n": row["n"],
            "Macro_std": row["Macro_std"], "Macro_cur": now.get("Macro"),
            "ToolRec_std": row["ToolRec_std"], "ToolRec_cur": now.get("ToolRec."),
            "Attempt_std": row["Attempt_std"], "Attempt_cur": now.get("Attempt"),
            "Blocked_std": row["Blocked_std"], "Blocked_cur": now.get("Blocked"),
            "Steps": row["Steps_std"],
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "std_vs_current.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    lines = ["| Cell | Family | Macro(标准) | Macro(当前) | ToolRec(标准) | ToolRec(当前) | Attempt(标准) | Attempt(当前) | Blocked(标准) | Blocked(当前) | Steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in table:
        lines.append("| {cell} | {family} | {Macro_std} | {Macro_cur} | {ToolRec_std} | {ToolRec_cur} | "
                     "{Attempt_std} | {Attempt_cur} | {Blocked_std} | {Blocked_cur} | {Steps} |".format(**row))
    (out / "std_vs_current.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[out] {out/'std_vs_current.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
