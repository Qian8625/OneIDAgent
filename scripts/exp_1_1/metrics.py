"""工具使用指标：Macro / Seq. / ToolRec. / Norm.Arg. / E2E / Attempt / Blocked / Steps。

口径（写死在代码里，报数时必须带上）:

    Seq.      有序非 Terminate 工具序列与 GT 完全一致的样本占比 (%)
    ToolRec.  工具多重集合召回 |G ∩ P| / |G|，逐样本平均 (%)
    Macro     逐工具 F1（precision/recall 取多重集合），再对"GT 出现过的工具集合"求宏平均 (%)；
              没调用任何工具时按 0 计
    Norm.Arg. 仅在预测与 GT 同位置、同工具的对齐位置上比较参数；参数经归一化
              （小写、去路径、浮点两位、检测列表按 (label, bbox) 排序）后逐键相等才算命中 (%)。
              需要 trace 里落盘了 `action_log[].args`；旧 trace 没有参数值时该项记 N/A
    E2E       answer_ok（计数 ∧ 属性逐字段全中）(%)
    Attempt   每样本工具调用尝试次数（含失败/被拦截的尝试）均值
    Blocked   每样本被拦截的动作数均值 = 策略拦截（如闭卷/无 OneID）+ duplicate_call_blocked
    Steps     每样本平均轮次

GT 序列从数据集原始对话里取（assistant 的 actions[].name，去掉 Terminate）。
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

METRICS = ("Macro", "Seq.", "ToolRec.", "Norm.Arg.", "E2E", "Attempt", "Blocked", "Steps")
FAMILIES = ("task4", "task5")


# --------------------------------------------------------------------------- #
# GT
# --------------------------------------------------------------------------- #
def _gold_actions_from_dataset(path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    from scripts.eval.e2e_task45.data import _family_of, _target_of

    gold: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            images = [str(p) for p in (item.get("images") or [])]
            family = _family_of(images)
            if family not in FAMILIES or len(images) < 2:
                continue
            question, actions_out = "", []
            for message in item.get("messages") or []:
                if message.get("role") == "user" and not question:
                    question = (message.get("content") or "").replace("<image>", "").strip()
                if message.get("role") != "assistant":
                    continue
                try:
                    actions = (json.loads(message.get("content") or "") or {}).get("actions") or []
                except Exception:
                    continue
                for action in actions:
                    name = (action or {}).get("name")
                    if name and name != "Terminate":
                        actions_out.append({"name": name, "args": (action or {}).get("arguments")})
            pair_id = "_".join(Path(images[0]).name.split("_")[:2])
            gold.setdefault((pair_id, _target_of(question)), actions_out)
    return gold


# --------------------------------------------------------------------------- #
# 归一化与逐样本指标
# --------------------------------------------------------------------------- #
def image_roles(pair_id: str, family: str, data_root: Path) -> Dict[str, str]:
    """把路径/符号引用统一成 img_1 / img_2，否则 GT 的 "img_1" 与预测的绝对路径永远不等。"""
    stem = ("match_images", "_RGB.png", "_SAR.png") if family == "task4" else ("time_images", "_time1.png", "_time2.png")
    first = data_root / stem[0] / f"{pair_id}{stem[1]}"
    second = data_root / stem[0] / f"{pair_id}{stem[2]}"
    roles: Dict[str, str] = {"img_1": "img_1", "img_2": "img_2"}
    for role, path in (("img_1", first), ("img_2", second)):
        roles[str(path)] = role
        roles[path.name] = role
        roles[str(path).lower()] = role
    roles.update({"$primary_image": "img_1", "$optical_image": "img_1", "$time1_image": "img_1",
                  "$sar_image": "img_2", "$time2_image": "img_2", "$latest_image": "$latest_image"})
    return roles


def _norm_value(value: Any, roles: Optional[Dict[str, str]] = None) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, str):
        text = value.strip()
        if roles and (text in roles or text.lower() in roles):
            return roles.get(text, roles.get(text.lower()))
        if "/" in text or "\\" in text:
            name = Path(text.replace("\\", "/")).name
            if roles and name in roles:
                return roles[name]
            text = name
        try:
            return round(float(text), 2)
        except ValueError:
            return text.lower()
    if isinstance(value, (list, tuple)):
        items = [_norm_value(item, roles) for item in value]
        if items and all(isinstance(item, dict) for item in items):
            items = sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
        return items
    if isinstance(value, dict):
        return {str(key).lower(): _norm_value(val, roles) for key, val in sorted(value.items())}
    return str(value).lower()


def norm_args(args: Optional[Dict[str, Any]], roles: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    if not isinstance(args, dict):
        return None
    return {str(key).lower(): _norm_value(val, roles) for key, val in args.items()}


def tool_prf(gold: List[str], pred: List[str]) -> Tuple[float, float, float]:
    g, p = Counter(gold), Counter(pred)
    overlap = sum(min(g[t], p[t]) for t in set(g) | set(p))
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(gold) if gold else 1.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def macro_tool_f1(gold: List[str], pred: List[str]) -> float:
    """GT 出现过的每个工具各算一次 F1（多重集合），再宏平均。"""
    g, p = Counter(gold), Counter(pred)
    tools = list(g) or list(dict.fromkeys(pred))
    if not tools:
        return 0.0
    scores = []
    for tool in tools:
        tp = min(g.get(tool, 0), p.get(tool, 0))
        precision = tp / p[tool] if p.get(tool) else 0.0
        recall = tp / g[tool] if g.get(tool) else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def norm_arg_match(gold_actions: List[Dict[str, Any]], pred_actions: List[Dict[str, Any]],
                   roles: Optional[Dict[str, str]] = None) -> Optional[float]:
    """对齐位置上同工具的参数归一化命中率；任一侧没有参数值时返回 None。"""
    comparable = matched = 0
    for gold, pred in zip(gold_actions, pred_actions):
        if gold.get("name") != pred.get("name"):
            continue
        gold_args, pred_args = gold.get("args"), pred.get("args")
        if gold_args is None or pred_args is None:
            continue
        comparable += 1
        if norm_args(gold_args, roles) == norm_args(pred_args, roles):
            matched += 1
    return (100.0 * matched / comparable) if comparable else None


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
def _pred_actions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"name": entry.get("api_name"), "args": entry.get("args")}
            for entry in (record.get("action_log") or [])
            if entry.get("api_name") and entry.get("api_name") != "Terminate"]


def _blocked_count(record: Dict[str, Any]) -> int:
    duplicates = sum(1 for entry in (record.get("action_log") or [])
                     if entry.get("error") == "duplicate_call_blocked")
    return duplicates + len(record.get("oneid_blocked") or [])


def compute(records: List[Dict[str, Any]], gold: Dict[Tuple[str, str], List[Dict[str, Any]]],
            data_root: Path = Path("/home/ubuntu/00_Data/RSTI")) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("probe"):
            continue
        buckets[(record["cell"], record["family"])].append(record)

    rows: List[Dict[str, Any]] = []
    for (cell, family), items in sorted(buckets.items()):
        n = len(items)
        seq = macro = toolrec = e2e = attempt = blocked = steps = 0.0
        arg_hits = arg_total = 0
        arg_samples = 0
        for record in items:
            gold_raw = gold.get((record["pair_id"], record["target"])) or []
            gold_tools = [item["name"] if isinstance(item, dict) else item for item in gold_raw]
            pred_tools = [name for name in (record.get("tool_sequence") or []) if name != "Terminate"]
            if gold_tools is not None:
                seq += 1.0 if pred_tools == gold_tools else 0.0
                macro += macro_tool_f1(gold_tools, pred_tools)
                _, recall, _ = tool_prf(gold_tools, pred_tools)
                toolrec += recall
            e2e += 1.0 if record["score"]["answer_ok"] else 0.0
            attempt += record["trace"]["raw_tool_calls"]
            blocked += _blocked_count(record)
            steps += record["trace"]["rounds"]
            gold_actions = gold.get((record["pair_id"], record["target"]))
            pred_actions = _pred_actions(record)
            if gold_actions and any(entry.get("args") is not None for entry in pred_actions):
                roles = image_roles(record["pair_id"], record["family"], data_root)
                value = norm_arg_match(gold_actions, pred_actions, roles)
                if value is not None:
                    arg_hits += value
                    arg_total += 1
                    arg_samples += 1

        rows.append({
            "cell": cell, "family": family, "n": n,
            "Macro": round(100 * macro / n, 2) if n else 0.0,
            "Seq.": round(100 * seq / n, 2) if n else 0.0,
            "ToolRec.": round(100 * toolrec / n, 2) if n else 0.0,
            "Norm.Arg.": round(arg_hits / arg_total, 2) if arg_total else None,
            "E2E": round(100 * e2e / n, 2) if n else 0.0,
            "Attempt": round(attempt / n, 2) if n else 0.0,
            "Blocked": round(blocked / n, 2) if n else 0.0,
            "Steps": round(steps / n, 2) if n else 0.0,
            "_norm_arg_samples": arg_samples,
        })
    return rows


def to_markdown(rows: List[Dict[str, Any]], cells: Iterable[str]) -> str:
    look = {(row["cell"], row["family"]): row for row in rows}
    lines = ["| Metric | Cell | n4 | n5 | task4 | task5 |", "| --- | --- | --- | --- | --- | --- |"]
    for metric in METRICS:
        for cell in cells:
            t4, t5 = look.get((cell, "task4")), look.get((cell, "task5"))
            if not t4 and not t5:
                continue
            def fmt(row):
                if not row:
                    return "—"
                value = row[metric]
                return "N/A" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))
            n4 = (t4 or {}).get("n", 0)
            n5 = (t5 or {}).get("n", 0)
            lines.append(f"| {metric} | {cell} | {n4} | {n5} | {fmt(t4)} | {fmt(t5)} |")
    return "\n".join(lines) + "\n"


def write(rows: List[Dict[str, Any]], out_dir: Path, cells: Iterable[str]) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["cell", "family", "n", *METRICS, "_norm_arg_samples"]
    csv_path = out_dir / "tool_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    md_path = out_dir / "tool_metrics.md"
    md_path.write_text("# 工具使用指标\n\n" + to_markdown(rows, cells), encoding="utf-8")
    return {"csv": csv_path, "md": md_path}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--dataset", default="/home/ubuntu/00_Data/RSTI/test.jsonl")
    parser.add_argument("--out", default="outputs/exp_1_1/report")
    parser.add_argument("--cells", default="base_det,base_oneid,sft_det,sft_oneid")
    parser.add_argument("--rescore", action="store_true",
                        help="用当前 score.py 口径重算分数（含属性归一化修复）")
    args = parser.parse_args(argv)

    from scripts.exp_1_1 import run as run_mod

    paths = []
    for item in args.traces:
        item = Path(item)
        paths.extend(sorted(item.glob("*.jsonl")) if item.is_dir() else [item])
    records = run_mod.load_records(paths)
    if args.rescore:
        from scripts.eval.e2e_task45.data import parse_gt
        from scripts.eval.e2e_task45.score import score_answer

        for record in records:
            if record.get("probe"):
                continue
            record["score"] = score_answer(parse_gt(record.get("label") or "", record["family"]),
                                           record.get("final_answer") or "", record["family"])
    gold = _gold_actions_from_dataset(Path(args.dataset))
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    rows = compute(records, gold, Path(args.dataset).parent)
    written = write(rows, Path(args.out), cells)
    print(to_markdown(rows, cells))
    print(f"[out] {written['csv']}\n      {written['md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
