#!/usr/bin/env python3
"""等四臂跑满 → 自动用当前口径重算并出最终报告。

产物 outputs/exp_1_1/report_final/：
  REPORT.md        2x2 主表 + G_E/G_O/G_S/DiD + McNemar（analyze --rescore）
  metrics.csv      逐 cell × family 的主指标
  tool_metrics.csv 8 项工具指标（Macro/Seq./ToolRec./Norm.Arg./E2E/Attempt/Blocked/Steps）
  FINAL.md         四臂对比表 + task4/task5 分组指标（本脚本拼装）

判满条件：四个 trace 文件都到 EXPECT 条；或检测到 run 进程已退出且 10 分钟内数量不再增长。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACES = ROOT / "outputs/exp_1_1/run_full2/traces"
OUT = ROOT / "outputs/exp_1_1/report_final"
LOG = ROOT / "outputs/exp_1_1/finalize.log"
CELLS = ("base_det", "base_oneid", "sft_det", "sft_oneid")
SEED = 61
EXPECT = 715
PY = "/home/ubuntu/miniconda3/envs/OEA/bin/python"


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def counts() -> Dict[str, int]:
    out = {}
    for cell in CELLS:
        path = TRACES / f"{cell}_seed{SEED}.jsonl"
        out[cell] = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
    return out


def running() -> int:
    result = subprocess.run(["pgrep", "-fc", "scripts.exp_1_1.run --data .*--out outputs/exp_1_1/run_full2"],
                            capture_output=True, text=True)
    try:
        return int(result.stdout.strip() or 0)
    except ValueError:
        return 0


def wait_done() -> Dict[str, int]:
    last = counts()
    log(f"等待四臂完成，当前 {last}")
    while True:
        time.sleep(300)
        now = counts()
        alive = running()
        log(f"进度 {now} run进程={alive}")
        if all(now[c] >= EXPECT for c in CELLS):
            return now
        if alive == 0 and now == last:
            log("run 进程已退出且计数不再增长，按当前进度收尾")
            return now
        last = now


def run_module(args: List[str]) -> None:
    log("执行: " + " ".join(args))
    result = subprocess.run([PY, *args], cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        log(f"[warn] 退出码 {result.returncode}: {result.stderr[-500:]}")
    else:
        log("ok")


def read_csv(path: Path) -> List[Dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def final_markdown(progress: Dict[str, int]) -> str:
    metrics = read_csv(OUT / "metrics.csv")
    tools = read_csv(OUT / "tool_metrics.csv")
    gains = read_csv(OUT / "gains.csv")

    def fmt(v: str, digits: int = 2) -> str:
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return v or "—"

    lines: List[str] = []
    lines.append("# exp_1_1 最终结果\n")
    lines.append(f"- 数据：`~/00_Data/RSTI/test.jsonl`（task4+task5），seed={SEED}，每臂 {EXPECT} 条")
    lines.append(f"- 实际条数：{progress}")
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip()
        lines.append(f"- 代码版本：`{head}`")
    except Exception:
        pass
    lines.append("- 口径：E2E = 计数 ∧ 4 属性逐字段相等；属性 GT 为空时要求预测留空；全部 `--rescore`\n")

    lines.append("## 1. 四臂对比（全样本）\n")
    lines.append("| 指标 | base_det | base_oneid | sft_det | sft_oneid |")
    lines.append("| --- | --- | --- | --- | --- |")
    order = ["counts_acc", "answer_acc", "attr_field_acc", "empty_rate", "oneid_call_rate", "avg_rounds", "max_rounds_rate"]
    labels = {"counts_acc": "计数全对率 %", "answer_acc": "E2E 整题全对率 %", "attr_field_acc": "属性字段准确率 %",
              "empty_rate": "空答案率 %", "oneid_call_rate": "OneID 调用率 %", "avg_rounds": "平均轮次",
              "max_rounds_rate": "打满轮次 %"}
    for key in order:
        cells = []
        for cell in CELLS:
            rows = [row for row in metrics if row["cell"] == cell]
            if not rows:
                cells.append("—")
                continue
            n = sum(int(row["n"] or 0) for row in rows)
            value = sum(float(row[key] or 0) * int(row["n"] or 0) for row in rows) / n if n else 0
            cells.append(fmt(value))
        lines.append(f"| {labels[key]} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 2. task4 / task5 分组\n")
    for family, title in (("task4", "task4（RGB/SAR 跨视图）"), ("task5", "task5（时相变化）")):
        lines.append(f"### {title}\n")
        lines.append("| 指标 | base_det | base_oneid | sft_det | sft_oneid |")
        lines.append("| --- | --- | --- | --- | --- |")
        for key in order:
            cells = []
            for cell in CELLS:
                row = next((r for r in metrics if r["cell"] == cell and r["family"] == family), None)
                cells.append(fmt(row[key]) if row else "—")
            lines.append(f"| {labels[key]} | " + " | ".join(cells) + " |")
        lines.append("")

    if tools:
        lines.append("## 3. 工具指标（逐 cell × family）\n")
        lines.append("| Metric | Cell | task4 | task5 |")
        lines.append("| --- | --- | --- | --- |")
        for metric in ("Macro", "Seq.", "ToolRec.", "Norm.Arg.", "E2E", "Attempt", "Blocked", "Steps"):
            for cell in CELLS:
                t4 = next((r for r in tools if r["cell"] == cell and r["family"] == "task4"), None)
                t5 = next((r for r in tools if r["cell"] == cell and r["family"] == "task5"), None)
                if not t4 and not t5:
                    continue
                a = (t4 or {}).get(metric, "—")
                b = (t5 or {}).get(metric, "—")
                lines.append(f"| {metric} | {cell} | {a} | {b} |")
        lines.append("")

    if gains:
        lines.append("## 4. 增益与检验\n")
        lines.append("| 任务 | 对象 | 量 | 差值(pp) | 95% CI | n | McNemar p |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in gains:
            lines.append(f"| {row.get('family','')} | {row.get('weights','')} | {row.get('metric','')} | "
                         f"{row.get('diff','—')} | {row.get('ci','—')} | {row.get('n','—')} | {row.get('mcnemar_p','—')} |")
        lines.append("")

    lines.append("## 5. 读表提醒\n")
    lines.append("- `counts_ok` 与 `E2E` 的差距 = 属性没答对（含 GT 为空却编造的情况），两者不要混着引用；")
    lines.append("- `oneid_call_rate` 必须与增益一起报：调用率低的臂，其增益只来自少数样本；")
    lines.append("- 单 seed 结果只能当趋势，正式结论需要 seed 2/3 复制。")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    progress = wait_done()

    run_module(["-m", "scripts.exp_1_1.analyze", "--traces", str(TRACES), "--rescore", "--out", str(OUT)])
    run_module(["-m", "scripts.exp_1_1.metrics", "--traces", str(TRACES), "--rescore",
                "--cells", ",".join(CELLS), "--out", str(OUT)])

    text = final_markdown(progress)
    (OUT / "FINAL.md").write_text(text, encoding="utf-8")
    log(f"完成：{OUT/'FINAL.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
