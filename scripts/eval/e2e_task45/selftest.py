#!/usr/bin/env python3
"""CPU 自检：不碰 GPU、不连 controller、不要 API key。

它证明的是"实验装置"本身没有错，而不是模型好坏：
  1. 两臂提示词只差 CrossModalInterpret 一条（结构断言，不是字符串感觉）；
  2. 依赖证据齐备时：无 OneID 臂在派发前被拦，OneID 臂真的派发出去；
  3. 依赖证据缺失时：两臂都先撞"依赖门禁"（门禁在臂策略之前），
     所以事后分析必须看 oneid_blocked 才能分清"关掉了"和"没证据"；
  4. 评分器对满分答案给满分、对编造属性给 0；
  5. 汇总表按 臂×任务 分行，mock 满分答案在表里是 100%。

需要带 torch 的解释器（AgentRunner 的导入链会拉 tool_server.utils.utils）：
    <python-with-torch> scripts/eval/e2e_task45/selftest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.eval.e2e_task45 import arms as arms_mod  # noqa: E402
from scripts.eval.e2e_task45 import data as data_mod  # noqa: E402
from scripts.eval.e2e_task45 import score as score_mod  # noqa: E402
from scripts.eval.e2e_task45.mocks import FakeTools, MockLLM, crossmodal_script  # noqa: E402

PASSED: list = []
FAILED: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    print(("PASS  " if ok else "FAIL  ") + name + (f" | {detail}" if detail else ""))
    return ok


def fixture_rows() -> list:
    system = {"role": "system", "content": "You are a remote-sensing tool-use agent."}

    def row(images, question, answer):
        return {
            "images": images,
            "messages": [
                system,
                {"role": "user", "content": "<image>\n<image>\n" + question},
                {"role": "assistant", "content": json.dumps(
                    {"thought": "done", "actions": [{"name": "Terminate", "arguments": {"ans": answer}}]})},
            ],
        }

    return [
        row(["match_images/task4_000001_RGB.png", "match_images/task4_000001_SAR.png"],
            "Use workbook replay for pair task4_000001 to cross-validate bridge objects in the RGB and SAR images. "
            "Report (a) the number of distinct bridge objects after cross-view deduplication together with the "
            "matched, RGB-only and SAR-only counts, and (b) the subtype, color, size_bin and orientation of the "
            "largest RGB bridge detection.",
            "Workbook replay found 1 distinct bridge object(s) in task4_000001: 0 matched, 1 RGB-only and 0 SAR-only. "
            "The largest RGB bridge detection has subtype=viaduct, color=gray, size_bin=large, orientation=horizontal."),
        row(["time_images/task5_000101_time1.png", "time_images/task5_000101_time2.png"],
            "Use workbook replay for pair task5_000101 to compare aircraft objects between the time1 and time2 images. "
            "Report (a) how many aircraft objects appeared, disappeared, stayed static and moved, and (b) the subtype, "
            "color, size_bin and orientation of the largest time1 aircraft detection.",
            "Workbook replay found 5 appeared, 0 disappeared, 1 static and 0 moved aircraft object(s) in task5_000101. "
            "The largest time1 aircraft detection has subtype=civilian_airliner, color=white, size_bin=medium, "
            "orientation=horizontal, visibility=partial."),
    ]


def run_arms(samples, tmp: Path, tag: str, script=None) -> list:
    """跑无 OneID / OneID 两臂，返回记录列表（不按 arm 去重，保留逐样本）。"""
    from scripts.eval.e2e_task45 import runner as runner_mod

    records = []
    for key in ("base", "base_oneid"):
        payload = script or []
        paths = runner_mod.run_arm(
            arms_mod.ARMS[key], samples, out_dir=tmp / tag / key, data_root=tmp,
            model_path="<mock>", resume=False,
            llm_factory=lambda prompt, s=payload: MockLLM(s),
            tool_manager_factory=FakeTools,
        )
        records.extend(runner_mod.load_records(paths))
    return records


def pick(records, arm: str, family: str) -> dict:
    for record in records:
        if record["arm"] == arm and record["family"] == family:
            return record
    raise AssertionError(f"没有 {arm}/{family} 的记录")


def main() -> int:
    # ---- 1. 提示词结构 -------------------------------------------------
    on_prompt = arms_mod.build_system_prompt(True)
    off_prompt = arms_mod.build_system_prompt(False)
    check("OneID 臂提示词与仓库正式版逐字符一致", on_prompt == arms_mod.RS_AGENT_PROMPT)
    removed = [t for t in arms_mod.bullet_tools(on_prompt) if t not in arms_mod.bullet_tools(off_prompt)]
    check("两臂工具条目只差 CrossModalInterpret", removed == [arms_mod.ONEID_TOOL], str(removed))
    check("非 OneID 臂不再宣传该工具", arms_mod.ONEID_TOOL not in off_prompt)
    check("两臂提示词指纹不同", arms_mod.prompt_sha(on_prompt) != arms_mod.prompt_sha(off_prompt))

    # ---- 2. 数据与 GT 解析 --------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="e2e_task45_"))
    dataset = tmp / "test.jsonl"
    dataset.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in fixture_rows()), encoding="utf-8")
    samples = data_mod.load_samples(dataset, question_mode="neutral")
    task4 = [s for s in samples if s["family"] == "task4"]
    check("切出 task4 + task5 两条样本", len(samples) == 2, str([s["family"] for s in samples]))
    check("样本 id 稳定且带 pair/target", task4[0]["sample_id"].endswith("task4_000001_bridge"),
          task4[0]["sample_id"])
    check("中性问法不出现工具名", all("workbook" not in s["question"].lower() for s in samples))
    gt4 = task4[0]["gt"]
    check("task4 计数解析", gt4["counts"] == {"matched": 0, "rgb_only": 1, "sar_only": 0}, str(gt4["counts"]))
    check("task4 属性解析", gt4["attributes"]["subtype"] == "viaduct", str(gt4["attributes"]))
    gt5 = next(s["gt"] for s in samples if s["family"] == "task5")
    check("task5 计数解析", gt5["counts"] == {"appeared": 5, "disappeared": 0, "static": 1, "moved": 0},
          str(gt5["counts"]))

    # ---- 3. 评分器 -----------------------------------------------------
    perfect = score_mod.score_answer(gt4, task4[0]["label"], "task4")
    check("GT 原文得满分", perfect["answer_ok"] and perfect["attr_hits"] == 4,
          json.dumps({"answer_ok": perfect["answer_ok"], "attr_hits": perfect["attr_hits"]}))
    wrong = score_mod.score_answer(gt4, task4[0]["label"].replace("viaduct", "suspension"), "task4")
    check("属性写错则整题不通过", not wrong["answer_ok"] and wrong["counts_ok"], str(wrong["attr_hits"]))
    empty_gt = {"counts": {"matched": 1, "rgb_only": 0, "sar_only": 0},
                "distinct": 1, "attributes": {k: None for k in data_mod.ATTR_KEYS}}
    invented = score_mod.score_answer(
        empty_gt, "1 matched, 0 RGB-only and 0 SAR-only. subtype=frigate, color=gray, "
                  "size_bin=large, orientation=horizontal.", "task4")
    check("GT 无属性时编造即扣分", not invented["attrs_ok"] and invented["counts_ok"])

    # ---- 4a. 有证据：臂策略生效（task4 才有两路检测证据） --------------
    records = run_arms(task4, tmp, "with_evidence", crossmodal_script("bridge"))
    off, on = pick(records, "base", "task4"), pick(records, "base_oneid", "task4")
    check("无 OneID 臂：OneID 在派发前被拦", off["oneid_blocked"] == [arms_mod.ONEID_TOOL],
          str(off["oneid_blocked"]))
    check("无 OneID 臂：从未真正派发", off["oneid_calls"] == 0)
    check("OneID 臂：真的派发了 OneID", on["oneid_calls"] >= 1, str(on["oneid_calls"]))
    check("两臂都拿到最终答案", bool(off["final_answer"]) and bool(on["final_answer"]))

    # ---- 4b. 无证据：依赖门禁先于臂策略 -------------------------------
    gate = run_arms(task4, tmp, "no_evidence",
                    [{"name": "CrossModalInterpret",
                      "arguments": {"target": "bridge", "task": "crosscheck_exists"}}])
    gate_on = pick(gate, "base_oneid", "task4")
    check("无证据时 OneID 臂撞依赖门禁", "invalid_dependency" in gate_on["trace"]["error_tags"],
          str(gate_on["trace"]["error_tags"]))
    check("无证据时 OneID 臂也没派发", gate_on["oneid_calls"] == 0)

    # ---- 5. 汇总（满分链路） ------------------------------------------
    plain = run_arms(samples, tmp, "plain")
    for record in plain:
        record["score"] = score_mod.score_answer(
            next(s["gt"] for s in samples if s["sample_id"] == record["sample_id"]),
            record["final_answer"], record["family"])
    rows = score_mod.aggregate(plain)
    check("汇总表按 臂×任务 分行 = 4", len(rows) == 4, str([(r["arm"], r["family"]) for r in rows]))
    check("每格样本数 = 1", all(row["n"] == 1 for row in rows), str([row["n"] for row in rows]))
    check("mock 满分答案在表里是 100%", all(row["answer_acc"] == 100.0 for row in rows),
          str([(r["arm"], r["answer_acc"]) for r in rows]))
    written = score_mod.write_outputs(plain, rows, tmp / "out")
    check("写出 csv/json/md", all(Path(p).is_file() for p in written.values()),
          ", ".join(p.name for p in written.values()))

    total = len(PASSED) + len(FAILED)
    print(f"\nselftest: {len(PASSED)}/{total} passed")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
