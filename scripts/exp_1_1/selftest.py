#!/usr/bin/env python3
"""CPU 自检：验证六个 cell 的提示词不变量、探针构造、统计函数和 mock 跑通。

不需要 GPU / controller / API key；但 AgentRunner 的导入链要 torch，
所以用带 torch 的解释器跑：

    <python-with-torch> -m scripts.exp_1_1.selftest
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.e2e_task45 import arms as harness_arms  # noqa: E402
from scripts.exp_1_1 import analyze, config  # noqa: E402

PASSED: list = []
FAILED: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    print(("PASS  " if ok else "FAIL  ") + name + (f" | {detail}" if detail else ""))
    return ok


def fixture_rows() -> list:
    def row(images, question, answer):
        return {"images": images, "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "<image>\n<image>\n" + question},
            {"role": "assistant", "content": json.dumps(
                {"thought": "d", "actions": [{"name": "Terminate", "arguments": {"ans": answer}}]})},
        ]}

    return [
        row(["match_images/task4_000001_RGB.png", "match_images/task4_000001_SAR.png"],
            "Compare the optical (RGB) and SAR images of pair task4_000001. Report (a) counts and (b) attributes.",
            "Workbook replay found 1 distinct bridge object(s) in task4_000001: 0 matched, 1 RGB-only and 0 SAR-only. "
            "The largest RGB bridge detection has subtype=viaduct, color=gray, size_bin=large, orientation=horizontal."),
        row(["time_images/task5_000101_time1.png", "time_images/task5_000101_time2.png"],
            "Compare the time1 and time2 images of pair task5_000101. Report (a) counts and (b) attributes.",
            "Workbook replay found 5 appeared, 0 disappeared, 1 static and 0 moved aircraft object(s) in task5_000101. "
            "The largest time1 aircraft detection has subtype=civilian_airliner, color=white, size_bin=medium, "
            "orientation=horizontal, visibility=partial."),
    ]


def test_config() -> None:
    prompts = {c: config.build_prompt(c) for c in config.CONDITIONS}
    none_tools = harness_arms.bullet_tools(prompts["none"])
    det_tools = harness_arms.bullet_tools(prompts["det"])
    oneid_tools = harness_arms.bullet_tools(prompts["oneid"])
    check("闭卷提示词没有工具条目", none_tools == [], str(none_tools[:3]))
    check("oneid 与 det 只差 CrossModalInterpret",
          set(oneid_tools) - set(det_tools) == {harness_arms.ONEID_TOOL})
    check("工具条目数为 0/21/22", (len(none_tools), len(det_tools), len(oneid_tools)) == (0, 21, 22),
          f"{len(none_tools)}/{len(det_tools)}/{len(oneid_tools)}")
    check("提示词按条件缓存（同一对象）", config.build_prompt("det") is prompts["det"])
    check("6 个 cell、4 臂", len(config.CELLS) == 6 and len(config.ARMS) == 4)
    check("cell 解析", config.split_cell("sft_oneid") == ("sft", "oneid"))
    check("闭卷提示词带硬性说明", "No tools are available" in prompts["none"])


def test_stats() -> None:
    keep = {"ObjectDetection", "DenoDetV2", "SmallObjectDetection", "CrossModalInterpret", "Terminate"}
    filtered = config.build_prompt_filtered("oneid", keep)
    recipe = config.ONEID_RECIPE.strip().splitlines()
    check("registry filtering preserves evidence recipe", all(line in filtered for line in recipe))
    check("registry filtering removes unavailable tools",
          set(harness_arms.bullet_tools(filtered)) == keep)


    check("McNemar 完全一致 -> p=1", analyze.mcnemar_exact(0, 0) == 1.0)
    check("McNemar 对称", analyze.mcnemar_exact(2, 9) == analyze.mcnemar_exact(9, 2))
    p = analyze.mcnemar_exact(10, 2)
    check("McNemar 偏向一侧 -> p 小", p < 0.05, f"p={p}")

    a = {"s1": 0.0, "s2": 1.0, "s3": 0.0, "s4": 0.0}
    b = {"s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 0.0}
    result = analyze.paired(a, b)
    check("配对差值计算", result["n"] == 4 and result["diff"] == 50.0, str(result))
    ci = analyze.bootstrap_diff(a, b, {"s1": "p1", "s2": "p2", "s3": "p3", "s4": "p4"}, rounds=500)
    check("bootstrap 返回区间且包含点估计", ci is not None and ci[0] <= 50.0 <= ci[1], str(ci))


def test_metrics() -> None:
    def rec(cell, family, ok):
        return {"cell": cell, "family": family, "probe": None, "oneid_calls": 0,
                "trace": {"tool_errors": 0, "raw_tool_calls": 1, "rounds": 2, "error_tags": []},
                "score": {"counts_ok": ok, "answer_ok": ok, "attr_hits": 4 if ok else 0,
                          "empty_answer": False}}

    rows = analyze.cell_metrics([rec("base_det", "task4", True), rec("base_det", "task4", False)])
    row = rows[0]
    check("cell_metrics 计算 50%", row["answer_acc"] == 50.0 and row["n"] == 2, str(row))


def test_run_mock(tmp: Path) -> None:
    from scripts.eval.e2e_task45.mocks import FakeTools, MockLLM
    from scripts.exp_1_1 import run as run_mod

    dataset = tmp / "test.jsonl"
    dataset.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in fixture_rows()), encoding="utf-8")
    from scripts.eval.e2e_task45 import data as harness_data

    samples = harness_data.load_samples(dataset, ("task4", "task5"))
    trace = run_mod.run_samples(
        "base_det", samples, seed=7, trace_path=tmp / "t.jsonl", data_root=tmp, resume=False,
        llm_factory=lambda prompt, seed: MockLLM(),
        tool_manager_factory=FakeTools,
    )
    records = run_mod.load_records(trace)
    check("mock 跑出两条记录", len(records) == 2)
    check("记录带 cell/condition/seed", all(r["cell"] == "base_det" and r["seed"] == 7 for r in records))
    check("mock 答案满分", all(r["score"]["answer_ok"] for r in records),
          str([r["score"]["answer_ok"] for r in records]))
    check("prompt_sha 与 config 一致",
          all(r["prompt_sha"] == harness_arms.prompt_sha(config.build_prompt("det")) for r in records))

    # 闭卷条件：任何工具调用都必须被拦截
    policy = run_mod.Policy(FakeTools(), "none")
    resp = policy.call_tool("ObjectDetection", {"image": "x"})
    check("闭卷拦截工具调用", resp["error_code"] == 3 and policy.blocked == ["ObjectDetection"], str(resp))
    policy_det = run_mod.Policy(FakeTools(), "det")
    resp = policy_det.call_tool("CrossModalInterpret", {})
    check("det 拦截 OneID", resp["error_code"] == 3 and policy_det.blocked == ["CrossModalInterpret"])


def main() -> int:
    test_config()
    test_stats()
    test_metrics()
    tmp = Path(tempfile.mkdtemp(prefix="exp_1_1_"))
    test_run_mock(tmp)
    analyze.main(["--traces", str(tmp / "t.jsonl"), "--out", str(tmp / "report")])
    check("analyze 产出 REPORT.md", (tmp / "report" / "REPORT.md").is_file())

    total = len(PASSED) + len(FAILED)
    print(f"\nselftest: {len(PASSED)}/{total} passed")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
