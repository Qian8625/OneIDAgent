#!/usr/bin/env python3
"""2x2 消融主入口：无 SFT / SFT × OneID 关 / 开，只在 task4/task5 上跑。

    # 0) 只做体检：数据、权重、环境开关
    python scripts/eval/e2e_task45/run_matrix.py --check

    # 1) 不跑模型，打印四臂将用的提示词指纹与样本量
    python scripts/eval/e2e_task45/run_matrix.py --dry-run

    # 2) 无 GPU 自检：mock LLM + fake tools，验证矩阵/闸门/评分链路
    python scripts/eval/e2e_task45/run_matrix.py --mock --limit 4

    # 3) 真跑（需要 controller + 最小栈已启动，见 README）
    ONEID_ABLATION_BASE_MODEL=/path/base \
    ONEID_ABLATION_SFT_MODEL=/path/sft \
    python scripts/eval/e2e_task45/run_matrix.py --tasks task4,task5

出口码：0 正常；1 有臂未配置或被跳过；2 参数/环境错误。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # 允许 `python scripts/eval/e2e_task45/run_matrix.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.eval.e2e_task45 import arms as arms_mod  # noqa: E402
from scripts.eval.e2e_task45 import data as data_mod  # noqa: E402
from scripts.eval.e2e_task45 import score as score_mod  # noqa: E402
from scripts.eval.e2e_task45 import stack as stack_mod  # noqa: E402


from scripts.eval.e2e_task45.mocks import FakeTools, MockLLM  # 自检替身


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.environ.get("ONEID_DATA_ROOT", ""),
                        help="数据集根目录（含 test.jsonl 与 match_images/time_images）")
    parser.add_argument("--dataset", default="", help="显式指定 jsonl；默认 <data>/test.jsonl")
    parser.add_argument("--tasks", default="task4,task5")
    parser.add_argument("--arms", default="all", help="all 或逗号分隔的 " + ",".join(arms_mod.ORDER))
    parser.add_argument("--question-mode", default="neutral", choices=("neutral", "original"))
    parser.add_argument("--out", default="outputs/e2e_task45")
    parser.add_argument("--limit", type=int, default=None, help="每个臂最多跑多少样本")
    parser.add_argument("--per-family", type=int, default=None,
                        help="每个任务各取 N 条（冒烟用，避免前 N 条全是 task4）")
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--greedy", action="store_true",
                        help="关掉采样（默认沿用 checkpoint 的 do_sample/temperature），让结果可复现")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--sft-model", default=None)
    parser.add_argument("--controller", default="http://127.0.0.1:20001")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--strict", action="store_true", help="体检出现 warn 也返回 1")
    return parser.parse_args(argv)


def _selected_arms(spec: str):
    if spec.strip() == "all":
        return [arms_mod.ARMS[key] for key in arms_mod.ORDER]
    keys = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = [key for key in keys if key not in arms_mod.ARMS]
    if unknown:
        raise SystemExit(f"未知臂：{', '.join(unknown)}；可选 {', '.join(arms_mod.ORDER)}")
    return [arms_mod.ARMS[key] for key in keys]


def main(argv=None) -> int:
    args = _parse_args(argv)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    bad = [t for t in tasks if t not in data_mod.FAMILIES]
    if bad:
        raise SystemExit(f"未知任务：{', '.join(bad)}；只支持 task4/task5")

    data_root = Path(args.data or ".").expanduser().resolve()
    dataset = Path(args.dataset).expanduser() if args.dataset else data_root / "test.jsonl"
    out_dir = Path(args.out).expanduser().resolve()
    selected = _selected_arms(args.arms)

    if not dataset.is_file():
        print(f"[error] 找不到数据集：{dataset}\n        用 --data 指向 ~/00_Data/RSTI 或 --dataset 指定 jsonl")
        return 2

    samples = data_mod.load_samples(dataset, tasks, limit=args.limit,
                                    question_mode=args.question_mode, per_family=args.per_family)
    stats = data_mod.summarize(samples)
    print(f"[data] {dataset}  样本 {stats['total']}  指纹 {data_mod.fingerprint(samples)}")
    print(f"       by_family={stats['by_family']} by_target={stats['by_target']} "
          f"with_attributes={stats['with_attributes']}")

    if args.check:
        checks = stack_mod.preflight(data_root, tasks, oneid_arms=any(arm.oneid for arm in selected))
        width = max(len(c["item"]) for c in checks)
        for check in checks:
            print(f"[{check['level']:5}] {check['item']:<{width}}  {check['detail']}")
        print()
        for arm in selected:
            model = arms_mod.resolve_model(arm, args.base_model, args.sft_model)
            ready = "ready" if model and Path(model).exists() else "MISSING"
            print(f"[arm] {arms_mod.describe(arm, model)}  [{ready}]")
        warn = any(c["level"] != "ok" for c in checks)
        missing = []
        for arm in selected:
            model = arms_mod.resolve_model(arm, args.base_model, args.sft_model)
            if not model or not Path(model).exists():
                missing.append(arm.key)
        if missing or (args.strict and warn):
            print(f"\n[exit 1] 未就绪的臂：{', '.join(missing) or '(权重齐备)'}")
            return 1
        print("\n[ok] 体检通过")
        return 0

    if args.dry_run:
        for arm in selected:
            prompt = arms_mod.build_system_prompt(arm.oneid)
            model = arms_mod.resolve_model(arm, args.base_model, args.sft_model)
            print(f"[dry] {arms_mod.describe(arm, model)}  prompt_sha={arms_mod.prompt_sha(prompt)} "
                  f"tools={len(arms_mod.bullet_tools(prompt))}")
        print(f"[dry] 未启动任何进程；样本 {stats['total']} 条，输出目录 {out_dir}")
        return 0

    from scripts.eval.e2e_task45 import runner as runner_mod

    paths = []
    failures = []
    for arm in selected:
        model = arms_mod.resolve_model(arm, args.base_model, args.sft_model)
        if not args.mock:
            if not model:
                print(f"[skip] {arm.key}: 未配置 {arm.model_env}")
                failures.append(arm.key)
                continue
            if not Path(model).exists():
                print(f"[skip] {arm.key}: 权重不存在 {model}")
                failures.append(arm.key)
                continue

        llm_factory = (lambda prompt, _c=args.mock: MockLLM()) if args.mock else None
        tools_factory = (lambda: FakeTools()) if args.mock else None

        os.environ["ONEID_CONTROLLER_ADDR"] = args.controller

        print(f"[run ] {arm.key}  oneid={'on' if arm.oneid else 'off'}  "
              f"model={model or '<mock>'}  samples={len(samples)}")
        path = runner_mod.run_arm(
            arm, samples, out_dir=out_dir / "traces", data_root=data_root,
            model_path=model, max_rounds=args.max_rounds, max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            llm_factory=llm_factory, tool_manager_factory=tools_factory,
            controller=args.controller, resume=not args.no_resume,
        )
        paths.append(path)
        # 一个臂一个 4B 模型；四个臂串行时及时释放显存，避免累计 OOM。
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    if not paths:
        print("[error] 没有任何臂可跑")
        return 1

    records = runner_mod.load_records(paths)
    gt_index = {sample["sample_id"]: sample["gt"] for sample in samples}
    scored = []
    for record in records:
        gt = gt_index.get(record["sample_id"])
        if gt is None:  # 续跑遗留的样本：不在本次 --limit 范围内，跳过而不是崩
            print(f"[skip] trace 里的 {record['sample_id']} 不在本次样本集内，未计分")
            continue
        record["score"] = score_mod.score_answer(gt, record["final_answer"], record["family"])
        scored.append(record)
    records = scored
    rows = score_mod.aggregate(records)
    written = score_mod.write_outputs(records, rows, out_dir)
    print(f"\n[out ] {written['csv']}\n       {written['md']}\n")
    print(score_mod.render_markdown(rows))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
