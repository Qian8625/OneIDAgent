"""exp_1_1 的 runner：按 cell（weights x condition）跑样本，落盘可审计的 trace。

与 e2e_task45 的 runner 的差别只有三点，其余（AgentRunner、registry、评分）完全复用：
  1. 多一个 condition 维度，决定提示词与工具白名单；
  2. 支持 --seed：每轮生成前 torch.manual_seed，保证同一 (cell, sample, seed) 可复现；
  3. trace 里多记 cell / condition / seed / 工具调用序列。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.e2e_task45 import arms as harness_arms  # noqa: E402
from scripts.eval.e2e_task45 import data as harness_data  # noqa: E402
from scripts.eval.e2e_task45 import score as harness_score  # noqa: E402
from scripts.exp_1_1 import config  # noqa: E402

ONEID_TOOL = "CrossModalInterpret"


def _compact_args(value, depth=0):
    """把参数压成可比较的形态：检测列表只留长度与首项，避免 trace 膨胀。"""
    if depth > 3:
        return "..."
    if isinstance(value, dict):
        return {str(k): _compact_args(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 3 and all(isinstance(item, dict) for item in value):
            return {"__n__": len(value), "__first__": _compact_args(value[0], depth + 1)}
        return [_compact_args(item, depth + 1) for item in value]
    if isinstance(value, str) and len(value) > 120:
        return value[:120]
    return value


class Policy:
    """工具白名单：闭卷全拦；det 拦 OneID；oneid 全放行。记录调用与拦截。"""

    def __init__(self, inner: Any, condition: str):
        self._inner = inner
        self.condition = condition
        self.calls: List[str] = []
        self.blocked: List[str] = []

    def call_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.condition == "none":
            self.blocked.append(name)
            return {"error_code": 3,
                    "text": "No tools are available in this condition. Answer from prior knowledge."}
        if name == ONEID_TOOL and self.condition != "oneid":
            self.blocked.append(name)
            return {"error_code": 3, "text": f"{ONEID_TOOL} is disabled in this arm."}
        self.calls.append(name)
        return self._inner.call_tool(name, params)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


def _seeded_llm(model_path: str, prompt: str, seed: int, max_new_tokens: int):
    from tool_server.tf_eval.agent_runtime.agent_runner import LocalLLM

    class SeededLocalLLM(LocalLLM):
        def __init__(self) -> None:
            super().__init__(model_path, system_prompt=prompt)
            self.seed = seed
            self.round = 0
            self.max_new_tokens = max_new_tokens

        def generate(self, conversation, extra_system_prompt: str = "") -> str:
            import torch

            torch.manual_seed(self.seed * 100000 + self.round)
            self.round += 1
            return super().generate(conversation, extra_system_prompt)

    return SeededLocalLLM()


def registry_for(sample: Dict[str, Any], data_root: Path,
                 override: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """override 用于探针：改写 primary/optical/sar/time1/time2 的图片路径。"""
    images = [(data_root / path).resolve() for path in sample["images"]]
    registry: Dict[str, Any] = {"sample_id": sample["sample_id"], "primary_image": str(images[0])}
    if sample["family"] == "task4":
        registry.update(optical_image=str(images[0]), sar_image=str(images[1]))
    else:
        registry.update(time1_image=str(images[0]), time2_image=str(images[1]))
    if override:
        registry.update(override)
    return registry


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                out.add(json.loads(line)["sample_id"])
            except Exception:
                continue
    return out


def run_samples(
    cell: str,
    samples: List[Dict[str, Any]],
    *,
    seed: int,
    trace_path: Path,
    data_root: Path,
    controller: str = "http://127.0.0.1:20001",
    max_rounds: int = 16,
    max_new_tokens: int = 512,
    model_path: Optional[str] = None,
    resume: bool = True,
    llm_factory: Optional[Callable[[str, int], Any]] = None,
    tool_manager_factory: Optional[Callable[[], Any]] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
    tools_mode: str = "prompt",
    registry_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Path:
    from tool_server.tf_eval.agent_runtime.agent_runner import AgentRunner
    from tool_server.tf_eval.agent_runtime.run_config import AgentRunConfig

    weights, condition = config.split_cell(cell)
    if tools_mode == "registry":
        keep = config.available_keep(controller)
        prompt = config.build_prompt_filtered(condition, keep)
        extra_fields = dict(extra_fields or {})
        extra_fields["prompt_tools"] = sorted(keep)
    else:
        prompt = config.build_prompt(condition)
    model_path = model_path or config.model_for(weights)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(trace_path) if resume else set()

    if tool_manager_factory is None:
        from tool_server.tool_workers.tool_manager.base_manager import ToolManager

        tool_manager_factory = lambda: ToolManager(controller)  # noqa: E731

    policy = Policy(tool_manager_factory(), condition)
    llm = (llm_factory(prompt, seed) if llm_factory
           else _seeded_llm(str(model_path), prompt, seed, max_new_tokens))

    class CellRunner(AgentRunner):
        def _get_llm(self) -> Any:
            if hasattr(self.llm, "system_prompt"):
                self.llm.system_prompt = prompt
            return self.llm

    runner = CellRunner(config=AgentRunConfig(max_rounds=max_rounds, pretrained_path=model_path),
                        llm=llm, tool_manager=policy)
    setter = getattr(llm, "set_answer", None)

    todo = [s for s in samples if s["sample_id"] not in done]
    with open(trace_path, "a", encoding="utf-8") as handle:
        for sample in todo:
            policy.calls.clear()
            policy.blocked.clear()
            if setter is not None:
                setter(sample["label"])
            started = time.perf_counter()
            try:
                registry = (registry_fn or (lambda s: registry_for(s, data_root)))(sample)
                trace = runner.run_sample(sample["question"], registry,
                                          max_rounds=max_rounds, sample_id=sample["sample_id"])
                error = None
            except Exception as exc:
                trace = {"sample_id": sample["sample_id"], "final_answer": "", "rounds": 0,
                         "raw_tool_calls": 0, "error_tags": ["harness_exception"]}
                error = f"{type(exc).__name__}: {exc}"

            score = harness_score.score_answer(sample["gt"], trace.get("final_answer", ""), sample["family"])
            record = {
                "cell": cell, "weights": weights, "condition": condition, "seed": seed,
                "model": model_path, "prompt_sha": harness_arms.prompt_sha(prompt),
                "sample_id": sample["sample_id"], "family": sample["family"],
                "pair_id": sample["pair_id"], "target": sample["target"],
                "question": sample["question"], "label": sample["label"],
                "final_answer": trace.get("final_answer", ""),
                "terminated": trace.get("terminated", False),
                "tool_sequence": list(policy.calls),
                "oneid_calls": sum(1 for n in policy.calls if n == ONEID_TOOL),
                "oneid_blocked": list(policy.blocked),
                "action_log": [
                    {"round": item.get("round"), "api_name": item.get("api_name"),
                     "error": item.get("error"),
                     "arg_keys": sorted((item.get("arguments") or {}).keys()),
                     "n_rgb": len((item.get("arguments") or {}).get("rgb_detections") or []),
                     "n_sar": len((item.get("arguments") or {}).get("sar_detections") or []),
                     "prov": sorted(((item.get("arguments") or {}).get("evidence_provenance") or {}).keys()),
                     "args": _compact_args(item.get("arguments") or {})}
                    for item in trace.get("actions", []) if isinstance(item, dict)
                ],
                "tool_log": [
                    {"round": item.get("round"), "api_name": item.get("api_name"),
                     "error_code": (item.get("response") or {}).get("error_code"),
                     "n_detections": len((item.get("response") or {}).get("detections") or [])
                     or len(((item.get("response") or {}).get("structured") or {}).get("detections") or []),
                     "text": str((item.get("response") or {}).get("text"))[:80]}
                    for item in trace.get("tool_responses", []) if isinstance(item, dict)
                ],
                "trace": {
                    "rounds": trace.get("rounds", 0),
                    "raw_tool_calls": trace.get("raw_tool_calls", 0),
                    "tool_errors": sum(1 for item in trace.get("tool_responses", [])
                                       if (item.get("response") or {}).get("error_code") not in (0, None)),
                    "error_tags": trace.get("error_tags", []),
                    "wall_time": round(time.perf_counter() - started, 3),
                    "llm_time": trace.get("llm_time", 0.0),
                    "tool_time": trace.get("tool_time", 0.0),
                },
                "score": score,
                "error": error,
            }
            if extra_fields:
                record.update(extra_fields)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return trace_path


def load_records(paths) -> List[Dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    records: List[Dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=os.environ.get("ONEID_DATA_ROOT", str(ROOT / "data")))
    parser.add_argument("--tasks", default="task4,task5")
    parser.add_argument("--cells", default="arms", help="arms(默认四格) | all(6格) | 逗号分隔")
    parser.add_argument("--tools", default="prompt", choices=("prompt", "registry"),
                        help="prompt=提示词宣告全部工具（旧行为）；registry=按 controller 实际注册裁剪")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--per-family", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--controller", default="http://127.0.0.1:20001")
    parser.add_argument("--out", default="outputs/exp_1_1")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    if args.cells == "arms":
        cells = list(config.ARMS)
    elif args.cells == "all":
        cells = list(config.CELLS)
    else:
        cells = [c.strip() for c in args.cells.split(",") if c.strip()]
        bad = [c for c in cells if c not in config.CELLS]
        if bad:
            raise SystemExit(f"unknown cells: {bad}; choose from {config.CELLS}")

    # 硬门槛：注册表为空或关键工具缺失，直接拒绝开跑（E2E 任何实验都必须先过这一关）
    from scripts.exp_1_1 import doctor

    try:
        tools = doctor.list_models(args.controller)
    except Exception as exc:
        print(f"[error] 取不到 controller 注册表：{type(exc).__name__}: {exc}")
        return 2
    print(f"[registry] {len(tools)} tools: {tools}")
    if not tools:
        print("[error] /list_models 为空：先 `python -m scripts.exp_1_1.doctor restart-all` 再跑")
        return 2
    required = set()
    for cell in cells:
        weights, condition = config.split_cell(cell)
        if condition in ("det", "oneid"):
            required |= {"ObjectDetection", "DenoDetV2", "SmallObjectDetection"}
        if condition == "oneid":
            required |= {"CrossModalInterpret"}
    missing = sorted(required - set(tools))
    if missing:
        print(f"[error] 缺少必需工具 {missing}；先 `python -m scripts.exp_1_1.doctor restart <名字>` 或 restart-all")
        return 2

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    data_root = Path(args.data).expanduser().resolve()
    dataset = data_root / "test.jsonl"
    if not dataset.is_file():
        print(f"[error] 找不到 {dataset}")
        return 2

    samples = harness_data.load_samples(dataset, tasks, limit=args.limit, per_family=args.per_family)
    print(f"[data] {dataset} samples={len(samples)} fingerprint={harness_data.fingerprint(samples)}")
    print("[cells]", ", ".join(cells))

    if args.check:
        missing = []
        for cell in cells:
            print("[cell]", config.describe(cell))
            weights = config.split_cell(cell)[0]
            path = config.model_for(weights)
            if not path or not Path(path).exists():
                missing.append(cell)
        if missing:
            print(f"[exit 1] 缺权重：{missing}")
            return 1
        print("[ok] 体检通过")
        return 0

    if args.dry_run:
        for cell in cells:
            weights, condition = config.split_cell(cell)
            prompt = config.build_prompt(condition)
            print(f"[dry] {cell:12} prompt_sha={harness_arms.prompt_sha(prompt)} "
                  f"tools={len(harness_arms.bullet_tools(prompt))} samples={len(samples)}")
        return 0

    out_dir = Path(args.out).expanduser().resolve() / "traces"
    paths = []
    for cell in cells:
        print(f"[run ] {cell} seed={args.seed} samples={len(samples)}", flush=True)
        paths.append(run_samples(
            cell, samples, seed=args.seed, trace_path=out_dir / f"{cell}_seed{args.seed}.jsonl",
            data_root=data_root, controller=args.controller, max_rounds=args.max_rounds,
            max_new_tokens=args.max_new_tokens, resume=not args.no_resume,
            tools_mode=args.tools))
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    records = load_records(paths)
    print(f"[done] {len(records)} records -> {out_dir}")
    for cell in cells:
        rows = [r for r in records if r["cell"] == cell]
        if not rows:
            continue
        ok = sum(1 for r in rows if r["score"]["answer_ok"])
        print(f"  {cell:12} n={len(rows):4} answer_ok={ok}/{len(rows)} "
              f"oneid_calls={sum(1 for r in rows if r['oneid_calls'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
