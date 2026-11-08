"""跑一个臂：驱动 AgentRunner，落盘可复现的 trace。

为什么不用 ``python -m tool_server.tf_eval``（``scripts/eval/eval_e2e.sh`` 的入口）：
它 ``from .models import get_model``，而 ``tool_server/tf_eval/models/`` 在当前树里
不存在（从未进过 git），所以那条路现在是断的。``AgentRunner`` 是 UI 与 CLI 共用的
同一条循环，且 ``llm``/``tool_manager`` 都可注入，正好用来做受控消融。

受控点：
* 系统提示词由臂决定（AgentRunner 会硬编码 RS_AGENT_PROMPT，这里用子类覆盖）；
* ``CrossModalInterpret`` 在"无 OneID"臂里由工具策略硬拦截，不依赖“worker 有没有注册”；
* 每个样本落盘 arm/model/prompt 指纹，事后能证明一行结果属于哪个臂。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import arms as arms_mod

ONEID_TOOL = arms_mod.ONEID_TOOL


class ToolPolicy:
    """工具策略代理：按臂放行/拦截 OneID，并统计调用。

    拦截发生在模型侧动作被接受之后、真正派发之前，所以 trace 里会留下
    ``oneid_blocked``，便于区分"没想调"和"想调但被关掉"。
    """

    def __init__(self, inner: Any, allow_oneid: bool):
        self._inner = inner
        self.allow_oneid = allow_oneid
        self.calls: List[str] = []
        self.blocked: List[str] = []

    def call_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if name == ONEID_TOOL and not self.allow_oneid:
            self.blocked.append(name)
            return {"error_code": 3, "text": f"{ONEID_TOOL} is disabled in this arm."}
        self.calls.append(name)
        return self._inner.call_tool(name, params)

    def __getattr__(self, item: str) -> Any:  # 其余属性透传
        return getattr(self._inner, item)


def _build_runner_class(prompt: str):
    from tool_server.tf_eval.agent_runtime.agent_runner import AgentRunner, LocalLLM

    class ArmRunner(AgentRunner):
        """只改系统提示词的注入方式，其余行为与线上完全一致。"""

        def _get_llm(self) -> Any:
            if self.llm is None:
                self.llm = LocalLLM(str(self.config.pretrained_path), system_prompt=prompt)
            elif hasattr(self.llm, "system_prompt"):
                self.llm.system_prompt = prompt
            return self.llm

    return ArmRunner, LocalLLM


def build_registry(sample: Dict[str, Any], data_root: Path) -> Dict[str, Any]:
    images = [(data_root / path).resolve() for path in sample["images"]]
    registry: Dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "primary_image": str(images[0]),
    }
    if sample["family"] == "task4":
        registry.update(optical_image=str(images[0]), sar_image=str(images[1]))
    else:
        registry.update(time1_image=str(images[0]), time2_image=str(images[1]))
    return registry


def read_done(path: Path) -> set:
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["sample_id"])
            except Exception:
                continue
    return done


def run_arm(
    arm: arms_mod.Arm,
    samples: List[Dict[str, Any]],
    *,
    out_dir: Path,
    data_root: Path,
    model_path: Optional[str] = None,
    max_rounds: int = 16,
    max_new_tokens: int = 512,
    greedy: bool = False,
    llm_factory: Optional[Callable[[str], Any]] = None,
    tool_manager_factory: Optional[Callable[[], Any]] = None,
    controller: Optional[str] = None,
    resume: bool = True,
    limit: Optional[int] = None,
) -> Path:
    from tool_server.tf_eval.agent_runtime.run_config import AgentRunConfig

    prompt = arms_mod.build_system_prompt(arm.oneid)
    ArmRunner, _ = _build_runner_class(prompt)

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"{arm.key}.jsonl"
    done = read_done(trace_path) if resume else set()

    if tool_manager_factory is None:
        from tool_server.tool_workers.tool_manager.base_manager import ToolManager

        tool_manager_factory = lambda: ToolManager(controller)  # noqa: E731

    policy = ToolPolicy(tool_manager_factory(), allow_oneid=arm.oneid)
    llm = llm_factory(prompt) if llm_factory else None
    # LocalLLM 默认 256 token；4B 模型的 thought 稍长就会被截断，JSON 不完整
    # 直接导致 no_action 死循环。四臂统一放宽到同一上限。
    if llm is not None and hasattr(llm, "max_new_tokens"):
        llm.max_new_tokens = max_new_tokens
    if greedy and llm is not None and hasattr(llm, "model"):
        # checkpoint 的 generation_config 默认 do_sample=True/temperature=0.7，
        # 同一 prompt 两次会给出不同答案；--greedy 关掉采样，让结果可复现。
        config = getattr(llm.model, "generation_config", None)
        if config is not None:
            config.do_sample = False
            for key in ("temperature", "top_p", "top_k"):
                if hasattr(config, key):
                    setattr(config, key, None)

    state = AgentRunConfig(max_rounds=max_rounds, pretrained_path=model_path)
    runner = ArmRunner(config=state, llm=llm, tool_manager=policy)

    todo = [s for s in samples if s["sample_id"] not in done]
    if limit is not None:
        todo = todo[:limit]

    with open(trace_path, "a", encoding="utf-8") as handle:
        for sample in todo:
            policy.calls.clear()
            policy.blocked.clear()
            # 自检钩子：mock LLM 需要知道本条 GT；真实模型没有该属性。
            setter = getattr(runner.llm, "set_answer", None)
            if setter is not None:
                setter(sample["label"])
            started = time.perf_counter()
            try:
                trace = runner.run_sample(
                    sample["question"],
                    build_registry(sample, data_root),
                    max_rounds=max_rounds,
                    sample_id=sample["sample_id"],
                )
                error = None
            except Exception as exc:  # 单样本失败不该中断整臂
                trace = {"sample_id": sample["sample_id"], "final_answer": "", "rounds": 0,
                         "raw_tool_calls": 0, "error_tags": ["harness_exception"]}
                error = f"{type(exc).__name__}: {exc}"

            record = {
                "arm": arm.key,
                "arm_label": arm.label,
                "oneid": arm.oneid,
                "model": model_path,
                "prompt_sha": arms_mod.prompt_sha(prompt),
                "sample_id": sample["sample_id"],
                "family": sample["family"],
                "pair_id": sample["pair_id"],
                "target": sample["target"],
                "question": sample["question"],
                "label": sample["label"],
                "final_answer": trace.get("final_answer", ""),
                "terminated": trace.get("terminated", False),
                "oneid_calls": sum(1 for name in policy.calls if name == ONEID_TOOL),
                "oneid_blocked": list(policy.blocked),
                "trace": {
                    "rounds": trace.get("rounds", 0),
                    "raw_tool_calls": trace.get("raw_tool_calls", 0),
                    "tool_errors": sum(
                        1 for item in trace.get("tool_responses", [])
                        if (item.get("response") or {}).get("error_code") not in (0, None)
                    ),
                    "error_tags": trace.get("error_tags", []),
                    "wall_time": round(time.perf_counter() - started, 3),
                    "llm_time": trace.get("llm_time", 0.0),
                    "tool_time": trace.get("tool_time", 0.0),
                },
                "error": error,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return trace_path


def load_records(paths) -> List[Dict[str, Any]]:
    """接受单个路径或路径列表。"""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    records: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records
