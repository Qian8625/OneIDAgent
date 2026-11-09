"""exp_1_1：SFT 记忆效应与 OneID 增益的分离实验（2x3 条件矩阵）。

    weights  ∈ {base, sft}
    condition∈ {none 闭卷, det 仅检测, oneid 检测+OneID}
    cell     = f"{weights}_{condition}"，共 6 个；用户要的 2x2 是其中 4 格：
               base_det / base_oneid / sft_det / sft_oneid

条件之间只在"能拿到什么证据"上不同，提示词仍从仓库唯一的那份派生，并做结构断言：
    none  0 个工具条目（追加"本轮无工具"的硬性说明）
    det   21 个
    oneid 22 个，且恰好多一个 CrossModalInterpret
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.e2e_task45 import arms as harness_arms  # noqa: E402

WEIGHTS = ("base", "sft")
CONDITIONS = ("none", "det", "oneid")
CELLS = tuple(f"{w}_{c}" for w in WEIGHTS for c in CONDITIONS)
#: 用户实验表里的 4 格
ARMS = ("base_det", "base_oneid", "sft_det", "sft_oneid")

MODEL_ENV: Dict[str, str] = {
    "base": "ONEID_ABLATION_BASE_MODEL",
    "sft": "ONEID_ABLATION_SFT_MODEL",
}
DEFAULT_MODEL: Dict[str, str] = {
    "base": "cpk/base/Qwen3-4B-Instruct-2507",
    "sft": "cpk/sft/qwen3_4b_rs_text_sft_14336",
}

NO_TOOL_NOTE = (
    "\n\nNo tools are available in this condition. Do not call any tool: answer from "
    "prior knowledge only, and finish with a single Terminate action that contains your "
    "final answer."
)

#: 只加在 oneid 条件后面：把"先取证、再调 OneID"写成硬性三步。
#: 两个 ON 臂共用同一段，因此不影响臂间比较；不加这段时 4B 模型会在
#: 没有证据的情况下直接调 CrossModalInterpret，被依赖门禁判 invalid_dependency。
ONEID_RECIPE = (
    "\n\nMandatory evidence recipe (this overrides any other ordering advice):\n"
    "- For a task4 (RGB/SAR match) pair: (1) call ObjectDetection on the optical image "
    "WITHOUT a text filter; (2) call DenoDetV2 on the SAR image WITHOUT a class filter; "
    "(3) then call CrossModalInterpret with source_mode=workbook_replay, dataset_id=task4, "
    "family=match, the pair_id and target from the question, task=joint_scene.\n"
    "- For a task5 (time1/time2) pair: (1) call SmallObjectDetection on the time1 image; "
    "(2) call SmallObjectDetection on the time2 image; (3) then call CrossModalInterpret "
    "with source_mode=workbook_replay, dataset_id=task5, family=time, the pair_id and "
    "target from the question, task=temporal_changes.\n"
    "- Omit rgb_detections and sar_detections entirely: the runtime injects them from the two "
    "successful detection calls. Never write placeholder text or hand-made boxes there.\n"
    "- Never call CrossModalInterpret before both detection calls have returned error_code 0. "
    "Do not hand-write detections and do not repeat a detection call that already succeeded.\n"
)

#: 进程内工具（不通过 controller 注册，但确实可用）
OFFLINE_TOOLS = ("GetBboxFromGeotiff", "DisplayOnGeotiff")
#: 永远保留：由 agent 循环自己实现，不是 worker
ALWAYS_TOOLS = ("Terminate",)

PROMPTS: Dict[str, str] = {}
PROMPTS_FILTERED: Dict[tuple, str] = {}


def registered_tools(controller: str = "http://127.0.0.1:20001", timeout: float = 5.0) -> set:
    """问 controller 现在到底注册了哪些工具。"""
    import json as _json
    import urllib.request

    request = urllib.request.Request(
        controller.rstrip("/") + "/list_models", data=b"{}",
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return set(_json.loads(response.read().decode("utf-8")).get("models") or [])


def build_prompt_filtered(condition: str, keep: set) -> str:
    """只保留 keep 里的工具条目（其余行原样保留）。

    动机：提示词宣告 21 个工具、实际只注册 6 个，模型会去调不存在的工具，
    拿到 "Tool X not found." 后重试，被重复调用熔断，最终以空答案终止。
    """
    key = (condition, frozenset(keep))
    if key in PROMPTS_FILTERED:
        return PROMPTS_FILTERED[key]

    lines = build_prompt(condition).splitlines()
    kept, dropped = [], []
    for line in lines:
        if line.startswith("- "):
            name = line[2:].split(":", 1)[0].strip()
            if name not in keep:
                dropped.append(name)
                continue
        kept.append(line)
    prompt = "\n".join(kept)
    PROMPTS_FILTERED[key] = prompt
    return prompt


def available_keep(controller: str = "http://127.0.0.1:20001") -> set:
    """提示词里应当保留的工具 = controller 已注册 + 进程内工具 + Terminate。"""
    return set(registered_tools(controller)) | set(OFFLINE_TOOLS) | set(ALWAYS_TOOLS)


def model_for(weights: str, override: Optional[str] = None) -> Optional[str]:
    raw = override or os.environ.get(MODEL_ENV[weights]) or DEFAULT_MODEL[weights]
    if not raw:
        return None
    path = Path(raw)
    return str(path if path.is_absolute() else (ROOT / path))


def _raw_prompt(condition: str) -> str:
    if condition == "oneid":
        return harness_arms.RS_AGENT_PROMPT.replace(
            "\n\nTo solve the problem:", ONEID_RECIPE + "\nTo solve the problem:")
    if condition == "det":
        return harness_arms.build_system_prompt(False)
    if condition == "none":
        det = harness_arms.build_system_prompt(False)
        kept = [line for line in det.splitlines() if not line.startswith("- ")]
        return "\n".join(kept).rstrip() + NO_TOOL_NOTE
    raise ValueError(f"unknown condition: {condition}")


def _validate(raw: Dict[str, str]) -> None:
    none_tools = harness_arms.bullet_tools(raw["none"])
    det_tools = harness_arms.bullet_tools(raw["det"])
    oneid_tools = harness_arms.bullet_tools(raw["oneid"])
    if none_tools:
        raise AssertionError(f"闭卷条件的提示词里仍有工具条目：{none_tools}")
    if set(oneid_tools) - set(det_tools) != {harness_arms.ONEID_TOOL}:
        raise AssertionError("oneid 条件与 det 条件必须只差 CrossModalInterpret")
    if len(det_tools) + 1 != len(oneid_tools):
        raise AssertionError(f"工具条目数应为 n / n+1，实际 {len(det_tools)} / {len(oneid_tools)}")


def build_prompt(condition: str) -> str:
    """返回该条件的系统提示词；首次调用时构建三者并校验不变量。"""
    if not PROMPTS:
        raw = {name: _raw_prompt(name) for name in CONDITIONS}
        _validate(raw)
        PROMPTS.update(raw)
    try:
        return PROMPTS[condition]
    except KeyError:
        raise ValueError(f"unknown condition: {condition}")


def split_cell(cell: str) -> Tuple[str, str]:
    weights, _, condition = cell.partition("_")
    if weights not in WEIGHTS or condition not in CONDITIONS:
        raise ValueError(f"unknown cell: {cell}")
    return weights, condition


def oneid_enabled(condition: str) -> bool:
    return condition == "oneid"


def describe(cell: str, override_model: Optional[str] = None) -> str:
    weights, condition = split_cell(cell)
    model = model_for(weights, override_model)
    tools = len(harness_arms.bullet_tools(build_prompt(condition)))
    return f"{cell:12} weights={weights:4} condition={condition:5} tools={tools:2} model={model}"
