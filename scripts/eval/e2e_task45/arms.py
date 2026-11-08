"""2x2 消融的"臂"定义：权重来源、提示词、OneID 开关。

四个臂只允许在两个维度上不同：权重（基座 vs SFT）和 OneID（关 vs 开）。
其余一切——工具集、提示词模板、数据顺序、解码参数、评分——必须完全一致，
否则增益无法归因。

提示词不是复制一份新的，而是从唯一的 ``RS_AGENT_PROMPT`` 派生并做结构断言：
``build_system_prompt(False)`` 的工具条目必须恰好等于 ``build_system_prompt(True)``
去掉 ``CrossModalInterpret`` 一条。这样上游改提示词时，两臂不会悄悄漂移。
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from tool_server.tf_eval.utils.rs_agent_prompt import RS_AGENT_PROMPT

ONEID_TOOL = "CrossModalInterpret"

#: 非 OneID 臂里必须改写的两处从句（整行删除会误删 ComputeDistance 的其它说明）。
_OFF_REPLACEMENTS = (
    (
        "To call ComputeDistance, first call AddPoisLayer to add POI layers "
        "or CrossModalInterpret with return_artifact=true to add detection layers.",
        "To call ComputeDistance, first call AddPoisLayer to add POI layers.",
    ),
    (" within a CrossModalInterpret dedup layer", ""),
)

_BULLET = re.compile(r"^- ([A-Za-z][A-Za-z0-9_]*):", re.MULTILINE)


@dataclass(frozen=True)
class Arm:
    """一个实验臂。``key`` 同时用作输出目录名。"""

    key: str
    label: str
    weights: str  # "base" | "sft"
    oneid: bool

    @property
    def model_env(self) -> str:
        return "ONEID_ABLATION_BASE_MODEL" if self.weights == "base" else "ONEID_ABLATION_SFT_MODEL"


ARMS: Dict[str, Arm] = {
    arm.key: arm
    for arm in (
        Arm("base", "No SFT · Base Agent + Tools", "base", False),
        Arm("base_oneid", "No SFT · Base Agent + Tools + OneID", "base", True),
        Arm("sft", "SFT · SFT Agent + Tools", "sft", False),
        Arm("sft_oneid", "SFT · SFT Agent + Tools + OneID", "sft", True),
    )
}

#: 报告里的行列顺序。
ORDER: tuple = ("base", "base_oneid", "sft", "sft_oneid")


def bullet_tools(prompt: str) -> List[str]:
    """提示词里 "- Name: ..." 形式的工具条目名，按出现顺序。"""
    return _BULLET.findall(prompt)


def build_system_prompt(oneid: bool) -> str:
    """返回该臂的系统提示词。``oneid=True`` 时逐字符等于仓库里的正式提示词。"""
    if oneid:
        return RS_AGENT_PROMPT

    text = RS_AGENT_PROMPT
    for old, new in _OFF_REPLACEMENTS:
        if old not in text:
            raise AssertionError(f"prompt drift: 找不到待改写片段：{old[:60]}...")
        text = text.replace(old, new)

    kept = [line for line in text.splitlines() if ONEID_TOOL not in line]
    prompt = "\n".join(kept)

    on_tools, off_tools = bullet_tools(RS_AGENT_PROMPT), bullet_tools(prompt)
    removed = [name for name in on_tools if name not in off_tools]
    if removed != [ONEID_TOOL]:
        raise AssertionError(f"两臂工具集必须只差 {ONEID_TOOL}，实际差：{removed}")
    if ONEID_TOOL in prompt:  # 兜底：正文里也不许残留
        raise AssertionError(f"非 OneID 臂的提示词里仍出现 {ONEID_TOOL}")
    return prompt


def prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def resolve_model(arm: Arm, base: Optional[str] = None, sft: Optional[str] = None) -> Optional[str]:
    """模型路径优先级：显式参数 > 环境变量 > 未配置（None）。"""
    if arm.weights == "base":
        return base or os.environ.get("ONEID_ABLATION_BASE_MODEL") or None
    return sft or os.environ.get("ONEID_ABLATION_SFT_MODEL") or None


def describe(arm: Arm, model: Optional[str]) -> str:
    state = model or "<未配置：设 {}>".format(arm.model_env)
    return f"{arm.key:10} oneid={'on ' if arm.oneid else 'off'}  {state}"
