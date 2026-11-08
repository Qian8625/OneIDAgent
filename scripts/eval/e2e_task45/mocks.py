"""CPU 自检用的替身：脚本化 LLM 和能产生检测证据的假工具。

放在单独模块，是因为两处都要用（run_matrix 的 --mock 与 selftest），
且它们必须与真实链路同构：``FakeTools`` 的返回值要能让
``update_registry_with_tool_response`` 真的登记出 RGB/SAR 检测证据，
否则 ``CrossModalInterpret`` 会先被依赖门禁拦下，测不到臂策略。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

RGB_TOOLS = ("ObjectDetection", "TextToBbox")
SAR_TOOLS = ("DenoDetV2",)
DETECTION_TOOLS = RGB_TOOLS + SAR_TOOLS


class MockLLM:
    """按脚本发动作；脚本用完后一律 Terminate 并给 GT 答案。"""

    def __init__(self, script: Optional[List[Dict[str, Any]]] = None):
        self.system_prompt = ""
        self.script = list(script or [])
        self._answer = ""
        self._step = 0

    def set_answer(self, answer: str) -> None:
        self._answer = answer
        self._step = 0

    def generate(self, conversation, extra_system_prompt: str = "") -> str:
        if self._step < len(self.script):
            action = self.script[self._step]
            self._step += 1
            return json.dumps({"thought": f"step {self._step}", "actions": [action]})
        return json.dumps({"thought": "done", "actions": [
            {"name": "Terminate", "arguments": {"ans": self._answer}}]})


class FakeTools:
    """检测工具返回可登记的证据，其余工具返回成功。"""

    def __init__(self):
        self.calls: List[str] = []

    def call_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.calls.append(name)
        if name in DETECTION_TOOLS:
            return {
                "error_code": 0,
                "text": f"{name} outputs: (10, 20, 110, 120), bridge, score 0.91",
                "structured": {"detections": [
                    {"bbox": [10, 20, 110, 120], "label": "bridge", "score": 0.91},
                ]},
            }
        if name == "CrossModalInterpret":
            return {"error_code": 0, "text": "1 matched, 0 RGB-only and 0 SAR-only",
                    "summary": {"matched_count": 1}}
        return {"error_code": 0, "text": f"{name} ok"}


def crossmodal_script(target: str = "bridge") -> List[Dict[str, Any]]:
    """task4 的标准前置序列：先产生两路检测证据，再调 OneID。"""
    return [
        {"name": "ObjectDetection", "arguments": {"image": "$primary_image", "text": target}},
        {"name": "DenoDetV2", "arguments": {"image": "$sar_image"}},
        {"name": "CrossModalInterpret", "arguments": {
            "target": target, "task": "crosscheck_exists", "output_mode": "full"}},
    ]
