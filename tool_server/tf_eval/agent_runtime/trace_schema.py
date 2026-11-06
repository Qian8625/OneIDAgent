from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentTrace(TypedDict):
    sample_id: str
    question: str
    actions: List[Dict[str, Any]]
    tool_responses: List[Dict[str, Any]]
    final_answer: str
    terminated: bool
    rounds: int
    raw_tool_calls: int
    total_tool_calls: int
    wall_time: float
    llm_time: float
    tool_time: float
    error_tags: List[str]


def create_agent_trace(sample_id: str, question: str) -> AgentTrace:
    return {
        "sample_id": sample_id or "",
        "question": question or "",
        "actions": [],
        "tool_responses": [],
        "final_answer": "",
        "terminated": False,
        "rounds": 0,
        "raw_tool_calls": 0,
        "total_tool_calls": 0,
        "wall_time": 0.0,
        "llm_time": 0.0,
        "tool_time": 0.0,
        "error_tags": [],
    }
