from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from tool_server.utils.paths import env_path


class RunMode(str, Enum):
    NO_SKILL = "no_skill"
    AUTO_SKILL = "auto_skill"
    ORACLE_SKILL = "oracle_skill"
    ORACLE_FORCED_SKILL = "oracle_forced_skill"

    @classmethod
    def normalize(cls, value: "RunMode | str") -> "RunMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            valid = ", ".join(mode.value for mode in cls)
            raise ValueError(f"Unknown run mode: {value}. Expected one of: {valid}") from exc


@dataclass
class AgentRunConfig:
    mode: RunMode | str = RunMode.AUTO_SKILL
    top_k: int = 3
    min_score: float = 2.5
    score_margin: float = 0.5
    max_rounds: int = 16
    pretrained_path: Optional[str] = None
    controller_url_location: Optional[str] = None

    def __post_init__(self) -> None:
        self.mode = RunMode.normalize(self.mode)
        if self.pretrained_path is None:
            self.pretrained_path = env_path("PRETRAINED_PATH", "${CHECKPOINT_ROOT}/OpenEarthAgent")
