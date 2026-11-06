from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from tool_server.utils.paths import env_path


@dataclass
class AgentRunConfig:
    max_rounds: int = 16
    pretrained_path: Optional[str] = None
    controller_url_location: Optional[str] = None

    def __post_init__(self) -> None:
        if self.pretrained_path is None:
            self.pretrained_path = env_path("PRETRAINED_PATH", "${CHECKPOINT_ROOT}/OpenEarthAgent")
