"""Shared output format for both experiment runners."""
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def write_jsonl(path: Path, pairs: Sequence[Mapping[str, Any]], predictions: Mapping[str, List[Dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps({
                "pair_id": pair["base_id"],
                "family": pair["family"],
                "objects": predictions.get(pair["base_id"], []),
            }, ensure_ascii=False) + "\n")
