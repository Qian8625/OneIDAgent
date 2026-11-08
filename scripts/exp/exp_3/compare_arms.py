"""Merge the three arm summaries into one comparison table.

Reads ``<arm>/summary.json`` produced by ``aggregate.py`` and writes
``comparison.md`` / ``comparison.json`` next to them.

Arms (see README):

* ``refit``   -- theta refitted on each fold's training pairs (out-of-fold);
* ``shipped`` -- the frozen ``contextual_theta_v1.json`` (in-sample on test folds);
* ``noctx``   -- contextual reliability disabled (table reliability baseline).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

METRICS = (
    "G4_end_to_end.category.accuracy",
    "G4_end_to_end.category.fill_rate",
    "G4_end_to_end.category.accuracy_on_filled",
    "G4_end_to_end.subtype.accuracy",
    "G4_end_to_end.color.accuracy",
    "G4_end_to_end.size_bin.accuracy",
    "G5_identity.composite.detection_accuracy",
    "G5_identity.composite.association_accuracy",
)

SHORT = {
    "G4_end_to_end.category.accuracy": "category acc (Truth Acc)",
    "G4_end_to_end.category.fill_rate": "category fill rate",
    "G4_end_to_end.category.accuracy_on_filled": "accuracy on filled",
    "G4_end_to_end.subtype.accuracy": "subtype acc",
    "G4_end_to_end.color.accuracy": "color acc",
    "G4_end_to_end.size_bin.accuracy": "size_bin acc",
    "G5_identity.composite.detection_accuracy": "detection acc (G5)",
    "G5_identity.composite.association_accuracy": "association acc (G5)",
}


def load_arm(path: Path) -> Optional[Dict[str, Any]]:
    summary = Path(path) / "summary.json"
    if not summary.is_file():
        return None
    return json.loads(summary.read_text(encoding="utf-8"))


def _cell(block: Mapping[str, Any], metric: str) -> str:
    entry = ((block.get("metrics") or {}).get(metric)) or {}
    if not entry:
        return "-"
    low, high = entry["ci95_t"]
    return f"{entry['mean']:.4f} [{low:.4f}, {high:.4f}]"


def render(arms: Mapping[str, Mapping[str, Any]]) -> str:
    lines: List[str] = ["# exp_3 — arm comparison (fold mean [95% t-CI])", ""]
    families = sorted({family for arm in arms.values() for family in (arm.get("families") or {})})
    for family in families:
        lines.append(f"## {family}")
        lines.append("")
        header = "| metric | " + " | ".join(f"`{name}`" for name in arms) + " |"
        lines.append(header)
        lines.append("| --- | " + " | ".join("---" for _ in arms) + " |")
        for metric in METRICS:
            cells = []
            for name, arm in arms.items():
                block = (arm.get("families") or {}).get(family) or {}
                cells.append(_cell(block, metric))
            lines.append(f"| {SHORT.get(metric, metric)} | " + " | ".join(cells) + " |")
        lines.append("")
        pooled_rows = []
        for metric in ("category", "subtype", "color", "size_bin"):
            cells = []
            for name, arm in arms.items():
                block = (arm.get("families") or {}).get(family) or {}
                entry = (block.get("pooled") or {}).get(metric)
                cells.append(f"{entry['accuracy']:.4f} ({entry['correct']}/{entry['support']})"
                             if entry else "-")
            pooled_rows.append((metric, cells))
        if any(any(cell != "-" for cell in cells) for _, cells in pooled_rows):
            lines.append("Pooled held-out accuracy (correct/support):")
            lines.append("")
            lines.append("| attribute | " + " | ".join(f"`{name}`" for name in arms) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in arms) + " |")
            for attribute, cells in pooled_rows:
                lines.append(f"| {attribute} | " + " | ".join(cells) + " |")
            lines.append("")
    quality = {name: (arm.get("claim_quality") or {}) for name, arm in arms.items()}
    if any(quality.values()):
        lines.append("## claim-level out-of-fold quality")
        lines.append("")
        lines.append("| arm | auc | log_loss | brier |")
        lines.append("| --- | --- | --- | --- |")
        for name, block in quality.items():
            auc = (block.get("auc") or {}).get("mean")
            loss = (block.get("log_loss") or {}).get("mean")
            brier = (block.get("brier") or {}).get("mean")
            lines.append(
                f"| `{name}` | {auc:.4f} | {loss:.4f} | {brier:.4f} |"
                if None not in (auc, loss, brier) else f"| `{name}` | - | - | - |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="directory holding <arm>/summary.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = Path(args.root)
    arms: Dict[str, Mapping[str, Any]] = {}
    for name in ("refit", "shipped", "noctx"):
        loaded = load_arm(root / name)
        if loaded:
            arms[name] = loaded
    if not arms:
        raise SystemExit(f"no summary.json found under {root}")
    out_dir = Path(args.out) if args.out else root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(
        json.dumps(arms, ensure_ascii=False, indent=2), encoding="utf-8")
    text = render(arms)
    (out_dir / "comparison.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
