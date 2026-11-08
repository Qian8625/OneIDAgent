"""Fold aggregation: mean, dispersion and 95% confidence intervals.

Two intervals are reported per metric:

* **t interval** over the k fold means -- the standard "mean of folds" reading;
* **percentile bootstrap** over folds (resampling folds with replacement),
  which makes no normality assumption.  With k=5 both are wide by construction;
  the point of the table is the *mean* and the *spread*, not a p-value.

``pooled`` is additionally reported for the end-to-end accuracies: sum of
correct over sum of support across folds, i.e. the accuracy a single reader
would compute by pooling every held-out object.  It has no CI because the folds
are not independent samples of one estimator -- it is the descriptive number.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: Metrics whose ``correct``/``support`` can be pooled across folds.
POOLABLE = ("category", "subtype", "color", "size_bin")


def _values(result: Mapping[str, Any], family: str, path: str) -> List[float]:
    values: List[float] = []
    for fold in result.get("fold_results") or []:
        node: Any = (fold.get("metrics") or {}).get(family) or {}
        if isinstance(node, Mapping) and path in node:
            value = node[path]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
            continue
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                node = None
                break
            node = node[part]
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            values.append(float(node))
    return values


def _t_critical(k: int) -> float:
    """Two-sided 95% t critical value for ``k-1`` degrees of freedom."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
    return table.get(k - 1, 1.96)


def summarize(values: Sequence[float], resamples: int = 10000, seed: int = 20261107) -> Dict[str, Any]:
    values = list(values)
    k = len(values)
    if not k:
        return {"n": 0}
    mean = sum(values) / k
    variance = sum((value - mean) ** 2 for value in values) / (k - 1) if k > 1 else 0.0
    sd = math.sqrt(variance)
    se = sd / math.sqrt(k) if k else 0.0
    half = _t_critical(k) * se if k > 1 else 0.0
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    low = means[int(0.025 * resamples)]
    high = means[int(0.975 * resamples) - 1]
    return {
        "n": k,
        "mean": round(mean, 6),
        "sd": round(sd, 6),
        "se": round(se, 6),
        "ci95_t": [round(mean - half, 6), round(mean + half, 6)],
        "ci95_boot": [round(low, 6), round(high, 6)],
        "values": [round(value, 6) for value in values],
    }


def pooled(result: Mapping[str, Any], family: str, attribute: str) -> Optional[Dict[str, Any]]:
    correct = _values(result, family, f"G4_end_to_end.{attribute}.correct")
    support = _values(result, family, f"G4_end_to_end.{attribute}.support")
    if not correct or not support:
        return None
    total_correct = int(round(sum(correct)))
    total_support = int(round(sum(support)))
    if total_support <= 0:
        return None
    return {
        "correct": total_correct,
        "support": total_support,
        "accuracy": round(total_correct / total_support, 6),
    }


def build(result: Mapping[str, Any]) -> Dict[str, Any]:
    metric_paths = list(result.get("metric_paths") or [])
    families = list(result.get("families") or [])
    out: Dict[str, Any] = {
        "arm": result.get("arm"),
        "folds": result.get("folds"),
        "seed": result.get("seed"),
        "families": {},
        "claim_quality": {},
    }
    for family in families:
        block: Dict[str, Any] = {"metrics": {}}
        for path in metric_paths:
            values = _values(result, family, path)
            if values:
                block["metrics"][path] = summarize(values)
        block["pooled"] = {
            attribute: pooled(result, family, attribute)
            for attribute in POOLABLE
        }
        block["pooled"] = {k: v for k, v in block["pooled"].items() if v}
        out["families"][family] = block

    quality_keys = ("auc", "log_loss", "brier", "scored", "unsupported")
    for key in quality_keys:
        values = []
        for fold in result.get("fold_results") or []:
            value = (fold.get("claim_quality") or {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        if values:
            out["claim_quality"][key] = summarize(values)
    return out


def _fmt(block: Mapping[str, Any], path: str, key: str = "mean") -> str:
    entry = (block.get("metrics") or {}).get(path) or {}
    if not entry:
        return "-"
    if key == "mean_ci":
        low, high = entry["ci95_t"]
        return f"{entry['mean']:.4f} [{low:.4f}, {high:.4f}]"
    if key == "boot":
        low, high = entry["ci95_boot"]
        return f"[{low:.4f}, {high:.4f}]"
    return str(entry.get(key, "-"))


def render_markdown(summary: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# exp_3 — 5-fold cross-fitting")
    lines.append("")
    lines.append(f"- arm: `{summary.get('arm')}`")
    lines.append(f"- folds: {summary.get('folds')}  seed: `{summary.get('seed')}`")
    lines.append("- grouping: pair-level (both views + annotations stay together)")
    lines.append("")
    for family, block in (summary.get("families") or {}).items():
        lines.append(f"## {family}")
        lines.append("")
        lines.append("| metric | fold mean | fold SD | 95% t-CI | 95% bootstrap CI |")
        lines.append("| --- | --- | --- | --- | --- |")
        for path in (result.get("metric_paths") or []):
            entry = (block.get("metrics") or {}).get(path)
            if not entry:
                continue
            lines.append(
                f"| `{path}` | {entry['mean']:.4f} | {entry['sd']:.4f} | "
                f"[{entry['ci95_t'][0]:.4f}, {entry['ci95_t'][1]:.4f}] | "
                f"[{entry['ci95_boot'][0]:.4f}, {entry['ci95_boot'][1]:.4f}] |")
        if block.get("pooled"):
            lines.append("")
            lines.append("Pooled held-out accuracy (sum correct / sum support):")
            lines.append("")
            lines.append("| attribute | correct / support | accuracy |")
            lines.append("| --- | --- | --- |")
            for attribute, entry in block["pooled"].items():
                lines.append(
                    f"| {attribute} | {entry['correct']} / {entry['support']} | "
                    f"{entry['accuracy']:.4f} |")
        lines.append("")
    quality = summary.get("claim_quality") or {}
    if quality:
        lines.append("## claim-level out-of-fold quality (theta)")
        lines.append("")
        lines.append("| metric | mean | SD | 95% t-CI |")
        lines.append("| --- | --- | --- | --- |")
        for key, entry in quality.items():
            lines.append(
                f"| {key} | {entry['mean']:.6f} | {entry['sd']:.6f} | "
                f"[{entry['ci95_t'][0]:.6f}, {entry['ci95_t'][1]:.6f}] |")
        lines.append("")
    lines.append("## per-fold values")
    lines.append("")
    for family, block in (summary.get("families") or {}).items():
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| metric | " + " | ".join(f"f{i}" for i in range(int(summary.get("folds") or 0))) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in range(int(summary.get("folds") or 0))) + " |")
        for path in (result.get("metric_paths") or []):
            entry = (block.get("metrics") or {}).get(path)
            if not entry:
                continue
            cells = " | ".join(f"{value:.4f}" for value in entry["values"])
            lines.append(f"| `{path}` | {cells} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="directory holding fold_results.json")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    result = json.loads((run_dir / "fold_results.json").read_text(encoding="utf-8"))
    summary = build(result)
    out_dir = Path(args.out) if args.out else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(summary, result), encoding="utf-8")
    print(render_markdown(summary, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
