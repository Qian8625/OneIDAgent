"""Measure per-(modality, attribute) informativeness from the property workbooks.

``DS_VI/models/shared_identity/attribute_consistency.py`` weights each attribute
of the candidate-consistency gate by how much it can actually separate values.
Those numbers are measurements, not opinions: when the annotation workbooks are
reissued (a new attribute column, a relabelled vocabulary, a previously empty
axis that is now filled), the hardcoded table silently stops describing the
data. This module recomputes it from the four workbooks with the evaluator's own
instance pairing, so the gate can be re-fitted with one command:

    python -B -m DS_VI.experiments.metrics.attribute_informativeness --family both

Definition, identical to the gate's contract:

.. code-block:: text

    acc      = agreement among *filled* pairs (both sides non-empty)
    C        = number of distinct ground-truth values of that attribute
    info     = max(0, (acc - 1/C) / (1 - 1/C))       in [0, 1]

``info = 0`` means the channel is at or below chance and must not vote. A channel
with no filled pairs (for example ``color`` on SAR) is reported as 0.0 rather
than guessed, so it abstains by construction.

The workbook order is irrelevant here: instances are joined to the ground truth
by IoU, exactly like G2.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from DS_VI.experiments.metrics.property_metrics import (
    FAMILY_FILES,
    IOU_THRESHOLD,
    match_image,
    normalized,
    read_book,
)

#: Attribute axes the consistency gate can read, in a stable order.
CANDIDATE_ATTRIBUTES: Tuple[str, ...] = (
    "category",
    "subtype",
    "color",
    "state",
    "size_bin",
    "orientation",
)

#: family -> view -> modality, mirroring ``attribute_consistency.VIEW_MODALITY``.
FAMILY_VIEW_MODALITY: Dict[str, Dict[str, str]] = {
    "match": {"rgb": "optical", "sar": "sar"},
    "time": {"time1": "optical", "time2": "optical"},
}


def matched_pairs(
    tool_path: Path,
    truth_path: Path,
    family: str,
    iou_threshold: float = IOU_THRESHOLD,
) -> Dict[str, List[Tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    """``modality -> [(tool row, truth row)]`` for every matched instance."""
    tool = read_book(Path(tool_path))
    truth = read_book(Path(truth_path))
    tool_images: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    truth_images: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in tool["instances"]:
        tool_images[(str(row.get("pair_id")), str(row.get("view")))].append(row)
    for row in truth["instances"]:
        truth_images[(str(row.get("pair_id")), str(row.get("view")))].append(row)
    shared = (
        {str(row.get("pair_id")) for row in tool["instances"]}
        & {str(row.get("pair_id")) for row in truth["instances"]}
    )
    view_modality = FAMILY_VIEW_MODALITY[family]
    out: Dict[str, List[Tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for key in sorted(set(tool_images) | set(truth_images)):
        pair_id, view = key
        if pair_id not in shared:
            continue
        modality = view_modality.get(view)
        if modality is None:
            continue
        record = match_image(tool_images.get(key, []), truth_images.get(key, []), iou_threshold)
        for match in record["matches"]:
            out[modality].append((match["tool"], match["truth"]))
    return out


def measure_attribute(
    pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]],
    attribute: str,
) -> Dict[str, Any]:
    """Agreement, domain size and chance-normalized informativeness of one axis."""
    scored = []
    gold_values = set()
    for tool_row, truth_row in pairs:
        gold = normalized(truth_row.get(attribute))
        if gold is None:
            continue
        gold_values.add(gold)
        predicted = normalized(tool_row.get(attribute))
        if predicted is None:
            continue
        scored.append((predicted, gold))
    correct = sum(1 for predicted, gold in scored if predicted == gold)
    domain = len(gold_values)
    accuracy = (correct / len(scored)) if scored else 0.0
    chance = (1.0 / domain) if domain else 1.0
    informativeness = (
        max(0.0, (accuracy - chance) / (1.0 - chance)) if domain > 1 else 0.0
    )
    return {
        "support": len(scored),
        "filled_gold": len(gold_values),
        "domain": domain,
        "accuracy": round(accuracy, 6),
        "fill_rate": round(len(scored) / len(pairs), 6) if pairs else 0.0,
        "informativeness": round(informativeness, 4),
    }


def measure_family(
    family: str,
    tool_path: Optional[Path] = None,
    truth_path: Optional[Path] = None,
    dataset_root: Path = Path("H:/1_RSTIBenchMark"),
    attributes: Sequence[str] = CANDIDATE_ATTRIBUTES,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``modality -> attribute -> measurement`` for one family."""
    tool_name, truth_name, _views = FAMILY_FILES[family]
    pairs = matched_pairs(
        Path(tool_path or (dataset_root / tool_name)),
        Path(truth_path or (dataset_root / truth_name)),
        family,
    )
    return {
        modality: {
            attribute: measure_attribute(rows, attribute) for attribute in attributes
        }
        for modality, rows in sorted(pairs.items())
    }


def informativeness_table(
    families: Sequence[str] = ("match", "time"),
    dataset_root: Path = Path("H:/1_RSTIBenchMark"),
    attributes: Sequence[str] = CANDIDATE_ATTRIBUTES,
) -> Dict[str, Dict[str, float]]:
    """``modality -> attribute -> informativeness``, pooled over the families.

    ``optical`` is pooled across ``match``'s RGB and ``time``'s two time points
    because the gate keys by modality, and a channel must be weighted by what it
    can do across the data it is used on.
    """
    collected: Dict[str, Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    domains: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for family in families:
        measured = measure_family(family, dataset_root=dataset_root, attributes=attributes)
        for modality, per_attribute in measured.items():
            for attribute, record in per_attribute.items():
                if not record["support"]:
                    continue
                collected[modality][attribute].append((record["accuracy"], record["support"]))
                domains[modality][attribute].add(record["domain"])
    table: Dict[str, Dict[str, float]] = {}
    for modality, per_attribute in collected.items():
        table[modality] = {}
        for attribute, entries in per_attribute.items():
            total = sum(n for _acc, n in entries)
            accuracy = sum(acc * n for acc, n in entries) / total if total else 0.0
            domain = max(domains[modality][attribute])
            chance = (1.0 / domain) if domain else 1.0
            table[modality][attribute] = round(
                max(0.0, (accuracy - chance) / (1.0 - chance)) if domain > 1 else 0.0, 4
            )
    return table


def as_python(table: Mapping[str, Mapping[str, float]]) -> str:
    """Render the table as the literal used by ``attribute_consistency``."""
    lines = ["ATTRIBUTE_INFORMATIVENESS: Dict[str, Dict[str, float]] = {"]
    for modality in sorted(table):
        cells = ", ".join(
            f'"{attribute}": {table[modality][attribute]}'
            for attribute in CANDIDATE_ATTRIBUTES
            if attribute in table[modality]
        )
        lines.append(f'    "{modality}": {{{cells}}},')
    lines.append("}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(FAMILY_FILES) + ["both"], default="both")
    parser.add_argument("--dataset-root", type=Path, default=Path("H:/1_RSTIBenchMark"))
    parser.add_argument("--out", type=Path, help="optional JSON output path")
    parser.add_argument("--python", action="store_true", help="print the literal for attribute_consistency.py")
    args = parser.parse_args(argv)

    families = sorted(FAMILY_FILES) if args.family == "both" else [args.family]
    details = {family: measure_family(family, dataset_root=args.dataset_root) for family in families}
    table = informativeness_table(families, args.dataset_root)

    print(json.dumps(details, ensure_ascii=False, indent=2))
    print("")
    print(as_python(table))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"details": details, "table": table}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
