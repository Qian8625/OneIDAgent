"""5-fold cross-fitting driver for the contextual reliability (theta) calibration.

For every fold:

1. fit theta on the **training** pairs only (``exp_3.theta.fit_theta``);
2. write a per-fold shared-identity YAML whose ``reliability.contextual.theta_path``
   points at that fold's artifact;
3. run the DS -> VI -> cascade chain on the held-out pairs
   (``DS_VI.experiments.run_detection.run_family``), with the workbook attribute
   claims and the visibility context the paper's ``F`` arm uses;
4. materialise the VI objects as an ``identities`` sheet and score it against the
   frozen truth workbook with ``property_metrics.evaluate_family`` -- the same
   G1/G2/G4/G5/G6 code path as the published tables.

The output of one fold is therefore a *held-out* measurement, and the mean over
folds is an honest estimate of what the calibration buys.

Usage
-----
::

    export ONEID_DATA_ROOT=/home/ubuntu/00_Data/RSTI
    python -m scripts.exp.exp_3.run_cv --families match,time --folds 5

    # smoke test: 3 pairs per fold, both families
    python -m scripts.exp.exp_3.run_cv --max-pairs-per-fold 3 --out outputs/exp3_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import yaml

from . import data as exp_data
from . import folds as exp_folds
from . import theta as exp_theta

REPO_ROOT = exp_data.REPO_ROOT

#: Base arm YAML per experiment arm.
ARM_CONFIG = {
    "refit": REPO_ROOT / "scripts/exp/arms/f_ctx.yaml",
    "shipped": REPO_ROOT / "scripts/exp/arms/f_ctx.yaml",
    "noctx": REPO_ROOT / "scripts/exp/arms/f_online.yaml",
}

#: Metric paths reported in the fold table.  ``G4_end_to_end`` is the paper's
#: Truth Acc (end-to-end attribute accuracy over matched objects).
METRIC_PATHS: Sequence[str] = (
    "G4_end_to_end.category.accuracy",
    "G4_end_to_end.category.fill_rate",
    "G4_end_to_end.category.accuracy_on_filled",
    "G4_end_to_end.subtype.accuracy",
    "G4_end_to_end.color.accuracy",
    "G4_end_to_end.size_bin.accuracy",
    "G5_identity.composite.detection_accuracy",
    "G5_identity.composite.association_accuracy",
    "G5_identity.composite.confirmed",
    "G5_identity.composite.gt_double_view",
)


def dataset_config(families: Sequence[str]) -> Dict[str, Any]:
    """Shipped D-S config with the data root pointed at the local checkout.

    ``DS_VI/configs/ds_default.json`` still carries the Windows authoring paths
    (``H:/1_RSTIBenchMark/...``); on Linux those are *relative*, so they must be
    rewritten or nothing loads.
    """
    payload = json.loads((REPO_ROOT / "DS_VI/configs/ds_default.json").read_text(encoding="utf-8"))
    root = Path(exp_data.DATA_ROOT)
    for family in families:
        spec = payload["families"][family]
        directory = root / exp_data.FAMILIES[family]["image_dir"]
        spec["image_dir"] = str(directory)
        spec["tool_dir"] = str(directory / "0_tool_json")
        spec["truth_dir"] = str(directory / "0_truth_json")
        spec["attributes_workbook"] = str(exp_data.tool_workbook(family))
        spec["context_workbook"] = str(exp_data.tool_workbook(family))
    return payload


def fold_vi_config(arm: str, theta_path: Optional[Path], out_path: Path) -> Path:
    """Write the arm YAML specialised for one fold."""
    base = ARM_CONFIG[arm]
    payload = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    if arm == "noctx":
        payload.setdefault("reliability", {}).setdefault("contextual", {})["enabled"] = False
        payload["reliability"]["contextual"].pop("theta_path", None)
    else:
        contextual = payload.setdefault("reliability", {}).setdefault("contextual", {})
        contextual["enabled"] = True
        contextual["theta_path"] = str(Path(theta_path).resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def subset_workbook(source: Path, pair_ids: Sequence[str], out_path: Path) -> Path:
    """Copy one workbook keeping only the given pairs.

    ``evaluate_family`` scores the whole truth workbook, so a held-out fold must
    be scored against a truth workbook restricted to the same pairs -- otherwise
    every fold's accuracy is diluted by the pairs it never predicted.
    """
    from openpyxl import Workbook, load_workbook

    wanted = {str(pair) for pair in pair_ids}
    source_book = load_workbook(source, read_only=True, data_only=True)
    out_book = Workbook()
    out_book.remove(out_book.active)
    for name in source_book.sheetnames:
        sheet = source_book[name]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            continue
        target = out_book.create_sheet(name)
        target.append(list(header))
        try:
            column = list(header).index("pair_id")
        except ValueError:
            column = None
        for row in rows:
            if column is None or str(row[column]) in wanted:
                target.append(list(row))
    source_book.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_book.save(out_path)
    return out_path


def _decided_attribute(obj: Mapping[str, Any], attribute: str) -> Optional[str]:
    """Highest-probability VI belief for one attribute, across time keys."""
    beliefs = obj.get("attribute_beliefs") or {}
    best: Optional[Mapping[str, Any]] = None
    for belief in beliefs.values():
        if str(belief.get("attribute")) != attribute:
            continue
        if belief.get("value") in (None, ""):
            continue
        if best is None or float(belief.get("probability") or 0.0) > float(best.get("probability") or 0.0):
            best = belief
    if best is None:
        return None
    return str(best.get("value"))


def materialize_workbook(
    tool_workbook: Path,
    predictions_by_pair: Mapping[str, List[Mapping[str, Any]]],
    out_path: Path,
    family: str,
    attribute_table: Mapping[str, Any],
    label_source: str = "vi",
) -> Path:
    """Replace the tool workbook's ``identities`` sheet with the system's answer.

    ``label_source``:

    * ``vi``       -- the attribute the shared-identity engine **decided**
      (``attribute_beliefs``), falling back to the detector only when the engine
      has no belief for that axis.  This is the quantity truth discovery is
      supposed to improve, and the only one that can differ between theta arms.
    * ``detector`` -- the raw detector label / workbook attribute.  Identical to
      the trunk behaviour; useful as a control.

    The instances sheet is copied verbatim because ``identity_metrics`` uses its
    instance ids as the bridge to annotated objects.
    """
    from openpyxl import load_workbook

    views = exp_data.family_views(family)
    header = ["object_id", "pair_id", "instance_a", "instance_b",
              "category", "subtype", "color", "size_bin", "association_score"]
    rows: List[List[Any]] = []
    dropped_members = 0
    for pair_id, objects in predictions_by_pair.items():
        for index, obj in enumerate(objects or []):
            members: Dict[str, Optional[str]] = {"a": None, "b": None}
            for side in ("a", "b"):
                detection_index = obj.get(f"view_{side}_index")
                if detection_index is None:
                    continue
                members[side] = f"{pair_id}:{views[side]}:{int(detection_index):04d}"
            if members["a"] is None and members["b"] is None:
                dropped_members += 1
                continue
            snapshot: Dict[str, str] = {}
            for side, member in members.items():
                if member is None:
                    continue
                values = ((attribute_table.get(pair_id) or {}).get(side) or {}).get(
                    int(member.rsplit(":", 1)[-1]))
                if values:
                    snapshot = values
                    break
            attributes: Dict[str, Optional[str]] = {}
            for attribute in ("category", "subtype", "color", "size_bin"):
                decided = _decided_attribute(obj, attribute) if label_source == "vi" else None
                if decided is None:
                    decided = snapshot.get(attribute)
                if attribute == "category" and decided is None:
                    decided = obj.get("label")
                attributes[attribute] = decided
            rows.append([
                f"{pair_id}:vi:{index:03d}",
                pair_id,
                members["a"],
                members["b"],
                attributes["category"],
                attributes["subtype"],
                attributes["color"],
                attributes["size_bin"],
                obj.get("score"),
            ])

    if dropped_members:
        print(f"[workbook {family}] dropped {dropped_members} objects without members")

    workbook = load_workbook(tool_workbook)
    if "identities" in workbook.sheetnames:
        workbook.remove(workbook["identities"])
    sheet = workbook.create_sheet("identities")
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(out_path)
    return out_path


def run_fold_engine(
    family: str,
    config: Mapping[str, Any],
    test_pairs: Sequence[str],
    vi_config: Path,
    attribute_table: Mapping[str, Any],
    context_table: Mapping[str, Any],
    out_dir: Path,
    inject_attributes: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run D-S -> VI -> cascade over the held-out pairs."""
    from DS_VI.experiments.property_attributes import DEFAULT_ATTRIBUTES

    from .engine import run_family_with_claims

    pairs = [{"base_id": pair_id, "family": family,
              "images": _image_paths(family, pair_id)} for pair_id in test_pairs]
    return run_family_with_claims(
        family,
        config,
        pairs,
        out_dir,
        vi_config=str(vi_config),
        attribute_table=attribute_table,
        context_table=context_table,
        attributes=DEFAULT_ATTRIBUTES,
        inject_attributes=inject_attributes,
    )


def _image_paths(family: str, pair_id: str) -> Dict[str, str]:
    views = exp_data.family_views(family)
    directory = Path(exp_data.DATA_ROOT) / exp_data.FAMILIES[family]["image_dir"]
    return {side: str(directory / f"{pair_id}_{tag}.png") for side, tag in views.items()}


def read_predictions(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """``predictions.jsonl`` -> ``{pair_id: [objects]}``."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[str(record["pair_id"])] = list(record.get("objects") or [])
    return out


def _get(metrics: Mapping[str, Any], path: str) -> Any:
    node: Any = metrics
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


#: Extra paths stored for the pooled accuracy (sum correct / sum support).
POOLABLE = ("category", "subtype", "color", "size_bin")
POOL_PATHS: Sequence[str] = tuple(
    f"G4_end_to_end.{attribute}.{field}"
    for attribute in POOLABLE for field in ("correct", "support")
)


def fold_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {path: _get(metrics, path) for path in METRIC_PATHS + POOL_PATHS}


def run(
    families: Sequence[str],
    k: int,
    seed: int,
    arm: str,
    out_root: Path,
    max_pairs_per_fold: int = 0,
    pair_override: Optional[Sequence[str]] = None,
    inject_attributes: bool = True,
    label_source: str = "vi",
) -> Dict[str, Any]:
    from DS_VI.experiments.property_attributes import (
        DEFAULT_ATTRIBUTES,
        load_attribute_table,
        load_context_table,
    )
    from DS_VI.experiments.metrics.property_metrics import evaluate_family

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()

    all_rows = exp_data.load_all_claims(families)
    pair_ids = sorted({row.pair_id for row in all_rows})
    if pair_override:
        wanted = set(pair_override)
        pair_ids = [pair_id for pair_id in pair_ids if pair_id in wanted]
    folds = exp_folds.make_folds(pair_ids, k=k, seed=seed)
    if max_pairs_per_fold:
        for fold in folds:
            fold["test"] = fold["test"][:max_pairs_per_fold]

    dataset = dataset_config(families)
    (out_root / "dataset_config.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

    tables: Dict[str, Dict[str, Any]] = {}
    for family in families:
        workbook = exp_data.tool_workbook(family)
        tables[family] = {
            "attribute": load_attribute_table(workbook, family, DEFAULT_ATTRIBUTES),
            "context": load_context_table(workbook, family, ("visibility",)),
        }

    rows_by_pair: Dict[str, List[exp_data.ClaimRow]] = {}
    for row in all_rows:
        rows_by_pair.setdefault(row.pair_id, []).append(row)

    fold_results: List[Dict[str, Any]] = []
    for fold in folds:
        index = int(fold["fold"])
        fold_dir = out_root / f"fold_{index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_pairs = set(fold["train"])
        test_pairs = list(fold["test"])

        train_rows = [row for pair in train_pairs for row in rows_by_pair.get(pair, [])]
        test_rows = [row for pair in test_pairs for row in rows_by_pair.get(pair, [])]
        if arm == "refit":
            artifact = exp_theta.fit_theta(
                train_rows, exp_data.DEFAULT_ATTRIBUTES,
                metadata={"fold": index, "train_pairs": len(train_pairs),
                          "test_pairs": len(test_pairs), "seed": seed})
            theta_path = fold_dir / "theta.json"
            exp_theta.write_artifact(theta_path, artifact)
        elif arm == "shipped":
            artifact = json.loads(exp_theta.SHIPPED_THETA.read_text(encoding="utf-8"))
            theta_path = exp_theta.SHIPPED_THETA
        else:  # noctx
            artifact = None
            theta_path = None

        claim_quality = exp_theta.artifact_log_loss(artifact, test_rows) if artifact else {}
        vi_config = fold_vi_config(arm, theta_path, fold_dir / "vi.yaml")

        per_family: Dict[str, Any] = {}
        for family in families:
            family_pairs = [pair for pair in test_pairs if exp_data.pair_family(pair) == family]
            if not family_pairs:
                continue
            engine_dir = fold_dir / family / "engine"
            tool_subset = subset_workbook(
                exp_data.tool_workbook(family), family_pairs, fold_dir / family / "tool.xlsx")
            truth_subset = subset_workbook(
                exp_data.truth_workbook(family), family_pairs, fold_dir / family / "truth.xlsx")
            predictions = run_fold_engine(
                family, dataset, family_pairs, vi_config,
                tables[family]["attribute"], tables[family]["context"], engine_dir,
                inject_attributes=inject_attributes)
            predicted_workbook = fold_dir / family / "predictions.xlsx"
            materialize_workbook(
                tool_subset, predictions, predicted_workbook,
                family, tables[family]["attribute"], label_source=label_source)
            metrics = evaluate_family(predicted_workbook, truth_subset, family)
            (fold_dir / family / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            per_family[family] = fold_metrics(metrics)

        fold_results.append({
            "fold": index,
            "train_pairs": len(train_pairs),
            "test_pairs": len(test_pairs),
            "claim_quality": claim_quality,
            "metrics": per_family,
        })
        print(f"[fold {index}] test={len(test_pairs)} done "
              f"({time.time() - started:.1f}s elapsed)")

    result = {
        "arm": arm,
        "families": list(families),
        "folds": k,
        "seed": seed,
        "metric_paths": list(METRIC_PATHS),
        "inject_attributes": bool(inject_attributes),
        "label_source": str(label_source),
        "fold_results": fold_results,
        "elapsed_seconds": round(time.time() - started, 2),
        "environment": {"python": sys.version.split()[0],
                        "data_root": str(exp_data.DATA_ROOT)},
    }
    (out_root / "fold_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", default="match,time")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--arm", choices=sorted(ARM_CONFIG), default="refit")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/exp3_crossfit")
    parser.add_argument("--max-pairs-per-fold", type=int, default=0,
                        help="debug: keep only the first N held-out pairs per fold")
    parser.add_argument("--pairs", default=None, help="comma separated pair ids")
    parser.add_argument("--label-source", choices=["vi", "detector"], default="vi",
                        help="score the engine's decided attributes (vi) or the raw "
                             "detector labels (detector, trunk behaviour)")
    parser.add_argument("--inject-attributes", choices=["on", "off"], default="on",
                        help="inject subtype/color/size_bin/orientation as VI claims; "
                             "off reproduces the trunk's category-only behaviour")
    args = parser.parse_args(argv)

    families = [item.strip() for item in args.families.split(",") if item.strip()]
    pairs = [item.strip() for item in args.pairs.split(",")] if args.pairs else None
    result = run(families, args.folds, args.seed, args.arm, args.out,
                 max_pairs_per_fold=args.max_pairs_per_fold, pair_override=pairs,
                 inject_attributes=(args.inject_attributes == "on"),
                 label_source=args.label_source)
    print(json.dumps({"arm": result["arm"], "folds": result["folds"],
                      "elapsed": result["elapsed_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
