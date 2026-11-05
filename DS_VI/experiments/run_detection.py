"""Run one DS -> VI pipeline on real tool detections, without reading truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from DS_VI.experiments.common import write_jsonl
from DS_VI.experiments.property_attributes import (
    DEFAULT_ATTRIBUTES,
    attribute_names,
    attributes_from_table,
    context_from_table,
    load_attribute_table,
    load_context_table,
    observation_ids,
)
from DS_VI.models.cascade import cascade_objects
from DS_VI.models.cascade.vi import BEST_CONFIG, resolve_cascade_settings, run_vi_pair
from DS_VI.models.ds_fusion.config import merge_overrides
from DS_VI.models.ds_fusion.config import ROOT, _resolve, load_config
from DS_VI.models.ds_fusion.pipeline import DualDetectionProvider, load_dataset, run_pair

#: Output policies understood by :func:`DS_VI.models.cascade.cascade_objects`.
POLICIES = ("status", "posterior", "rank", "blend", "all")

#: Reported per-object status vocabulary (see ``DS_VI.models.cascade.status``):
#: match -> match / rgb-only / sar-only / unknown;
#: time  -> static / moved / time1-only / time2-only.


def run_family(
    family: str,
    config: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    vi_config: str = BEST_CONFIG,
    attribute_table: Optional[Dict[str, Any]] = None,
    context_table: Optional[Dict[str, Any]] = None,
    attributes: Sequence[str] = DEFAULT_ATTRIBUTES,
    policy: Optional[str] = None,
    policy_threshold: Optional[float] = None,
    extra_links: Optional[int] = None,
    extra_mode: Optional[str] = None,
    extra_prior_floor: Optional[float] = None,
    extra_policy: Optional[str] = None,
    extra_margin: Optional[float] = None,
    associator: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    # Everything the cascade adds beyond the D-S handoff comes from the shipped
    # per-family settings; an explicit argument overrides them.
    settings = resolve_cascade_settings(
        vi_config, family, {"provenance": {"grouping_mode": "observation"}})
    policy = settings["policy"] if policy is None else policy
    if policy_threshold is None:
        policy_threshold = settings["blend_weight"]
    extra_links = settings["extra_links"] if extra_links is None else extra_links
    extra_mode = settings["extra_mode"] if extra_mode is None else extra_mode
    extra_prior_floor = settings["extra_prior_floor"] if extra_prior_floor is None else extra_prior_floor
    extra_policy = settings["extra_policy"] if extra_policy is None else extra_policy
    extra_margin = settings["extra_margin"] if extra_margin is None else extra_margin
    provider = DualDetectionProvider(config, family)
    names = attribute_names(attributes) if attribute_table is not None else ("category",)
    predictions: Dict[str, List[Dict[str, Any]]] = {}
    diagnostics: List[Dict[str, Any]] = []
    for pair in pairs:
        base = pair["base_id"]
        detections = {view: provider.detect(base, view, config["targets"]) for view in ("a", "b")}
        pair_config = config
        vi_overrides = {"provenance": {"grouping_mode": "observation"}}
        # The fitted D-S acceptance point lives in the shared-identity YAML's
        # ``cascade`` section so one file selects the operating point for both
        # the entry point and the comparison arms. ``None`` = keep the D-S config.
        decision_override = {}
        if settings.get("accept_betp") is not None:
            decision_override["accept_betp"] = float(settings["accept_betp"])
            if settings.get("single_accept_betp") is not None:
                decision_override["single_accept_betp"] = float(settings["single_accept_betp"])
        if decision_override:
            pair_config = merge_overrides(pair_config, {"decision": decision_override})
        if associator is not None:
            pair_config = merge_overrides(
                pair_config, {"association": {"associator": str(associator)}})
        ds = run_pair(pair, detections, pair_config)
        matched = [obj for obj in ds["objects"] if obj.get("source") == "matched"]
        singles = [obj for obj in ds["objects"] if obj.get("source") != "matched"]

        extra_claims = {}
        extra_context = {}
        attribute_snapshot = {}
        if attribute_table is not None or context_table is not None:
            by_view_index = observation_ids(pair, detections, config, family)
            if attribute_table is not None:
                attribute_snapshot = attributes_from_table(attribute_table, base, by_view_index, attributes)
            if context_table is not None:
                extra_context = context_from_table(context_table, base, by_view_index)

        vi = run_vi_pair(
            pair, detections, config, family, matched=matched, config_path=vi_config,
            decision_overrides=vi_overrides,
            extra_claims=extra_claims,
            attribute_names=names,
            extra_context=extra_context,
            attribute_snapshot=attribute_snapshot,
            extra_links=extra_links,
            extra_mode=extra_mode,
            extra_prior_floor=extra_prior_floor,
            extra_margin=extra_margin,
        )
        extras = list(vi.get("extras") or [])
        gate = float(settings.get("extra_displacement_gate") or 0.0)
        rejected_displacement = 0
        if gate > 0.0 and extras:
            from DS_VI.models.cascade.displacement import (
                filter_extras_by_displacement, shifts_from_matched)

            extras, rejected_displacement = filter_extras_by_displacement(
                extras, shifts_from_matched(matched), gate)
        objects = cascade_objects(
            matched, singles, vi["objects"], policy=policy, threshold=policy_threshold,
            extra_records=extras, extra_policy=extra_policy, family=family,
        )
        for index, obj in enumerate(objects):
            obj["object_id"] = f"{base}:pred:{index}"
        predictions[base] = objects
        vi_diagnostics = dict(vi.get("diagnostics") or {})
        vi_diagnostics["extra_rejected_displacement"] = rejected_displacement
        diagnostics.append({
            "pair_id": base,
            "ds": ds,
            "vi": dict(vi, diagnostics=vi_diagnostics),
            "injected_claims": sum(len(v) for v in extra_claims.values()),
        })
        print(f"[{family}] {base}: {len(objects)} objects")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", pairs, predictions)
    with (output_dir / "diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "config.json").write_text(json.dumps({
        "family": family, "pair_ids": [pair["base_id"] for pair in pairs],
        "config": config, "vi_config": str(vi_config), "policy": policy,
        "policy_threshold": float(policy_threshold),
        "attributes": list(attributes) if attribute_table is not None else ["category"],
        "context_covariates": list(context_table is not None and ["visibility"] or []),
        "extra_links": int(extra_links),
        "extra_mode": str(extra_mode),
        "extra_policy": str(extra_policy),
        "extra_prior_floor": float(extra_prior_floor),
        "extra_margin": float(extra_margin),
        "accept_betp": settings.get("accept_betp"),
        "single_accept_betp": settings.get("single_accept_betp"),
        "extra_displacement_gate": float(settings.get("extra_displacement_gate") or 0.0),
        "extra_consistency": list(settings["extra_consistency"]),
        "consistency_mode": settings["consistency_mode"],
        "consistency_tau": float(settings["consistency_tau"]),
        "associator": str((config.get("association") or {}).get("associator", "hungarian")),
        "provenance_grouping": "observation", "evidence_source": "tool_detections",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["match", "time"], default="match")
    parser.add_argument("--pair-id", action="append", help="Repeat to select specific pairs")
    parser.add_argument("--pair-limit", type=int, default=0, help="0 means all pairs")
    parser.add_argument("--config", type=Path, help="JSON dataset/config overrides")
    parser.add_argument("--vi-config", default=BEST_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--attributes-workbook", type=Path,
                        help="tool property workbook supplying subtype/color/size_bin/orientation")
    parser.add_argument("--attributes", default=",".join(DEFAULT_ATTRIBUTES),
                        help="comma separated attribute axes to inject")
    parser.add_argument("--policy", choices=POLICIES, default=None,
                        help="VI -> prediction policy; default: the shipped cascade.policy")
    parser.add_argument("--policy-threshold", type=float, default=None,
                        help="threshold for posterior/rank, mixing weight for blend; "
                             "default: cascade.blend_weight")
    parser.add_argument("--context-workbook", type=Path,
                        help="tool property workbook supplying the visibility covariate")
    parser.add_argument("--extra-links", type=int, default=None,
                        help=">0 lets VI also consider detections D-S left unused; "
                             "default: cascade.extra_links")
    parser.add_argument("--extra-mode", choices=["mutual", "pruned"], default=None,
                        help="mutual = one candidate per leftover; pruned = k nearest; "
                             "default: cascade.extra_mode")
    parser.add_argument("--extra-prior-floor", type=float, default=None,
                        help="neutral prior the independent candidates start from; "
                             "default: cascade.extra_prior_floor")
    parser.add_argument("--extra-policy", choices=["confirmed", "all", "posterior"],
                        default=None, help="how independent candidates are reported; "
                             "default: cascade.extra_policy")
    parser.add_argument("--associator", choices=["hungarian", "ot"], default=None,
                        help="pairing rule for the D-S stage; default: association.associator "
                             "(hungarian). 'ot' needs requirements-torch.txt")
    parser.add_argument("--extra-margin", type=float, default=None,
                        help="ambiguity ratio: skip a leftover whose best and second-best "
                             "candidates score within this fraction; default: per-family "
                             "cascade.extra_margin")
    args = parser.parse_args()
    if args.pair_limit < 0:
        parser.error("--pair-limit must be non-negative")

    config = load_config({"decision": {"accept_betp": 0.3, "matched_fallback": True}})
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        families = overrides.pop("families", {})
        for family, spec in families.items():
            config["families"][family].update(spec)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key].update(value)
            else:
                config[key] = value

    attributes = tuple(value.strip() for value in args.attributes.split(",") if value.strip())
    spec = config["families"][args.family]

    # The shipped consistency gate and the theta calibration need the tool
    # attribute workbook; when the flag is absent fall back to the dataset spec
    # so the plain command reproduces the frozen configuration.
    attributes_workbook = args.attributes_workbook
    if attributes_workbook is None and spec.get("attributes_workbook"):
        candidate = _resolve(spec["attributes_workbook"])
        attributes_workbook = candidate if candidate.is_file() else None
    attribute_table = None
    if attributes_workbook:
        if not attributes_workbook.is_file():
            parser.error(f"attribute workbook not found: {attributes_workbook}")
        attribute_table = load_attribute_table(attributes_workbook, args.family, attributes)

    context_workbook = args.context_workbook
    if context_workbook is None and spec.get("context_workbook"):
        candidate = _resolve(spec["context_workbook"])
        context_workbook = candidate if candidate.is_file() else None
    context_table = None
    if context_workbook:
        if not context_workbook.is_file():
            parser.error(f"context workbook not found: {context_workbook}")
        context_table = load_context_table(context_workbook, args.family)

    for key in ("tool_dir", "image_dir"):
        if not _resolve(spec[key]).is_dir():
            parser.error(f"Missing {key}: {_resolve(spec[key])}")
    pairs = load_dataset(config, args.family, args.pair_id)
    missing = set(args.pair_id or []) - {pair["base_id"] for pair in pairs}
    if missing:
        parser.error(f"Unknown pair ids: {sorted(missing)}")
    if args.pair_limit:
        pairs = pairs[: args.pair_limit]
    if not pairs:
        parser.error("No input pairs found")
    output = args.output_dir or ROOT / "DS_VI/results" / args.family
    if any((output / name).exists() for name in ("predictions.jsonl", "diagnostics.jsonl", "config.json")):
        parser.error(f"Output already exists; choose a new --output-dir: {output}")
    run_family(
        args.family, config, pairs, output, args.vi_config,
        attribute_table=attribute_table,
        context_table=context_table,
        attributes=attributes,
        policy=args.policy,
        policy_threshold=args.policy_threshold,
        extra_links=None if args.extra_links is None else max(0, args.extra_links),
        extra_mode=args.extra_mode,
        extra_prior_floor=args.extra_prior_floor,
        extra_policy=args.extra_policy,
        extra_margin=None if args.extra_margin is None else max(0.0, args.extra_margin),
        associator=args.associator,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
