"""Engine entry point for exp_3, with attribute claims actually injected.

``DS_VI.experiments.run_detection.run_family`` is the trunk entry point and it
deliberately keeps ``extra_claims = {}``: only ``category`` reaches the inference,
while subtype/color/size_bin/orientation are handed to the consistency gate as a
snapshot.  That is the right call for the shipped agent path, but it makes the
contextual reliability model almost inert -- its whole point is to weight the
*attribute* claims, and attribute claim accuracy is 0.40-0.80, not 0.99.

This module mirrors ``run_family`` step for step and adds the evidence channel
that ``property_attributes.claims_from_table`` was written for.  With
``inject_attributes=False`` it must reproduce ``run_family`` byte for byte; the
parity check lives in ``scripts/exp/exp_3/check_parity.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from DS_VI.experiments.property_attributes import (
    attribute_names,
    attributes_from_table,
    claims_from_table,
    context_from_table,
    observation_ids,
)
from DS_VI.models.cascade import cascade_objects
from DS_VI.models.cascade.vi import resolve_cascade_settings, run_vi_pair
from DS_VI.models.ds_fusion.config import merge_overrides
from DS_VI.models.ds_fusion.pipeline import DualDetectionProvider, run_pair


def run_family_with_claims(
    family: str,
    config: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    vi_config: Optional[str] = None,
    attribute_table: Optional[Mapping[str, Any]] = None,
    context_table: Optional[Mapping[str, Any]] = None,
    attributes: Sequence[str] = (),
    inject_attributes: bool = True,
    policy: Optional[str] = None,
    policy_threshold: Optional[float] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """D-S -> VI -> cascade over ``pairs``; returns ``{pair_id: [objects]}``."""
    from DS_VI.experiments.common import write_jsonl

    settings = resolve_cascade_settings(
        vi_config or "DS_VI/configs/shared_identity/runtime_best.yaml",
        family,
        {"provenance": {"grouping_mode": "observation"}},
    )
    policy = settings["policy"] if policy is None else policy
    if policy_threshold is None:
        policy_threshold = settings["blend_weight"]
    extra_links = settings["extra_links"]
    extra_mode = settings["extra_mode"]
    extra_policy = settings["extra_policy"]
    extra_prior_floor = settings["extra_prior_floor"]
    extra_margin = settings["extra_margin"]

    provider = DualDetectionProvider(config, family)
    names = attribute_names(attributes) if (attribute_table is not None and inject_attributes) else ("category",)

    predictions: Dict[str, List[Dict[str, Any]]] = {}
    diagnostics: List[Dict[str, Any]] = []
    for pair in pairs:
        base = pair["base_id"]
        detections = {view: provider.detect(base, view, config["targets"]) for view in ("a", "b")}

        pair_config = config
        decision_override: Dict[str, Any] = {}
        if settings.get("accept_betp") is not None:
            decision_override["accept_betp"] = float(settings["accept_betp"])
            if settings.get("single_accept_betp") is not None:
                decision_override["single_accept_betp"] = float(settings["single_accept_betp"])
        if decision_override:
            pair_config = merge_overrides(pair_config, {"decision": decision_override})

        ds = run_pair(pair, detections, pair_config)
        matched = [obj for obj in ds["objects"] if obj.get("source") == "matched"]
        singles = [obj for obj in ds["objects"] if obj.get("source") != "matched"]

        extra_claims: Dict[str, List[Dict[str, Any]]] = {}
        extra_context: Dict[str, Dict[str, Any]] = {}
        attribute_snapshot: Dict[str, Dict[str, str]] = {}
        if attribute_table is not None or context_table is not None:
            by_view_index = observation_ids(pair, detections, config, family)
            if attribute_table is not None:
                attribute_snapshot = attributes_from_table(
                    attribute_table, base, by_view_index, attributes)
                if inject_attributes:
                    extra_claims = claims_from_table(
                        attribute_table, base, by_view_index, attributes)
            if context_table is not None:
                extra_context = context_from_table(context_table, base, by_view_index)

        vi = run_vi_pair(
            pair, detections, config, family, matched=matched, config_path=vi_config,
            decision_overrides={"provenance": {"grouping_mode": "observation"}},
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
            "injected_claims": sum(len(value) for value in extra_claims.values()),
        })
        print(f"[{family}] {base}: {len(objects)} objects")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", pairs, predictions)
    with (output_dir / "diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for row in diagnostics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return predictions


__all__ = ["run_family_with_claims"]
