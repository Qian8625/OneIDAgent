#!/usr/bin/env python3
"""OneID —— DS+VI 共享身份在线适配器。

把工具侧的 RGB/SAR 检测结果装配成 DS+VI 共享身份链路的输入，并按上游
``experiments/run_detection.py`` 的模板执行三段式推理：

    run_pair(D-S 配对) -> run_vi_pair(共享身份证据) -> cascade_objects(装配)

设计约束
--------
* 上游 ``DS_VI/`` 保持 1:1 vendor，本模块是唯一新增的在线实现；
* 对外契约沿用 CrossModalInterpret 的既有字段（summary / selected_targets /
  final_bbox / dedup_detections），只是数值来源换成 cascade 输出；
* 三级降级全部显式返回 ``mode`` / ``degraded`` / ``degrade_reason``，
  不允许静默降级；
* 在线链路不读任何工作簿、真值或拟合 artifact，属性证据只有 ``category``。

对外只暴露 :func:`interpret`。所有上游异常都收敛成 :class:`AdapterError`，
由 worker 决定降级到 ``vi_only`` / ``legacy_hard``。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
CONFIG_DIR = _HERE / "configs" / "ds_vi"
DEFAULT_DS_CONFIG = CONFIG_DIR / "online_match.json"
DEFAULT_VI_CONFIG = CONFIG_DIR / "runtime_online.yaml"

FAMILY = "match"
MODE_SHARED_IDENTITY = "shared_identity"
MODE_VI_ONLY = "vi_only"

#: 观测数上限：超过即走 vi_only，避免 EM 迭代把并发打满。
MAX_OBSERVATIONS = int(os.environ.get("ONEID_DSVI_MAX_OBSERVATIONS", "800") or 800)
#: 单次推理的软预算秒数；超出后在阶段边界降级。
DEFAULT_BUDGET_SECONDS = float(os.environ.get("ONEID_DSVI_BUDGET_SECONDS", "120") or 120)

#: compact / summary 模式必须剔除的大字段。
BIG_PAYLOAD_KEYS = (
    "primitive_observations",
    "claims",
    "candidate_edges",
    "inference_detail",
)

_API_CACHE: Dict[str, Any] = {}


class AdapterError(RuntimeError):
    """Raised when a stage of the DS+VI chain fails or runs out of budget."""

    def __init__(self, message: str, stage: str = MODE_SHARED_IDENTITY):
        super().__init__(message)
        self.stage = stage


def _api() -> Dict[str, Any]:
    """Import the vendored DS+VI modules once (lazy so start-up stays tolerant)."""
    if _API_CACHE:
        return _API_CACHE
    from DS_VI.models.cascade.attributes import cascade_objects
    from DS_VI.models.cascade.displacement import (
        filter_extras_by_displacement,
        shifts_from_matched,
    )
    from DS_VI.models.cascade.vi import (
        build_observations,
        resolve_cascade_settings,
        run_vi_pair,
        vi_boxes,
    )
    from DS_VI.models.ds_fusion.config import merge_overrides
    from DS_VI.models.ds_fusion.pipeline import normalize_class, run_pair

    _API_CACHE.update({
        "cascade_objects": cascade_objects,
        "filter_extras_by_displacement": filter_extras_by_displacement,
        "shifts_from_matched": shifts_from_matched,
        "build_observations": build_observations,
        "resolve_cascade_settings": resolve_cascade_settings,
        "run_vi_pair": run_vi_pair,
        "vi_boxes": vi_boxes,
        "merge_overrides": merge_overrides,
        "normalize_class": normalize_class,
        "run_pair": run_pair,
    })
    return _API_CACHE


# --------------------------------------------------------------------------- #
# 输入装配
# --------------------------------------------------------------------------- #

def load_ds_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the path-free online D-S config (``online_match.json``)."""
    target = Path(path) if path else DEFAULT_DS_CONFIG
    if not target.is_file():
        raise AdapterError(f"D-S config not found: {target}", "config")
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise AdapterError(f"D-S config is not valid JSON: {target}: {exc}", "config")


def canonical_label(value: Any, interpreter: Any) -> str:
    """Map a tool label onto the shared-identity value domain (plan D4).

    ``TARGET_ALIASES`` first (airplane -> aircraft), then ``normalize_class``
    (cars -> vehicle).  Unknown labels come back as the normalised text and are
    dropped by :func:`_stage_detections`.
    """
    text = interpreter.normalize_label(value)
    if not text:
        return ""
    text = interpreter.TARGET_ALIASES.get(text, text)
    try:
        return str(_api()["normalize_class"](text))
    except Exception:  # pragma: no cover - normalize_class never raises in practice
        return text


def _stage_detections(
    detections: Sequence[Mapping[str, Any]],
    target: str,
    interpreter: Any,
    view: str,
    unmapped: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Canonicalise, filter and index one view's detections.

    Only boxes of the requested target participate in pairing; everything else
    is counted in ``unmapped_labels`` instead of silently disappearing.
    """
    staged: List[Dict[str, Any]] = []
    final_targets = getattr(interpreter, "FINAL_TARGETS", set())
    for index, raw in enumerate(detections or []):
        if not isinstance(raw, Mapping):
            continue
        bbox = raw.get("bbox") or []
        if len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        label = canonical_label(raw.get("label") or raw.get("class_name") or "", interpreter)
        if label not in final_targets or label != target:
            key = str(raw.get("label") or raw.get("class_name") or "unknown")
            unmapped[key] = unmapped.get(key, 0) + 1
            continue
        try:
            score = float(raw.get("score") if raw.get("score") is not None else interpreter.DEFAULT_DETECTION_CONFIDENCE)
        except (TypeError, ValueError):
            score = float(interpreter.DEFAULT_DETECTION_CONFIDENCE)
        staged.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "score": max(0.0, min(1.0, score)),
            "index": index,
            "view": view,
        })
    return staged


def _pair_id(kept: Mapping[str, List[Dict[str, Any]]], target: str, explicit: Any) -> str:
    if explicit:
        return str(explicit)
    key = json.dumps({
        "target": target,
        "a": [[round(v, 2) for v in d["bbox"]] for d in kept.get("a", [])],
        "b": [[round(v, 2) for v in d["bbox"]] for d in kept.get("b", [])],
    }, sort_keys=True, ensure_ascii=False)
    return "dsvi-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 三段式推理
# --------------------------------------------------------------------------- #

def _stage_settings(vi_config: str) -> Dict[str, Any]:
    api = _api()
    return dict(api["resolve_cascade_settings"](
        vi_config, FAMILY, {"provenance": {"grouping_mode": "observation"}}))


def _physical_config(vi_config: str) -> Dict[str, Any]:
    """Read the shared-identity physical weights for reporting.

    ``summary.weights`` must not claim the legacy 0.50/0.30/0.10 weights while
    the answer came from the shared-identity operating point; the numbers are
    therefore taken from the same YAML the cascade used.
    """
    try:
        import yaml

        data = yaml.safe_load(Path(vi_config).read_text(encoding="utf-8")) or {}
        physical = dict(data.get("physical") or {})
        return {
            "weights": dict(physical.get("weights") or {}),
            "gates": dict(physical.get("gates") or {}),
            "center_scale_factor": physical.get("center_scale_factor"),
            "null_prior": physical.get("null_prior"),
        }
    except Exception:  # pragma: no cover - reporting only
        return {}


def _apply_decision_point(config: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Fold the fitted D-S acceptance point into the per-request config."""
    api = _api()
    decision: Dict[str, Any] = {}
    if settings.get("accept_betp") is not None:
        decision["accept_betp"] = float(settings["accept_betp"])
        if settings.get("single_accept_betp") is not None:
            decision["single_accept_betp"] = float(settings["single_accept_betp"])
    if not decision:
        return dict(config)
    return api["merge_overrides"](dict(config), {"decision": decision})


def _full_chain(
    pair: Mapping[str, Any],
    detections: Mapping[str, List[Dict[str, Any]]],
    config: Mapping[str, Any],
    vi_config: str,
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    """D-S -> VI -> cascade, the shipped operating point."""
    api = _api()
    ds = api["run_pair"](pair, detections, config)
    matched = [obj for obj in ds.get("objects") or [] if obj.get("source") == "matched"]
    singles = [obj for obj in ds.get("objects") or [] if obj.get("source") != "matched"]

    vi = api["run_vi_pair"](
        pair,
        detections,
        config,
        FAMILY,
        matched=matched,
        config_path=vi_config,
        decision_overrides={"provenance": {"grouping_mode": "observation"}},
        # Online there are no attribute workbooks: category evidence only.
        attribute_names=("category",),
        extra_links=settings["extra_links"],
        extra_mode=settings["extra_mode"],
        extra_prior_floor=settings["extra_prior_floor"],
        extra_margin=settings["extra_margin"],
    )

    extras = list(vi.get("extras") or [])
    rejected_displacement = 0
    gate = float(settings.get("extra_displacement_gate") or 0.0)
    if gate > 0.0 and extras:
        extras, rejected_displacement = api["filter_extras_by_displacement"](
            extras, api["shifts_from_matched"](matched), gate)

    objects = api["cascade_objects"](
        matched,
        singles,
        vi.get("objects") or [],
        policy=str(settings["policy"]),
        threshold=float(settings["blend_weight"]),
        extra_records=extras,
        extra_policy=str(settings["extra_policy"]),
        family=FAMILY,
    )
    return {
        "objects": objects,
        "ds": ds,
        "vi": vi,
        "extra_rejected_displacement": rejected_displacement,
    }


def _vi_only_chain(
    pair: Mapping[str, Any],
    detections: Mapping[str, List[Dict[str, Any]]],
    config: Mapping[str, Any],
    vi_config: str,
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    """VI alone: no D-S handoff, candidates come from the shared-identity graph."""
    api = _api()
    vi = api["run_vi_pair"](
        pair,
        detections,
        config,
        FAMILY,
        matched=None,
        config_path=vi_config,
        decision_overrides={"provenance": {"grouping_mode": "observation"}},
        attribute_names=("category",),
    )
    policy = str(settings.get("policy") or "status")
    if policy not in {"status", "posterior", "rank"}:
        policy = "posterior"
    boxes = api["vi_boxes"](
        {"objects": vi.get("objects") or [], "singles": vi.get("singles") or []},
        policy=policy,
        threshold=float(settings.get("blend_weight") or 0.5),
    )
    return {"objects": boxes, "ds": {}, "vi": vi, "extra_rejected_displacement": 0}


# --------------------------------------------------------------------------- #
# 输出映射：cascade 对象 -> legacy dedup 契约
# --------------------------------------------------------------------------- #

def _bbox(interp: Any, bbox: Sequence[Any]) -> Optional[List[float]]:
    if not bbox or len(bbox) < 4:
        return None
    try:
        return [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return None


def _source_of(obj: Mapping[str, Any]) -> str:
    status = str(obj.get("status") or obj.get("vi_status") or "")
    if status == "match" or obj.get("source") == "matched":
        return "matched"
    if status == "rgb-only":
        return "rgb_only"
    if status == "sar-only":
        return "sar_only"
    if obj.get("source") in {"view_a_only", "rgb_only"}:
        return "rgb_only"
    if obj.get("source") in {"view_b_only", "sar_only"}:
        return "sar_only"
    if obj.get("view_a_bbox") and obj.get("view_b_bbox"):
        return "matched"
    if obj.get("view_a_bbox"):
        return "rgb_only"
    if obj.get("view_b_bbox"):
        return "sar_only"
    return "sar_only" if str(obj.get("view")) == "b" else "rgb_only"


def _dedup_entries(
    objects: Sequence[Mapping[str, Any]],
    detections: Mapping[str, List[Dict[str, Any]]],
    target: str,
    interp: Any,
) -> List[Dict[str, Any]]:
    """Translate cascade objects into the legacy ``dedup_detections`` schema.

    ``match_score`` carries the cascade score (VI blend probability), which is
    the field downstream skills compare against a threshold (plan G3).
    """
    by_view: Dict[str, Dict[int, Mapping[str, Any]]] = {"a": {}, "b": {}}
    for view in ("a", "b"):
        for det in detections.get(view) or []:
            by_view[view][int(det["index"])] = det

    dedup: List[Dict[str, Any]] = []
    counters = {"matched": 0, "rgb_only": 0, "sar_only": 0}
    for obj in objects:
        bbox = _bbox(interp, obj.get("bbox") or [])
        if bbox is None:
            continue
        source = _source_of(obj)
        view_a_bbox = _bbox(interp, obj.get("view_a_bbox") or [])
        view_b_bbox = _bbox(interp, obj.get("view_b_bbox") or [])
        if source == "matched" and not (view_a_bbox and view_b_bbox):
            # cascade may keep a pair without echoed member boxes; recover them.
            det_a = by_view["a"].get(obj.get("view_a_index")) if obj.get("view_a_index") is not None else None
            det_b = by_view["b"].get(obj.get("view_b_index")) if obj.get("view_b_index") is not None else None
            view_a_bbox = _bbox(interp, det_a["bbox"]) if det_a else None
            view_b_bbox = _bbox(interp, det_b["bbox"]) if det_b else None
            if not (view_a_bbox and view_b_bbox):
                source = "rgb_only" if view_a_bbox else "sar_only" if view_b_bbox else source

        counters[source] = counters.get(source, 0) + 1
        score = float(obj.get("score") or 0.0)
        dedup.append({
            "name": f"{target}_{source}_{counters[source] - 1}",
            "source": source,
            "label": str(obj.get("label") or target),
            "bbox": [round(v, 3) for v in bbox],
            "center": interp.bbox_center(bbox),
            "area": round(interp.bbox_area(bbox), 3),
            "score": round(score, 6),
            "match_score": round(score, 6),
            "betp": obj.get("betp"),
            "vi_status": obj.get("vi_status") or obj.get("status"),
            "vi_p_class": obj.get("vi_p_class") or obj.get("p_class"),
            "vi_association": obj.get("vi_association") or obj.get("association"),
            "rule": obj.get("rule"),
            "rgb_bbox": view_a_bbox,
            "sar_bbox": view_b_bbox,
            "rgb_label": str(obj.get("label") or target) if view_a_bbox else None,
            "sar_label": str(obj.get("label") or target) if view_b_bbox else None,
            "attribute_beliefs": dict(obj.get("attribute_beliefs") or {}),
        })
    return dedup


def _select(
    interp: Any,
    task: str,
    dedup: Sequence[Dict[str, Any]],
    preview_top_k: int,
    pixel_size_m: float,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, Any]], str]:
    """Mirror of ``CrossModalInterpreter.generate``'s task selection.

    Kept in lock-step with upstream on purpose: the selection semantics must not
    drift between ``legacy_hard`` and ``shared_identity``.
    """
    selected: Optional[Dict[str, Any]] = None
    selected_targets: Dict[str, Any] = {}
    distance_pair: Optional[Dict[str, Any]] = None
    fallback = interp.conclusion_from_counts({
        "matched_count": sum(1 for d in dedup if d["source"] == "matched"),
        "rgb_filtered_count": sum(1 for d in dedup if d["source"] in {"matched", "rgb_only"}),
        "sar_filtered_count": sum(1 for d in dedup if d["source"] in {"matched", "sar_only"}),
    })

    if task == "crosscheck_exists":
        selected = interp.pick_best_match(dedup) or interp.pick_highest_score(dedup)
        selected_targets["best_evidence"] = interp.compact_target(selected)
    elif task == "select_largest":
        selected = interp.pick_largest(dedup)
        selected_targets["largest"] = interp.compact_target(selected)
    elif task == "select_highest_score":
        selected = interp.pick_highest_score(dedup)
        selected_targets["highest_score"] = interp.compact_target(selected)
    elif task == "joint_scene":
        selected = interp.pick_best_match(dedup) or interp.pick_largest(dedup)
        selected_targets["representative_targets"] = interp.representative_targets(
            dedup, top_k=max(1, preview_top_k))
    elif task == "distance_largest_smallest":
        valid = [d for d in dedup if float(d.get("area") or 0.0) > 0]
        largest = sorted(
            valid,
            key=lambda d: (float(d.get("area") or 0.0), float(d.get("score") or 0.0),
                           float(d.get("match_score") or 0.0)),
            reverse=True,
        )[0] if valid else None
        smallest = None
        if largest:
            for cand in sorted(valid, key=lambda d: (float(d.get("area") or 0.0),
                                                     -float(d.get("score") or 0.0),
                                                     str(d.get("name") or ""))):
                if cand.get("name") != largest.get("name"):
                    smallest = cand
                    break
        selected = largest
        selected_targets["largest"] = interp.compact_target(largest)
        selected_targets["smallest"] = interp.compact_target(smallest)
        if largest and smallest:
            distance_pair = {
                "source": largest.get("name"),
                "target": smallest.get("name"),
                "source_role": "largest",
                "target_role": "smallest",
                "distance_m": interp.center_distance_m(largest["center"], smallest["center"], pixel_size_m),
                "pixel_size_m": pixel_size_m,
            }
    return selected, selected_targets, distance_pair, fallback


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def interpret(
    interpreter: Any,
    params: Mapping[str, Any],
    target: str,
    rgb_detections: Sequence[Mapping[str, Any]],
    sar_detections: Sequence[Mapping[str, Any]],
    *,
    budget_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the shared-identity chain and return a CrossModalInterpret response.

    ``target`` must already be canonical and in ``FINAL_TARGETS`` (the caller
    validates it).  Raises :class:`AdapterError` when no rung of the ladder can
    produce a result; the worker then falls back to ``legacy_hard``.
    """
    started = time.time()
    budget = float(DEFAULT_BUDGET_SECONDS if budget_seconds is None else budget_seconds)

    def out_of_budget() -> bool:
        return budget > 0 and (time.time() - started) > budget

    try:
        api = _api()
    except Exception as exc:
        raise AdapterError(f"DS+VI modules unavailable: {exc}", "import")

    ds_path = params.get("ds_config_path") or os.environ.get("ONEID_DS_CONFIG") or None
    vi_config = str(params.get("vi_config_path") or os.environ.get("ONEID_VI_CONFIG") or DEFAULT_VI_CONFIG)
    config = load_ds_config(ds_path)
    try:
        settings = _stage_settings(vi_config)
    except Exception as exc:
        raise AdapterError(f"shared-identity config rejected: {exc}", "config")

    unmapped: Dict[str, int] = {}
    kept = {
        "a": _stage_detections(rgb_detections, target, interpreter, "a", unmapped),
        "b": _stage_detections(sar_detections, target, interpreter, "b", unmapped),
    }
    total = len(kept["a"]) + len(kept["b"])
    pair = {
        "base_id": _pair_id(kept, target, params.get("pair_id")),
        "family": FAMILY,
        "images": {
            "a": str(params.get("rgb_image") or "rgb:online"),
            "b": str(params.get("sar_image") or "sar:online"),
        },
    }

    mode = MODE_SHARED_IDENTITY
    degraded = False
    degrade_reason: Optional[str] = None
    chain: Dict[str, Any]
    request_config = _apply_decision_point(config, settings)

    def _empty_result() -> Dict[str, Any]:
        return {"objects": [], "ds": {"objects": [], "counts": {}}, "vi": {"diagnostics": {}},
                "extra_rejected_displacement": 0}

    if total == 0:
        # No usable detections: an empty, successful answer instead of an error
        # (the agent must not loop "re-run detection" forever, plan G5).
        chain = _empty_result()
    elif total > MAX_OBSERVATIONS:
        degraded, degrade_reason = True, f"observation_cap_exceeded:{total}>{MAX_OBSERVATIONS}"
        try:
            chain = _vi_only_chain(pair, kept, request_config, vi_config, settings)
            mode = MODE_VI_ONLY
        except Exception as exc:
            raise AdapterError(f"vi_only stage failed: {exc}", MODE_VI_ONLY)
    else:
        try:
            chain = _full_chain(pair, kept, request_config, vi_config, settings)
        except Exception as exc:
            degraded, degrade_reason = True, f"shared_identity_failed:{exc}"
            try:
                chain = _vi_only_chain(pair, kept, request_config, vi_config, settings)
                mode = MODE_VI_ONLY
            except Exception as exc2:
                raise AdapterError(
                    f"shared_identity failed ({exc}); vi_only failed ({exc2})", MODE_VI_ONLY)
        if out_of_budget():
            degraded = True
            degrade_reason = "budget_exceeded:shared_identity"

    dedup = _dedup_entries(chain.get("objects") or [], kept, target, interpreter)
    preview_top_k = int(params.get("preview_top_k") or params.get("top_k") or 3)
    preview_top_k = max(0, min(preview_top_k, 10))
    pixel_size_m = float(params.get("pixel_size_m") or params.get("gsd_m_per_pixel") or 1.0)

    task = interpreter.canonical_task(params.get("task"))
    if task is None:  # pragma: no cover - caller validates
        raise AdapterError(f"unsupported task: {params.get('task')}", "request")
    selected, selected_targets, distance_pair, fallback = _select(
        interpreter, task, dedup, preview_top_k, pixel_size_m)
    conclusion_type = interpreter.conclusion_from_selected(selected, fallback)
    final_bbox = selected.get("bbox") if selected else None

    matched_items = [d for d in dedup if d["source"] == "matched"]
    rgb_only_items = [d for d in dedup if d["source"] == "rgb_only"]
    sar_only_items = [d for d in dedup if d["source"] == "sar_only"]

    summary = interpreter.build_summary(
        target, str(params.get("target") or target),
        list(rgb_detections or []), list(sar_detections or []),
        kept["a"], kept["b"],
        matched_items, rgb_only_items, sar_only_items, dedup,
        float(settings.get("blend_weight") or 0.5),
    )
    physical = _physical_config(vi_config)
    if physical.get("weights"):
        summary["weights"] = physical["weights"]
    summary.update({
        "mode": mode,
        "degraded": bool(degraded),
        "degrade_reason": degrade_reason,
        "match_method": "shared_identity_cascade" if mode == MODE_SHARED_IDENTITY else "vi_only",
        "unmapped_labels": dict(sorted(unmapped.items())),
        "dsvi": {
            "family": FAMILY,
            "pair_id": pair["base_id"],
            "observations": total,
            "policy": str(settings.get("policy")),
            "blend_weight": float(settings.get("blend_weight") or 0.5),
            "accept_betp": settings.get("accept_betp"),
            "single_accept_betp": settings.get("single_accept_betp"),
            "attribute_names": ["category"],
            "physical": physical,
            "extra_rejected_displacement": chain.get("extra_rejected_displacement", 0),
        },
    })
    if mode == MODE_SHARED_IDENTITY and not degraded:
        summary["match_threshold"] = float(settings.get("blend_weight") or 0.5)

    text = (
        f"CrossModalInterpret: task={task}, target={target}, mode={mode}"
        + (" (degraded)" if degraded else "")
        + f", {summary['rgb_filtered_count']} RGB / {summary['sar_filtered_count']} SAR after filtering, "
        f"matched={summary['matched_count']}, rgb_only={summary['rgb_only_count']}, "
        f"sar_only={summary['sar_only_count']}, dedup_count={summary['dedup_count']}."
    )
    if selected:
        text += f" Selected {selected.get('name')} from {selected.get('source')} at {selected.get('bbox')}."

    output_mode = str(params.get("output_mode") or "compact").strip().lower()
    if output_mode not in {"summary", "compact", "full"}:
        output_mode = "compact"

    vi_diagnostics = dict((chain.get("vi") or {}).get("diagnostics") or {})
    evidence_summary: Dict[str, Any] = {
        "observations": total,
        "objects": len(chain.get("objects") or []),
        "matched": len(matched_items),
        "rgb_only": len(rgb_only_items),
        "sar_only": len(sar_only_items),
        "iterations": vi_diagnostics.get("iterations"),
        "converged": vi_diagnostics.get("converged"),
        "mean_null_mass": vi_diagnostics.get("mean_null_mass"),
        "extra_candidates": vi_diagnostics.get("extra_candidates"),
        "extra_confirmed": vi_diagnostics.get("extra_confirmed"),
        "extra_rejected_inconsistent": vi_diagnostics.get("extra_rejected_inconsistent"),
        "extra_rejected_displacement": chain.get("extra_rejected_displacement", 0),
        "policy": str(settings.get("policy")),
        "blend_weight": float(settings.get("blend_weight") or 0.5),
        "capability": {"attributes": ["category"], "time_family": False, "registration": "identity"},
    }
    inference = {
        "iterations": vi_diagnostics.get("iterations"),
        "converged": vi_diagnostics.get("converged"),
        "mean_null_mass": vi_diagnostics.get("mean_null_mass"),
        "cascade_settings": vi_diagnostics.get("cascade_settings"),
    }

    hypotheses = [dict(obj) for obj in (chain.get("objects") or [])]
    if output_mode != "full":
        # 上下文预算：compact 只保留有限条对象假设。
        hypotheses = hypotheses[:max(preview_top_k, 5)]

    response: Dict[str, Any] = {
        "text": text,
        "error_code": 0,
        "task": task,
        "target": target,
        "output_mode": output_mode,
        "truncated": output_mode != "full",
        "mode": mode,
        "degraded": bool(degraded),
        "degrade_reason": degrade_reason,
        "summary": summary,
        "conclusion_type": conclusion_type,
        "final_bbox": final_bbox,
        "selected_targets": selected_targets,
        "dedup_count": len(dedup),
        "source_counts": summary["source_counts"],
        "evidence_summary": evidence_summary,
        "inference": inference,
        "object_hypotheses": hypotheses,
    }
    if task == "joint_scene":
        if params.get("scene_description") or params.get("scene_context"):
            response["scene_description"] = str(
                params.get("scene_description") or params.get("scene_context"))
    if distance_pair is not None:
        response["distance_pair"] = distance_pair
        response["distance_m"] = distance_pair["distance_m"]
    if output_mode in {"compact", "full"}:
        response["preview"] = {
            "top_matches": [interpreter.compact_target(d) for d in sorted(
                matched_items, key=lambda d: float(d.get("match_score") or 0.0), reverse=True)[:preview_top_k]],
            "top_rgb_only": [interpreter.compact_target(d) for d in rgb_only_items[:preview_top_k]],
            "top_sar_only": [interpreter.compact_target(d) for d in sar_only_items[:preview_top_k]],
        }
    if output_mode == "full":
        response["dedup_detections"] = dedup
        response["objects"] = [dict(obj) for obj in (chain.get("objects") or [])]
        try:
            built = api["build_observations"](pair, kept, request_config, FAMILY)
            response["primitive_observations"] = [obs.to_dict() if hasattr(obs, "to_dict") else str(obs)
                                                  for obs in built.get("observations") or []]
            response["claims"] = [cl.to_dict() if hasattr(cl, "to_dict") else str(cl)
                                  for cl in built.get("claims") or []]
        except Exception as exc:  # pragma: no cover - diagnostics only
            response["diagnostics_error"] = str(exc)
        if chain.get("ds"):
            response["ds_stage"] = {
                "counts": chain["ds"].get("counts"),
                "registration": chain["ds"].get("registration"),
            }
        response["vi_diagnostics"] = vi_diagnostics

    # 旧贪心结果保留为对照，不参与任何选目标逻辑。
    try:
        legacy_matched, legacy_rgb_only, legacy_sar_only = interpreter.match_detections(
            kept["a"], kept["b"], float(params.get("match_threshold") or interpreter.MATCH_THRESHOLD),
            None if params.get("max_center_distance") in (None, "") else float(params.get("max_center_distance")),
        )
        if output_mode == "full":
            response["legacy_match"] = {
                "matched_pairs": legacy_matched,
                "rgb_only": legacy_rgb_only,
                "sar_only": legacy_sar_only,
                "note": "legacy greedy assignment, provided for comparison only",
            }
        else:
            # 上下文预算：compact 只给计数，要逐对细节请用 output_mode=full。
            response["legacy_match"] = {
                "matched_count": len(legacy_matched),
                "rgb_only_count": len(legacy_rgb_only),
                "sar_only_count": len(legacy_sar_only),
                "note": "legacy greedy counts only in compact mode; use output_mode=full for pairs",
            }
    except Exception:  # pragma: no cover - comparison data only
        response["legacy_match"] = {"matched_pairs": [], "rgb_only": [], "sar_only": [],
                                    "note": "legacy greedy assignment unavailable"}

    if out_of_budget():
        response["summary"]["budget_exceeded"] = True
        response["summary"]["elapsed_seconds"] = round(time.time() - started, 3)
    return response


def strip_big_fields(response: Dict[str, Any], output_mode: str) -> Dict[str, Any]:
    """Enforce the context budget of compact/summary modes (plan G1)."""
    if str(output_mode) == "full":
        return response
    for key in BIG_PAYLOAD_KEYS:
        response.pop(key, None)
    return response


__all__ = [
    "AdapterError",
    "BIG_PAYLOAD_KEYS",
    "DEFAULT_DS_CONFIG",
    "DEFAULT_VI_CONFIG",
    "MODE_SHARED_IDENTITY",
    "MODE_VI_ONLY",
    "canonical_label",
    "interpret",
    "load_ds_config",
    "strip_big_fields",
]
