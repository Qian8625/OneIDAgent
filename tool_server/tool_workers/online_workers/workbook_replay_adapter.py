"""Isolated controlled property replay through D-S, VI and cascade.

Detections are live: they arrive as ``rgb_detections`` / ``sar_detections`` from
``ObjectDetection`` / ``DenoDetV2`` through the same registry gate as the online
path.  The workbook is read for one thing only — the requested target's
attributes — and those are attached to the live observations by the frozen
snapshot index.

Binding rule (the one contract this module owns)
------------------------------------------------
``instance_id`` carries the 0-based position of the box in the raw detector JSON
order, and ``_stage_detections`` keeps that same position in
``detection["index"]``.  So ``(view, index)`` alone binds a live box to its
workbook attribute row — no geometric threshold decides what an object *is*.
The row's frozen bbox is only used to audit the binding: every bound pair gets
an IoU in ``replay_diagnostics.attribute_binding``, and a snapshot that has
drifted so far that no bound box overlaps at all fails explicitly.
"""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from tool_server.tool_workers.workbook_replay import ATTRIBUTES, ReplayError, load_property_workbook

HERE = Path(__file__).resolve().parent
VI_CONFIG = HERE / "configs/ds_vi/runtime_online.yaml"
CONFIG = HERE / "configs/ds_vi/workbook_replay.json"

#: Arguments this mode accepts.  Detection payloads are required; server-side
#: switches (workbook path, config paths) are not accepted.
ALLOWED = {
    "source_mode", "family", "dataset_id", "pair_id", "target", "task", "output_mode", "mode",
    "preview_top_k", "pixel_size_m", "gsd_m_per_pixel", "rgb_image", "sar_image",
    "time1_image", "time2_image", "rgb_detections", "sar_detections",
    "rgb_detection_set", "sar_detection_set", "evidence_provenance",
}


def _bbox_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _live(params, *keys):
    for key in keys:
        value = params.get(key)
        if isinstance(value, (list, tuple)) and len(value):
            return list(value)
    return []


def interpret(interpreter, params):
    from DS_VI.models.cascade.attributes import cascade_objects
    from DS_VI.models.cascade.vi import build_observations, run_vi_pair, shipped_cascade_settings
    from DS_VI.models.shared_identity.config import load_config
    from DS_VI.models.ds_fusion.pipeline import run_pair, time_change_label
    from DS_VI.models.cascade.status import object_status
    from tool_server.tool_workers.online_workers import ds_vi_adapter
    if set(params) - ALLOWED:
        raise ReplayError("Unsupported replay arguments: " + ", ".join(sorted(set(params) - ALLOWED)))
    family = params.get("family")
    mode = params.get("mode", "shared_identity")
    if mode not in {"shared_identity", "vi_only"}:
        raise ReplayError("Replay supports shared_identity or vi_only; no attribute-free fallback")
    target = interpreter.canonical_target(params.get("target"))
    if target not in interpreter.FINAL_TARGETS:
        raise ReplayError("Explicit supported target required")
    task = params.get("task") or ("temporal_changes" if family == "time" else "joint_scene")
    if family == "time":
        if task not in {"temporal_changes", "joint_scene"}:
            raise ReplayError("Time replay supports temporal_changes or joint_scene")
    else:
        task = interpreter.canonical_task(task)
        if task is None:
            raise ReplayError("Unsupported match task")
    output = params.get("output_mode", "compact")
    if output not in {"summary", "compact", "full"}:
        raise ReplayError("Invalid output_mode")
    pixel_size = float(params.get("pixel_size_m", params.get("gsd_m_per_pixel", 1.0)))
    if not __import__("math").isfinite(pixel_size) or pixel_size <= 0:
        raise ReplayError("Invalid pixel size")
    replay = load_property_workbook(params.get("dataset_id"), family, params.get("pair_id"), target)
    expected_images = replay["pair"]["images"]
    # The family's own keys win: app wiring binds rgb_image/sar_image from the
    # registry for every mode, so a stray optical image must not override the
    # declared time1/time2 pair of a time request.
    if family == "time":
        supplied = {"a": params.get("time1_image") or params.get("rgb_image"), "b": params.get("time2_image") or params.get("sar_image")}
    else:
        supplied = {"a": params.get("rgb_image") or params.get("time1_image"), "b": params.get("sar_image") or params.get("time2_image")}
    for key, value in supplied.items():
        if value and str(Path(str(value)).resolve()) != str(Path(expected_images[key]).resolve()):
            raise ReplayError("Runtime image does not match frozen workbook sample")
    # Detections come from the live tools, not from the workbook.
    raw = {"a": _live(params, "rgb_detections", "rgb_detection_set"), "b": _live(params, "sar_detections", "sar_detection_set")}
    # 空列表是"该视图零目标"的合法证据（rgb_only / sar_only 的一半语义），
    # 只有"键缺失"才算没取证。
    if raw["a"] is None or raw["b"] is None:
        raise ReplayError("Replay needs live detections on both views; the workbook supplies attributes only")
    detections, unmapped = {}, {}
    for key in ("a", "b"):
        unmapped[key] = {}
        detections[key] = ds_vi_adapter._stage_detections(raw[key], target, interpreter, key, unmapped[key])
    total = sum(len(rows) for rows in detections.values())
    if total == 0:
        raise ReplayError("Live detections contain no " + target + " to pair with workbook attributes: " + json.dumps(unmapped, sort_keys=True))
    if total > ds_vi_adapter.MAX_OBSERVATIONS:
        raise ReplayError("Replay observation cap exceeded; no silent estimator fallback")
    pair = replay["pair"]
    frozen = json.loads(CONFIG.read_text())
    overrides = frozen["vi_overrides"]
    shared = load_config(VI_CONFIG, overrides=overrides)
    config = ds_vi_adapter.load_ds_config()
    config["families"]["time"] = frozen["time_family"]
    settings = shipped_cascade_settings(shared, family)
    config = ds_vi_adapter._apply_decision_point(config, settings)
    built = build_observations(pair, detections, config, family)
    claims, context, snapshot, mapping, counts = {}, {}, {}, [], Counter()
    binding = {"bound": 0, "unbound_live": 0, "unbound_workbook": 0, "ious": []}
    # 位置绑定假定了"检测器跨运行输出顺序稳定"。同一模型重跑时分数会有 ~1e-3 抖动，
    # 近似并列的框会互换位置，于是 (view, index) 绑不上。默认仍是纯位置绑定；
    # 显式设置 ONEID_REPLAY_BINDING=iou_assisted 时，位置绑定失配的框改用
    # 同视图、未占用、IoU 最高的行（阈值 0.5），并在诊断里如实记账。
    binding_mode = os.environ.get("ONEID_REPLAY_BINDING", "positional").strip().lower()
    assist_iou = 0.5
    used_rows = {view: set() for view in ("a", "b")}
    assisted = 0
    for view in ("a", "b"):
        rows = replay["attributes"][view]
        # Positional binding: ``instance_id`` freezes a box's index in the raw
        # detector order and staging kept that index, so the row carrying the
        # same ``(view, index)`` is this box's attribute row.  Geometry never
        # decides identity here; the row bbox only audits the binding below.
        for det in detections[view]:
            oid = built["by_view_index"][(view, det["index"])]
            row = rows.get(det["index"])
            row_key = det["index"]
            if binding_mode == "iou_assisted" and rows:
                positional_iou = _bbox_iou(row["bbox"], det["bbox"]) if row is not None else 0.0
                if row is None or positional_iou < assist_iou or row_key in used_rows[view]:
                    best_key, best_iou = None, 0.0
                    for candidate_key, candidate in rows.items():
                        if candidate_key in used_rows[view]:
                            continue
                        candidate_iou = _bbox_iou(candidate["bbox"], det["bbox"])
                        if candidate_iou > best_iou:
                            best_key, best_iou = candidate_key, candidate_iou
                    if best_key is not None and best_iou >= assist_iou:
                        row, row_key = rows[best_key], best_key
                        assisted += 1
            if row is None:
                binding["unbound_live"] += 1
                mapping.append(dict(observation_id=oid, index=det["index"], view=view, attribute_source="none"))
                continue
            used_rows[view].add(row_key)
            iou = _bbox_iou(row["bbox"], det["bbox"])
            for attribute, value in row["attributes"].items():
                if value not in shared["value_domains"].get(attribute, []):
                    raise ReplayError("Invalid attribute value: " + attribute + "=" + value)
            binding["bound"] += 1
            binding["ious"].append(iou)
            claims[oid] = [{"attribute": a, "value": v, "source_tool": "property_workbook", "source_channel": "sar" if family == "match" and view == "b" else "optical"} for a, v in row["attributes"].items()]
            context[oid] = dict(row["context"], replay_instance_id=row["instance_id"], replay_sha256=replay["provenance"]["sha256"], workbook_iou=round(iou, 4))
            snapshot[oid] = row["attributes"]
            counts.update(row["attributes"].keys())
            mapping.append(dict(observation_id=oid, instance_id=row["instance_id"], index=det["index"], view=view, sheet_row=row["sheet_row"], workbook_iou=round(iou, 4), fields=sorted(row["attributes"])))
    binding["unbound_workbook"] = sum(len(rows) for rows in replay["attributes"].values()) - binding["bound"]
    binding["snapshot_rows"] = dict(replay["instance_counts"])
    binding["live_payload"] = {view: len(raw[view]) for view in ("a", "b")}
    binding["live_target_payload"] = {view: len(detections[view]) for view in ("a", "b")}
    binding["binding_mode"] = "iou_assisted" if binding_mode == "iou_assisted" else "positional_index"
    if binding_mode == "iou_assisted":
        binding["iou_assisted_bindings"] = assisted
    binding["live_detections"] = {"a": len(detections["a"]), "b": len(detections["b"])}
    binding["workbook_target_rows"] = {view: len(rows) for view, rows in replay["attributes"].items()}
    binding["unmapped_labels"] = unmapped
    # A shorter live payload cannot be compared positionally with the snapshot;
    # equal lengths do not prove shared order, a difference proves it is not
    # shared.  Reported, never trusted.
    binding["index_space_matches"] = binding["snapshot_rows"] == binding["live_payload"]
    binding["min_iou"] = round(min(binding["ious"]), 4) if binding["ious"] else None
    binding["max_iou"] = round(max(binding["ious"]), 4) if binding["ious"] else None
    binding["iou_below_half"] = sum(1 for iou in binding["ious"] if iou < 0.5)
    binding["zero_iou_pairs"] = sum(1 for iou in binding["ious"] if iou == 0.0)
    del binding["ious"]
    if not binding["bound"] and any(replay["attributes"].values()):
        raise ReplayError("Attribute binding failed: no live target box matched a workbook row at the same index")
    if binding["bound"] and binding["zero_iou_pairs"] == binding["bound"]:
        raise ReplayError("Attribute binding audit failed: no bound box overlaps its frozen snapshot row, so the workbook has drifted from this pair's detections")
    if mode == "vi_only":
        ds, matched, singles, matched_arg = {"objects": []}, [], [], None
    else:
        ds = run_pair(pair, detections, config)
        matched = [o for o in ds["objects"] if o.get("source") == "matched"]
        singles = [o for o in ds["objects"] if o.get("source") != "matched"]
        matched_arg = matched
    vi = run_vi_pair(pair, detections, config, family, matched=matched_arg, config_path=str(VI_CONFIG), decision_overrides=overrides, extra_claims=claims, extra_context=context, attribute_names=("category",) + ATTRIBUTES, attribute_snapshot=snapshot)
    if mode == "shared_identity":
        gate = float(settings.get("extra_displacement_gate") or 0.0)
        if gate and vi.get("extras"):
            from DS_VI.models.cascade.displacement import filter_extras_by_displacement, shifts_from_matched
            vi["extras"], rejected = filter_extras_by_displacement(vi.get("extras") or [], shifts_from_matched(matched), gate)
            vi.setdefault("diagnostics", {})["extra_rejected_displacement"] = rejected
    if mode == "vi_only":
        objects = []
        # Keep canonical VI membership and posteriors, not lossy bbox exports.
        index_of = {oid: key for key, oid in built["by_view_index"].items()}
        for record in vi["objects"]:
            if not record.get("bbox"):
                continue
            if record.get("status") not in {"Confirmed", "Weak Support"} or record.get("value") != record.get("class"):
                continue
            members = [index_of[m] for m in record.get("members", []) if m in index_of]
            indices = {view: index for view, index in members}
            objects.append(dict(record, label=record["class"], score=record.get("confidence", 0.0), source="matched", view_a_index=indices.get("a"), view_b_index=indices.get("b")))
        for single in vi.get("singles", []):
            view, index = index_of[single["observation_id"]]
            objects.append(dict(single, label=single["class"], source="view_a_only" if view == "a" else "view_b_only", view_a_index=index if view == "a" else None, view_b_index=index if view == "b" else None))
    else:
        objects = cascade_objects(matched, singles, vi["objects"], policy=settings["policy"], threshold=settings["blend_weight"], extra_records=vi.get("extras"), extra_policy=settings["extra_policy"], family=family)
    by_index = {v: {d["index"]: d for d in detections[v]} for v in ("a", "b")}
    for i, obj in enumerate(objects):
        a, b = by_index["a"].get(obj.get("view_a_index")), by_index["b"].get(obj.get("view_b_index"))
        obj["object_id"] = pair["base_id"] + ":replay:" + str(i)
        obj["view_a_bbox"], obj["view_b_bbox"] = a["bbox"] if a else None, b["bbox"] if b else None
        if family == "time":
            obj["bbox_t0"], obj["bbox_t1"] = obj["view_a_bbox"], obj["view_b_bbox"]
            obj["change"] = time_change_label(obj["bbox_t0"], obj["bbox_t1"], config["time"]["stayed_diag_ratio"])
        obj["status"] = object_status(family, a is not None, b is not None, obj.get("change"))
    diagnostics = {"detection_source": "live_detectors", "attribute_source": "property_workbook", "target": target, "attribute_binding": binding, "observations": len(mapping), "category_claims": len(mapping), "attribute_claims": sum(counts.values()), "attribute_claims_by_name": dict(counts), "context_observations": sum(bool(value) for value in context.values()), "mapping": mapping, "reliability_status": frozen["reliability_status"], "theta_enabled": False, "provenance_grouping": "observation", "vi": vi["diagnostics"], "vi_config_sha256": hashlib.sha256(VI_CONFIG.read_bytes()).hexdigest(), "replay_config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest()}
    selected, selected_targets, distance = None, {}, None
    if family == "match":
        dedup = ds_vi_adapter._dedup_entries(objects, detections, target, interpreter)
        selected, selected_targets, distance, fallback = ds_vi_adapter._select(interpreter, task, dedup, 3, pixel_size)
        conclusion = interpreter.conclusion_from_selected(selected, fallback)
    else:
        dedup, conclusion = objects, "temporal_changes"
    summary = dict(family=family, pair_id=pair["base_id"], count=len(objects), source_counts=dict(Counter(o["status"] for o in objects)), source_mode="workbook_replay", detection_source="live_detectors", attribute_source="property_workbook")
    limit = len(objects) if output == "full" else min(10, max(1, int(params.get("preview_top_k", 5))))
    capability = dict(attributes=["category"] + list(ATTRIBUTES), time_family=family == "time", registration="identity", truth_data=False, source_mode="workbook_replay", detection_source="live_detectors", attribute_source="property_workbook")
    result = dict(error_code=0, status="ok", mode=mode, source_mode="workbook_replay", family=family, dataset_id=params["dataset_id"], pair_id=pair["base_id"], task=task, target=target, output_mode=output, degraded=False, degrade_reason=None, text="Live detections; the target's attributes come from the frozen property workbook; fixed priors are not calibrated for replay.", count=len(objects), dedup_count=len(objects), summary=summary, source_counts=summary["source_counts"], selected_targets=selected_targets, final_bbox=selected.get("bbox") if selected else None, conclusion_type=conclusion, capability=capability, replay_provenance=replay["provenance"], replay_diagnostics=diagnostics, object_hypotheses=objects[:limit], truncated=len(objects) > limit)
    if distance:
        result.update(distance_pair=distance, distance_m=distance["distance_m"])
    if output == "full":
        result.update(objects=objects, dedup_detections=dedup, attribute_claims=claims, observation_context=context, vi_objects=vi["objects"])
    else:
        diagnostics.pop("mapping")
    result["structured"] = {k: v for k, v in result.items() if k not in {"text", "error_code", "structured"}}
    return result
