"""Dual-view, multi-class D-S pipeline.

Two families share one implementation:

* ``match``: view A = RGB, view B = SAR -> fused detection boxes.
* ``time``:  view A = time1, view B = time2 (both optical) -> cross-time identity
  pairs plus a change label (stayed / moved / left / appeared). Boxes are never
  fused across time; the pair members are reported separately.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .ds_rules import adaptive_fusion, make_decision, validate_bpa
from .evidence import build_matched_evidence, build_unmatched_evidence
from .geometry import (
    affine_registration,
    associate_ot,
    hungarian_association,
    rotated_nms,
)
from .config import _resolve

CANONICAL_CLASSES = ["aircraft", "bridge", "harbor", "ship", "tank", "vehicle"]

CLASS_ALIASES = {
    "airplane": "aircraft",
    "airplanes": "aircraft",
    "plane": "aircraft",
    "planes": "aircraft",
    "helicopter": "aircraft",
    "helicopters": "aircraft",
    "boat": "ship",
    "boats": "ship",
    "ships": "ship",
    "vessel": "ship",
    "vessels": "ship",
    "storage tank": "tank",
    "oil tank": "tank",
    "oil storage tank": "tank",
    "storage_tank": "tank",
    "car": "vehicle",
    "cars": "vehicle",
}

_SIZE_CACHE: Dict[str, tuple] = {}


def normalize_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    return CLASS_ALIASES.get(text, text)


def image_size(path: str) -> tuple:
    key = str(path)
    if key not in _SIZE_CACHE:
        from PIL import Image

        with Image.open(key) as image:
            _SIZE_CACHE[key] = tuple(image.size)
    return _SIZE_CACHE[key]


def load_dataset(
    config: Mapping[str, Any],
    family: str = "match",
    pair_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    spec = dict(config["families"][family])
    tool_dir = _resolve(spec["tool_dir"])
    image_dir = _resolve(spec["image_dir"])
    suffix_a = str(spec["views"]["a"]["tool_suffix"]) + ".json"
    ids = sorted(path.name[: -len(suffix_a)] for path in tool_dir.glob(f"*{suffix_a}"))
    if pair_ids:
        wanted = set(pair_ids)
        ids = [pair_id for pair_id in ids if pair_id in wanted]
    pairs = []
    for base_id in ids:
        images = {
            key: str(image_dir / f"{base_id}{view['image_suffix']}.png")
            for key, view in spec["views"].items()
        }
        pairs.append({"base_id": base_id, "family": family, "images": images})
    return pairs


class DualDetectionProvider:
    """Tool-JSON detection source for one family (match or time)."""

    def __init__(self, config: Mapping[str, Any], family: str = "match"):
        spec = dict(config["families"][family])
        self.family = family
        self.views = {key: dict(view) for key, view in spec["views"].items()}
        self.score_threshold = float(dict(config.get("detection") or {}).get("score_threshold", 0.0))
        tool_dir = _resolve(spec["tool_dir"])
        self._cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        suffixes = [(str(view["tool_suffix"]), key) for key, view in self.views.items()]
        for path in sorted(tool_dir.glob("*.json")):
            stem = path.stem
            for suffix, key in suffixes:
                if stem.endswith(suffix):
                    base_id = stem[: -len(suffix)]
                    with path.open("r", encoding="utf-8") as handle:
                        payload = json.load(handle)
                    self._cache.setdefault(base_id, {})[key] = payload.get("detections") or []
                    break

    def detect(
        self,
        base_id: str,
        view: str,
        targets: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        wanted = set(targets) if targets else None
        raw = self._cache.get(base_id, {}).get(view, [])
        boxes = []
        for index, detection in enumerate(raw):
            label = normalize_class(detection.get("label") or detection.get("class_name"))
            if wanted is not None and label not in wanted:
                continue
            bbox = detection.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            except (TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                score = max(0.0, min(1.0, float(detection.get("score", 0.0))))
            except (TypeError, ValueError):
                score = 0.0
            if score < self.score_threshold:
                continue
            boxes.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "score": score,
                "index": index,
                "view": view,
                "source": "tool_json",
            })
        return boxes


def _diag(bbox: Sequence[float]) -> float:
    from math import hypot

    return max(1.0, hypot(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])))


def _center(bbox: Sequence[float]) -> tuple:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def time_change_label(
    view_a_bbox: Optional[Sequence[float]],
    view_b_bbox: Optional[Sequence[float]],
    stayed_diag_ratio: float = 0.5,
) -> str:
    if view_a_bbox is None and view_b_bbox is None:
        return "none"
    if view_a_bbox is None:
        return "appeared"
    if view_b_bbox is None:
        return "left"
    from math import hypot

    ca, cb = _center(view_a_bbox), _center(view_b_bbox)
    distance = hypot(ca[0] - cb[0], ca[1] - cb[1])
    threshold = float(stayed_diag_ratio) * (_diag(view_a_bbox) + _diag(view_b_bbox)) / 2.0
    return "stayed" if distance <= threshold else "moved"


def fuse_box(a_bbox: Sequence[float], b_bbox: Sequence[float], a_score: float, b_score: float) -> List[float]:
    w_a = max(float(a_score), 1e-6)
    w_b = max(float(b_score), 1e-6)
    total = w_a + w_b
    return [(w_a * float(a_bbox[i]) + w_b * float(b_bbox[i])) / total for i in range(4)]


def _decision_for_group(
    group: Mapping[str, Any],
    fusion_cfg: Mapping[str, Any],
    decision_cfg: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    mass, fusion_info = adaptive_fusion(
        group["m1"],
        group["m2"],
        theta=float(fusion_cfg.get("theta", 0.5)),
        rule=str(fusion_cfg.get("rule", "adaptive")),
        conflict_fallback=bool(fusion_cfg.get("conflict_fallback", True)),
        with_info=True,
    )
    try:
        mass = validate_bpa(mass)
        # Single-side detections have no counterpart to support them, so the pair
        # threshold is not a sensible retention threshold for them: raising
        # ``accept_betp`` to reject bad *pairs* would silently delete most valid
        # single-view detections. ``single_accept_betp`` decouples the two; when it
        # is absent the historic behaviour (one threshold for everything) holds.
        single_only = group.get("kind") in ("view_a_only", "view_b_only")
        single_threshold = decision_cfg.get("single_accept_betp")
        accept = float(decision_cfg.get("accept_betp", 0.5))
        if single_only and single_threshold is not None:
            accept = float(single_threshold)
        decision = make_decision(
            mass,
            accept_betp=accept,
            abstain_band=float(decision_cfg.get("abstain_band", 0.0)),
        )
    except (ValueError, TypeError):
        return None
    return {"mass": mass, "decision": decision, "fusion": fusion_info}


def run_pair(
    pair: Mapping[str, Any],
    detections: Mapping[str, List[Dict[str, Any]]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    family = str(pair.get("family") or "match")
    spec = dict(config["families"][family])
    targets = list(config.get("targets") or [config.get("target", "ship")])
    classes = [target for target in targets if target in CANONICAL_CLASSES] or list(targets)

    nms_cfg = dict(config.get("nms") or {})
    assoc_cfg = dict(config.get("association") or {})
    fusion_cfg = dict(config.get("fusion") or {})
    evidence_cfg = dict(config.get("evidence") or {})
    decision_cfg = dict(config.get("decision") or {})
    reg_cfg = dict(config.get("registration") or {})

    boxes_a = [dict(item) for item in detections.get("a", [])]
    boxes_b = [dict(item) for item in detections.get("b", [])]
    if bool(nms_cfg.get("enabled", True)):
        boxes_a = rotated_nms(boxes_a, float(nms_cfg.get("iou_threshold", 0.5)))
        boxes_b = rotated_nms(boxes_b, float(nms_cfg.get("iou_threshold", 0.5)))

    mode = str(spec.get("registration") or reg_cfg.get("mode", "identity"))
    boxes_a, boxes_b, registration = affine_registration(
        boxes_a,
        boxes_b,
        pair["images"]["a"],
        pair["images"]["b"],
        mode=mode,
        min_inliers=int(reg_cfg.get("min_inliers", 8)),
        max_rotation_deg=float(reg_cfg.get("max_rotation_deg", 15.0)),
        max_shift_px=float(reg_cfg.get("max_shift_px", 200.0)),
    )

    if family == "match":
        evidence_cfg_view = {
            **evidence_cfg,
            "alpha_rgb": evidence_cfg.get("alpha_rgb", 0.8),
            "alpha_sar": evidence_cfg.get("alpha_sar", 0.7),
        }
    else:
        evidence_cfg_view = {
            **evidence_cfg,
            "alpha_rgb": float(spec.get("alpha_a", 0.8)),
            "alpha_sar": float(spec.get("alpha_b", 0.8)),
        }

    family_assoc = {**assoc_cfg, **dict(spec.get("association") or {})}
    associator = str(family_assoc.get("associator", "hungarian") or "hungarian").lower()
    ot_cfg = dict(family_assoc.get("ot") or {})
    weights = {
        key: float(family_assoc.get(key, default))
        for key, default in (("w_iou", 0.5), ("w_center", 0.3), ("w_area", 0.2))
    }
    center_scale = float(family_assoc.get("center_scale", 0.75))
    gate_iou = float(family_assoc.get("gate_iou", 0.05))
    gate_score = float(family_assoc.get("gate_score", 0.35))
    stayed_ratio = float(dict(config.get("time") or {}).get("stayed_diag_ratio", 0.5))

    objects: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {
        "view_a_raw": len(boxes_a),
        "view_b_raw": len(boxes_b),
        "matched": 0,
        "view_a_only": 0,
        "view_b_only": 0,
        "output": 0,
    }
    for target in classes:
        a_items = [item for item in boxes_a if item.get("label") == target]
        b_items = [item for item in boxes_b if item.get("label") == target]
        if associator == "ot":
            matched, a_only, b_only = associate_ot(
                a_items,
                b_items,
                weights=weights,
                center_scale=center_scale,
                gate_iou=gate_iou,
                gate_score=gate_score,
                **ot_cfg,
            )
        else:
            matched, a_only, b_only = hungarian_association(
                a_items,
                b_items,
                weights=weights,
                center_scale=center_scale,
                gate_iou=gate_iou,
                gate_score=gate_score,
            )
        counts["matched"] += len(matched)
        counts["view_a_only"] += len(a_only)
        counts["view_b_only"] += len(b_only)

        groups: List[Dict[str, Any]] = []
        for pair_info in matched:
            m1, m2 = build_matched_evidence(
                pair_info["rgb_box"],
                pair_info["sar_box"],
                pair_info,
                evidence_cfg_view,
            )
            groups.append({
                "kind": "matched",
                "m1": m1,
                "m2": m2,
                "view_a_box": pair_info["rgb_box"],
                "view_b_box": pair_info["sar_box"],
                "geometry": {
                    key: pair_info[key]
                    for key in ("iou", "center_distance", "center_score", "area_ratio", "match_score")
                },
            })
        if not bool(decision_cfg.get("drop_unmatched", False)):
            for item in a_only:
                box = next((candidate for candidate in boxes_a if candidate.get("index") == item["rgb_index"] and candidate.get("label") == target), None)
                m1, m2 = build_unmatched_evidence("view_a", box, None, None, evidence_cfg_view)
                groups.append({"kind": "view_a_only", "m1": m1, "m2": m2, "view_a_box": box, "view_b_box": None, "geometry": {}})
            for item in b_only:
                box = next((candidate for candidate in boxes_b if candidate.get("index") == item["sar_index"] and candidate.get("label") == target), None)
                m1, m2 = build_unmatched_evidence("view_b", box, None, None, evidence_cfg_view)
                groups.append({"kind": "view_b_only", "m1": m1, "m2": m2, "view_a_box": None, "view_b_box": box, "geometry": {}})

        matched_fallback = bool(decision_cfg.get("matched_fallback", False))

        def _emit(view_a_box, view_b_box, source, score_value, outcome):
            if view_a_box is not None and view_b_box is not None:
                bbox = fuse_box(view_a_box["bbox"], view_b_box["bbox"], view_a_box["score"], view_b_box["score"])
            elif view_a_box is not None:
                bbox = list(view_a_box["bbox"])
            else:
                bbox = list(view_b_box["bbox"])
            item = {
                "pair_id": pair["base_id"],
                "family": family,
                "label": target,
                "bbox": [round(float(v), 3) for v in bbox],
                "score": round(float(score_value), 6),
                "betp": round(float(score_value), 6),
                "source": source,
                "view_a_bbox": [round(float(v), 3) for v in view_a_box["bbox"]] if view_a_box else None,
                "view_b_bbox": [round(float(v), 3) for v in view_b_box["bbox"]] if view_b_box else None,
                "view_a_score": round(float(view_a_box.get("score") or 0.0), 6) if view_a_box else None,
                "view_b_score": round(float(view_b_box.get("score") or 0.0), 6) if view_b_box else None,
                "view_a_index": view_a_box.get("index") if view_a_box else None,
                "view_b_index": view_b_box.get("index") if view_b_box else None,
                "rule": outcome["fusion"].get("rule"),
                "distance": round(float(outcome["fusion"].get("distance", 0.0)), 6),
            }
            if family == "time":
                item["change"] = time_change_label(item["view_a_bbox"], item["view_b_bbox"], stayed_ratio)
                item["bbox_t0"] = item["view_a_bbox"]
                item["bbox_t1"] = item["view_b_bbox"]
            objects.append(item)

        for group in groups:
            outcome = _decision_for_group(group, fusion_cfg, decision_cfg)
            if outcome is None:
                continue
            decision = outcome["decision"]
            betp = float(decision["betp_ship"])
            view_a_box = group["view_a_box"]
            view_b_box = group["view_b_box"]
            if decision.get("decision") == "ship":
                if view_a_box is not None and view_b_box is not None:
                    source = "matched"
                elif view_a_box is not None:
                    source = "view_a_only"
                else:
                    source = "view_b_only"
                _emit(view_a_box, view_b_box, source, betp, outcome)
            elif matched_fallback and group.get("kind") == "matched":
                # A rejected association must not silently delete both boxes:
                # degrade it to two single-view detections so recall is preserved.
                if view_a_box is not None:
                    _emit(view_a_box, None, "view_a_only", betp, outcome)
                if view_b_box is not None:
                    _emit(None, view_b_box, "view_b_only", betp, outcome)
    counts["output"] = len(objects)
    return {
        "pair_id": pair["base_id"],
        "family": family,
        "registration": {key: value for key, value in registration.items() if key != "matrix"},
        "counts": counts,
        "objects": objects,
    }
