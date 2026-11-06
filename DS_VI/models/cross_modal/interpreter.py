#!/usr/bin/env python3
"""
RGB/SAR cross-modal reference interpreter.

Pure model implementation; no online service or worker dependencies.

Supported tasks:
1. crosscheck_exists             - verify whether a target exists in RGB/SAR.
2. select_largest                - verify the largest target and return final position.
3. select_highest_score          - verify the highest-confidence target and return final position.
4. joint_scene                   - report deduplicated count and representative targets.
5. distance_largest_smallest     - compute distance between largest and smallest target.

Default output is compact: full matching is done internally, but only summary and task-relevant evidence
are returned to the LLM context. Use output_mode='full' for debugging.
"""

import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import logging

# Shared physical primitives. There must be exactly one implementation of
# IoU / center / area / confidence agreement in the repository.
from DS_VI.models.shared_identity.physical_features import (
    bbox_area as shared_bbox_area,
)
from DS_VI.models.shared_identity.physical_features import (
    bbox_area_ratio as shared_bbox_area_ratio,
)
from DS_VI.models.shared_identity.physical_features import (
    bbox_center as shared_bbox_center,
)
from DS_VI.models.shared_identity.physical_features import (
    bbox_center_distance as shared_center_distance,
)
from DS_VI.models.shared_identity.physical_features import (
    bbox_diagonal as shared_bbox_diagonal,
)
from DS_VI.models.shared_identity.physical_features import (
    bbox_iou as shared_bbox_iou,
)
from DS_VI.models.shared_identity.physical_features import (
    compute_pair_features,
    legacy_center_score,
)
import os


def _artifact_dir() -> Path:
    """Directory for tool artifacts, from TOOLS_OUTPUT_DIR.

    Never the current working directory: that made artifact location depend
    on how the worker happened to be launched.
    """
    return Path(os.environ.get("TOOLS_OUTPUT_DIR") or (Path(__file__).resolve().parents[2] / "results/tool_artifacts"))


logger = logging.getLogger(__name__)


class CrossModalInterpreter:
    FINAL_TARGETS = {"vehicle", "tank", "ship", "aircraft", "bridge", "harbor"}

    TARGET_ALIASES = {
        "vehicles": "vehicle", "vehicle": "vehicle", "car": "vehicle", "cars": "vehicle",
        "small vehicle": "vehicle", "large vehicle": "vehicle",
        "tank": "tank", "storage tank": "tank", "oil tank": "tank", "oil storage tank": "tank",
        "ship": "ship", "ships": "ship", "boat": "ship", "boats": "ship",
        "aircraft": "aircraft", "airplane": "aircraft", "airplanes": "aircraft",
        "plane": "aircraft", "planes": "aircraft", "helicopter": "aircraft", "helicopters": "aircraft",
        "bridge": "bridge", "harbor": "harbor",
    }

    OPTICAL_EQUIVALENTS = {
        "vehicle": {"vehicle", "small vehicle", "large vehicle", "car", "cars"},
        "tank": {"tank", "storage tank", "oil tank", "oil storage tank"},
        "ship": {"ship", "ships", "boat", "boats"},
        "aircraft": {"aircraft", "airplane", "airplanes", "plane", "planes", "helicopter", "helicopters"},
        "bridge": {"bridge"},
        "harbor": {"harbor"},
    }

    SAR_EQUIVALENTS = {
        "vehicle": {"vehicle", "car", "cars"},
        "tank": {"tank", "storage tank", "oil tank", "oil storage tank"},
        "ship": {"ship", "ships", "boat", "boats"},
        "aircraft": {"aircraft", "airplane", "airplanes", "plane", "planes", "helicopter", "helicopters"},
        "bridge": {"bridge"},
        "harbor": {"harbor"},
    }

    TASK_ALIASES = {
        "crosscheck": "crosscheck_exists", "exists": "crosscheck_exists", "crosscheck_exists": "crosscheck_exists",
        "largest": "select_largest", "largest_area": "select_largest", "select_largest": "select_largest",
        "highest_score": "select_highest_score", "highest_confidence": "select_highest_score",
        "select_highest_score": "select_highest_score",
        "joint": "joint_scene", "joint_scene": "joint_scene", "joint_detect_scene": "joint_scene",
        "distance": "distance_largest_smallest", "ship_distance": "distance_largest_smallest",
        "largest_smallest_distance": "distance_largest_smallest",
        "distance_largest_smallest": "distance_largest_smallest",
    }

    MATCH_THRESHOLD = 0.45
    # Legacy weights, retained only for the legacy_hard path so historic
    # benchmark numbers stay reproducible.
    W_IOU = 0.50
    W_CENTER = 0.30
    W_AREA = 0.10
    W_CONF = 0.10
    STRONG_IOU_THRESHOLD = 0.35
    CENTER_SUPPORTED_THRESHOLD = 0.75
    DEFAULT_DETECTION_CONFIDENCE = 0.5

    # Reasoning modes. Default stays legacy_hard so existing benchmarks are
    # untouched until the new layer is evaluated separately.
    MODE_LEGACY_HARD = "legacy_hard"
    MODE_CANDIDATE_GRAPH = "candidate_graph"
    MODE_SHARED_IDENTITY = "shared_identity"
    VALID_MODES = (MODE_LEGACY_HARD, MODE_CANDIDATE_GRAPH, MODE_SHARED_IDENTITY)
    DEFAULT_MODE = MODE_LEGACY_HARD

    @classmethod
    def canonical_mode(cls, value: Any) -> str:
        """Normalize the ``mode`` parameter, falling back to ``legacy_hard``."""
        text = str(value or cls.DEFAULT_MODE).strip().lower().replace("-", "_")
        aliases = {
            "legacy": cls.MODE_LEGACY_HARD,
            "hard": cls.MODE_LEGACY_HARD,
            "legacy_hard": cls.MODE_LEGACY_HARD,
            "graph": cls.MODE_CANDIDATE_GRAPH,
            "candidate": cls.MODE_CANDIDATE_GRAPH,
            "candidate_graph": cls.MODE_CANDIDATE_GRAPH,
            "shared": cls.MODE_SHARED_IDENTITY,
            "shared_identity": cls.MODE_SHARED_IDENTITY,
            "soft": cls.MODE_SHARED_IDENTITY,
        }
        return aliases.get(text, cls.DEFAULT_MODE)

    @staticmethod
    def error_response(text: str, error_code: int, status: str, failure_reason: Optional[str] = None) -> Dict[str, Any]:
        structured = {
            "type": "CrossModalInterpretation",
            "schema_version": "1.0",
            "status": status,
        }
        if failure_reason:
            structured["failure_reason"] = failure_reason
        return {"text": text, "error_code": error_code, "status": status, "structured": structured}

    @staticmethod
    def normalize_label(value: Any) -> str:
        text = str(value or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def canonical_target(self, target: str) -> str:
        label = self.normalize_label(target)
        return self.TARGET_ALIASES.get(label, label)

    def canonical_task(self, task: Optional[str]) -> Optional[str]:
        """Normalize task names robustly.

        normalize_label() converts underscores to spaces, so values such as
        "select_largest" become "select largest".  The original version
        looked up only underscore-style keys and therefore silently converted
        every underscore-style task to crosscheck_exists.
        """
        label = self.normalize_label(task or "crosscheck_exists")
        underscore_label = label.replace(" ", "_")
        return (
            self.TASK_ALIASES.get(underscore_label)
            or self.TASK_ALIASES.get(label)
        )

    @staticmethod
    def to_bool(value: Any, default: bool = False) -> bool:
        if value in [None, ""]:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y", "是", "需要"}:
            return True
        if text in {"false", "0", "no", "n", "否", "不需要"}:
            return False
        return default

    @staticmethod
    def clamp_score(value: Any, default: float = DEFAULT_DETECTION_CONFIDENCE) -> float:
        try:
            score = float(value)
        except Exception:
            score = default
        return max(0.0, min(1.0, score))

    @staticmethod
    def parse_detection_text(text: str) -> List[Dict[str, Any]]:
        """Parse the pre-structured worker text format for legacy callers only."""
        detections: List[Dict[str, Any]] = []
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue

            m = re.match(
                r"^\(([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+)\),\s*(.*?),\s*score\s*([-+]?\d*\.?\d+)$",
                line,
            )
            if m:
                x1, y1, x2, y2, label, score = m.groups()
                detections.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "bbox_format": "xyxy", "label": label.strip(), "score": float(score)})
                continue

            m = re.match(
                r"^\(([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+),\s*([-+]?\d+)\),\s*([-+]?\d*\.?\d+)$",
                line,
            )
            if m:
                x1, y1, x2, y2, score = m.groups()
                detections.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "bbox_format": "xyxy", "label": "object", "score": float(score)})
        return detections

    @staticmethod
    def _detections_from_set(value: Any) -> Optional[List[Dict[str, Any]]]:
        if isinstance(value, list):
            return value
        if not isinstance(value, dict):
            return None
        if isinstance(value.get("detections"), list):
            return value["detections"]
        structured = value.get("structured")
        if isinstance(structured, dict) and isinstance(structured.get("detections"), list):
            return structured["detections"]
        # `data` is the legacy DenoDetV2 structured alias.
        if isinstance(value.get("data"), list):
            return value["data"]
        return None

    def load_detections(self, params: Dict[str, Any], list_key: str, set_key: str, text_key: str) -> List[Dict[str, Any]]:
        value = params.get(list_key) if list_key in params else params.get(set_key)
        detections = self._detections_from_set(value)
        if detections is not None:
            return self.normalize_detections(detections)
        text = params.get(text_key)
        if isinstance(text, str):
            logger.warning(f"Using deprecated legacy detection text input: {text_key}")
            return self.normalize_detections(self.parse_detection_text(text))
        return []

    def normalize_detections(self, detections: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for det in detections:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox") or []
            if len(bbox) < 4:
                continue
            try:
                bbox = [int(round(float(v))) for v in bbox[:4]]
            except Exception:
                continue
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            label = det.get("label") or det.get("class_name") or "object"
            item = dict(det)
            item.update({"bbox": bbox, "bbox_format": "xyxy", "label": str(label), "score": self.clamp_score(det.get("score"))})
            out.append(item)
        out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return out

    def filter_by_target(self, detections: List[Dict[str, Any]], target: str, modality: str, category_match_mode: str = "exact") -> List[Dict[str, Any]]:
        target = self.canonical_target(target)
        alias_set = self.OPTICAL_EQUIVALENTS.get(target, {target}) if modality == "rgb" else self.SAR_EQUIVALENTS.get(target, {target})
        aliases = {self.normalize_label(x) for x in alias_set}
        out = []
        for det in detections:
            label = self.normalize_label(det.get("label") or det.get("class_name"))
            if not label or label == "object":
                out.append(det)
                continue
            if category_match_mode == "substring":
                if any(a in label or label in a for a in aliases):
                    out.append(det)
            else:
                if label in aliases:
                    out.append(det)
        return out

    # Geometry delegates to DS_VI.models.shared_identity.physical_features so all
    # callers share one implementation.
    @staticmethod
    def bbox_area(bbox: Sequence[int]) -> float:
        return shared_bbox_area(bbox)

    @staticmethod
    def bbox_center(bbox: Sequence[int]) -> List[float]:
        center = shared_bbox_center(bbox)
        if center is None:
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        return [round(center[0], 3), round(center[1], 3)]

    @staticmethod
    def bbox_diag(bbox: Sequence[int]) -> float:
        return shared_bbox_diagonal(bbox)

    @staticmethod
    def bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
        return shared_bbox_iou(a, b)

    @staticmethod
    def center_distance(a: Sequence[int], b: Sequence[int]) -> float:
        distance = shared_center_distance(a, b)
        return distance if math.isfinite(distance) else 0.0

    @staticmethod
    def fuse_bbox(a: Sequence[int], b: Sequence[int]) -> List[int]:
        return [int(round((float(x) + float(y)) / 2.0)) for x, y in zip(a[:4], b[:4], strict=False)]

    def compute_pair_score(self, rgb_det: Dict[str, Any], sar_det: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy-weighted pair score, computed from shared primitives.

        The features come from the single shared implementation; only the
        weighting (0.50/0.30/0.10/0.10) and the legacy center scale remain local
        so ``legacy_hard`` output stays numerically identical to the baseline.
        """
        rgb_bbox, sar_bbox = rgb_det["bbox"], sar_det["bbox"]
        iou = shared_bbox_iou(rgb_bbox, sar_bbox)
        dist = self.center_distance(rgb_bbox, sar_bbox)
        center_score = legacy_center_score(rgb_bbox, sar_bbox)
        area_score = shared_bbox_area_ratio(rgb_bbox, sar_bbox)
        conf_score = (self.clamp_score(rgb_det.get("score")) + self.clamp_score(sar_det.get("score"))) / 2.0
        score = self.W_IOU * iou + self.W_CENTER * center_score + self.W_AREA * area_score + self.W_CONF * conf_score
        return {
            "match_score": round(score, 4), "iou": round(iou, 4), "center_distance": round(dist, 3),
            "center_score": round(center_score, 4), "area_score": round(area_score, 4), "confidence_score": round(conf_score, 4),
        }

    def compute_shared_pair_features(self, rgb_det: Dict[str, Any], sar_det: Dict[str, Any]) -> Dict[str, Any]:
        """Canonical shared PairFeatures for candidate-graph / shared-identity modes."""
        return compute_pair_features(
            rgb_det.get("bbox"),
            sar_det.get("bbox"),
            observation_confidence=self.clamp_score(rgb_det.get("score")),
            object_confidence=self.clamp_score(sar_det.get("score")),
        ).to_dict()

    def match_type(self, info: Dict[str, Any]) -> str:
        if float(info.get("iou", 0.0)) >= self.STRONG_IOU_THRESHOLD:
            return "strong_iou_match"
        if float(info.get("center_score", 0.0)) >= self.CENTER_SUPPORTED_THRESHOLD:
            return "center_supported_match"
        return "hybrid_score_match"

    def match_detections(self, rgb_dets: List[Dict[str, Any]], sar_dets: List[Dict[str, Any]], match_threshold: Optional[float] = None, max_center_distance: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        threshold = self.MATCH_THRESHOLD if match_threshold is None else float(match_threshold)
        candidates: List[Dict[str, Any]] = []
        for ri, rd in enumerate(rgb_dets):
            for si, sd in enumerate(sar_dets):
                info = self.compute_pair_score(rd, sd)
                if max_center_distance is not None and info["center_distance"] > float(max_center_distance):
                    continue
                if info["match_score"] < threshold:
                    continue
                candidates.append({
                    "rgb_index": ri, "sar_index": si, **info, "match_type": self.match_type(info),
                    "rgb_bbox": rd["bbox"], "sar_bbox": sd["bbox"],
                    "rgb_score": rd.get("score"), "sar_score": sd.get("score"),
                    "rgb_label": rd.get("label"), "sar_label": sd.get("label"),
                })
        candidates.sort(key=lambda x: (float(x.get("match_score", 0.0)), float(x.get("iou", 0.0)), -float(x.get("center_distance", 0.0))), reverse=True)
        used_rgb, used_sar, matched = set(), set(), []
        for item in candidates:
            if item["rgb_index"] in used_rgb or item["sar_index"] in used_sar:
                continue
            used_rgb.add(item["rgb_index"])
            used_sar.add(item["sar_index"])
            matched.append(item)
        rgb_only = [{"rgb_index": i, "rgb_bbox": d["bbox"], "rgb_score": d.get("score"), "rgb_label": d.get("label")} for i, d in enumerate(rgb_dets) if i not in used_rgb]
        sar_only = [{"sar_index": i, "sar_bbox": d["bbox"], "sar_score": d.get("score"), "sar_label": d.get("label")} for i, d in enumerate(sar_dets) if i not in used_sar]
        return matched, rgb_only, sar_only

    def build_dedup_detections(self, matched: List[Dict[str, Any]], rgb_only: List[Dict[str, Any]], sar_only: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
        dedup: List[Dict[str, Any]] = []
        for idx, item in enumerate(matched):
            bbox = self.fuse_bbox(item["rgb_bbox"], item["sar_bbox"])
            dedup.append({
                "name": f"{target}_matched_{idx}", "source": "matched", "label": target, "bbox": bbox,
                "center": self.bbox_center(bbox), "area": round(self.bbox_area(bbox), 3),
                "score": max(float(item.get("rgb_score") or 0.0), float(item.get("sar_score") or 0.0)),
                "match_score": item.get("match_score"), "match_type": item.get("match_type"), "iou": item.get("iou"),
                "center_distance": item.get("center_distance"), "center_score": item.get("center_score"),
                "area_score": item.get("area_score"), "confidence_score": item.get("confidence_score"),
                "rgb_bbox": item.get("rgb_bbox"), "sar_bbox": item.get("sar_bbox"),
                "rgb_label": item.get("rgb_label"), "sar_label": item.get("sar_label"),
            })
        for idx, item in enumerate(rgb_only):
            bbox = item["rgb_bbox"]
            dedup.append({
                "name": f"{target}_rgb_only_{idx}", "source": "rgb_only", "label": item.get("rgb_label") or target,
                "bbox": bbox, "center": self.bbox_center(bbox), "area": round(self.bbox_area(bbox), 3),
                "score": float(item.get("rgb_score") or 0.0), "rgb_bbox": bbox, "sar_bbox": None,
            })
        for idx, item in enumerate(sar_only):
            bbox = item["sar_bbox"]
            dedup.append({
                "name": f"{target}_sar_only_{idx}", "source": "sar_only", "label": item.get("sar_label") or target,
                "bbox": bbox, "center": self.bbox_center(bbox), "area": round(self.bbox_area(bbox), 3),
                "score": float(item.get("sar_score") or 0.0), "rgb_bbox": None, "sar_bbox": bbox,
            })
        return dedup

    @staticmethod
    def source_counts(dedup: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"matched": 0, "rgb_only": 0, "sar_only": 0}
        for d in dedup:
            source = str(d.get("source", "unknown"))
            counts[source] = counts.get(source, 0) + 1
        return counts

    @staticmethod
    def compact_target(det: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not det:
            return None
        keys = ["name", "source", "label", "bbox", "center", "area", "score", "match_score", "match_type", "iou", "center_distance", "rgb_bbox", "sar_bbox"]
        return {k: det[k] for k in keys if k in det and det[k] is not None}

    @staticmethod
    def pick_best_match(dedup: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        items = [d for d in dedup if d.get("source") == "matched"]
        if not items:
            return None
        return sorted(items, key=lambda d: (float(d.get("match_score") or 0.0), float(d.get("iou") or 0.0), float(d.get("score") or 0.0), float(d.get("area") or 0.0)), reverse=True)[0]

    @staticmethod
    def pick_largest(dedup: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        items = [d for d in dedup if float(d.get("area") or 0.0) > 0]
        return sorted(items, key=lambda d: (float(d.get("area") or 0.0), float(d.get("score") or 0.0)), reverse=True)[0] if items else None

    @staticmethod
    def pick_smallest(dedup: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        items = [d for d in dedup if float(d.get("area") or 0.0) > 0]
        return sorted(items, key=lambda d: (float(d.get("area") or 0.0), -float(d.get("score") or 0.0)))[0] if items else None

    @staticmethod
    def pick_highest_score(dedup: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not dedup:
            return None
        return sorted(dedup, key=lambda d: (float(d.get("score") or 0.0), float(d.get("match_score") or 0.0), float(d.get("area") or 0.0)), reverse=True)[0]

    def representative_targets(self, dedup: Sequence[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        candidates = [self.pick_best_match(dedup), self.pick_largest(dedup), self.pick_highest_score(dedup)]
        reps, seen = [], set()
        for item in candidates:
            if not item or item.get("name") in seen:
                continue
            seen.add(item.get("name"))
            reps.append(self.compact_target(item))
        rest = sorted(dedup, key=lambda d: (float(d.get("match_score") or 0.0), float(d.get("score") or 0.0), float(d.get("area") or 0.0)), reverse=True)
        for item in rest:
            if len(reps) >= top_k:
                break
            if item.get("name") in seen:
                continue
            seen.add(item.get("name"))
            reps.append(self.compact_target(item))
        return reps[:top_k]

    @staticmethod
    def conclusion_from_counts(summary: Dict[str, Any]) -> str:
        if int(summary.get("matched_count", 0)) > 0:
            return "confirmed"
        rgb = int(summary.get("rgb_filtered_count", 0))
        sar = int(summary.get("sar_filtered_count", 0))
        if rgb > 0 and sar > 0:
            return "spatial_conflict"
        if rgb > 0 and sar == 0:
            return "rgb_possible"
        if sar > 0 and rgb == 0:
            return "sar_possible"
        return "not_found"

    @staticmethod
    def conclusion_from_selected(selected: Optional[Dict[str, Any]], fallback: str) -> str:
        if not selected:
            return fallback
        source = selected.get("source")
        if source == "matched":
            return "confirmed"
        if source == "rgb_only":
            return "rgb_possible"
        if source == "sar_only":
            return "sar_possible"
        return fallback

    @staticmethod
    def center_distance_m(a: Sequence[float], b: Sequence[float], pixel_size_m: float) -> float:
        return round(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])) * pixel_size_m, 3)

    def build_summary(self, target: str, raw_target: str, rgb_raw: List[Dict[str, Any]], sar_raw: List[Dict[str, Any]], rgb_dets: List[Dict[str, Any]], sar_dets: List[Dict[str, Any]], matched: List[Dict[str, Any]], rgb_only: List[Dict[str, Any]], sar_only: List[Dict[str, Any]], dedup: List[Dict[str, Any]], match_threshold: float) -> Dict[str, Any]:
        return {
            "target": target, "raw_target": raw_target,
            "rgb_input_count": len(rgb_raw), "sar_input_count": len(sar_raw),
            "rgb_filtered_count": len(rgb_dets), "sar_filtered_count": len(sar_dets),
            "matched_count": len(matched), "rgb_only_count": len(rgb_only), "sar_only_count": len(sar_only),
            "dedup_count": len(dedup), "source_counts": self.source_counts(dedup),
            "match_method": "global_sorted_greedy_hybrid_score", "match_threshold": match_threshold,
            "weights": {"iou": self.W_IOU, "center": self.W_CENTER, "area": self.W_AREA, "confidence": self.W_CONF},
        }

    def build_evidence_layer(
        self,
        rgb_dets: List[Dict[str, Any]],
        sar_dets: List[Dict[str, Any]],
        target: str,
        run_inference: bool,
        acquisition_ids: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build observations, claims, candidate edges and physical prior.

        Optical and SAR detections become independent :class:`PrimitiveObservation`
        instances; object hypotheses are seeded from their supports.  No greedy
        pairing happens here.  When ``run_inference`` is set, the shared joint
        A-Q inference is invoked through the shared module — this worker never
        reimplements the algorithm.
        """
        from DS_VI.models.shared_identity.candidate_graph import build_candidate_graph
        from DS_VI.models.shared_identity.config import load_config
        from DS_VI.models.shared_identity.observation_builder import (
            build_claim,
            build_observation,
            cross_image_object_hypotheses,
        )
        from DS_VI.models.shared_identity.physical_prior import compute_physical_prior
        from DS_VI.models.shared_identity.schema import (
            InferenceProblem,
            Modality,
        )

        config = load_config()
        ids = dict(acquisition_ids or {})
        observations = []
        claims = []
        relation_group_id = (
            f"cross_modal:{ids.get('rgb') or 'rgb'}:{ids.get('sar') or 'sar'}"
        )

        for modality, detections, channel, acquisition in (
            (Modality.OPTICAL, rgb_dets, "optical", ids.get("rgb")),
            (Modality.SAR, sar_dets, "sar", ids.get("sar")),
        ):
            image_group_id = f"cross_modal:{channel}:{acquisition or channel}"
            other_acquisition = ids.get("sar") if channel == "optical" else ids.get("rgb")
            other_channel = "sar" if channel == "optical" else "optical"
            for index, detection in enumerate(detections):
                observation = build_observation(
                    source_tool="CrossModalInterpret",
                    modality=modality,
                    bbox=detection.get("bbox"),
                    detection_confidence=self.clamp_score(detection.get("score")),
                    acquisition_id=acquisition,
                    frame_id=acquisition,
                    input_roots=[a for a in [acquisition] if a],
                    context={
                        "channel": channel,
                        "detector_index": index,
                        "relation_group_id": relation_group_id,
                        "image_group_id": image_group_id,
                        "related_image_group_ids": [
                            f"cross_modal:{other_channel}:{other_acquisition or other_channel}"
                        ],
                    },
                    index=index,
                )
                observations.append(observation)
                label = detection.get("label") or target
                claims.append(
                    build_claim(
                        observation,
                        "category",
                        label,
                        source_channel=channel,
                        raw_confidence=self.clamp_score(detection.get("score")),
                    )
                )
                for attribute, keys in (
                    ("subtype", ("subtype", "subclass", "fine_grained_label")),
                    ("heading", ("heading", "orientation", "direction")),
                ):
                    for key in keys:
                        value = detection.get(key)
                        if value not in (None, ""):
                            claims.append(
                                build_claim(
                                    observation,
                                    attribute,
                                    value,
                                    source_channel=channel,
                                    raw_confidence=self.clamp_score(detection.get("score")),
                                )
                            )
                            break

        default_capacity = dict(config.get("inference") or {}).get("default_capacity")
        pair_capacity = dict(config.get("association") or {}).get("pair_capacity")
        pair_capacity = None if pair_capacity is None else max(1, int(pair_capacity))
        objects = cross_image_object_hypotheses(observations, pair_capacity)
        for obj in objects:
            obj.capacity = default_capacity
            obj.attribute_hints = {"category": target}
            obj.context["origin"] = "cross_image_pair"
        edges = build_candidate_graph(observations, objects, config)
        prior = compute_physical_prior(edges, config)

        payload: Dict[str, Any] = {
            "primitive_observations": [o.to_dict() for o in observations],
            "claims": [c.to_dict() for c in claims],
            "object_hypotheses": [o.to_dict() for o in objects],
            "candidate_edges": [e.to_dict() for e in edges],
            "physical_prior": {k: dict(v) for k, v in prior.items()},
            "evidence_summary": {
                "observation_count": len(observations),
                "claim_count": len(claims),
                "object_count": len(objects),
                "edge_count": len(edges),
                "mean_candidates": (
                    len(edges) / len(observations) if observations else 0.0
                ),
            },
        }

        if run_inference and observations:
            from DS_VI.models.shared_identity.inference import run_joint_inference

            problem = InferenceProblem(
                observations=observations,
                claims=claims,
                objects=objects,
                candidate_edges=edges,
                capacities={o.object_id: default_capacity for o in objects},
            )
            state = run_joint_inference(problem, config)
            payload["inference"] = state.to_dict()
        return payload

    @staticmethod
    def write_full_artifact(payload: Dict[str, Any], prefix: str = "cross_modal_interpret") -> Dict[str, Any]:
        out_dir = _artifact_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{prefix}_{uuid.uuid4().hex[:8]}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return {"id": "full_cross_modal_result", "type": "json", "path": str(path)}

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        provenance = params.get("evidence_provenance")
        allow_unprovenanced = self.to_bool(params.get("allow_unprovenanced_text"), False)
        if (params.get("rgb_text") or params.get("sar_text")) and not provenance and not allow_unprovenanced:
            return self.error_response(
                "Missing valid RGB/SAR detection evidence from successful previous tools.",
                2,
                "invalid_dependency",
                "missing evidence_provenance",
            )
        text_blob = json.dumps({"rgb_text": params.get("rgb_text"), "sar_text": params.get("sar_text")}, ensure_ascii=False)
        if not provenance and (
            "(10, 20, 50, 80), ship, score 0.73" in text_blob
            or "(12, 22, 52, 82), ship, score 0.86" in text_blob
        ):
            return self.error_response(
                "Prompt example detections are not valid evidence.",
                2,
                "example_leakage_detected",
                "prompt example bbox leakage",
            )
        raw_target = str(params.get("target") or "").strip()
        if not raw_target:
            return self.error_response("Missing required parameter: target", 2, "invalid_request", "missing target")
        target = self.canonical_target(raw_target)
        if target not in self.FINAL_TARGETS:
            return self.error_response(
                f"Unsupported target: {raw_target}. Supported targets are: {sorted(self.FINAL_TARGETS)}",
                2,
                "invalid_request",
                "unsupported target",
            )

        try:
            task = self.canonical_task(params.get("task"))
            if task is None:
                return self.error_response(
                    f"Unsupported task: {params.get('task')}",
                    2,
                    "invalid_request",
                    "unsupported task",
                )
            output_mode = str(params.get("output_mode") or "compact").strip().lower()
            if output_mode not in {"summary", "compact", "full"}:
                output_mode = "compact"
            mode = self.canonical_mode(params.get("mode") or params.get("reasoning_mode"))
            preview_top_k = max(0, min(int(params.get("preview_top_k", params.get("top_k", 3))), 10))
            category_match_mode = str(params.get("category_match_mode") or "exact").strip().lower()
            if category_match_mode not in {"exact", "substring"}:
                category_match_mode = "exact"
            match_threshold = float(params.get("match_threshold", self.MATCH_THRESHOLD))
            pixel_size_m = float(params.get("pixel_size_m", params.get("gsd_m_per_pixel", 1.0)))
            max_center_distance = params.get("max_center_distance")
            max_center_distance = None if max_center_distance in [None, ""] else float(max_center_distance)
        except (TypeError, ValueError) as exc:
            return self.error_response(
                f"Invalid numeric parameter: {exc}", 2, "invalid_request", str(exc)
            )

        try:
            rgb_raw = self.load_detections(params, "rgb_detections", "rgb_detection_set", "rgb_text")
            sar_raw = self.load_detections(params, "sar_detections", "sar_detection_set", "sar_text")
            rgb_dets = self.filter_by_target(rgb_raw, target, "rgb", category_match_mode)
            sar_dets = self.filter_by_target(sar_raw, target, "sar", category_match_mode)
            matched, rgb_only, sar_only = self.match_detections(rgb_dets, sar_dets, match_threshold, max_center_distance)
            dedup = self.build_dedup_detections(matched, rgb_only, sar_only, target)
            summary = self.build_summary(target, raw_target, rgb_raw, sar_raw, rgb_dets, sar_dets, matched, rgb_only, sar_only, dedup, match_threshold)
            fallback = self.conclusion_from_counts(summary)

            selected = None
            selected_targets: Dict[str, Any] = {}
            distance_pair = None

            if task == "crosscheck_exists":
                selected = self.pick_best_match(dedup) or self.pick_highest_score(dedup)
                selected_targets["best_evidence"] = self.compact_target(selected)
            elif task == "select_largest":
                selected = self.pick_largest(dedup)
                selected_targets["largest"] = self.compact_target(selected)
            elif task == "select_highest_score":
                selected = self.pick_highest_score(dedup)
                selected_targets["highest_score"] = self.compact_target(selected)
            elif task == "joint_scene":
                selected = self.pick_best_match(dedup) or self.pick_largest(dedup)
                selected_targets["representative_targets"] = self.representative_targets(dedup, top_k=max(1, preview_top_k))
            elif task == "distance_largest_smallest":
                # Pick two distinct valid targets.  The old logic called
                # pick_largest() and pick_smallest() independently; when all
                # candidates had the same area, both functions could select the
                # same item and distance_m would be omitted.
                valid = [d for d in dedup if float(d.get("area") or 0.0) > 0]
                largest = sorted(
                    valid,
                    key=lambda d: (float(d.get("area") or 0.0), float(d.get("score") or 0.0), float(d.get("match_score") or 0.0)),
                    reverse=True,
                )[0] if valid else None
                smallest = None
                if largest:
                    for cand in sorted(valid, key=lambda d: (float(d.get("area") or 0.0), -float(d.get("score") or 0.0), str(d.get("name") or ""))):
                        if cand.get("name") != largest.get("name"):
                            smallest = cand
                            break
                selected = largest
                selected_targets["largest"] = self.compact_target(largest)
                selected_targets["smallest"] = self.compact_target(smallest)
                if largest and smallest:
                    distance_pair = {
                        "source": largest.get("name"), "target": smallest.get("name"),
                        "source_role": "largest", "target_role": "smallest",
                        "distance_m": self.center_distance_m(largest["center"], smallest["center"], pixel_size_m),
                        "pixel_size_m": pixel_size_m,
                    }

            conclusion_type = self.conclusion_from_selected(selected, fallback)
            final_bbox = selected.get("bbox") if selected else None
            text = (
                f"CrossModalInterpret: task={task}, target={target}, "
                f"{summary['rgb_filtered_count']} RGB / {summary['sar_filtered_count']} SAR after filtering, "
                f"matched={summary['matched_count']}, rgb_only={summary['rgb_only_count']}, "
                f"sar_only={summary['sar_only_count']}, dedup_count={summary['dedup_count']}."
            )
            if selected:
                text += f" Selected {selected.get('name')} from {selected.get('source')} at {selected.get('bbox')}."

            response: Dict[str, Any] = {
                "text": text, "error_code": 0, "task": task, "target": target,
                "output_mode": output_mode, "truncated": output_mode != "full",
                "mode": mode,
                "summary": summary, "conclusion_type": conclusion_type, "final_bbox": final_bbox,
                "selected_targets": selected_targets,
            }

            # candidate_graph / shared_identity add the evidence layer alongside
            # the legacy fields; legacy_hard output is byte-for-byte unchanged
            # apart from the new "mode" echo.
            if mode in {self.MODE_CANDIDATE_GRAPH, self.MODE_SHARED_IDENTITY}:
                try:
                    evidence = self.build_evidence_layer(
                        rgb_dets,
                        sar_dets,
                        target,
                        run_inference=(mode == self.MODE_SHARED_IDENTITY),
                        acquisition_ids={
                            "rgb": params.get("rgb_image") or params.get("optical_image"),
                            "sar": params.get("sar_image"),
                        },
                    )
                    response.update(evidence)
                    response["legacy_match"] = {
                        "matched_pairs": matched,
                        "rgb_only": rgb_only,
                        "sar_only": sar_only,
                        "note": "legacy greedy assignment, provided for comparison only",
                    }
                except Exception as exc:
                    logger.exception(f"shared-identity evidence layer failed: {exc}")
                    response["evidence_layer_error"] = str(exc)
            if task == "joint_scene":
                if params.get("scene_description") or params.get("scene_context"):
                    response["scene_description"] = str(params.get("scene_description") or params.get("scene_context"))
                response["dedup_count"] = summary["dedup_count"]
                response["source_counts"] = summary["source_counts"]
            if distance_pair is not None:
                response["distance_pair"] = distance_pair
                response["distance_m"] = distance_pair["distance_m"]
            if output_mode in {"compact", "full"}:
                response["preview"] = {
                    "top_matches": [self.compact_target(d) for d in sorted([x for x in dedup if x.get("source") == "matched"], key=lambda x: float(x.get("match_score") or 0.0), reverse=True)[:preview_top_k]],
                    "top_rgb_only": [self.compact_target(d) for d in [x for x in dedup if x.get("source") == "rgb_only"][:preview_top_k]],
                    "top_sar_only": [self.compact_target(d) for d in [x for x in dedup if x.get("source") == "sar_only"][:preview_top_k]],
                }

            full_payload = {
                "rgb_detections": rgb_dets, "sar_detections": sar_dets,
                "matched": matched, "matched_pairs": matched, "rgb_only": rgb_only, "sar_only": sar_only,
                "dedup_detections": dedup, "summary": summary,
            }
            if output_mode == "full":
                response.update(full_payload)
                response["count"] = summary["dedup_count"]
                response["dedup_count"] = summary["dedup_count"]
            elif self.to_bool(params.get("return_artifact"), False):
                artifact = self.write_full_artifact(full_payload, prefix=f"cross_modal_{target}")
                response["artifacts"] = [artifact]
                response["result_id"] = Path(artifact["path"]).stem

            structured = {
                "type": "CrossModalInterpretation",
                "schema_version": "1.0",
                "status": "ok",
                "task": task,
                "target": target,
                "output_mode": output_mode,
                "mode": mode,
                "summary": summary,
                "conclusion_type": conclusion_type,
                "final_bbox": final_bbox,
                "selected_targets": selected_targets,
            }
            for key in (
                "dedup_count", "source_counts", "distance_pair", "distance_m", "preview",
                "rgb_detections", "sar_detections", "matched", "matched_pairs", "rgb_only",
                "sar_only", "dedup_detections", "count", "artifacts", "result_id", "scene_description",
                "primitive_observations", "claims", "object_hypotheses", "candidate_edges",
                "physical_prior", "evidence_summary", "inference", "legacy_match",
                "evidence_layer_error",
            ):
                if key in response:
                    structured[key] = response[key]
            response["structured"] = structured
            return response
        except Exception as exc:
            logger.exception(f"CrossModalInterpret failed: {exc}")
            return self.error_response(f"CrossModalInterpret failed: {exc}", 1, "error", str(exc))

