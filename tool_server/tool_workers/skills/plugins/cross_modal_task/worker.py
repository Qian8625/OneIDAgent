"""Task-level optical-SAR reasoning skill."""
import argparse
import ast
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"CrossModalTaskSkill_worker_{worker_id}.log")


class CrossModalTaskSkillWorker(BaseToolWorker):
    INTERNAL_TOOL_WEIGHTS = {
        "CrossModalMatchSkill": 4,
        "ObjectDetectSelectSkill": 3,
        "SAREvidenceExtractSkill": 4,
        "SARToRGB": 4,
        "DenoDetV2": 4,
    }

    VALID_TASKS = {
        "optical_sar_verify",
        "sar_optical_attribution",
        "sar_candidate_recall",
        "optical_sar_consistency",
        "cross_modal_select_largest",
        "cross_modal_select_highest_score",
        "cross_modal_distance_largest_smallest",
    }

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "CrossModalTaskSkill",
        limit_model_concurrency: int = 2,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        wait_timeout: float = 180.0,
        task_timeout: float = 240.0,
    ):
        self.tool_addr_cache: Dict[str, str] = {}
        super().__init__(
            controller_addr=controller_addr,
            worker_addr=worker_addr,
            worker_id=worker_id,
            no_register=no_register,
            model_name=model_name,
            limit_model_concurrency=limit_model_concurrency,
            host=host,
            port=port,
            wait_timeout=wait_timeout,
            task_timeout=task_timeout,
        )

    def init_model(self):
        logger.info(f"{self.model_name} initialized successfully.")

    def _resolve_tool_addr(self, tool_name: str) -> str:
        cached = self.tool_addr_cache.get(tool_name)
        if cached:
            return cached
        ret = requests.post(
            self.controller_addr + "/get_worker_address",
            json={"model": tool_name},
            timeout=5,
        )
        ret.raise_for_status()
        addr = (ret.json().get("address") or "").strip()
        if not addr:
            raise RuntimeError(f"No worker address returned for tool '{tool_name}'")
        self.tool_addr_cache[tool_name] = addr
        return addr

    def _call_tool(self, tool_name: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
        addr = self._resolve_tool_addr(tool_name)
        ret = requests.post(
            addr + "/worker_generate",
            headers={"User-Agent": self.model_name},
            json=payload,
            timeout=timeout,
        )
        ret.raise_for_status()
        response = ret.json()
        if not isinstance(response, dict):
            return {"text": f"{tool_name} returned non-dict response: {response}", "error_code": 1}
        return response

    @classmethod
    def _estimate_internal_cost(cls, internal_tool_calls: List[Dict[str, Any]]) -> float:
        return float(sum(cls.INTERNAL_TOOL_WEIGHTS.get(str(item.get("tool")), 2) for item in internal_tool_calls or []))

    @staticmethod
    def _append_internal_tool_call(internal_tool_calls: List[Dict[str, Any]], tool_name: str, response: Dict[str, Any], elapsed: float, reason: str) -> None:
        internal_tool_calls.append({"tool": tool_name, "success": response.get("error_code") == 0, "error_code": response.get("error_code", 1), "elapsed": round(elapsed, 6), "reason": reason})
        nested_calls = response.get("internal_tool_calls") if isinstance(response, dict) else None
        if isinstance(nested_calls, list):
            for item in nested_calls:
                if isinstance(item, dict):
                    nested = dict(item)
                    nested.setdefault("parent_tool", tool_name)
                    internal_tool_calls.append(nested)

    @staticmethod
    def _to_bool(value: Any, default: bool = False) -> bool:
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
    def _literal_or_raw(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return value
        try:
            return ast.literal_eval(text)
        except Exception:
            return value

    @classmethod
    def _as_list(cls, value: Any) -> List[Dict[str, Any]]:
        value = cls._literal_or_raw(value)
        if value in [None, ""]:
            return []
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    @classmethod
    def _first_list(cls, params: Dict[str, Any], names: Sequence[str]) -> List[Dict[str, Any]]:
        for name in names:
            items = cls._as_list(params.get(name))
            if items:
                return items
        structured = params.get("structured")
        if isinstance(structured, dict):
            for name in names:
                items = cls._as_list(structured.get(name))
                if items:
                    return items
            evidence = structured.get("evidence")
            if isinstance(evidence, dict):
                for name in names:
                    items = cls._as_list(evidence.get(name))
                    if items:
                        return items
        return []

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _bbox_from_item(item: Dict[str, Any], *names: str) -> Optional[List[float]]:
        for name in names:
            bbox = item.get(name)
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    return [float(v) for v in bbox[:4]]
                except Exception:
                    return None
        return None

    @staticmethod
    def _label_from_item(item: Dict[str, Any], *names: str) -> str:
        for name in names:
            value = item.get(name)
            if value not in [None, ""]:
                return str(value)
        return "object"

    @classmethod
    def _normalize_detections(cls, items: List[Dict[str, Any]], modality: str) -> List[Dict[str, Any]]:
        normalized = []
        for item in items:
            if modality == "rgb":
                bbox = cls._bbox_from_item(item, "bbox", "rgb_bbox", "box")
                score = item.get("score", item.get("rgb_score", item.get("confidence")))
                label = cls._label_from_item(item, "label", "rgb_label", "class_name", "name")
            else:
                bbox = cls._bbox_from_item(item, "bbox", "sar_bbox", "box")
                score = item.get("score", item.get("sar_score", item.get("confidence")))
                label = cls._label_from_item(item, "label", "sar_label", "class_name", "name")
            if bbox is None:
                continue
            record = dict(item)
            record.update({"bbox": bbox, "score": cls._score(score), "label": label})
            normalized.append(record)
        return normalized

    @classmethod
    def _normalize_sar_only(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for idx, item in enumerate(items):
            bbox = cls._bbox_from_item(item, "sar_bbox", "bbox", "box")
            if bbox is None:
                continue
            normalized.append(
                {
                    **item,
                    "sar_index": item.get("sar_index", idx),
                    "sar_bbox": bbox,
                    "sar_score": cls._score(item.get("sar_score", item.get("score", item.get("confidence")))),
                    "sar_label": cls._label_from_item(item, "sar_label", "label", "class_name", "name"),
                }
            )
        return normalized

    @classmethod
    def _normalize_rgb_only(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for idx, item in enumerate(items):
            bbox = cls._bbox_from_item(item, "rgb_bbox", "bbox", "box")
            if bbox is None:
                continue
            normalized.append(
                {
                    **item,
                    "rgb_index": item.get("rgb_index", idx),
                    "rgb_bbox": bbox,
                    "rgb_score": cls._score(item.get("rgb_score", item.get("score", item.get("confidence")))),
                    "rgb_label": cls._label_from_item(item, "rgb_label", "label", "class_name", "name"),
                }
            )
        return normalized

    @staticmethod
    def _best_match(matched_pairs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not matched_pairs:
            return None
        return sorted(
            matched_pairs,
            key=lambda item: (
                float(item.get("match_score") or 0.0),
                float(item.get("iou") or 0.0),
                float(item.get("center_score") or 0.0),
                float(item.get("confidence_score") or 0.0),
            ),
            reverse=True,
        )[0]

    @staticmethod
    def _match_confidence(match: Optional[Dict[str, Any]]) -> float:
        if not match:
            return 0.0
        for key in ["match_score", "confidence_score", "iou", "score"]:
            if match.get(key) not in [None, ""]:
                return max(0.0, min(1.0, float(match.get(key))))
        rgb_score = match.get("rgb_score")
        sar_score = match.get("sar_score")
        if rgb_score is not None or sar_score is not None:
            return max(0.0, min(1.0, (float(rgb_score or 0.0) + float(sar_score or 0.0)) / 2.0))
        return 0.0

    @classmethod
    def _selection_candidates(cls, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        def add(source: str, item: Dict[str, Any], bbox_names: Sequence[str], score_names: Sequence[str]) -> None:
            bbox = cls._bbox_from_item(item, *bbox_names)
            if bbox is None:
                return
            score = 0.0
            for name in score_names:
                if item.get(name) not in [None, ""]:
                    score = cls._score(item.get(name))
                    break
            candidates.append({
                "source": source,
                "bbox": bbox,
                "area": max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]),
                "score": score,
                "label": cls._label_from_item(item, "rgb_label", "sar_label", "label", "class_name", "name"),
                "raw": item,
            })

        for item in context.get("matched_pairs", []):
            add("matched", item, ["rgb_bbox", "sar_bbox", "bbox"], ["match_score", "confidence_score", "rgb_score", "sar_score", "score"])
        for item in context.get("rgb_only", []):
            add("rgb_only", item, ["rgb_bbox", "bbox"], ["rgb_score", "score", "confidence"])
        for item in context.get("sar_only", []):
            add("sar_only", item, ["sar_bbox", "bbox"], ["sar_score", "score", "confidence"])
        for item in context.get("rgb_detections", []):
            add("rgb", item, ["rgb_bbox", "bbox"], ["rgb_score", "score", "confidence"])
        for item in context.get("sar_detections", []):
            add("sar", item, ["sar_bbox", "bbox"], ["sar_score", "score", "confidence"])
        return candidates

    @staticmethod
    def _center(bbox: Sequence[float]) -> Tuple[float, float]:
        return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)

    @staticmethod
    def _is_reasonable_match(match: Optional[Dict[str, Any]], threshold: float) -> bool:
        if not match:
            return False
        match_type = str(match.get("match_type") or "").lower()
        if "strong" in match_type:
            return True
        if float(match.get("iou") or 0.0) >= 0.2:
            return True
        if "center_supported" in match_type and float(match.get("center_score") or 0.0) >= 0.5:
            return True
        return float(match.get("match_score") or 0.0) >= max(0.25, threshold)

    @staticmethod
    def _labels_compatible(target: str, match: Optional[Dict[str, Any]]) -> bool:
        if not match:
            return False
        labels = [str(match.get("rgb_label") or ""), str(match.get("sar_label") or ""), str(target or "")]
        normalized = [" ".join(label.lower().replace("_", " ").split()) for label in labels]
        rgb_label, sar_label, target_label = normalized
        if not target_label:
            return True
        if target_label in rgb_label or target_label in sar_label:
            return True
        if rgb_label and sar_label and (rgb_label in sar_label or sar_label in rgb_label):
            return True
        return False

    def _match_context(self, params: Dict[str, Any], skill_trace: List[Dict[str, Any]], internal_tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        matched_pairs = self._first_list(params, ["matched_pairs", "matched"])
        rgb_detections = self._normalize_detections(self._first_list(params, ["rgb_detections", "rgb_evidence"]), "rgb")
        sar_detections = self._normalize_detections(self._first_list(params, ["sar_detections", "sar_candidates", "sar_evidence"]), "sar")
        rgb_only = self._normalize_rgb_only(self._first_list(params, ["rgb_only"]))
        sar_only = self._normalize_sar_only(self._first_list(params, ["sar_only"]))

        if matched_pairs or rgb_only or sar_only:
            return {
                "matched_pairs": matched_pairs,
                "rgb_detections": rgb_detections,
                "sar_detections": sar_detections,
                "rgb_only": rgb_only,
                "sar_only": sar_only,
                "source": "provided",
                "summary": {},
            }

        target = str(params.get("target") or "")
        if rgb_detections and sar_detections:
            payload = {
                "target": target,
                "rgb_image": params.get("rgb_image"),
                "sar_image": params.get("sar_image"),
                "rgb_detections": rgb_detections,
                "sar_candidates": sar_detections,
                "iou_threshold": params.get("iou_threshold", params.get("match_threshold", 0.2)),
                "score_threshold": params.get("confidence_threshold", 0.3),
                "match_strategy": params.get("match_strategy", "hybrid"),
                "visualize": False,
            }
            started = time.perf_counter()
            response = self._call_tool("CrossModalMatchSkill", payload, timeout=180)
            elapsed = time.perf_counter() - started
            self._append_internal_tool_call(internal_tool_calls, "CrossModalMatchSkill", response, elapsed, "match provided RGB and SAR detections")
            skill_trace.append({"tool": "CrossModalMatchSkill", "status": "success" if response.get("error_code") == 0 else "error", "text": response.get("text", "")})
            if response.get("error_code") != 0:
                raise RuntimeError(response.get("text") or "CrossModalMatchSkill failed")
            return {
                "matched_pairs": self._as_list(response.get("matched_pairs") or response.get("matched")),
                "rgb_detections": self._normalize_detections(self._as_list(response.get("rgb_detections")) or rgb_detections, "rgb"),
                "sar_detections": self._normalize_detections(self._as_list(response.get("sar_detections")) or sar_detections, "sar"),
                "rgb_only": self._normalize_rgb_only(self._as_list(response.get("rgb_only"))),
                "sar_only": self._normalize_sar_only(self._as_list(response.get("sar_only"))),
                "source": "CrossModalMatchSkill",
                "summary": response.get("match_summary", response.get("summary", {})),
            }

        if params.get("rgb_image") and params.get("sar_image"):
            payload = {
                "rgb_image": params.get("rgb_image"),
                "sar_image": params.get("sar_image"),
                "target": target,
                "score_threshold": params.get("confidence_threshold", 0.3),
                "iou_threshold": params.get("iou_threshold", params.get("match_threshold", 0.2)),
                "match_strategy": params.get("match_strategy", "hybrid"),
                "visualize": False,
                "max_draw": params.get("max_draw", 20),
            }
            started = time.perf_counter()
            response = self._call_tool("CrossModalMatchSkill", payload, timeout=300)
            elapsed = time.perf_counter() - started
            self._append_internal_tool_call(internal_tool_calls, "CrossModalMatchSkill", response, elapsed, "derive cross-modal evidence from images")
            skill_trace.append({"tool": "CrossModalMatchSkill", "status": "success" if response.get("error_code") == 0 else "error", "text": response.get("text", "")})
            if response.get("error_code") != 0:
                raise RuntimeError(response.get("text") or "CrossModalMatchSkill failed")
            return {
                "matched_pairs": self._as_list(response.get("matched_pairs") or response.get("matched")),
                "rgb_detections": self._normalize_detections(self._as_list(response.get("rgb_detections")), "rgb"),
                "sar_detections": self._normalize_detections(self._as_list(response.get("sar_detections")), "sar"),
                "rgb_only": self._normalize_rgb_only(self._as_list(response.get("rgb_only"))),
                "sar_only": self._normalize_sar_only(self._as_list(response.get("sar_only"))),
                "source": "CrossModalMatchSkill",
                "summary": response.get("match_summary", response.get("summary", {})),
            }

        return {
            "matched_pairs": [],
            "rgb_detections": rgb_detections,
            "sar_detections": sar_detections,
            "rgb_only": [],
            "sar_only": [],
            "source": "provided_partial",
            "summary": {},
        }

    @staticmethod
    def _evidence_summary(context: Dict[str, Any]) -> str:
        return (
            f"matched_pairs={len(context.get('matched_pairs', []))}, "
            f"rgb_evidence={len(context.get('rgb_detections', []))}, "
            f"sar_evidence={len(context.get('sar_detections', []))}, "
            f"rgb_only={len(context.get('rgb_only', []))}, "
            f"sar_only={len(context.get('sar_only', []))}."
        )

    def _build_evidence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        evidence = {
            "matched_pairs": context.get("matched_pairs", []),
            "rgb_evidence": context.get("rgb_detections", []),
            "sar_evidence": context.get("sar_detections", []),
            "rgb_only": context.get("rgb_only", []),
            "sar_only": context.get("sar_only", []),
            "evidence_summary": self._evidence_summary(context),
            "source": context.get("source"),
        }
        if context.get("summary"):
            evidence["match_summary"] = context.get("summary")
        return evidence

    def _decide_task(
        self,
        task_type: str,
        target: str,
        context: Dict[str, Any],
        confidence_threshold: float,
        evidence_required: bool,
    ) -> Tuple[Dict[str, Any], Optional[str], str]:
        matched_pairs = context.get("matched_pairs", [])
        rgb_evidence = context.get("rgb_detections", [])
        sar_evidence = context.get("sar_detections", [])
        sar_only = context.get("sar_only", [])
        rgb_only = context.get("rgb_only", [])
        best_match = self._best_match(matched_pairs)
        match_ok = self._is_reasonable_match(best_match, confidence_threshold)
        confidence = self._match_confidence(best_match)

        if evidence_required and not (matched_pairs or rgb_evidence or sar_evidence or sar_only or rgb_only):
            return {
                "label": self._negative_label(task_type),
                "confidence": 0.0,
                "answer": "No cross-modal evidence is available for a task-level positive conclusion.",
            }, "no cross-modal evidence available", "insufficient_evidence"

        if task_type == "optical_sar_verify":
            if match_ok:
                return {
                    "label": "supported",
                    "confidence": confidence,
                    "answer": f"The optical {target} is supported by SAR evidence via {best_match.get('match_type', 'matched_pair')}.",
                }, None, "success"
            if not sar_evidence and not sar_only and not matched_pairs:
                return {
                    "label": "unsupported",
                    "confidence": 0.7 if rgb_evidence else 0.0,
                    "answer": f"No SAR response or SAR candidate supports the optical {target}.",
                }, None if rgb_evidence else "missing RGB/SAR evidence", "success" if rgb_evidence else "insufficient_evidence"
            return {
                "label": "unsupported",
                "confidence": max(0.2, min(0.8, 1.0 - confidence)),
                "answer": f"SAR evidence exists, but no reasonable optical-SAR match supports the {target}.",
            }, None, "success"

        if task_type == "sar_optical_attribution":
            if match_ok:
                rgb_label = best_match.get("rgb_label") or target
                return {
                    "label": "attributed",
                    "confidence": confidence,
                    "answer": f"The SAR candidate corresponds to optical target '{rgb_label}' at RGB bbox {best_match.get('rgb_bbox')}.",
                }, None, "success"
            return {
                "label": "unattributed",
                "confidence": max(0.2, min(0.8, 1.0 - confidence)),
                "answer": "No matched optical region provides a reliable attribution for the SAR candidate.",
            }, None if sar_evidence or sar_only else "missing SAR candidate evidence", "success" if sar_evidence or sar_only else "insufficient_evidence"

        if task_type == "sar_candidate_recall":
            trusted_sar_only = [item for item in sar_only if self._score(item.get("sar_score", item.get("score"))) >= confidence_threshold]
            if trusted_sar_only:
                best = sorted(trusted_sar_only, key=lambda item: self._score(item.get("sar_score", item.get("score"))), reverse=True)[0]
                return {
                    "label": "recalled",
                    "confidence": max(0.0, min(1.0, self._score(best.get("sar_score", best.get("score"))))),
                    "answer": f"SAR provides a credible optical-missed candidate at SAR bbox {best.get('sar_bbox') or best.get('bbox')}.",
                }, None, "success"
            return {
                "label": "not_recalled",
                "confidence": 0.7 if sar_evidence or matched_pairs or rgb_evidence else 0.0,
                "answer": "No trusted SAR-only candidate is available as a recall target.",
            }, None if sar_evidence or matched_pairs or rgb_evidence else "missing SAR/RGB matching evidence", "success" if sar_evidence or matched_pairs or rgb_evidence else "insufficient_evidence"

        if task_type == "optical_sar_consistency":
            label_ok = self._labels_compatible(target, best_match)
            if match_ok and label_ok:
                return {
                    "label": "consistent",
                    "confidence": confidence,
                    "answer": f"Optical target and SAR response are spatially and semantically consistent for {target}.",
                }, None, "success"
            reason = "no reasonable match" if not match_ok else "matched labels are not compatible with the target"
            if rgb_only or sar_only:
                reason += f"; rgb_only={len(rgb_only)}, sar_only={len(sar_only)}"
            return {
                "label": "inconsistent",
                "confidence": max(0.2, min(0.85, 1.0 - confidence)),
                "answer": f"Optical target and SAR response are inconsistent: {reason}.",
            }, None if matched_pairs or rgb_evidence or sar_evidence or rgb_only or sar_only else "missing consistency evidence", "success" if matched_pairs or rgb_evidence or sar_evidence or rgb_only or sar_only else "insufficient_evidence"

        if task_type in {"cross_modal_select_largest", "cross_modal_select_highest_score"}:
            candidates = [
                item for item in self._selection_candidates(context)
                if item.get("source") not in {"rgb", "rgb_only"}
            ]
            if not candidates:
                mode = "largest-target" if task_type == "cross_modal_select_largest" else "highest-confidence"
                return {
                    "label": "not_selected",
                    "confidence": 0.0,
                    "answer": f"No valid {target} target is available for {mode} selection.",
                    "selection_mode": mode,
                }, "missing selectable cross-modal evidence", "insufficient_evidence"
            if task_type == "cross_modal_select_largest":
                selected = sorted(candidates, key=lambda item: (item["area"], item["score"]), reverse=True)[0]
                mode = "largest-target"
            else:
                selected = sorted(candidates, key=lambda item: (item["score"], item["area"]), reverse=True)[0]
                mode = "highest-confidence"
            bbox = [int(round(v)) for v in selected["bbox"]]
            return {
                "label": "selected",
                "confidence": max(0.0, min(1.0, float(selected["score"] or confidence))),
                "answer": f"The {mode} {target} target is selected from {selected['source']} and its final position is {bbox}.",
                "final_bbox": bbox,
                "selected_target": target,
                "selection_mode": mode,
                "source": selected["source"],
            }, None, "success"

        if task_type == "cross_modal_distance_largest_smallest":
            candidates = [
                item for item in self._selection_candidates(context)
                if item.get("source") not in {"rgb", "rgb_only"}
            ]
            if len(candidates) < 2:
                return {
                    "label": "not_measured",
                    "confidence": 0.0,
                    "answer": f"Insufficient valid {target} targets are available to measure largest-smallest distance.",
                }, "fewer than two selectable targets", "insufficient_evidence"
            ordered = sorted(candidates, key=lambda item: item["area"])
            smallest, largest = ordered[0], ordered[-1]
            sx, sy = self._center(smallest["bbox"])
            lx, ly = self._center(largest["bbox"])
            distance_px = math.hypot(lx - sx, ly - sy)
            return {
                "label": "measured",
                "confidence": max(float(smallest.get("score") or 0.0), float(largest.get("score") or 0.0)),
                "answer": f"The pixel distance between the largest and smallest {target} targets is {distance_px:.2f} px.",
                "distance_px": round(distance_px, 4),
                "largest_bbox": [int(round(v)) for v in largest["bbox"]],
                "smallest_bbox": [int(round(v)) for v in smallest["bbox"]],
            }, None, "success"

        raise ValueError(f"Unsupported task_type: {task_type}")

    @staticmethod
    def _negative_label(task_type: str) -> str:
        return {
            "optical_sar_verify": "unsupported",
            "sar_optical_attribution": "unattributed",
            "sar_candidate_recall": "not_recalled",
            "optical_sar_consistency": "inconsistent",
            "cross_modal_select_largest": "not_selected",
            "cross_modal_select_highest_score": "not_selected",
            "cross_modal_distance_largest_smallest": "not_measured",
        }.get(task_type, "unsupported")

    @staticmethod
    def _draw_box(draw: ImageDraw.ImageDraw, bbox: Sequence[float], color: Tuple[int, int, int], label: str, font: ImageFont.ImageFont):
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        for offset in range(2):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        label_y = max(0, y1 - 14)
        draw.rectangle((x1, label_y, x1 + max(42, len(label) * 6), label_y + 13), fill=color)
        draw.text((x1 + 2, label_y + 1), label[:40], fill=(255, 255, 255), font=font)

    def _draw_visualization(self, rgb_image: str, sar_image: str, context: Dict[str, Any], result_label: str, max_draw: int) -> Optional[str]:
        try:
            rgb = Image.open(rgb_image).convert("RGB")
            sar = Image.open(sar_image).convert("RGB")
            font = ImageFont.load_default()
            rgb_draw = ImageDraw.Draw(rgb)
            sar_draw = ImageDraw.Draw(sar)

            for idx, match in enumerate(context.get("matched_pairs", [])[:max_draw]):
                rgb_bbox = self._bbox_from_item(match, "rgb_bbox")
                sar_bbox = self._bbox_from_item(match, "sar_bbox")
                if rgb_bbox:
                    self._draw_box(rgb_draw, rgb_bbox, (0, 170, 0), f"match {idx}:{self._match_confidence(match):.2f}", font)
                if sar_bbox:
                    self._draw_box(sar_draw, sar_bbox, (0, 120, 255), f"match {idx}:{self._match_confidence(match):.2f}", font)

            for idx, item in enumerate(context.get("rgb_only", [])[:max_draw]):
                bbox = self._bbox_from_item(item, "rgb_bbox", "bbox")
                if bbox:
                    self._draw_box(rgb_draw, bbox, (255, 140, 0), f"rgb_only {idx}", font)
            for idx, item in enumerate(context.get("sar_only", [])[:max_draw]):
                bbox = self._bbox_from_item(item, "sar_bbox", "bbox")
                if bbox:
                    self._draw_box(sar_draw, bbox, (255, 0, 0), f"sar_only {idx}", font)

            header_h = 28
            footer_h = 32
            gap = 8
            canvas_w = rgb.width + sar.width + gap
            canvas_h = max(rgb.height, sar.height) + header_h + footer_h
            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            canvas.paste(rgb, (0, header_h))
            canvas.paste(sar, (rgb.width + gap, header_h))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 6), "Optical / RGB", fill=(0, 0, 0), font=font)
            draw.text((rgb.width + gap + 4, 6), "SAR", fill=(0, 0, 0), font=font)
            draw.text((4, header_h + max(rgb.height, sar.height) + 8), f"CrossModalTask: {result_label}", fill=(0, 0, 0), font=font)

            save_dir = Path(os.getcwd()) / "tools_output"
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"cross_modal_task_{uuid.uuid4().hex[:8]}.png"
            canvas.save(output_path)
            return str(output_path)
        except Exception as exc:
            logger.warning(f"Failed to draw cross-modal task visualization: {exc}")
            return None

    #: DS+VI 可能给出的"不确定"裁决。
    VI_WEAK_STATES = {
        "Weak Support", "Conflict", "Uncertain",
        "interpret_unavailable", "interpret_empty", "interpret_unmapped", "unsupported_target",
    }
    #: 明确的肯定裁决。
    VI_STRONG_STATES = {"Confirmed", "match", "single"}

    @classmethod
    def _vi_evidence_states(cls, evidence: Any) -> set:
        """收集匹配结果里出现的 VI 裁决词。"""
        if not isinstance(evidence, dict):
            return set()
        states = set()
        for key in ("matched_pairs", "rgb_only", "sar_only", "dedup_detections"):
            for item in evidence.get(key) or []:
                if isinstance(item, dict) and item.get("vi_status"):
                    states.add(str(item["vi_status"]))
        pairing = evidence.get("pairing") or {}
        reason = pairing.get("degrade_reason")
        if pairing.get("degraded") and reason:
            states.add(str(reason))
        return states

    @classmethod
    def _normalize_evidence_status(cls, status: str, failure_reason: Optional[str], states: set):
        """G4：unknown/abstain 一律降级为 insufficient_evidence，绝不升级。"""
        if status != "success" or not states:
            return status, failure_reason
        strong = {s for s in states if s in cls.VI_STRONG_STATES}
        if strong:
            return status, failure_reason
        weak = {
            s for s in states
            if s in cls.VI_WEAK_STATES
            or s.startswith(("interpret_", "shared_identity_failed", "observation_cap", "degraded"))
        }
        if not weak:
            return status, failure_reason
        return "insufficient_evidence", failure_reason or "vi_status_unknown"

    @classmethod
    def _response(cls, text: str, error_code: int, target: str, task_type: str, task_result: Dict[str, Any], evidence: Dict[str, Any], failure_reason: Optional[str], status: str, artifacts: Optional[List[Dict[str, Any]]] = None, skill_trace: Optional[List[Dict[str, Any]]] = None, internal_tool_calls: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        internal_calls = internal_tool_calls or []
        internal_cost = cls._estimate_internal_cost(internal_calls)
        vi_states = cls._vi_evidence_states(evidence)
        status, failure_reason = cls._normalize_evidence_status(status, failure_reason, vi_states)
        structured = {
            "target": target,
            "task_type": task_type,
            "task_result": task_result,
            "evidence": evidence,
            "failure_reason": failure_reason,
            "status": status,
            "vi_evidence": sorted(vi_states),
            "internal_tool_calls": internal_calls,
            "internal_cost": internal_cost,
        }
        return {
            "text": text,
            "error_code": error_code,
            "task_result": task_result,
            "evidence": evidence,
            "structured": structured,
            "artifacts": artifacts or [],
            "skill_trace": skill_trace or [],
            "internal_tool_calls": internal_calls,
            "internal_cost": internal_cost,
        }

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rgb_image = params.get("rgb_image") or params.get("optical_image")
        sar_image = params.get("sar_image") or params.get("radar_image")
        target = str(params.get("target") or "").strip()
        task_type = str(params.get("task_type") or "").strip()
        if not rgb_image:
            return self._response("Missing required parameter: rgb_image", 2, target, task_type, {"label": "unsupported", "confidence": 0.0, "answer": "Missing rgb_image."}, {}, "missing rgb_image", "missing_inputs")
        if not sar_image:
            return self._response("Missing required parameter: sar_image", 2, target, task_type, {"label": "unsupported", "confidence": 0.0, "answer": "Missing sar_image."}, {}, "missing sar_image", "missing_inputs")
        if not target:
            return self._response("Missing required parameter: target", 2, "", task_type, {"label": "unsupported", "confidence": 0.0, "answer": "Missing target."}, {}, "missing target", "missing_inputs")
        if not task_type:
            return self._response("Missing required parameter: task_type", 2, target, "", {"label": "unsupported", "confidence": 0.0, "answer": "Missing task_type."}, {}, "missing task_type", "missing_inputs")
        if task_type not in self.VALID_TASKS:
            return self._response(f"Unsupported task_type: {task_type}", 2, target, task_type, {"label": "unsupported", "confidence": 0.0, "answer": f"Unsupported task_type: {task_type}"}, {}, "unsupported task_type", "missing_inputs")
        if not os.path.isfile(str(rgb_image)):
            return self._response(f"RGB image not found: {rgb_image}", 3, target, task_type, {"label": self._negative_label(task_type), "confidence": 0.0, "answer": "RGB image path is invalid."}, {}, "rgb image not found", "missing_inputs")
        if not os.path.isfile(str(sar_image)):
            return self._response(f"SAR image not found: {sar_image}", 3, target, task_type, {"label": self._negative_label(task_type), "confidence": 0.0, "answer": "SAR image path is invalid."}, {}, "sar image not found", "missing_inputs")

        try:
            confidence_threshold = float(params.get("confidence_threshold", 0.3))
            evidence_required = self._to_bool(params.get("evidence_required"), True)
            visualize = self._to_bool(params.get("visualize"), True)
            max_draw = max(0, int(params.get("max_draw", 20)))
        except Exception as exc:
            return self._response(f"Invalid parameter value: {exc}", 2, target, task_type, {"label": self._negative_label(task_type), "confidence": 0.0, "answer": str(exc)}, {}, "invalid parameter", "missing_inputs")

        skill_trace: List[Dict[str, Any]] = []
        internal_tool_calls: List[Dict[str, Any]] = []
        try:
            context = self._match_context(params, skill_trace, internal_tool_calls)
            task_result, failure_reason, status = self._decide_task(
                task_type=task_type,
                target=target,
                context=context,
                confidence_threshold=confidence_threshold,
                evidence_required=evidence_required,
            )
            evidence = self._build_evidence(context)
            artifacts: List[Dict[str, Any]] = []
            if visualize and status != "missing_inputs":
                visual_path = self._draw_visualization(str(rgb_image), str(sar_image), context, task_result["label"], max_draw)
                if visual_path:
                    artifacts.append({"id": "cross_modal_task_visualization", "type": "image", "path": visual_path, "name": os.path.basename(visual_path), "source": self.model_name})

            text = f"{task_type}: {task_result['label']} for '{target}'. {task_result['answer']} Evidence: {evidence['evidence_summary']}"
            return self._response(text, 0, target, task_type, task_result, evidence, failure_reason, status, artifacts, skill_trace, internal_tool_calls)
        except Exception as exc:
            logger.exception(f"Error in {self.model_name}: {exc}")
            task_result = {"label": self._negative_label(task_type), "confidence": 0.0, "answer": f"Task failed: {exc}"}
            return self._response(f"Error in {self.model_name}: {exc}", 1, target, task_type, task_result, {}, str(exc), "insufficient_evidence", [], skill_trace, internal_tool_calls)

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Task-level optical-SAR reasoning over matched cross-modal evidence.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rgb_image": {"type": "string", "description": "Path to the optical/RGB image."},
                        "sar_image": {"type": "string", "description": "Path to the SAR image."},
                        "target": {"type": "string", "description": "Target category or description."},
                        "task_type": {"type": "string", "description": "optical_sar_verify, sar_optical_attribution, sar_candidate_recall, optical_sar_consistency, cross_modal_select_largest, cross_modal_select_highest_score, or cross_modal_distance_largest_smallest."},
                        "matched_pairs": {"type": "array", "description": "Optional precomputed optical-SAR matched pairs."},
                        "rgb_detections": {"type": "array", "description": "Optional RGB detections."},
                        "sar_candidates": {"type": "array", "description": "Optional SAR candidates/evidence."},
                        "sar_detections": {"type": "array", "description": "Alias for optional SAR candidates/evidence."},
                        "rgb_only": {"type": "array", "description": "Optional RGB-only detections from previous matching."},
                        "sar_only": {"type": "array", "description": "Optional SAR-only candidates from previous matching."},
                        "question": {"type": "string", "description": "Original task question."},
                        "evidence_required": {"type": "boolean", "description": "Require evidence before positive labels."},
                        "confidence_threshold": {"type": "number", "description": "Confidence threshold for weak matches and recall candidates."},
                        "match_threshold": {"type": "number", "description": "Alias for IoU/match threshold passed to matching."},
                        "iou_threshold": {"type": "number", "description": "IoU threshold passed to matching."},
                        "visualize": {"type": "boolean", "description": "Whether to output an evidence visualization."},
                        "max_draw": {"type": "integer", "description": "Maximum boxes to draw in the evidence visualization."},
                    },
                    "required": ["rgb_image", "sar_image", "target", "task_type"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20131)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="CrossModalTaskSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = CrossModalTaskSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
