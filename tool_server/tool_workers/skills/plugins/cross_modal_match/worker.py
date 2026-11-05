"""Target-level optical-SAR matching skill."""
import argparse
import ast
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"CrossModalMatchSkill_worker_{worker_id}.log")


class CrossModalMatchSkillWorker(BaseToolWorker):
    VALID_STRATEGIES = {"iou", "center_distance", "hybrid", "global"}
    # Shared-identity delegation (plan D-3 / M5).
    SHARED_TARGETS = {"vehicle", "tank", "ship", "aircraft", "bridge", "harbor"}
    SHARED_TARGET_ALIASES = {
        "vehicles": "vehicle", "car": "vehicle", "cars": "vehicle",
        "small vehicle": "vehicle", "large vehicle": "vehicle",
        "storage tank": "tank", "oil tank": "tank", "oil storage tank": "tank",
        "ships": "ship", "boat": "ship", "boats": "ship",
        "airplane": "aircraft", "airplanes": "aircraft", "plane": "aircraft",
        "planes": "aircraft", "helicopter": "aircraft", "helicopters": "aircraft",
    }
    #: CrossModalInterpret 单次预算，避免技能链超时叠加。
    INTERPRET_TIMEOUT = 120
    INTERNAL_TOOL_WEIGHTS = {
        "ObjectDetectSelectSkill": 3,
        "SAREvidenceExtractSkill": 4,
    }

    @classmethod
    def _estimate_internal_cost(cls, internal_tool_calls: List[Dict[str, Any]]) -> float:
        return float(sum(cls.INTERNAL_TOOL_WEIGHTS.get(str(item.get("tool")), 2) for item in internal_tool_calls or []))

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "CrossModalMatchSkill",
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
        ret = requests.post(self.controller_addr + "/get_worker_address", json={"model": tool_name}, timeout=5)
        ret.raise_for_status()
        addr = (ret.json().get("address") or "").strip()
        if not addr:
            raise RuntimeError(f"No worker address returned for tool '{tool_name}'")
        self.tool_addr_cache[tool_name] = addr
        return addr

    def _call_tool(self, tool_name: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
        addr = self._resolve_tool_addr(tool_name)
        ret = requests.post(addr + "/worker_generate", headers={"User-Agent": self.model_name}, json=payload, timeout=timeout)
        ret.raise_for_status()
        response = ret.json()
        if not isinstance(response, dict):
            return {"text": f"{tool_name} returned non-dict response: {response}", "error_code": 1}
        return response

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

    @staticmethod
    def _score(value: Any, default: float = 1.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _bbox(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in value[:4]]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @classmethod
    def _normalize_detections(cls, items: List[Dict[str, Any]], modality: str, score_threshold: float) -> List[Dict[str, Any]]:
        records = []
        for idx, item in enumerate(items):
            if modality == "rgb":
                bbox = cls._bbox(item.get("bbox") or item.get("rgb_bbox") or item.get("box"))
                score = cls._score(item.get("score", item.get("rgb_score", item.get("confidence"))), 1.0)
                label = item.get("label") or item.get("rgb_label") or item.get("class_name") or item.get("name") or "object"
            else:
                bbox = cls._bbox(item.get("bbox") or item.get("sar_bbox") or item.get("box"))
                score = cls._score(item.get("score", item.get("sar_score", item.get("confidence"))), 1.0)
                label = item.get("label") or item.get("sar_label") or item.get("class_name") or item.get("name") or "object"
            if bbox is None or score < score_threshold:
                continue
            records.append({**item, "bbox": bbox, "score": score, "label": str(label), "source_index": item.get("index", idx)})
        records.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        return records

    @staticmethod
    def _area(bbox: Sequence[float]) -> float:
        return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))

    @staticmethod
    def _center(bbox: Sequence[float]) -> Tuple[float, float]:
        return (float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0

    @classmethod
    def _diag(cls, bbox: Sequence[float]) -> float:
        return math.hypot(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))

    @classmethod
    def _iou(cls, a: Sequence[float], b: Sequence[float]) -> float:
        x1 = max(float(a[0]), float(b[0]))
        y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2]))
        y2 = min(float(a[3]), float(b[3]))
        inter = cls._area([x1, y1, x2, y2]) if x2 > x1 and y2 > y1 else 0.0
        union = cls._area(a) + cls._area(b) - inter
        return 0.0 if union <= 0 else inter / union

    @classmethod
    def _center_distance(cls, a: Sequence[float], b: Sequence[float]) -> float:
        ax, ay = cls._center(a)
        bx, by = cls._center(b)
        return math.hypot(ax - bx, ay - by)

    @classmethod
    def _area_ratio(cls, a: Sequence[float], b: Sequence[float]) -> float:
        area_a = cls._area(a)
        area_b = cls._area(b)
        if area_a <= 0 or area_b <= 0:
            return 0.0
        return min(area_a, area_b) / max(area_a, area_b)

    @classmethod
    def _auto_center_threshold(cls, rgb_bbox: Sequence[float], sar_bbox: Sequence[float]) -> float:
        return max(cls._diag(rgb_bbox), cls._diag(sar_bbox), 1.0) * 0.75

    @classmethod
    def _pair_info(
        cls,
        rgb_idx: int,
        sar_idx: int,
        rgb: Dict[str, Any],
        sar: Dict[str, Any],
        iou_threshold: float,
        center_distance_threshold: Any,
        strategy: str,
    ) -> Dict[str, Any]:
        rgb_bbox = rgb["bbox"]
        sar_bbox = sar["bbox"]
        iou = cls._iou(rgb_bbox, sar_bbox)
        center_distance = cls._center_distance(rgb_bbox, sar_bbox)
        center_threshold = cls._auto_center_threshold(rgb_bbox, sar_bbox) if center_distance_threshold == "auto" else float(center_distance_threshold)
        center_score = max(0.0, 1.0 - center_distance / max(center_threshold, 1.0))
        area_ratio = cls._area_ratio(rgb_bbox, sar_bbox)
        confidence_score = (float(rgb.get("score") or 0.0) + float(sar.get("score") or 0.0)) / 2.0
        hybrid_score = 0.45 * iou + 0.30 * center_score + 0.15 * area_ratio + 0.10 * confidence_score
        if strategy == "iou":
            match_score = iou
        elif strategy == "center_distance":
            match_score = center_score
        else:
            match_score = hybrid_score
        if iou >= iou_threshold:
            match_type = "strong_match"
            match_reason = f"IoU {iou:.3f} >= threshold {iou_threshold:.3f}."
        else:
            match_type = "weak_match"
            match_reason = f"IoU {iou:.3f} < threshold {iou_threshold:.3f}, center distance {center_distance:.2f}px <= {center_threshold:.2f}px."
        return {
            "rgb_index": rgb_idx,
            "sar_index": sar_idx,
            "rgb_bbox": rgb_bbox,
            "sar_bbox": sar_bbox,
            "rgb_score": float(rgb.get("score") or 0.0),
            "sar_score": float(sar.get("score") or 0.0),
            "rgb_label": rgb.get("label"),
            "sar_label": sar.get("label"),
            "iou": round(iou, 4),
            "center_distance_px": round(center_distance, 3),
            "area_ratio": round(area_ratio, 4),
            "center_score": round(center_score, 4),
            "confidence_score": round(confidence_score, 4),
            "match_score": round(match_score, 4),
            "match_type": match_type,
            "match_reason": match_reason,
        }

    @staticmethod
    def _passes_strategy(pair: Dict[str, Any], strategy: str, iou_threshold: float, score_threshold: float) -> bool:
        if strategy == "iou":
            return float(pair["iou"]) >= iou_threshold
        if strategy == "center_distance":
            return float(pair["center_score"]) > 0.0
        if float(pair["iou"]) >= iou_threshold:
            return True
        return float(pair["match_score"]) >= score_threshold and float(pair["center_score"]) > 0.0

    def _canonical_shared_target(self, target: Any) -> str:
        text = re.sub(r"[^a-z0-9]+", " ", str(target or "").strip().lower())
        text = re.sub(r"\s+", " ", text).strip()
        return self.SHARED_TARGET_ALIASES.get(text, text)

    @staticmethod
    def _index_of(detections: Sequence[Dict[str, Any]], bbox: Any) -> Optional[int]:
        if not bbox or len(bbox) < 4:
            return None
        wanted = [round(float(v), 3) for v in bbox[:4]]
        for index, det in enumerate(detections):
            candidate = det.get("bbox") or []
            if len(candidate) < 4:
                continue
            if [round(float(v), 3) for v in candidate[:4]] == wanted:
                return index
        return None

    def _shared_identity_match(
        self,
        params: Dict[str, Any],
        target: str,
        rgb_detections: List[Dict[str, Any]],
        sar_candidates: List[Dict[str, Any]],
        iou_threshold: float,
        center_distance_threshold: Any,
        match_strategy: str,
        skill_trace: List[Dict[str, Any]],
        internal_tool_calls: List[Dict[str, Any]],
    ):
        """把配对委托给 CrossModalInterpret(mode=shared_identity)。

        返回 ``(matched_pairs, rgb_only, sar_only, meta)``；返回 ``None`` 表示调用方
        必须退回本地贪心，原因写在 ``self._shared_identity_reason``。
        """
        self._shared_identity_reason = None
        canonical = self._canonical_shared_target(target)
        if canonical not in self.SHARED_TARGETS:
            # G2: interpret 只认 6 个 canonical target，其他类别直接本地贪心。
            self._shared_identity_reason = "unsupported_target"
            return None

        payload = {
            "target": canonical,
            "task": "crosscheck_exists",
            "mode": "shared_identity",
            "output_mode": "full",
            "rgb_detections": rgb_detections,
            "sar_detections": sar_candidates,
            "rgb_image": params.get("rgb_image") or params.get("optical_image"),
            "sar_image": params.get("sar_image") or params.get("radar_image"),
        }
        started = time.perf_counter()
        response = self._call_tool("CrossModalInterpret", payload, timeout=self.INTERPRET_TIMEOUT)
        elapsed = time.perf_counter() - started
        ok = isinstance(response, dict) and response.get("error_code") == 0
        internal_tool_calls.append({
            "tool": "CrossModalInterpret",
            "success": bool(ok),
            "error_code": (response or {}).get("error_code", 1),
            "elapsed": round(elapsed, 6),
            "reason": "shared-identity pairing (D-S -> evidence -> cascade)",
        })
        skill_trace.append({
            "tool": "CrossModalInterpret",
            "status": "success" if ok else "error",
            "text": (response or {}).get("text", ""),
        })
        if not ok:
            self._shared_identity_reason = "interpret_unavailable"
            return None

        structured = response.get("structured") or {}
        dedup = response.get("dedup_detections") or structured.get("dedup_detections") or []
        if not dedup:
            self._shared_identity_reason = "interpret_empty"
            return None

        matched_pairs: List[Dict[str, Any]] = []
        rgb_only: List[Dict[str, Any]] = []
        sar_only: List[Dict[str, Any]] = []
        for item in dedup:
            source = str(item.get("source") or "")
            if source == "matched":
                rgb_index = self._index_of(rgb_detections, item.get("rgb_bbox"))
                sar_index = self._index_of(sar_candidates, item.get("sar_bbox"))
                if rgb_index is None or sar_index is None:
                    continue
                pair = self._pair_info(
                    rgb_index, sar_index,
                    rgb_detections[rgb_index], sar_candidates[sar_index],
                    iou_threshold, center_distance_threshold, match_strategy,
                )
                # G3: 下游阈值读的是 match_score，这里必须是 VI 的 blend 分；
                # 几何分另存，供重新标定阈值时使用。
                pair["geometry_match_score"] = pair["match_score"]
                pair["match_score"] = round(float(item.get("score") or 0.0), 4)
                pair["vi_status"] = item.get("vi_status")
                pair["vi_p_class"] = item.get("vi_p_class")
                pair["vi_association"] = item.get("vi_association")
                pair["match_type"] = "shared_identity_match"
                pair["match_reason"] = "shared-identity cascade (VI blend); " + str(pair.get("match_reason") or "")
                matched_pairs.append(pair)
            elif source == "rgb_only":
                rgb_index = self._index_of(rgb_detections, item.get("rgb_bbox"))
                if rgb_index is None:
                    continue
                det = rgb_detections[rgb_index]
                rgb_only.append({
                    "rgb_index": rgb_index, "rgb_bbox": det["bbox"],
                    "rgb_score": det.get("score"), "rgb_label": det.get("label"),
                    "vi_status": item.get("vi_status"),
                })
            elif source == "sar_only":
                sar_index = self._index_of(sar_candidates, item.get("sar_bbox"))
                if sar_index is None:
                    continue
                det = sar_candidates[sar_index]
                sar_only.append({
                    "sar_index": sar_index, "sar_bbox": det["bbox"],
                    "sar_score": det.get("score"), "sar_label": det.get("label"),
                    "vi_status": item.get("vi_status"),
                })

        if not matched_pairs and not rgb_only and not sar_only:
            self._shared_identity_reason = "interpret_unmapped"
            return None

        summary = response.get("summary") or {}
        dsvi = summary.get("dsvi") or {}
        meta = {
            "mode": str(response.get("mode") or "shared_identity"),
            "degraded": bool(response.get("degraded")),
            "degrade_reason": response.get("degrade_reason"),
            "paired_by": "CrossModalInterpret",
            "blend_weight": dsvi.get("blend_weight"),
            "accept_betp": dsvi.get("accept_betp"),
            "elapsed": round(elapsed, 3),
        }
        return matched_pairs, rgb_only, sar_only, meta

    def _match(
        self,
        rgb_detections: List[Dict[str, Any]],
        sar_candidates: List[Dict[str, Any]],
        iou_threshold: float,
        center_distance_threshold: Any,
        score_threshold: float,
        match_strategy: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        pair_candidates = []
        for rgb_idx, rgb in enumerate(rgb_detections):
            for sar_idx, sar in enumerate(sar_candidates):
                pair = self._pair_info(rgb_idx, sar_idx, rgb, sar, iou_threshold, center_distance_threshold, match_strategy)
                if self._passes_strategy(pair, match_strategy, iou_threshold, score_threshold):
                    pair_candidates.append(pair)
        pair_candidates.sort(
            key=lambda item: (
                float(item.get("match_score") or 0.0),
                float(item.get("iou") or 0.0),
                -float(item.get("center_distance_px") or 0.0),
            ),
            reverse=True,
        )
        used_rgb, used_sar, matched_pairs = set(), set(), []
        for pair in pair_candidates:
            if pair["rgb_index"] in used_rgb or pair["sar_index"] in used_sar:
                continue
            used_rgb.add(pair["rgb_index"])
            used_sar.add(pair["sar_index"])
            matched_pairs.append(pair)

        rgb_only = []
        for idx, item in enumerate(rgb_detections):
            if idx not in used_rgb:
                rgb_only.append({"rgb_index": idx, "rgb_bbox": item["bbox"], "rgb_score": item.get("score"), "rgb_label": item.get("label")})
        sar_only = []
        for idx, item in enumerate(sar_candidates):
            if idx not in used_sar:
                sar_only.append({"sar_index": idx, "sar_bbox": item["bbox"], "sar_score": item.get("score"), "sar_label": item.get("label")})
        return matched_pairs, rgb_only, sar_only

    @classmethod
    def _dedup(cls, target: str, matched_pairs: List[Dict[str, Any]], rgb_only: List[Dict[str, Any]], sar_only: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dedup = []
        for idx, item in enumerate(matched_pairs):
            fused = [round((float(a) + float(b)) / 2.0, 3) for a, b in zip(item["rgb_bbox"], item["sar_bbox"])]
            dedup.append(
                {
                    "name": f"{target}_matched_{idx}",
                    "source": "matched",
                    "label": item.get("rgb_label") or item.get("sar_label") or target,
                    "bbox": fused,
                    "rgb_bbox": item.get("rgb_bbox"),
                    "sar_bbox": item.get("sar_bbox"),
                    "score": max(float(item.get("rgb_score") or 0.0), float(item.get("sar_score") or 0.0)),
                    "match_score": item.get("match_score"),
                    "match_type": item.get("match_type"),
                }
            )
        for idx, item in enumerate(rgb_only):
            dedup.append({"name": f"{target}_rgb_only_{idx}", "source": "rgb_only", "label": item.get("rgb_label") or target, "bbox": item.get("rgb_bbox"), "rgb_bbox": item.get("rgb_bbox"), "sar_bbox": None, "score": float(item.get("rgb_score") or 0.0)})
        for idx, item in enumerate(sar_only):
            dedup.append({"name": f"{target}_sar_only_{idx}", "source": "sar_only", "label": item.get("sar_label") or target, "bbox": item.get("sar_bbox"), "rgb_bbox": None, "sar_bbox": item.get("sar_bbox"), "score": float(item.get("sar_score") or 0.0)})
        return dedup

    def _fallback_rgb(self, image: str, target: str, score_threshold: float, top1: bool, skill_trace: List[Dict[str, Any]], internal_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload = {"image": image, "target": target, "selection_mode": "all", "confidence_threshold": score_threshold, "visualize": False, "top_k": 1 if top1 else 100}
        started = time.perf_counter()
        response = self._call_tool("ObjectDetectSelectSkill", payload, timeout=240)
        elapsed = time.perf_counter() - started
        internal_tool_calls.append({"tool": "ObjectDetectSelectSkill", "success": response.get("error_code") == 0, "error_code": response.get("error_code", 1), "elapsed": round(elapsed, 6), "reason": "fallback RGB detection because rgb_detections were not provided"})
        skill_trace.append({"tool": "ObjectDetectSelectSkill", "status": "success" if response.get("error_code") == 0 else "error", "text": response.get("text", "")})
        if response.get("error_code") != 0:
            raise RuntimeError(response.get("text") or "ObjectDetectSelectSkill failed")
        return self._as_list(response.get("detections") or response.get("selected_detections"))

    def _fallback_sar(self, image: str, target: str, score_threshold: float, top1: bool, skill_trace: List[Dict[str, Any]], internal_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload = {"sar_image": image, "target": target, "confidence_threshold": score_threshold, "visualize": False, "max_candidates": 1 if top1 else 100}
        started = time.perf_counter()
        response = self._call_tool("SAREvidenceExtractSkill", payload, timeout=300)
        elapsed = time.perf_counter() - started
        internal_tool_calls.append({"tool": "SAREvidenceExtractSkill", "success": response.get("error_code") == 0, "error_code": response.get("error_code", 1), "elapsed": round(elapsed, 6), "reason": "fallback SAR evidence extraction because sar_candidates were not provided"})
        skill_trace.append({"tool": "SAREvidenceExtractSkill", "status": "success" if response.get("error_code") == 0 else "error", "text": response.get("text", "")})
        if response.get("error_code") != 0:
            raise RuntimeError(response.get("text") or "SAREvidenceExtractSkill failed")
        return self._as_list(response.get("sar_candidates"))

    def _draw_visualization(self, rgb_image: str, sar_image: str, matched_pairs: List[Dict[str, Any]], rgb_only: List[Dict[str, Any]], sar_only: List[Dict[str, Any]], max_draw: int) -> Optional[str]:
        try:
            rgb = Image.open(rgb_image).convert("RGB")
            sar = Image.open(sar_image).convert("RGB")
            font = ImageFont.load_default()
            rgb_draw = ImageDraw.Draw(rgb)
            sar_draw = ImageDraw.Draw(sar)
            for idx, item in enumerate(matched_pairs[:max_draw]):
                self._draw_box(rgb_draw, item["rgb_bbox"], (0, 170, 0), f"M{idx}:{item['match_score']:.2f}", font)
                self._draw_box(sar_draw, item["sar_bbox"], (0, 120, 255), f"M{idx}:{item['match_score']:.2f}", font)
            for idx, item in enumerate(rgb_only[:max_draw]):
                self._draw_box(rgb_draw, item["rgb_bbox"], (255, 140, 0), f"RGB{idx}", font)
            for idx, item in enumerate(sar_only[:max_draw]):
                self._draw_box(sar_draw, item["sar_bbox"], (255, 0, 0), f"SAR{idx}", font)

            header_h, footer_h, gap = 28, 32, 8
            canvas_w = rgb.width + sar.width + gap
            canvas_h = max(rgb.height, sar.height) + header_h + footer_h
            canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
            canvas.paste(rgb, (0, header_h))
            canvas.paste(sar, (rgb.width + gap, header_h))
            draw = ImageDraw.Draw(canvas)
            draw.text((4, 6), "Optical / RGB", fill=(0, 0, 0), font=font)
            draw.text((rgb.width + gap + 4, 6), "SAR", fill=(0, 0, 0), font=font)
            draw.text((4, header_h + max(rgb.height, sar.height) + 8), f"matches={len(matched_pairs)}, rgb_only={len(rgb_only)}, sar_only={len(sar_only)}", fill=(0, 0, 0), font=font)
            save_dir = Path(os.getcwd()) / "tools_output"
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"cross_modal_match_{uuid.uuid4().hex[:8]}.png"
            canvas.save(output_path)
            return str(output_path)
        except Exception as exc:
            logger.warning(f"Failed to draw cross-modal match visualization: {exc}")
            return None

    @staticmethod
    def _draw_box(draw: ImageDraw.ImageDraw, bbox: Sequence[float], color: Tuple[int, int, int], label: str, font: ImageFont.ImageFont):
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        for offset in range(2):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        label_y = max(0, y1 - 14)
        draw.rectangle((x1, label_y, x1 + max(36, len(label) * 6), label_y + 13), fill=color)
        draw.text((x1 + 2, label_y + 1), label[:32], fill=(255, 255, 255), font=font)

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rgb_image = params.get("rgb_image") or params.get("optical_image")
        sar_image = params.get("sar_image") or params.get("radar_image")
        target = str(params.get("target") or "").strip()
        if not rgb_image:
            return self._missing_response(target, params.get("match_strategy", "hybrid"), "Missing required parameter: rgb_image")
        if not sar_image:
            return self._missing_response(target, params.get("match_strategy", "hybrid"), "Missing required parameter: sar_image")
        if not target:
            return self._missing_response("", params.get("match_strategy", "hybrid"), "Missing required parameter: target")
        if not os.path.isfile(str(rgb_image)):
            return self._missing_response(target, params.get("match_strategy", "hybrid"), f"RGB image not found: {rgb_image}", error_code=3)
        if not os.path.isfile(str(sar_image)):
            return self._missing_response(target, params.get("match_strategy", "hybrid"), f"SAR image not found: {sar_image}", error_code=3)

        try:
            iou_threshold = float(params.get("iou_threshold", 0.2))
            score_threshold = float(params.get("score_threshold", params.get("confidence_threshold", 0.3)))
            center_distance_threshold = params.get("center_distance_threshold", "auto")
            if not (isinstance(center_distance_threshold, str) and center_distance_threshold.strip().lower() == "auto"):
                center_distance_threshold = float(center_distance_threshold)
            match_strategy = str(params.get("match_strategy", "hybrid")).strip().lower().replace("-", "_")
            if match_strategy not in self.VALID_STRATEGIES:
                match_strategy = "hybrid"
            visualize = self._to_bool(params.get("visualize"), True)
            top1 = self._to_bool(params.get("top1"), False)
            max_draw = max(0, int(params.get("max_draw", 20)))
        except Exception as exc:
            return self._missing_response(target, params.get("match_strategy", "hybrid"), f"Invalid parameter value: {exc}", error_code=2)

        skill_trace: List[Dict[str, Any]] = []
        internal_tool_calls: List[Dict[str, Any]] = []
        try:
            has_rgb_input = any(key in params for key in ["rgb_detections", "detections", "rgb_boxes"])
            has_sar_input = any(key in params for key in ["sar_candidates", "sar_detections", "sar_boxes"])
            raw_rgb = self._as_list(params.get("rgb_detections") or params.get("detections") or params.get("rgb_boxes"))
            raw_sar = self._as_list(params.get("sar_candidates") or params.get("sar_detections") or params.get("sar_boxes"))
            if not raw_rgb and not has_rgb_input:
                raw_rgb = self._fallback_rgb(str(rgb_image), target, score_threshold, top1, skill_trace, internal_tool_calls)
            if not raw_sar and not has_sar_input:
                raw_sar = self._fallback_sar(str(sar_image), target, score_threshold, top1, skill_trace, internal_tool_calls)

            rgb_detections = self._normalize_detections(raw_rgb, "rgb", score_threshold)
            sar_candidates = self._normalize_detections(raw_sar, "sar", score_threshold)
            if top1:
                rgb_detections = rgb_detections[:1]
                sar_candidates = sar_candidates[:1]

            shared = self._shared_identity_match(
                params, target, rgb_detections, sar_candidates,
                iou_threshold, center_distance_threshold, match_strategy,
                skill_trace, internal_tool_calls,
            )
            if shared is not None:
                matched_pairs, rgb_only, sar_only, match_meta = shared
            else:
                matched_pairs, rgb_only, sar_only = self._match(
                    rgb_detections, sar_candidates, iou_threshold,
                    center_distance_threshold, score_threshold, match_strategy)
                match_meta = {
                    "mode": "legacy_local",
                    "degraded": True,
                    "degrade_reason": getattr(self, "_shared_identity_reason", None) or "shared_identity_unavailable",
                    "paired_by": "local_greedy",
                }
            dedup_detections = self._dedup(target, matched_pairs, rgb_only, sar_only)
            match_summary = {
                "rgb_count": len(rgb_detections),
                "sar_count": len(sar_candidates),
                "matched_count": len(matched_pairs),
                "rgb_only_count": len(rgb_only),
                "sar_only_count": len(sar_only),
                "dedup_count": len(dedup_detections),
            }
            status = "success" if matched_pairs else "no_match"

            artifacts: List[Dict[str, Any]] = []
            if visualize:
                visual_path = self._draw_visualization(str(rgb_image), str(sar_image), matched_pairs, rgb_only, sar_only, max_draw)
                if visual_path:
                    artifacts.append({"id": "cross_modal_match_visualization", "type": "image", "path": visual_path, "name": os.path.basename(visual_path), "source": self.model_name})

            structured = {
                "target": target,
                "match_strategy": match_strategy,
                "matched_pairs": matched_pairs,
                "rgb_only": rgb_only,
                "sar_only": sar_only,
                "dedup_detections": dedup_detections,
                "match_summary": match_summary,
                "status": status,
                "mode": match_meta.get("mode"),
                "degraded": match_meta.get("degraded"),
                "degrade_reason": match_meta.get("degrade_reason"),
                "pairing": match_meta,
                "internal_tool_calls": internal_tool_calls,
                "internal_cost": self._estimate_internal_cost(internal_tool_calls),
            }
            return {
                "text": f"CrossModalMatch: matched={len(matched_pairs)}, rgb_only={len(rgb_only)}, sar_only={len(sar_only)}, dedup_count={len(dedup_detections)}.",
                "error_code": 0,
                "match_summary": match_summary,
                "matched_pairs": matched_pairs,
                "rgb_only": rgb_only,
                "sar_only": sar_only,
                "dedup_detections": dedup_detections,
                "structured": structured,
                "artifacts": artifacts,
                "skill_trace": skill_trace,
                "internal_tool_calls": internal_tool_calls,
                "internal_cost": self._estimate_internal_cost(internal_tool_calls),
            }
        except Exception as exc:
            logger.exception(f"Error in {self.model_name}: {exc}")
            internal_cost = self._estimate_internal_cost(internal_tool_calls)
            return {"text": f"Error in {self.model_name}: {exc}", "error_code": 1, "match_summary": {}, "matched_pairs": [], "rgb_only": [], "sar_only": [], "dedup_detections": [], "structured": {"target": target, "match_strategy": match_strategy, "status": "missing_inputs", "internal_tool_calls": internal_tool_calls, "internal_cost": internal_cost}, "artifacts": [], "skill_trace": skill_trace, "internal_tool_calls": internal_tool_calls, "internal_cost": internal_cost}

    @staticmethod
    def _missing_response(target: str, match_strategy: Any, text: str, error_code: int = 2) -> Dict[str, Any]:
        summary = {"rgb_count": 0, "sar_count": 0, "matched_count": 0, "rgb_only_count": 0, "sar_only_count": 0, "dedup_count": 0}
        structured = {"target": target, "match_strategy": str(match_strategy or "hybrid"), "matched_pairs": [], "rgb_only": [], "sar_only": [], "dedup_detections": [], "match_summary": summary, "status": "missing_inputs"}
        return {"text": text, "error_code": error_code, "match_summary": summary, "matched_pairs": [], "rgb_only": [], "sar_only": [], "dedup_detections": [], "structured": structured, "artifacts": []}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Match optical/RGB detections with SAR candidates and return one-to-one cross-modal target pairs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rgb_image": {"type": "string", "description": "Path to optical/RGB image."},
                        "sar_image": {"type": "string", "description": "Path to SAR image."},
                        "target": {"type": "string", "description": "Target class or description."},
                        "rgb_detections": {"type": "array", "description": "Optional RGB detections."},
                        "sar_candidates": {"type": "array", "description": "Optional SAR candidates."},
                        "sar_detections": {"type": "array", "description": "Alias for optional SAR candidates."},
                        "match_strategy": {"type": "string", "description": "iou, center_distance, hybrid, or global."},
                        "iou_threshold": {"type": "number", "description": "IoU threshold for strong matches."},
                        "center_distance_threshold": {"type": "string", "description": "auto or numeric center distance threshold in pixels."},
                        "score_threshold": {"type": "number", "description": "Minimum detection/candidate score and weak-match score."},
                        "visualize": {"type": "boolean", "description": "Whether to draw match visualization."},
                        "top1": {"type": "boolean", "description": "Keep only the top-scoring candidate from each modality."},
                        "max_draw": {"type": "integer", "description": "Maximum boxes to draw in the match visualization."},
                    },
                    "required": ["rgb_image", "sar_image", "target"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20132)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="CrossModalMatchSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")
    worker = CrossModalMatchSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
