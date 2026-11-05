"""
Object detection and target selection skill.

This worker only detects candidate targets and selects final detections by a
selection rule. It deliberately avoids distance, GSD, segmentation, and
real-world area measurement tasks.
"""
import argparse
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image, ImageDraw, ImageFont

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ObjectDetectSelectSkill_worker_{worker_id}.log")


class ObjectDetectSelectSkillWorker(BaseToolWorker):
    VALID_SELECTION_MODES = {
        "all",
        "smallest",
        "largest",
        "nearest",
        "farthest",
        "leftmost",
        "rightmost",
        "topmost",
        "bottommost",
        "nth",
    }

    SMALL_OBJECT_HINTS = {
        "airplane",
        "aircraft",
        "plane",
        "bridge",
        "harbor",
        "harbour",
        "ship",
        "boat",
        "vessel",
        "storage tank",
        "tank",
        "vehicle",
        "car",
        "truck",
        "bus",
    }

    SELECTION_WORDS = {
        "all",
        "smallest",
        "largest",
        "biggest",
        "nearest",
        "closest",
        "farthest",
        "furthest",
        "leftmost",
        "rightmost",
        "topmost",
        "bottommost",
        "left",
        "right",
        "top",
        "bottom",
        "nth",
    }

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "ObjectDetectSelectSkill",
        limit_model_concurrency: int = 2,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        wait_timeout: float = 180.0,
        task_timeout: float = 180.0,
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

    def _call_tool(self, tool_name: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
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

    @classmethod
    def _target_for_detection(cls, target: str) -> str:
        query = str(target or "").strip()
        if not query:
            return query
        for word in sorted(cls.SELECTION_WORDS, key=len, reverse=True):
            query = re.sub(rf"\b{re.escape(word)}\b", " ", query, flags=re.IGNORECASE)
        for word in ["最小", "最大", "最近", "最远", "最左", "最右", "最上", "最下", "上方", "下方", "选择", "找出", "检测", "定位"]:
            query = query.replace(word, " ")
        query = re.sub(r"\s+", " ", query).strip(" ,.;:")
        return query or str(target or "").strip()

    @classmethod
    def _looks_like_small_object_target(cls, target: str) -> bool:
        normalized = " ".join(str(target or "").lower().replace("_", " ").split())
        return any(hint in normalized for hint in cls.SMALL_OBJECT_HINTS)

    @staticmethod
    def _normalize_detector_name(detector: Any) -> str:
        value = str(detector or "auto").strip().lower().replace("-", "_")
        aliases = {
            "object": "object_detection",
            "od": "object_detection",
            "text": "text_to_bbox",
            "text2bbox": "text_to_bbox",
            "text_to_box": "text_to_bbox",
            "small": "small_object_detection",
            "small_object": "small_object_detection",
            "sod": "small_object_detection",
        }
        return aliases.get(value, value)

    def _detector_order(self, detector: str, target: str) -> List[str]:
        detector = self._normalize_detector_name(detector)
        if detector == "auto":
            if self._looks_like_small_object_target(target):
                return ["SmallObjectDetection", "ObjectDetection", "TextToBbox"]
            return ["ObjectDetection", "TextToBbox"]
        if detector == "object_detection":
            return ["ObjectDetection"]
        if detector == "text_to_bbox":
            return ["TextToBbox"]
        if detector == "small_object_detection":
            return ["SmallObjectDetection"]
        raise ValueError(f"Unsupported detector: {detector}")

    @staticmethod
    def _normalize_bbox(bbox: Any, width: int, height: int) -> Optional[List[int]]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return None

        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    @classmethod
    def _parse_text_detections(cls, text: str, width: int, height: int, fallback_label: str) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue

            match = re.match(
                r"^\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\),\s*(.*?),\s*score\s*([-+]?\d*\.?\d+)$",
                line,
            )
            if match:
                x1, y1, x2, y2, label, score = match.groups()
                bbox = cls._normalize_bbox([x1, y1, x2, y2], width, height)
                if bbox:
                    detections.append({"bbox": bbox, "score": float(score), "label": label.strip() or fallback_label})
                continue

            match = re.match(
                r"^\(([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?),\s*([-+]?\d+(?:\.\d+)?)\),\s*([-+]?\d*\.?\d+)$",
                line,
            )
            if match:
                x1, y1, x2, y2, score = match.groups()
                bbox = cls._normalize_bbox([x1, y1, x2, y2], width, height)
                if bbox:
                    detections.append({"bbox": bbox, "score": float(score), "label": fallback_label})
        return detections

    @classmethod
    def _extract_detections(cls, resp: Dict[str, Any], width: int, height: int, fallback_label: str) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        raw_items = resp.get("detections")
        if isinstance(raw_items, list):
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                bbox = cls._normalize_bbox(item.get("bbox"), width, height)
                if not bbox:
                    continue
                try:
                    score = float(item.get("score", 1.0))
                except Exception:
                    score = 1.0
                label = item.get("label") or item.get("class_name") or item.get("category") or fallback_label
                detections.append({"bbox": bbox, "score": score, "label": str(label)})

        detections.extend(cls._parse_text_detections(resp.get("text", ""), width, height, fallback_label))
        return detections

    @staticmethod
    def _bbox_area(bbox: Sequence[int]) -> float:
        x1, y1, x2, y2 = bbox[:4]
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    @staticmethod
    def _bbox_centroid(bbox: Sequence[int]) -> List[float]:
        x1, y1, x2, y2 = bbox[:4]
        return [round((x1 + x2) / 2.0, 3), round((y1 + y2) / 2.0, 3)]

    @staticmethod
    def _bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _nms(self, detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for det in sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True):
            if all(self._bbox_iou(det["bbox"], old["bbox"]) <= iou_threshold for old in kept):
                kept.append(det)
        return kept

    def _detect_targets(
        self,
        image: str,
        target: str,
        detector: str,
        confidence_threshold: float,
        iou_threshold: float,
        width: int,
        height: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        query = self._target_for_detection(target)
        trace: List[Dict[str, Any]] = []
        all_detections: List[Dict[str, Any]] = []

        for tool_name in self._detector_order(detector, query):
            if tool_name == "SmallObjectDetection":
                payload = {
                    "image": image,
                    "text": query,
                    "score_threshold": confidence_threshold,
                    "top1": False,
                }
                timeout = 180
            elif tool_name == "ObjectDetection":
                payload = {
                    "image": image,
                    "text": query,
                    "box_threshold": confidence_threshold,
                    "top1": False,
                }
                timeout = 180
            else:
                payload = {"image": image, "text": query, "top1": False}
                timeout = 180

            try:
                resp = self._call_tool(tool_name, payload, timeout=timeout)
                detections = self._extract_detections(resp, width, height, query)
                detections = [det for det in detections if float(det.get("score", 0.0)) >= confidence_threshold]
                trace.append(
                    {
                        "tool": tool_name,
                        "status": "success" if resp.get("error_code") == 0 else "failed",
                        "text": resp.get("text", ""),
                        "parsed_detections": len(detections),
                    }
                )
                if resp.get("error_code") == 0 and detections:
                    all_detections.extend(detections)
                    break
            except Exception as exc:
                trace.append({"tool": tool_name, "status": "failed", "text": str(exc), "parsed_detections": 0})

        return self._nms(all_detections, iou_threshold=iou_threshold), trace

    def _build_records(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for rank, det in enumerate(sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True), start=1):
            bbox = [int(v) for v in det["bbox"]]
            records.append(
                {
                    "bbox": bbox,
                    "score": float(det.get("score", 0.0)),
                    "label": str(det.get("label") or "object"),
                    "area_px": self._bbox_area(bbox),
                    "centroid": self._bbox_centroid(bbox),
                    "rank": rank,
                }
            )
        return records

    def _select_records(
        self,
        records: List[Dict[str, Any]],
        selection_mode: str,
        top_k: int,
        nth: Optional[int],
        width: int,
        height: int,
    ) -> List[Dict[str, Any]]:
        if not records:
            return []

        if selection_mode == "all":
            ordered = list(records)
            reason = "selected all detections"
        elif selection_mode == "smallest":
            ordered = sorted(records, key=lambda item: (item["area_px"], -item["score"], item["rank"]))
            reason = "smallest bbox pixel area"
        elif selection_mode == "largest":
            ordered = sorted(records, key=lambda item: (-item["area_px"], -item["score"], item["rank"]))
            reason = "largest bbox pixel area"
        elif selection_mode == "leftmost":
            ordered = sorted(records, key=lambda item: (item["centroid"][0], -item["score"], item["rank"]))
            reason = "leftmost centroid"
        elif selection_mode == "rightmost":
            ordered = sorted(records, key=lambda item: (-item["centroid"][0], -item["score"], item["rank"]))
            reason = "rightmost centroid"
        elif selection_mode == "topmost":
            ordered = sorted(records, key=lambda item: (item["centroid"][1], -item["score"], item["rank"]))
            reason = "topmost centroid"
        elif selection_mode == "bottommost":
            ordered = sorted(records, key=lambda item: (-item["centroid"][1], -item["score"], item["rank"]))
            reason = "bottommost centroid"
        elif selection_mode in {"nearest", "farthest"}:
            center_x, center_y = width / 2.0, height / 2.0

            def distance_sq(item: Dict[str, Any]) -> float:
                cx, cy = item["centroid"]
                return (cx - center_x) ** 2 + (cy - center_y) ** 2

            if selection_mode == "nearest":
                ordered = sorted(records, key=lambda item: (distance_sq(item), -item["score"], item["rank"]))
                reason = "nearest to image center"
            else:
                ordered = sorted(records, key=lambda item: (-distance_sq(item), -item["score"], item["rank"]))
                reason = "farthest from image center"
        elif selection_mode == "nth":
            index = max(1, int(nth if nth is not None else top_k)) - 1
            if index >= len(records):
                return []
            item = dict(records[index])
            item["selection_reason"] = f"nth detection by confidence rank: {index + 1}"
            return [item]
        else:
            return []

        selected = []
        limit = len(ordered) if selection_mode == "all" else max(1, int(top_k))
        for item in ordered[:limit]:
            selected_item = dict(item)
            selected_item["selection_reason"] = reason
            selected.append(selected_item)
        return selected

    @staticmethod
    def _draw_box(draw: ImageDraw.ImageDraw, bbox: Sequence[int], color: Tuple[int, int, int], label: str, font: ImageFont.ImageFont):
        x1, y1, x2, y2 = bbox[:4]
        for offset in range(2):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        label_y = max(0, y1 - 14)
        draw.rectangle((x1, label_y, x1 + max(40, len(label) * 6), label_y + 13), fill=color)
        draw.text((x1 + 2, label_y + 1), label, fill=(255, 255, 255), font=font)

    def _draw_candidate_visualization(
        self,
        image: str,
        records: List[Dict[str, Any]],
        selected: List[Dict[str, Any]],
        max_draw: int,
    ) -> Optional[str]:
        try:
            selected_ranks = {item.get("rank") for item in selected}
            canvas = Image.open(image).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()

            for item in records[: max(0, max_draw)]:
                rank = item.get("rank")
                color = (0, 170, 0) if rank in selected_ranks else (255, 140, 0)
                label = f"#{rank}:{item.get('label', 'object')}:{float(item.get('score', 0.0)):.2f}"
                self._draw_box(draw, item["bbox"], color, label[:32], font)

            save_dir = Path(os.getcwd()) / "tools_output"
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"object_detect_select_{uuid.uuid4().hex[:8]}.png"
            canvas.save(output_path)
            return str(output_path)
        except Exception as exc:
            logger.warning(f"Failed to draw candidate visualization: {exc}")
            return None

    def _empty_response(self, text: str, target: str, selection_mode: str, error_code: int = 0) -> Dict[str, Any]:
        structured = {
            "target": target,
            "selection_mode": selection_mode,
            "num_candidates": 0,
            "detections": [],
            "selected": [],
            "status": "no_detection" if error_code == 0 else "error",
        }
        return {
            "text": text,
            "error_code": error_code,
            "detections": [],
            "selected_detections": [],
            "structured": structured,
            "artifacts": [],
        }

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image = params.get("image")
        target = params.get("target") or params.get("text")
        selection_mode = str(params.get("selection_mode") or "").strip().lower()

        if not image:
            return {"text": "Missing required parameter: image", "error_code": 2}
        if not target:
            return {"text": "Missing required parameter: target", "error_code": 2}
        if not selection_mode:
            return {"text": "Missing required parameter: selection_mode", "error_code": 2}
        if not os.path.isfile(image):
            return {"text": f"Image not found: {image}", "error_code": 3}
        if selection_mode not in self.VALID_SELECTION_MODES:
            return {"text": f"Unsupported selection_mode: {selection_mode}", "error_code": 2}

        try:
            top_k = max(1, int(params.get("top_k", 1)))
            confidence_threshold = float(params.get("confidence_threshold", 0.25))
            iou_threshold = float(params.get("iou_threshold", 0.5))
            detector = params.get("detector", "auto")
            visualize = self._to_bool(params.get("visualize"), True)
            max_draw = int(params.get("max_draw", 20))
            nth = params.get("nth")
            nth = int(nth) if nth not in [None, ""] else None
        except Exception as exc:
            return {"text": f"Invalid parameter value: {exc}", "error_code": 2}

        skill_trace: List[Dict[str, Any]] = []

        try:
            with Image.open(image) as img:
                width, height = img.size

            detections, detect_trace = self._detect_targets(
                image=image,
                target=str(target),
                detector=str(detector),
                confidence_threshold=confidence_threshold,
                iou_threshold=max(0.0, min(1.0, iou_threshold)),
                width=width,
                height=height,
            )
            skill_trace.extend(detect_trace)

            records = self._build_records(detections)
            if not records:
                result = self._empty_response(
                    text=f"No detections found for '{target}'. No bbox is fabricated.",
                    target=str(target),
                    selection_mode=selection_mode,
                    error_code=0,
                )
                result["skill_trace"] = skill_trace
                return result

            selected = self._select_records(
                records=records,
                selection_mode=selection_mode,
                top_k=top_k,
                nth=nth,
                width=width,
                height=height,
            )

            artifacts: List[Dict[str, Any]] = []
            if visualize:
                visual_path = self._draw_candidate_visualization(
                    image=image,
                    records=records,
                    selected=selected,
                    max_draw=max_draw,
                )
                if visual_path:
                    artifacts.append(
                        {
                            "id": "candidate_visualization",
                            "type": "image",
                            "path": visual_path,
                            "name": os.path.basename(visual_path),
                            "source": self.model_name,
                        }
                    )

            structured = {
                "target": str(target),
                "selection_mode": selection_mode,
                "num_candidates": len(records),
                "detections": records,
                "selected": selected,
                "status": "success",
            }

            return {
                "text": (
                    f"Detected {len(records)} candidate(s) for '{target}'. "
                    f"Selected {len(selected)} detection(s) using selection_mode='{selection_mode}'."
                ),
                "error_code": 0,
                "detections": records,
                "selected_detections": selected,
                "structured": structured,
                "artifacts": artifacts,
                "skill_trace": skill_trace,
            }
        except Exception as exc:
            logger.exception(f"{self.model_name} failed: {exc}")
            return {
                "text": f"{self.model_name} failed: {exc}",
                "error_code": 1,
                "detections": [],
                "selected_detections": [],
                "structured": {
                    "target": str(target),
                    "selection_mode": selection_mode,
                    "num_candidates": 0,
                    "detections": [],
                    "selected": [],
                    "status": "no_detection",
                },
                "artifacts": [],
                "skill_trace": skill_trace,
            }

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Detect candidate targets in one training image and select final target detections. "
                    "It does not compute distance, GSD, segmentation, or real-world area measurements."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Path to the input image."},
                        "target": {"type": "string", "description": "Target object class or phrase, e.g. storage tank or airplane."},
                        "selection_mode": {
                            "type": "string",
                            "description": "One of all, smallest, largest, nearest, farthest, leftmost, rightmost, topmost, bottommost, nth.",
                        },
                        "top_k": {"type": "integer", "description": "Number of selected detections for non-nth modes. Default 1."},
                        "confidence_threshold": {"type": "number", "description": "Detector score threshold. Default 0.25."},
                        "iou_threshold": {"type": "number", "description": "NMS IoU threshold for duplicate candidates. Default 0.5."},
                        "detector": {"type": "string", "description": "auto, object_detection, text_to_bbox, or small_object_detection."},
                        "visualize": {"type": "boolean", "description": "Whether to create a candidate-debug visualization artifact."},
                        "max_draw": {"type": "integer", "description": "Maximum candidate boxes to draw in the debug artifact."},
                        "nth": {"type": "integer", "description": "Ordinal detection rank for selection_mode=nth."},
                    },
                    "required": ["image", "target", "selection_mode"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20127)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ObjectDetectSelectSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = ObjectDetectSelectSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
