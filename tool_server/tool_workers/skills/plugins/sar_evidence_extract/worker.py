"""SAR-only evidence extraction skill."""
import argparse
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
logger = build_logger(__file__, f"SAREvidenceExtractSkill_worker_{worker_id}.log")


class SAREvidenceExtractSkillWorker(BaseToolWorker):
    INTERNAL_TOOL_WEIGHTS = {
        "SARToRGB": 4,
        "DenoDetV2": 4,
    }

    CLASS_ALIASES = {
        "ship": {"ship", "boat", "vessel", "舰船", "船", "船只"},
        "aircraft": {"aircraft", "airplane", "plane", "飞机", "飞行器"},
        "car": {"car", "vehicle", "truck", "bus", "车辆", "汽车", "车"},
        "tank": {"tank", "storage tank", "储罐", "坦克"},
        "bridge": {"bridge", "桥", "桥梁"},
        "harbor": {"harbor", "harbour", "port", "港口", "码头"},
    }

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "SAREvidenceExtractSkill",
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
    def _normalize_preprocess_mode(value: Any) -> str:
        mode = str(value or "auto").strip().lower().replace("-", "_")
        aliases = {
            "none": "direct",
            "raw": "direct",
            "passthrough": "direct",
            "pass_through": "direct",
            "sar_preprocess": "sar_to_rgb",
            "sartorgb": "sar_to_rgb",
            "sar2rgb": "sar_to_rgb",
            "rgb": "sar_to_rgb",
        }
        return aliases.get(mode, mode)

    @staticmethod
    def _normalize_detector(value: Any) -> str:
        detector = str(value or "auto").strip().lower().replace("-", "_")
        aliases = {
            "deno": "denodetv2",
            "denodet": "denodetv2",
            "denodet_v2": "denodetv2",
            "sar": "denodetv2",
            "sar_detector": "denodetv2",
        }
        return aliases.get(detector, detector)

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

    @staticmethod
    def _bbox_area(bbox: Sequence[int]) -> int:
        x1, y1, x2, y2 = bbox[:4]
        return max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))

    @staticmethod
    def _bbox_centroid(bbox: Sequence[int]) -> List[float]:
        x1, y1, x2, y2 = bbox[:4]
        return [(float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0]

    @classmethod
    def _target_classes(cls, target: str) -> Optional[set]:
        normalized = " ".join(str(target or "").lower().replace("_", " ").split())
        if not normalized:
            return None
        matched = set()
        for class_name, aliases in cls.CLASS_ALIASES.items():
            if any(alias.lower() in normalized for alias in aliases):
                matched.add(class_name)
        return matched or None

    @classmethod
    def _target_matches_label(cls, target_classes: Optional[set], label: str) -> bool:
        if not target_classes:
            return True
        normalized_label = " ".join(str(label or "").lower().replace("_", " ").split())
        for class_name in target_classes:
            aliases = cls.CLASS_ALIASES.get(class_name, {class_name})
            if normalized_label == class_name or any(alias.lower() == normalized_label for alias in aliases):
                return True
        return False

    @staticmethod
    def _response_image(image_path: str) -> Image.Image:
        return Image.open(image_path).convert("L")

    @staticmethod
    def _response_stats(gray: Image.Image, bbox: Sequence[int]) -> Dict[str, float]:
        crop = gray.crop(tuple(bbox[:4]))
        values = list(crop.getdata())
        if not values:
            return {"response_mean": 0.0, "response_max": 0.0, "response_mean_norm": 0.0, "response_max_norm": 0.0}
        mean_value = float(sum(values)) / float(len(values))
        max_value = float(max(values))
        return {
            "response_mean": mean_value,
            "response_max": max_value,
            "response_mean_norm": mean_value / 255.0,
            "response_max_norm": max_value / 255.0,
        }

    @staticmethod
    def _passes_response_threshold(candidate: Dict[str, Any], threshold: Any) -> bool:
        if threshold in [None, ""]:
            return True
        if isinstance(threshold, str) and threshold.strip().lower() == "auto":
            return True
        threshold_value = float(threshold)
        if threshold_value <= 1.0:
            return float(candidate.get("response_max_norm", 0.0)) >= threshold_value
        return float(candidate.get("response_max", 0.0)) >= threshold_value

    @classmethod
    def _estimate_internal_cost(cls, internal_tool_calls: List[Dict[str, Any]]) -> float:
        return float(sum(cls.INTERNAL_TOOL_WEIGHTS.get(str(item.get("tool")), 2) for item in internal_tool_calls or []))

    def _preprocess_image(self, sar_image: str, preprocess_mode: str, skill_trace: List[Dict[str, Any]], internal_tool_calls: List[Dict[str, Any]]) -> str:
        if preprocess_mode in {"auto", "direct", "none", "raw"}:
            skill_trace.append({"stage": "preprocess_sar", "status": "skipped", "mode": preprocess_mode})
            return sar_image
        if preprocess_mode != "sar_to_rgb":
            raise ValueError(f"Unsupported preprocess_mode: {preprocess_mode}")

        started = time.perf_counter()
        response = self._call_tool("SARToRGB", {"image": sar_image}, timeout=240)
        elapsed = time.perf_counter() - started
        internal_tool_calls.append({"tool": "SARToRGB", "success": response.get("error_code") == 0, "error_code": response.get("error_code", 1), "elapsed": round(elapsed, 6), "reason": "SAR preprocessing requested sar_to_rgb"})
        skill_trace.append(
            {
                "stage": "preprocess_sar",
                "status": "success" if response.get("error_code") == 0 else "error",
                "mode": preprocess_mode,
                "tool": "SARToRGB",
                "tool_text": response.get("text", ""),
            }
        )
        if response.get("error_code") != 0:
            raise RuntimeError(response.get("text") or "SARToRGB preprocessing failed")
        output_path = response.get("image") or response.get("output_path")
        if not output_path or not os.path.isfile(output_path):
            raise RuntimeError("SARToRGB did not return a valid output image")
        return str(output_path)

    def _run_detector(
        self,
        image: str,
        detector: str,
        confidence_threshold: float,
        skill_trace: List[Dict[str, Any]],
        internal_tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if detector not in {"auto", "denodetv2"}:
            raise ValueError(f"Unsupported detector: {detector}")

        payload = {
            "image": image,
            "box_threshold": confidence_threshold,
            "top1": False,
            "return_json": True,
        }
        started = time.perf_counter()
        response = self._call_tool("DenoDetV2", payload, timeout=240)
        elapsed = time.perf_counter() - started
        internal_tool_calls.append({"tool": "DenoDetV2", "success": response.get("error_code") == 0, "error_code": response.get("error_code", 1), "elapsed": round(elapsed, 6), "reason": "SAR evidence candidate detection"})
        skill_trace.append(
            {
                "stage": "detect_sar_candidates",
                "status": "success" if response.get("error_code") == 0 else "error",
                "tool": "DenoDetV2",
                "confidence_threshold": confidence_threshold,
                "tool_text": response.get("text", ""),
            }
        )
        if response.get("error_code") != 0:
            if response.get("error_code") == 4:
                return []
            raise RuntimeError(response.get("text") or "DenoDetV2 detection failed")
        return list(response.get("data", []) or [])

    def _build_candidates(
        self,
        detections: List[Dict[str, Any]],
        sar_image: str,
        target: str,
        min_area: Optional[float],
        max_area: Optional[float],
        response_threshold: Any,
        max_candidates: int,
    ) -> List[Dict[str, Any]]:
        with Image.open(sar_image) as image:
            width, height = image.size
        gray = self._response_image(sar_image)
        target_classes = self._target_classes(target)

        candidates = []
        for detection in detections:
            bbox = self._normalize_bbox(detection.get("bbox"), width, height)
            if not bbox:
                continue
            label = str(detection.get("class_name") or detection.get("label") or "object")
            if not self._target_matches_label(target_classes, label):
                continue

            area_px = self._bbox_area(bbox)
            if min_area is not None and area_px < float(min_area):
                continue
            if max_area is not None and area_px > float(max_area):
                continue

            stats = self._response_stats(gray, bbox)
            candidate = {
                "bbox": bbox,
                "bbox_format": "xyxy",
                "centroid": self._bbox_centroid(bbox),
                "area_px": area_px,
                "score": float(detection.get("score", 0.0)),
                "label": label,
                "class_id": detection.get("class_id"),
                "target_match": bool(target_classes),
                **stats,
            }
            if not self._passes_response_threshold(candidate, response_threshold):
                continue
            candidates.append(candidate)

        candidates.sort(key=lambda item: (-float(item.get("score", 0.0)), -float(item.get("response_max", 0.0)), -int(item.get("area_px", 0))))
        limited = candidates[: max(0, int(max_candidates))]
        for rank, item in enumerate(limited, start=1):
            item["rank"] = rank
        return limited

    @staticmethod
    def _draw_box(draw: ImageDraw.ImageDraw, bbox: Sequence[int], label: str, font: ImageFont.ImageFont):
        x1, y1, x2, y2 = bbox[:4]
        color = (255, 60, 0)
        for offset in range(2):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        label_y = max(0, y1 - 14)
        draw.rectangle((x1, label_y, x1 + max(42, len(label) * 6), label_y + 13), fill=color)
        draw.text((x1 + 2, label_y + 1), label, fill=(255, 255, 255), font=font)

    def _draw_visualization(self, sar_image: str, candidates: List[Dict[str, Any]], max_draw: int) -> Optional[str]:
        try:
            canvas = Image.open(sar_image).convert("RGB")
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            for candidate in candidates[: max(0, int(max_draw))]:
                label = f"#{candidate.get('rank')}:{candidate.get('label')}:{float(candidate.get('score', 0.0)):.2f}"
                self._draw_box(draw, candidate["bbox"], label[:36], font)

            save_dir = Path(os.getcwd()) / "tools_output"
            save_dir.mkdir(parents=True, exist_ok=True)
            output_path = save_dir / f"sar_evidence_extract_{uuid.uuid4().hex[:8]}.png"
            canvas.save(output_path)
            return str(output_path)
        except Exception as exc:
            logger.warning(f"Failed to draw SAR evidence visualization: {exc}")
            return None

    @staticmethod
    def _empty_response(text: str, target: str, status: str = "no_sar_response") -> Dict[str, Any]:
        structured = {
            "target": target,
            "status": status,
            "num_candidates": 0,
            "sar_candidates": [],
            "detector": None,
            "preprocess_mode": None,
        }
        return {
            "text": text,
            "error_code": 0,
            "sar_candidates": [],
            "structured": structured,
            "artifacts": [],
        }

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sar_image = params.get("sar_image") or params.get("image")
        target = params.get("target") or params.get("text")
        if not sar_image:
            return {"text": "Missing required parameter: sar_image", "error_code": 2}
        if not target:
            return {"text": "Missing required parameter: target", "error_code": 2}
        if not os.path.isfile(str(sar_image)):
            return {"text": f"SAR image not found: {sar_image}", "error_code": 3}

        try:
            preprocess_mode = self._normalize_preprocess_mode(params.get("preprocess_mode", "auto"))
            detector = self._normalize_detector(params.get("detector", "auto"))
            confidence_threshold = float(params.get("confidence_threshold", params.get("box_threshold", 0.25)))
            response_threshold = params.get("response_threshold", "auto")
            if not (isinstance(response_threshold, str) and response_threshold.strip().lower() == "auto"):
                response_threshold = float(response_threshold)
            min_area = params.get("min_area")
            max_area = params.get("max_area")
            min_area = float(min_area) if min_area not in [None, ""] else None
            max_area = float(max_area) if max_area not in [None, ""] else None
            max_candidates = max(0, int(params.get("max_candidates", 50)))
            max_draw = int(params.get("max_draw", max_candidates or 50))
            visualize = self._to_bool(params.get("visualize"), True)
        except Exception as exc:
            return {"text": f"Invalid parameter value: {exc}", "error_code": 2}

        skill_trace: List[Dict[str, Any]] = []
        internal_tool_calls: List[Dict[str, Any]] = []

        try:
            detector_image = self._preprocess_image(str(sar_image), preprocess_mode, skill_trace, internal_tool_calls)
            detections = self._run_detector(detector_image, detector, confidence_threshold, skill_trace, internal_tool_calls)
            candidates = self._build_candidates(
                detections=detections,
                sar_image=str(sar_image),
                target=str(target),
                min_area=min_area,
                max_area=max_area,
                response_threshold=response_threshold,
                max_candidates=max_candidates,
            )

            if not candidates:
                result = self._empty_response(
                    text=f"No SAR evidence candidates found for '{target}'. No candidate is fabricated.",
                    target=str(target),
                )
                result["structured"].update(
                    {
                        "detector": detector,
                        "preprocess_mode": preprocess_mode,
                        "confidence_threshold": confidence_threshold,
                        "response_threshold": response_threshold,
                        "internal_tool_calls": internal_tool_calls,
                        "internal_cost": self._estimate_internal_cost(internal_tool_calls),
                    }
                )
                result["skill_trace"] = skill_trace
                result["internal_tool_calls"] = internal_tool_calls
                result["internal_cost"] = self._estimate_internal_cost(internal_tool_calls)
                return result

            artifacts: List[Dict[str, Any]] = []
            if visualize:
                visual_path = self._draw_visualization(str(sar_image), candidates, max_draw)
                if visual_path:
                    artifacts.append(
                        {
                            "id": "sar_candidate_visualization",
                            "type": "image",
                            "path": visual_path,
                            "name": os.path.basename(visual_path),
                            "source": self.model_name,
                        }
                    )

            structured = {
                "target": str(target),
                "status": "success",
                "num_candidates": len(candidates),
                "sar_candidates": candidates,
                "detector": detector,
                "preprocess_mode": preprocess_mode,
                "detector_image": detector_image,
                "confidence_threshold": confidence_threshold,
                "response_threshold": response_threshold,
                "min_area": min_area,
                "max_area": max_area,
                "internal_tool_calls": internal_tool_calls,
                "internal_cost": self._estimate_internal_cost(internal_tool_calls),
            }
            return {
                "text": f"Extracted {len(candidates)} SAR evidence candidate(s) for '{target}'.",
                "error_code": 0,
                "sar_candidates": candidates,
                "structured": structured,
                "artifacts": artifacts,
                "skill_trace": skill_trace,
                "internal_tool_calls": internal_tool_calls,
                "internal_cost": self._estimate_internal_cost(internal_tool_calls),
            }
        except Exception as exc:
            logger.exception(f"Error in {self.model_name}: {exc}")
            internal_cost = self._estimate_internal_cost(internal_tool_calls)
            return {"text": f"Error in {self.model_name}: {exc}", "error_code": 1, "sar_candidates": [], "structured": {"status": "error", "internal_tool_calls": internal_tool_calls, "internal_cost": internal_cost}, "artifacts": [], "skill_trace": skill_trace, "internal_tool_calls": internal_tool_calls, "internal_cost": internal_cost}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Extract SAR-only target evidence candidates with response statistics and visualization artifacts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sar_image": {"type": "string", "description": "Path to the SAR image."},
                        "target": {"type": "string", "description": "Target class or evidence description."},
                        "preprocess_mode": {"type": "string", "description": "auto, direct, or sar_to_rgb."},
                        "confidence_threshold": {"type": "number", "description": "Detector confidence threshold."},
                        "response_threshold": {"type": "string", "description": "auto or numeric SAR response threshold."},
                        "min_area": {"type": "number", "description": "Minimum candidate area in pixels."},
                        "max_area": {"type": "number", "description": "Maximum candidate area in pixels."},
                        "visualize": {"type": "boolean", "description": "Whether to draw candidate boxes."},
                        "max_candidates": {"type": "integer", "description": "Maximum number of SAR candidates to return."},
                        "max_draw": {"type": "integer", "description": "Maximum number of candidate boxes to draw."},
                        "detector": {"type": "string", "description": "auto or denodetv2."},
                    },
                    "required": ["sar_image", "target"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20130)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="SAREvidenceExtractSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = SAREvidenceExtractSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
