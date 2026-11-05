"""Geometry measurement skill for detected or provided object boxes."""
import argparse
import ast
import math
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ObjectMeasureSkill_worker_{worker_id}.log")


class ObjectMeasureSkillWorker(BaseToolWorker):
    VALID_MEASURE_TYPES = {"area", "distance", "centroid", "bbox_size", "pair_distance", "pixel_dimensions"}
    INTERNAL_TOOL_WEIGHTS = {
        "ObjectDetectSelectSkill": 3,
    }
    VALID_SELECTORS = {
        "first",
        "second",
        "smallest",
        "largest",
        "nearest",
        "farthest",
        "leftmost",
        "rightmost",
        "topmost",
        "bottommost",
    }

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "ObjectMeasureSkill",
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
    def _image_size(image: str) -> Optional[Tuple[int, int]]:
        try:
            with Image.open(image) as img:
                return img.size
        except Exception:
            return None

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
    def _coerce_bbox(cls, value: Any) -> Optional[List[float]]:
        value = cls._literal_or_raw(value)
        if isinstance(value, dict):
            value = value.get("bbox") or value.get("box")
        if isinstance(value, str):
            nums = re.findall(r"[-+]?\d*\.?\d+", value)
            if len(nums) >= 4:
                value = nums[:4]
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
    def _coerce_bbox_list(cls, value: Any) -> List[List[float]]:
        value = cls._literal_or_raw(value)
        if isinstance(value, dict):
            bbox = cls._coerce_bbox(value)
            return [bbox] if bbox else []
        if isinstance(value, str):
            groups = re.findall(r"\(([^()]+)\)", value)
            boxes = [cls._coerce_bbox(group) for group in groups]
            boxes = [box for box in boxes if box]
            if boxes:
                return boxes
        if isinstance(value, (list, tuple)) and len(value) >= 4 and not isinstance(value[0], (list, tuple, dict)):
            bbox = cls._coerce_bbox(value)
            return [bbox] if bbox else []
        if isinstance(value, (list, tuple)):
            boxes = [cls._coerce_bbox(item) for item in value]
            return [box for box in boxes if box]
        return []

    @staticmethod
    def _area_px(bbox: Sequence[float]) -> float:
        x1, y1, x2, y2 = bbox[:4]
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    @staticmethod
    def _centroid(bbox: Sequence[float]) -> List[float]:
        x1, y1, x2, y2 = bbox[:4]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    @classmethod
    def _make_object(cls, bbox: Sequence[float], role: str, reason: str, source: str, rank: Optional[int] = None) -> Dict[str, Any]:
        clean_bbox = [float(v) for v in bbox[:4]]
        obj = {
            "role": role,
            "bbox": clean_bbox,
            "centroid": cls._centroid(clean_bbox),
            "area_px": cls._area_px(clean_bbox),
            "selection_reason": reason,
            "source": source,
        }
        if rank is not None:
            obj["rank"] = rank
        return obj

    @classmethod
    def _objects_from_detections(cls, detections: Any, role: str, source: str) -> List[Dict[str, Any]]:
        detections = cls._literal_or_raw(detections)
        if not isinstance(detections, list):
            return []
        objects = []
        for idx, item in enumerate(detections, start=1):
            bbox = cls._coerce_bbox(item)
            if not bbox:
                continue
            reason = "provided detection"
            rank = idx
            if isinstance(item, dict):
                reason = str(item.get("selection_reason") or reason)
                rank = item.get("rank") or idx
            try:
                rank = int(rank)
            except Exception:
                rank = idx
            obj = cls._make_object(bbox, role, reason, source, rank=rank)
            if isinstance(item, dict):
                for key in ["label", "class_name", "name", "target"]:
                    if item.get(key) not in [None, ""]:
                        obj["label"] = " ".join(str(item.get(key)).lower().replace("_", " ").split())
                        break
            objects.append(obj)
        return objects

    @staticmethod
    def _normalize_label(value: Any) -> str:
        return " ".join(str(value or "").lower().replace("_", " ").split())

    def _objects_from_structured(self, structured: Any, source: str) -> List[Dict[str, Any]]:
        if not isinstance(structured, dict):
            return []
        selected = self._objects_from_detections(structured.get("selected"), "target", f"{source}.selected")
        if selected:
            return selected
        selected = self._objects_from_detections(structured.get("selected_detections"), "target", f"{source}.selected_detections")
        if selected:
            return selected
        return self._objects_from_detections(structured.get("detections"), "target", f"{source}.detections")

    def _latest_structured_candidates(self, input_registry: Any) -> List[Tuple[str, Dict[str, Any]]]:
        if not isinstance(input_registry, dict):
            return []
        candidates = []
        latest = input_registry.get("latest_structured")
        if isinstance(latest, dict):
            candidates.append(("input_registry.latest_structured", latest))
        for key, value in input_registry.items():
            if key == "latest_structured":
                continue
            if isinstance(key, str) and key.endswith("_structured") and isinstance(value, dict):
                candidates.append((f"input_registry.{key}", value))
        return candidates

    def _extract_objects(self, params: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
        trace = []

        boxes = self._coerce_bbox_list(params.get("bbox"))
        if boxes:
            objects = []
            for idx, bbox in enumerate(boxes):
                role = "target" if len(boxes) == 1 else ("a" if idx == 0 else "b")
                objects.append(self._make_object(bbox, role, "provided bbox", "bbox", rank=idx + 1))
            trace.append(f"bbox source used: {len(objects)} object(s)")
            return objects, trace

        for key in ["selected_detections", "detections"]:
            objects = self._objects_from_detections(params.get(key), "target", key)
            if objects:
                trace.append(f"{key} source used: {len(objects)} object(s)")
                return objects, trace

        objects = self._objects_from_structured(params.get("structured"), "structured")
        if objects:
            trace.append(f"structured source used: {len(objects)} object(s)")
            return objects, trace

        for source, structured in self._latest_structured_candidates(params.get("input_registry")):
            objects = self._objects_from_structured(structured, source)
            if objects:
                trace.append(f"{source} source used: {len(objects)} object(s)")
                return objects, trace

        return [], trace

    @classmethod
    def _filter_objects_by_target(cls, objects: List[Dict[str, Any]], target: Any) -> List[Dict[str, Any]]:
        target_text = cls._normalize_label(target)
        if not target_text:
            return objects
        filtered = []
        for obj in objects:
            label = cls._normalize_label(obj.get("label"))
            if not label or target_text in label or label in target_text:
                filtered.append(obj)
        return filtered or objects

    @staticmethod
    def _normalize_measure_type(value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return {
            "pixel": "pixel_dimensions",
            "pixels": "pixel_dimensions",
            "pixel_size": "pixel_dimensions",
            "pixel_dimension": "pixel_dimensions",
            "dimensions": "bbox_size",
            "size": "bbox_size",
            "bbox": "bbox_size",
            "bbox_dimensions": "bbox_size",
            "pairwise_distance": "pair_distance",
            "between_distance": "pair_distance",
        }.get(text, text)

    @classmethod
    def _estimate_internal_cost(cls, internal_tool_calls: Optional[List[Dict[str, Any]]]) -> float:
        if not internal_tool_calls:
            return 0.0
        return float(sum(cls.INTERNAL_TOOL_WEIGHTS.get(str(item.get("tool")), 2) for item in internal_tool_calls))

    def _fallback_detect(self, image: str, target: str, detector: str, reference_target: Optional[str]) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        trace = ["fallback:ObjectDetectSelectSkill used because no bbox/detections were provided"]
        objects: List[Dict[str, Any]] = []
        internal_tool_calls: List[Dict[str, Any]] = []

        def run(query: str, role: str):
            payload = {
                "image": image,
                "target": query,
                "selection_mode": "all",
                "detector": detector,
                "visualize": False,
            }
            started = time.perf_counter()
            try:
                resp = self._call_tool("ObjectDetectSelectSkill", payload, timeout=240)
            except Exception as exc:
                elapsed = time.perf_counter() - started
                internal_tool_calls.append(
                    {
                        "tool": "ObjectDetectSelectSkill",
                        "success": False,
                        "error_code": 1,
                        "elapsed": round(elapsed, 6),
                        "reason": "fallback detection because no bbox was provided",
                        "text": str(exc),
                    }
                )
                trace.append(f"fallback detection for {role}: exception={exc}")
                return []
            elapsed = time.perf_counter() - started
            internal_tool_calls.append(
                {
                    "tool": "ObjectDetectSelectSkill",
                    "success": resp.get("error_code") == 0,
                    "error_code": resp.get("error_code", 1),
                    "elapsed": round(elapsed, 6),
                    "reason": "fallback detection because no bbox was provided",
                    "text": resp.get("text", ""),
                }
            )
            trace.append(f"fallback detection for {role}: error_code={resp.get('error_code')}, text={resp.get('text', '')}")
            if resp.get("error_code") != 0:
                return []
            structured = resp.get("structured") if isinstance(resp.get("structured"), dict) else {}
            detected = self._objects_from_structured(structured, f"fallback.{role}")
            for obj in detected:
                obj["role"] = role
                obj["selection_reason"] = f"fallback detection for {role}"
            return detected

        objects.extend(run(target, "target" if not reference_target else "a"))
        if reference_target:
            objects.extend(run(reference_target, "b"))
        return objects, trace, internal_tool_calls

    def _select_object(
        self,
        objects: List[Dict[str, Any]],
        selector: Optional[str],
        role: str,
        image_size: Optional[Tuple[int, int]],
        exclude: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        candidates = [obj for obj in objects if obj is not exclude]
        role_candidates = [obj for obj in candidates if obj.get("role") == role]
        if role_candidates:
            candidates = role_candidates
        if not candidates:
            return None
        selector = (selector or "first").strip().lower()
        if selector not in self.VALID_SELECTORS:
            selector = "first"

        if selector == "first":
            selected = candidates[0]
            reason = "first available object"
        elif selector == "second":
            selected = candidates[1] if len(candidates) > 1 else None
            reason = "second available object"
        elif selector == "smallest":
            selected = min(candidates, key=lambda obj: (obj["area_px"], obj.get("rank", 999999)))
            reason = "smallest bbox pixel area"
        elif selector == "largest":
            selected = max(candidates, key=lambda obj: (obj["area_px"], -obj.get("rank", 999999)))
            reason = "largest bbox pixel area"
        elif selector == "leftmost":
            selected = min(candidates, key=lambda obj: (obj["centroid"][0], obj.get("rank", 999999)))
            reason = "leftmost centroid"
        elif selector == "rightmost":
            selected = max(candidates, key=lambda obj: (obj["centroid"][0], -obj.get("rank", 999999)))
            reason = "rightmost centroid"
        elif selector == "topmost":
            selected = min(candidates, key=lambda obj: (obj["centroid"][1], obj.get("rank", 999999)))
            reason = "topmost centroid"
        elif selector == "bottommost":
            selected = max(candidates, key=lambda obj: (obj["centroid"][1], -obj.get("rank", 999999)))
            reason = "bottommost centroid"
        else:
            if image_size:
                center_x, center_y = image_size[0] / 2.0, image_size[1] / 2.0
            else:
                center_x, center_y = 0.0, 0.0

            def dist_sq(obj: Dict[str, Any]) -> float:
                cx, cy = obj["centroid"]
                return (cx - center_x) ** 2 + (cy - center_y) ** 2

            if selector == "nearest":
                selected = min(candidates, key=lambda obj: (dist_sq(obj), obj.get("rank", 999999)))
                reason = "nearest to image center" if image_size else "nearest to origin"
            else:
                selected = max(candidates, key=lambda obj: (dist_sq(obj), -obj.get("rank", 999999)))
                reason = "farthest from image center" if image_size else "farthest from origin"

        if selected is None:
            return None
        result = dict(selected)
        result["role"] = role
        result["selection_reason"] = reason
        return result

    @staticmethod
    def _distance_px(a: Dict[str, Any], b: Dict[str, Any]) -> float:
        ax, ay = a["centroid"]
        bx, by = b["centroid"]
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    @staticmethod
    def _format_measurement_text(target: str, reference_target: str, measure_type: str, output_unit: str, measurement: Dict[str, Any]) -> str:
        if measure_type in {"distance", "pair_distance"}:
            subject = f"between the {target} and the {reference_target}" if reference_target else f"for '{target}'"
            if output_unit == "meter" and measurement.get("distance_m") is not None:
                return f"The distance {subject} is {float(measurement['distance_m']):.2f} meters."
            if measurement.get("distance_px") is not None:
                return f"The distance {subject} is {float(measurement['distance_px']):.2f} pixels."
        if measure_type == "area":
            if output_unit == "meter" and measurement.get("area_m2") is not None:
                return f"The area of the {target} is {float(measurement['area_m2']):.2f} square meters."
            if measurement.get("area_px") is not None:
                return f"The area of the {target} is {float(measurement['area_px']):.2f} square pixels."
        if measure_type in {"bbox_size", "pixel_dimensions"}:
            width = measurement.get("width_px")
            height = measurement.get("height_px")
            if width is not None and height is not None:
                return f"Pixel width ≈ {float(width):.2f} px. Pixel length ≈ {float(height):.2f} px."
        if measure_type == "centroid" and measurement.get("centroid") is not None:
            return f"The centroid of the {target} is {measurement['centroid']}."
        return f"Measured {measure_type} for '{target}' in {output_unit} units."

    def _base_result(
        self,
        text: str,
        error_code: int,
        target: str,
        measure_type: str,
        unit: str,
        gsd: Optional[float],
        objects_used: List[Dict[str, Any]],
        measurement: Dict[str, Any],
        calculation_trace: List[str],
        status: str,
        internal_tool_calls: Optional[List[Dict[str, Any]]] = None,
        internal_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        internal_tool_calls = internal_tool_calls or []
        if internal_cost is None:
            internal_cost = self._estimate_internal_cost(internal_tool_calls)
        structured = {
            "target": target,
            "measure_type": measure_type,
            "unit": unit,
            "gsd_m_per_pixel": gsd,
            "objects_used": objects_used,
            "measurement": measurement,
            "calculation_trace": calculation_trace,
            "status": status,
            "internal_tool_calls": internal_tool_calls,
            "internal_cost": internal_cost,
        }
        return {
            "text": text,
            "error_code": error_code,
            "task_result": {"label": status, "confidence": 1.0 if error_code == 0 else 0.0, "answer": text},
            "measurement": measurement,
            "structured": structured,
            "artifacts": [],
            "internal_tool_calls": internal_tool_calls,
            "internal_cost": internal_cost,
        }

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image = params.get("image")
        target = params.get("target") or params.get("text")
        reference_target = params.get("reference_target") or params.get("target_b") or params.get("reference")
        measure_type = self._normalize_measure_type(params.get("measure_type"))
        unit_pref = str(params.get("unit") or "pixel").strip().lower()
        unit_pref = "pixel" if unit_pref in {"pixel", "pixels", "px", "像素"} else "meter"
        gsd = params.get("gsd_m_per_pixel")
        detector = params.get("detector", "auto")

        if params.get("normalizer_error"):
            return {"text": str(params.get("normalizer_error")), "error_code": 2, "measurement": {}, "structured": {}, "artifacts": []}
        if not image:
            return {"text": "Missing required parameter: image", "error_code": 2, "measurement": {}, "structured": {}, "artifacts": []}
        if not target:
            return {"text": "Missing required parameter: target", "error_code": 2, "measurement": {}, "structured": {}, "artifacts": []}
        if not measure_type:
            return {"text": "Missing required parameter: measure_type", "error_code": 2, "measurement": {}, "structured": {}, "artifacts": []}
        if measure_type not in self.VALID_MEASURE_TYPES:
            return {"text": f"Unsupported measure_type: {measure_type}", "error_code": 2, "measurement": {}, "structured": {}, "artifacts": []}
        if not os.path.isfile(image):
            return {"text": f"Image not found: {image}", "error_code": 3, "measurement": {}, "structured": {}, "artifacts": []}
        if gsd not in [None, ""]:
            try:
                gsd = float(gsd)
            except Exception:
                return {"text": f"Invalid gsd_m_per_pixel: {gsd}", "error_code": 2, "measurement": {}, "structured": {}, "artifacts": []}
            if gsd <= 0:
                return {"text": f"gsd_m_per_pixel must be positive: {gsd}", "error_code": 2, "measurement": {}, "structured": {}, "artifacts": []}
        else:
            gsd = None

        calculation_trace: List[str] = []
        internal_tool_calls: List[Dict[str, Any]] = []
        objects, source_trace = self._extract_objects(params)
        calculation_trace.extend(source_trace)

        if not objects:
            try:
                objects, fallback_trace, fallback_internal_calls = self._fallback_detect(
                    image=image,
                    target=str(target),
                    detector=str(detector),
                    reference_target=reference_target,
                )
                calculation_trace.extend(fallback_trace)
                internal_tool_calls.extend(fallback_internal_calls)
            except Exception as exc:
                calculation_trace.append(f"fallback:ObjectDetectSelectSkill failed: {exc}")

        if not objects:
            return self._base_result(
                text=f"No bbox or detection is available for measuring '{target}'.",
                error_code=0,
                target=str(target),
                measure_type=measure_type,
                unit="pixel",
                gsd=gsd,
                objects_used=[],
                measurement={},
                calculation_trace=calculation_trace,
                status="no_detection" if any(item.startswith("fallback:") for item in calculation_trace) else "missing_bbox",
                internal_tool_calls=internal_tool_calls,
            )

        image_size = self._image_size(image)
        output_unit = "meter" if gsd is not None and unit_pref == "meter" else "pixel"
        status = "missing_gsd" if unit_pref == "meter" and gsd is None and measure_type != "centroid" else "success"

        select_a = params.get("select_a")
        select_b = params.get("select_b")
        objects_used: List[Dict[str, Any]] = []
        measurement: Dict[str, Any] = {}

        if measure_type in {"area", "centroid", "bbox_size", "pixel_dimensions"}:
            target_objects = self._filter_objects_by_target(objects, target)
            obj = self._select_object(target_objects, select_a, "target", image_size=image_size)
            if not obj:
                return self._base_result(
                    text="No valid bbox is available for the requested measurement.",
                    error_code=0,
                    target=str(target),
                    measure_type=measure_type,
                    unit="pixel",
                    gsd=gsd,
                    objects_used=[],
                    measurement={},
                    calculation_trace=calculation_trace,
                    status="missing_bbox",
                    internal_tool_calls=internal_tool_calls,
                )
            objects_used = [obj]
            width_px = obj["bbox"][2] - obj["bbox"][0]
            height_px = obj["bbox"][3] - obj["bbox"][1]

            if measure_type == "area":
                measurement["area_px"] = obj["area_px"]
                calculation_trace.append("area_px = width * height")
                if gsd is not None:
                    measurement["area_m2"] = obj["area_px"] * (gsd ** 2)
                    calculation_trace.append("area_m2 = area_px * gsd_m_per_pixel^2")
            elif measure_type == "centroid":
                measurement["centroid"] = obj["centroid"]
                output_unit = "pixel"
                status = "success"
                calculation_trace.append("centroid = [(x1+x2)/2, (y1+y2)/2]")
            elif measure_type == "bbox_size":
                measurement["width_px"] = width_px
                measurement["height_px"] = height_px
                calculation_trace.append("bbox_size_px = [x2-x1, y2-y1]")
                if gsd is not None:
                    measurement["width_m"] = width_px * gsd
                    measurement["height_m"] = height_px * gsd
                    calculation_trace.append("bbox_size_m = bbox_size_px * gsd_m_per_pixel")
            else:
                ratio = params.get("length_width_ratio")
                try:
                    ratio = float(ratio) if ratio not in [None, ""] else None
                except Exception:
                    ratio = None
                if ratio and ratio > 0:
                    width_px_from_area = math.sqrt(obj["area_px"] / ratio)
                    measurement["width_px"] = width_px_from_area
                    measurement["height_px"] = width_px_from_area * ratio
                    calculation_trace.append("pixel_dimensions derived from bbox area and length_width_ratio")
                else:
                    measurement["width_px"] = width_px
                    measurement["height_px"] = height_px
                    calculation_trace.append("pixel_dimensions_px = [bbox width, bbox height]")
                output_unit = "pixel"
                status = "success"
        else:
            if not select_a:
                select_a = "first"
            if not select_b:
                select_b = "second"
            objects_a = self._filter_objects_by_target(objects, target)
            objects_b = self._filter_objects_by_target(objects, reference_target) if reference_target else objects
            obj_a = self._select_object(objects_a, select_a, "a", image_size=image_size)
            obj_b = self._select_object(objects_b, select_b, "b", image_size=image_size, exclude=obj_a)
            if not obj_a or not obj_b:
                return self._base_result(
                    text="Two valid bboxes are required for distance measurement.",
                    error_code=0,
                    target=str(target),
                    measure_type=measure_type,
                    unit="pixel",
                    gsd=gsd,
                    objects_used=[obj for obj in [obj_a, obj_b] if obj],
                    measurement={},
                    calculation_trace=calculation_trace,
                    status="missing_bbox",
                    internal_tool_calls=internal_tool_calls,
                )
            objects_used = [obj_a, obj_b]
            distance_px = self._distance_px(obj_a, obj_b)
            measurement["distance_px"] = distance_px
            calculation_trace.append("distance_px = sqrt((cx1-cx2)^2 + (cy1-cy2)^2)")
            if gsd is not None:
                measurement["distance_m"] = distance_px * gsd
                calculation_trace.append("distance_m = distance_px * gsd_m_per_pixel")

        if gsd is None and unit_pref == "meter" and measure_type != "centroid":
            status = "missing_gsd"
            output_unit = "pixel"

        text = self._format_measurement_text(str(target), str(reference_target or ""), measure_type, output_unit, measurement)
        return self._base_result(
            text=text,
            error_code=0,
            target=str(target),
            measure_type=measure_type,
            unit=output_unit,
            gsd=gsd,
            objects_used=objects_used,
            measurement=measurement,
            calculation_trace=calculation_trace,
            status=status,
            internal_tool_calls=internal_tool_calls,
        )

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": (
                    "Measure geometry for provided object bboxes or selected detections. "
                    "Supports area, distance, centroid, bbox_size, pixel_dimensions, and pair_distance with optional GSD conversion."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Path to the input image."},
                        "target": {"type": "string", "description": "Target object name."},
                        "measure_type": {"type": "string", "description": "area, distance, centroid, bbox_size, pixel_dimensions, or pair_distance."},
                        "bbox": {"type": "array", "description": "One bbox [x1,y1,x2,y2] or a list of bboxes."},
                        "detections": {"type": "array", "description": "Candidate detections with bbox fields."},
                        "selected_detections": {"type": "array", "description": "Selected detections with bbox fields."},
                        "reference_target": {"type": "string", "description": "Optional reference target for fallback distance detection."},
                        "select_a": {"type": "string", "description": "Selection rule for role a/target."},
                        "select_b": {"type": "string", "description": "Selection rule for role b."},
                        "gsd_m_per_pixel": {"type": "number", "description": "Ground sampling distance in meters per pixel."},
                        "unit": {"type": "string", "description": "pixel or meter. Default is pixel; meter output requires gsd_m_per_pixel."},
                        "visualize": {"type": "boolean", "description": "Accepted for compatibility; this skill does not draw final annotations."},
                        "detector": {"type": "string", "description": "Detector preference for fallback ObjectDetectSelectSkill calls."},
                        "structured": {"type": "object", "description": "Previous skill structured output containing detections or selected boxes."},
                        "input_registry": {"type": "object", "description": "Runtime registry with latest structured outputs."},
                        "normalizer_error": {"type": "string", "description": "Internal normalized-parameter error propagated by SkillExecutor."},
                    },
                    "required": ["image", "target", "measure_type"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20128)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ObjectMeasureSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = ObjectMeasureSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
