"""Final visualization skill for existing object geometry outputs."""
import argparse
import ast
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"ObjectVisualOutputSkill_worker_{worker_id}.log")


class ObjectVisualOutputSkillWorker(BaseToolWorker):
    VALID_VISUAL_TYPES = {"box", "point", "plot", "box_and_label", "measurement_overlay"}
    COLOR_MAP = {
        "red": (255, 0, 0),
        "green": (0, 170, 0),
        "blue": (0, 90, 255),
        "yellow": (255, 210, 0),
        "orange": (255, 140, 0),
        "cyan": (0, 180, 220),
        "magenta": (220, 0, 220),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
    }

    def __init__(
        self,
        controller_addr: str,
        worker_addr: str = "auto",
        worker_id: str = worker_id,
        no_register: bool = False,
        model_name: str = "ObjectVisualOutputSkill",
        limit_model_concurrency: int = 2,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        wait_timeout: float = 120.0,
        task_timeout: float = 120.0,
    ):
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
    def _color(cls, value: Any) -> Tuple[int, int, int]:
        text = str(value or "red").strip().lower()
        if text in cls.COLOR_MAP:
            return cls.COLOR_MAP[text]
        if re.match(r"^#[0-9a-fA-F]{6}$", text):
            return tuple(int(text[i:i + 2], 16) for i in (1, 3, 5))
        return cls.COLOR_MAP["red"]

    @staticmethod
    def _labels(value: Any) -> List[str]:
        value = ObjectVisualOutputSkillWorker._literal_or_raw(value)
        if isinstance(value, list):
            return [str(item) for item in value]
        if value not in [None, ""]:
            return [str(value)]
        return []

    @classmethod
    def _coerce_bbox(cls, value: Any, width: int, height: int) -> Optional[List[float]]:
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
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width), x2))
        y2 = max(0.0, min(float(height), y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @classmethod
    def _coerce_point(cls, value: Any, width: int, height: int) -> Optional[List[float]]:
        value = cls._literal_or_raw(value)
        if isinstance(value, dict):
            value = value.get("point") or value.get("centroid") or value.get("center")
        if isinstance(value, str):
            nums = re.findall(r"[-+]?\d*\.?\d+", value)
            if len(nums) >= 2:
                value = nums[:2]
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return None
        try:
            x, y = [float(v) for v in value[:2]]
        except Exception:
            return None
        x = max(0.0, min(float(width - 1), x))
        y = max(0.0, min(float(height - 1), y))
        return [x, y]

    @staticmethod
    def _centroid_from_bbox(bbox: Sequence[float]) -> List[float]:
        x1, y1, x2, y2 = bbox[:4]
        return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    def _objects_from_values(self, value: Any, source: str, labels: List[str], width: int, height: int) -> List[Dict[str, Any]]:
        value = self._literal_or_raw(value)
        if value in [None, ""]:
            return []
        values = value if isinstance(value, list) and value and isinstance(value[0], (list, tuple, dict)) else [value]
        objects = []
        for idx, item in enumerate(values):
            label = labels[idx] if idx < len(labels) else None
            if isinstance(item, dict):
                label = item.get("label") or item.get("selection_reason") or label
            bbox = self._coerce_bbox(item, width, height)
            if bbox:
                objects.append({"type": "bbox", "bbox": bbox, "label": label, "source": source})
                continue
            point = self._coerce_point(item, width, height)
            if point:
                objects.append({"type": "point", "point": point, "label": label, "source": source})
        return objects

    def _objects_from_structured(self, structured: Any, source: str, width: int, height: int) -> List[Dict[str, Any]]:
        if not isinstance(structured, dict):
            return []
        objects: List[Dict[str, Any]] = []
        for key in ["selected", "selected_detections", "objects_used", "detections"]:
            objects.extend(self._objects_from_values(structured.get(key), f"{source}.{key}", [], width, height))
            if objects and key in {"selected", "selected_detections", "objects_used"}:
                break
        measurement = structured.get("measurement")
        if isinstance(measurement, dict):
            point = self._coerce_point(measurement.get("centroid") or measurement.get("point"), width, height)
            if point:
                objects.append({"type": "point", "point": point, "label": "centroid", "source": f"{source}.measurement"})
        return objects

    def _latest_structured_objects(self, input_registry: Any, width: int, height: int) -> List[Dict[str, Any]]:
        if not isinstance(input_registry, dict):
            return []
        objects: List[Dict[str, Any]] = []
        latest = input_registry.get("latest_structured")
        if isinstance(latest, dict):
            objects.extend(self._objects_from_structured(latest, "input_registry.latest_structured", width, height))
        for key, value in input_registry.items():
            if key != "latest_structured" and isinstance(key, str) and key.endswith("_structured") and isinstance(value, dict):
                objects.extend(self._objects_from_structured(value, f"input_registry.{key}", width, height))
        return objects

    def _extract_objects(self, params: Dict[str, Any], width: int, height: int) -> List[Dict[str, Any]]:
        labels = self._labels(params.get("labels"))
        objects: List[Dict[str, Any]] = []
        objects.extend(self._objects_from_values(params.get("visual_objects"), "visual_objects", labels, width, height))
        if objects:
            return objects
        objects.extend(self._objects_from_values(params.get("bbox"), "bbox_input", labels, width, height))
        objects.extend(self._objects_from_values(params.get("bboxes"), "bbox_input", labels, width, height))
        objects.extend(self._objects_from_values(params.get("points"), "points_input", labels, width, height))
        objects.extend(self._objects_from_values(params.get("selected_detections"), "selected_detections", labels, width, height))
        objects.extend(self._objects_from_values(params.get("detections"), "detections", labels, width, height))
        objects.extend(self._objects_from_structured(params.get("structured"), "structured", width, height))
        objects.extend(self._latest_structured_objects(params.get("input_registry"), width, height))
        return objects

    @staticmethod
    def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, color: Tuple[int, int, int], font: ImageFont.ImageFont):
        if not text:
            return
        x, y = xy
        y = max(0, y - 14)
        draw.rectangle((x, y, x + max(36, len(text) * 6), y + 13), fill=color)
        draw.text((x + 2, y + 1), text, fill=(255, 255, 255), font=font)

    def _draw_bbox(self, draw: ImageDraw.ImageDraw, bbox: Sequence[float], color: Tuple[int, int, int], thickness: int):
        x1, y1, x2, y2 = bbox[:4]
        for offset in range(max(1, thickness)):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)

    def _draw_point(self, draw: ImageDraw.ImageDraw, point: Sequence[float], color: Tuple[int, int, int], thickness: int):
        x, y = point[:2]
        radius = max(3, thickness * 2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)
        draw.line((x - radius * 2, y, x + radius * 2, y), fill=color, width=max(1, thickness))
        draw.line((x, y - radius * 2, x, y + radius * 2), fill=color, width=max(1, thickness))

    @staticmethod
    def _measurement_text(measurement: Any) -> str:
        if not isinstance(measurement, dict):
            return ""
        parts = []
        for key in ["distance_m", "distance_px", "area_m2", "area_px", "width_m", "height_m", "width_px", "height_px"]:
            if key in measurement:
                value = measurement[key]
                if isinstance(value, (int, float)):
                    parts.append(f"{key}={value:.3f}")
                else:
                    parts.append(f"{key}={value}")
        return ", ".join(parts)

    def _prepare_draw_objects(self, visual_type: str, objects: List[Dict[str, Any]], max_draw: int) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for obj in objects:
            if len(prepared) >= max_draw:
                break
            if visual_type in {"point", "plot"}:
                point = obj.get("point")
                if point is None and obj.get("bbox"):
                    point = self._centroid_from_bbox(obj["bbox"])
                if point is not None:
                    prepared.append({"type": "point", "point": point, "label": obj.get("label"), "source": obj.get("source")})
            else:
                if obj.get("bbox"):
                    prepared.append({"type": "bbox", "bbox": obj["bbox"], "label": obj.get("label"), "source": obj.get("source")})
                elif obj.get("point"):
                    prepared.append({"type": "point", "point": obj["point"], "label": obj.get("label"), "source": obj.get("source")})
        return prepared

    def _output_path(self, image: str, output_path: Any) -> str:
        if isinstance(output_path, str) and output_path.strip():
            path = Path(output_path).expanduser()
            if not path.is_absolute():
                path = Path(os.getcwd()) / path
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        save_dir = Path(os.getcwd()) / "tools_output"
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image).stem or "image"
        return str(save_dir / f"{stem}_object_visual_{uuid.uuid4().hex[:8]}.png")

    def _missing_response(self, visual_type: str, text: str) -> Dict[str, Any]:
        structured = {
            "visual_type": visual_type,
            "num_visual_objects": 0,
            "visual_objects": [],
            "output_path": None,
            "status": "missing_visual_target",
        }
        return {"text": text, "error_code": 0, "image": None, "output_path": None, "structured": structured, "artifacts": []}

    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        image = params.get("image")
        visual_type = str(params.get("visual_type") or "box").strip().lower()
        if not image:
            return {"text": "Missing required parameter: image", "error_code": 2, "image": None, "output_path": None, "structured": {}, "artifacts": []}
        if visual_type not in self.VALID_VISUAL_TYPES:
            return {"text": f"Unsupported visual_type: {visual_type}", "error_code": 2, "image": None, "output_path": None, "structured": {}, "artifacts": []}
        if not os.path.isfile(image):
            return {"text": f"Image not found: {image}", "error_code": 3, "image": None, "output_path": None, "structured": {}, "artifacts": []}

        try:
            thickness = max(1, int(params.get("thickness", 2)))
            max_draw = max(1, int(params.get("max_draw", 20)))
        except Exception as exc:
            return {"text": f"Invalid drawing parameter: {exc}", "error_code": 2, "image": None, "output_path": None, "structured": {}, "artifacts": []}

        canvas = Image.open(image).convert("RGB")
        width, height = canvas.size
        objects = self._extract_objects(params, width, height)
        draw_objects = self._prepare_draw_objects(visual_type, objects, max_draw=max_draw)

        measurement = params.get("measurement")
        structured = params.get("structured") if isinstance(params.get("structured"), dict) else {}
        if not isinstance(measurement, dict) and isinstance(structured, dict):
            measurement = structured.get("measurement")
        if not draw_objects:
            return self._missing_response(visual_type, "No bbox or point was provided for final visualization; no position is fabricated.")

        draw = ImageDraw.Draw(canvas)
        color = self._color(params.get("color", "red"))
        font = ImageFont.load_default()
        visual_objects: List[Dict[str, Any]] = []

        for idx, obj in enumerate(draw_objects, start=1):
            label = str(obj.get("label") or f"#{idx}")
            if obj["type"] == "bbox":
                bbox = obj["bbox"]
                self._draw_bbox(draw, bbox, color, thickness)
                if visual_type in {"box_and_label", "measurement_overlay"}:
                    self._draw_label(draw, (bbox[0], bbox[1]), label, color, font)
                visual_objects.append({"type": "bbox", "bbox": bbox, "label": label if visual_type in {"box_and_label", "measurement_overlay"} else obj.get("label"), "source": obj.get("source")})
            elif obj["type"] == "point":
                point = obj["point"]
                self._draw_point(draw, point, color, thickness)
                if visual_type in {"box_and_label", "measurement_overlay", "plot"}:
                    self._draw_label(draw, (point[0] + 6, point[1] - 6), label, color, font)
                visual_objects.append({"type": "point", "point": point, "label": label if visual_type in {"box_and_label", "measurement_overlay", "plot"} else obj.get("label"), "source": obj.get("source")})

        if visual_type == "measurement_overlay":
            overlay_text = self._measurement_text(measurement)
            if overlay_text:
                draw.rectangle((4, 4, min(width - 1, 10 + len(overlay_text) * 6), 22), fill=color)
                draw.text((8, 8), overlay_text, fill=(255, 255, 255), font=font)
                visual_objects.append({"type": "text", "label": overlay_text, "source": "measurement"})

        output_path = self._output_path(image, params.get("output_path"))
        canvas.save(output_path)
        structured_resp = {
            "visual_type": visual_type,
            "num_visual_objects": len([obj for obj in visual_objects if obj.get("type") in {"bbox", "point"}]),
            "visual_objects": visual_objects,
            "output_path": output_path,
            "status": "success",
        }
        artifacts = [
            {
                "id": "visual_output",
                "type": "image",
                "path": output_path,
                "name": os.path.basename(output_path),
                "source": self.model_name,
            }
        ]
        return {
            "text": f"Created final visualization with {structured_resp['num_visual_objects']} object(s).",
            "error_code": 0,
            "image": output_path,
            "output_path": output_path,
            "structured": structured_resp,
            "artifacts": artifacts,
        }

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": "Final visualization layer for drawing existing boxes, points, labels, plots, or measurement overlays.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Path to the input image."},
                        "visual_type": {"type": "string", "description": "box, point, plot, box_and_label, or measurement_overlay."},
                        "bbox": {"type": "array", "description": "Single bbox [x1,y1,x2,y2]."},
                        "bboxes": {"type": "array", "description": "List of bboxes."},
                        "points": {"type": "array", "description": "List of points or one point."},
                        "detections": {"type": "array", "description": "Detection objects with bbox fields."},
                        "selected_detections": {"type": "array", "description": "Selected detection objects with bbox fields."},
                        "labels": {"type": "array", "description": "Labels aligned with boxes or points."},
                        "measurement": {"type": "object", "description": "Existing measurement values for overlay text."},
                        "structured": {"type": "object", "description": "Previous skill structured output containing visual targets."},
                        "input_registry": {"type": "object", "description": "Runtime registry with latest structured outputs."},
                        "visual_objects": {"type": "array", "description": "Pre-normalized visual target objects with bbox or point fields."},
                        "output_path": {"type": "string", "description": "Optional output image path."},
                        "color": {"type": "string", "description": "Color name or #RRGGBB."},
                        "thickness": {"type": "integer", "description": "Line thickness."},
                        "max_draw": {"type": "integer", "description": "Maximum visual objects to draw."},
                    },
                    "required": ["image", "visual_type"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20129)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="ObjectVisualOutputSkill")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    worker = ObjectVisualOutputSkillWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
    )
    worker.run()
