import argparse
import uuid
import os
import re
import base64
from io import BytesIO
from typing import Optional, Tuple
from PIL import Image

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger


def parse_multi_float(input_str: str, number: Optional[int] = None) -> Tuple[float, ...]:
    pattern = r'([-+]?\d*\.?\d+)'
    matches = re.findall(pattern, input_str)
    if number is not None and len(matches) != number:
        raise ValueError(f'Expected {number} numbers, got {input_str}.')
    return tuple(float(num) for num in matches)


def normalize_bbox(bbox) -> Tuple[float, float, float, float]:
    if isinstance(bbox, (list, tuple)):
        if len(bbox) != 4:
            raise ValueError(f"Expected 4 bbox values, got {bbox}.")
        values = tuple(float(value) for value in bbox)
    elif isinstance(bbox, str):
        if "<" in bbox or ">" in bbox:
            raise ValueError(f"Invalid bbox placeholder: {bbox}.")
        values = parse_multi_float(bbox, 4)
    else:
        raise ValueError(f"Invalid bbox type: {type(bbox).__name__}.")
    return values


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"crop_region_worker_{worker_id}.log")


class CropRegionWorker(BaseToolWorker):
    """Crop an arbitrary pixel region out of an image and expose it as a reusable image artifact."""

    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="CropRegion",
                 host="0.0.0.0",
                 port=None,
                 limit_model_concurrency=5,
                 model_semaphore=None,
                 save_path=None,
                 wait_timeout=120.0,
                 task_timeout=30.0,
                 ):
        self.save_path = save_path
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,
            None,
            model_name,
            False,
            False,
            "cpu",
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info("CropRegionWorker does not need a model. Ready to run.")
        if self.save_path and os.path.isdir(self.save_path):
            logger.info(f"Outputs will be saved to: {self.save_path}")

    def _load_image(self, image_path: str) -> Image.Image:
        if os.path.exists(image_path):
            return Image.open(image_path).convert("RGB")
        return Image.open(BytesIO(base64.b64decode(image_path))).convert("RGB")

    @staticmethod
    def _resolve_output_dir(worker_save_path: Optional[str], params) -> str:
        param_save_path = params.get("save_path") or params.get("output_dir")
        if param_save_path and os.path.isdir(param_save_path):
            return param_save_path
        if worker_save_path and os.path.isdir(worker_save_path):
            return worker_save_path
        save_dir = os.path.join(os.getcwd(), "tools_output")
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def generate(self, params):
        try:
            if "image" not in params:
                txt_e = "Missing required parameter: image"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 2}
            if "bbox" not in params:
                txt_e = "Missing required parameter: bbox"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 2}

            image = params.get("image")
            bbox = params.get("bbox")
            margin = float(params.get("margin", 0.0) or 0.0)
            scale = float(params.get("scale", 1.0) or 1.0)

            if margin < 0 or margin > 0.5:
                txt_e = "margin must be in [0, 0.5]"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}
            if scale <= 0 or scale > 8:
                txt_e = "scale must be in (0, 8]"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            try:
                img = self._load_image(image)
            except Exception as exc:
                txt_e = f"Failed to load image: {exc}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}

            try:
                x1, y1, x2, y2 = normalize_bbox(bbox)
            except Exception as exc:
                txt_e = f"Invalid bbox format: {bbox}. Expected four numeric values '(x1,y1,x2,y2)'. {exc}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            W, H = img.size
            if margin:
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                dx, dy = margin * bw, margin * bh
                x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy

            x1 = max(0.0, min(float(x1), float(W)))
            y1 = max(0.0, min(float(y1), float(H)))
            x2 = max(0.0, min(float(x2), float(W)))
            y2 = max(0.0, min(float(y2), float(H)))

            ix1, iy1 = int(round(x1)), int(round(y1))
            ix2, iy2 = int(round(x2)), int(round(y2))
            if not (ix2 > ix1 and iy2 > iy1):
                txt_e = f"Invalid bbox region: {(ix1, iy1, ix2, iy2)} must satisfy x2>x1 and y2>y1 within image size {(W, H)}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 4}

            crop = img.crop((ix1, iy1, ix2, iy2))
            cw, ch = crop.size
            if scale != 1.0:
                crop = crop.resize((max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))), Image.LANCZOS)

            save_dir = self._resolve_output_dir(self.save_path, params)
            stem = os.path.splitext(os.path.basename(str(image).split("?")[0]))[0] or "image"
            new_filename = f"{stem}_crop_{ix1}_{iy1}_{ix2}_{iy2}_s{scale:g}.png"
            save_path = os.path.join(save_dir, new_filename)
            crop.save(save_path)

            txt = (f"Cropped region ({ix1}, {iy1}, {ix2}, {iy2}) from a {W}x{H} image, "
                   f"output size {crop.size[0]}x{crop.size[1]}, scale {scale:g}. "
                   f"Reuse it with image=$latest_image. "
                   f"Map crop coordinates back with x_global = x_crop / {scale:g} + {ix1}, y_global = y_crop / {scale:g} + {iy1}.")
            return {
                "text": txt,
                "image": save_path,
                "error_code": 0,
                "probe": {
                    "bbox": [ix1, iy1, ix2, iy2],
                    "offset": [ix1, iy1],
                    "scale": scale,
                    "source_image": (image if len(str(image)) <= 300 else str(image)[:300] + f"...(len={len(str(image))})"),
                    "source_size": [W, H],
                    "crop_size": list(crop.size),
                },
            }
        except Exception as exc:
            txt_e = f"Error in CropRegion: {exc}"
            logger.exception(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "CropRegion",
                "description": (
                    "Crop an arbitrary pixel region out of an image and return it as a reusable image "
                    "artifact ($latest_image). Use it to re-inspect a region with a detector, e.g. crop the "
                    "region paired with a detection from the other view, then run DenoDetV2 or "
                    "SmallObjectDetection on $latest_image."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string", "description": "Path, img_1/img_2 or $artifact reference."},
                        "bbox": {"type": "string", "description": "Pixel box in format '(x1, y1, x2, y2)'."},
                        "margin": {"type": "number", "description": "Optional normalized margin (0-0.5) expanded around the box."},
                        "scale": {"type": "number", "description": "Optional upscale factor in (0, 8], default 1."},
                    },
                    "required": ["image", "bbox"],
                    "optional": ["margin", "scale"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20040)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-name", type=str, default="CropRegion")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = CropRegionWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_name=args.model_name,
        limit_model_concurrency=args.limit_model_concurrency,
        save_path=args.save_path,
        host=args.host,
        port=args.port,
    )
    worker.run()
