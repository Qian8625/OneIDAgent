import argparse
import base64
import importlib
import os
import sys
import tempfile
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker
from tool_server.utils.server_utils import build_logger
from tool_server.utils.paths import env_path


worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"denodetv2_worker_{worker_id}.log")


class DenoDetV2Worker(BaseToolWorker):
    """DenoDetV2 SAR target detection worker.

    The SARDet-100K checkpoint config in this repository was exported with
    `sareye.*` module names. Runtime config patching maps those names to the
    local `groksar.*` package used by DenoDetV2.
    """

    DEFAULT_CLASSES = ("ship", "aircraft", "car", "tank", "bridge", "harbor")
    SAREYE_TO_GROKSAR = {
        "sareye.FFTResNet": "groksar.FFTResNet",
        "sareye.SAR_Det_Finegrained_Dataset": "groksar.SAR_Det_Finegrained_Dataset",
        "sareye.CocoShowMetric": "groksar.CocoShowMetric",
    }

    def __init__(
        self,
        controller_addr,
        worker_addr="auto",
        worker_id=worker_id,
        no_register=False,
        model_path=env_path("DENO_DET_V2_MODEL_PATH", "${LOCAL_MODEL_ROOT}/cpk/SARDet-100K/best_coco_bbox_mAP_epoch_12.pth"),
        model_config=env_path("DENO_DET_V2_CONFIG_PATH", "${LOCAL_MODEL_ROOT}/cpk/SARDet-100K/gfl_fft_attention_r50_fpn_1x_SARDet_100k.py"),
        model_name="DenoDetV2",
        load_8bit=False,
        load_4bit=False,
        device="cuda",
        limit_model_concurrency=2,
        host="0.0.0.0",
        port=None,
        model_semaphore=None,
        wait_timeout=180.0,
        task_timeout=180.0,
        denodet_code_dir=env_path("DENO_DET_V2_CODE_DIR", "${LOCAL_MODEL_ROOT}/DenoDetV2"),
        default_score_thr=0.30,
    ):
        self.model_path = model_path
        self.model_config = model_config
        self.denodet_code_dir = denodet_code_dir
        self.default_score_thr = float(default_score_thr)
        self.model = None
        self.cfg = None
        self.classes: Tuple[str, ...] = self.DEFAULT_CLASSES

        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            self.model_path,
            None,
            model_name,
            load_8bit,
            load_4bit,
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            wait_timeout,
            task_timeout,
        )

    def init_model(self):
        logger.info(f"Initializing DenoDetV2 model {self.model_name}...")
        logger.info(f"CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")

        if not os.path.exists(self.model_config):
            raise FileNotFoundError(f"model_config not found: {self.model_config}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model_path not found: {self.model_path}")

        self._register_groksar_modules()
        self._patch_torch_load_weights_only()

        self.cfg = Config.fromfile(self.model_config)
        self.cfg = self._patch_config_for_groksar(self.cfg)
        self._disable_training_only_config(self.cfg)

        logger.info(f"Loading DenoDetV2 config={self.model_config}")
        logger.info(f"Loading DenoDetV2 checkpoint={self.model_path}")

        self.model = init_detector(self.cfg, self.model_path, device=self.device)
        self.classes = self._infer_classes(self.model)
        self._apply_classes_to_model(self.classes)

        logger.info(f"DenoDetV2 initialized successfully. classes={self.classes}")

    def _register_groksar_modules(self):
        if self.denodet_code_dir and os.path.isdir(self.denodet_code_dir):
            if self.denodet_code_dir not in sys.path:
                sys.path.insert(0, self.denodet_code_dir)

        for module_name in (
            "groksar",
            "groksar.models",
            "groksar.models.backbones.FFTresnet",
            "groksar.datasets.SAR_Det",
        ):
            try:
                importlib.import_module(module_name)
                logger.info(f"Imported DenoDetV2 module: {module_name}")
            except Exception as exc:
                logger.warning(f"Failed to import DenoDetV2 module {module_name}: {exc}")

    def _patch_torch_load_weights_only(self):
        if getattr(torch.load, "_denodetv2_weights_only_patched", False):
            return

        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        patched_load._denodetv2_weights_only_patched = True
        torch.load = patched_load

    def _patch_config_for_groksar(self, cfg: Config) -> Config:
        cfg_dict = self._replace_sareye_references(cfg.to_dict())
        return Config(cfg_dict)

    def _replace_sareye_references(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._replace_sareye_references(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._replace_sareye_references(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._replace_sareye_references(item) for item in value)
        if isinstance(value, str):
            return self.SAREYE_TO_GROKSAR.get(value, value)
        return value

    def _disable_training_only_config(self, cfg: Config):
        try:
            if hasattr(cfg, "model") and "backbone" in cfg.model:
                cfg.model.backbone.pretrained = None
        except Exception:
            logger.warning("Could not set cfg.model.backbone.pretrained=None; continuing.")

        try:
            cfg.model.train_cfg = None
        except Exception:
            pass

    def _infer_classes(self, model: Any) -> Tuple[str, ...]:
        dataset_meta = getattr(model, "dataset_meta", None)
        if isinstance(dataset_meta, dict) and dataset_meta.get("classes"):
            return tuple(dataset_meta["classes"])

        model_classes = getattr(model, "CLASSES", None)
        if model_classes:
            return tuple(model_classes)

        return self.DEFAULT_CLASSES

    def _apply_classes_to_model(self, classes: Sequence[str]):
        if self.model is None or not classes:
            return
        classes = tuple(classes)
        try:
            self.model.CLASSES = classes
        except Exception:
            pass
        try:
            if not hasattr(self.model, "dataset_meta") or self.model.dataset_meta is None:
                self.model.dataset_meta = {}
            self.model.dataset_meta["classes"] = classes
        except Exception:
            pass

    def _parse_classes_param(self, value: Any) -> Optional[Tuple[str, ...]]:
        if value is None:
            return None
        if isinstance(value, str):
            separator = "." if "." in value else "," if "," in value else None
            parts = value.split(separator) if separator else value.split()
            out = tuple(part.strip() for part in parts if part.strip())
            return out or None
        if isinstance(value, (list, tuple)):
            out = tuple(str(item).strip() for item in value if str(item).strip())
            return out or None
        return None

    def _image_to_path(self, image_input: str) -> Tuple[str, Optional[str]]:
        if not isinstance(image_input, str) or not image_input.strip():
            raise ValueError("image must be a non-empty string path or base64 image")

        if os.path.exists(image_input):
            return image_input, None

        raw = image_input
        if "," in raw and raw.lstrip().lower().startswith("data:image"):
            raw = raw.split(",", 1)[1]

        try:
            image = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
        except Exception as exc:
            raise FileNotFoundError(f"Image path does not exist and base64 decoding failed: {exc}") from exc

        fd, tmp_path = tempfile.mkstemp(prefix="denodetv2_", suffix=".png")
        os.close(fd)
        image.save(tmp_path)
        return tmp_path, tmp_path

    def _normalize_result(self, result: Any) -> List[Dict[str, Any]]:
        if not hasattr(result, "pred_instances"):
            raise TypeError(f"Unsupported DenoDetV2 inference result type: {type(result)}")

        pred = result.pred_instances
        bboxes = pred.bboxes.detach().cpu().numpy() if hasattr(pred, "bboxes") else np.empty((0, 4))
        scores = pred.scores.detach().cpu().numpy() if hasattr(pred, "scores") else np.empty((0,))
        labels = pred.labels.detach().cpu().numpy() if hasattr(pred, "labels") else np.zeros((len(scores),), dtype=np.int64)

        detections = []
        for bbox, score, label in zip(bboxes, scores, labels):
            detections.append(
                {
                    "class_id": int(label),
                    "bbox": np.asarray(bbox, dtype=float).reshape(-1).tolist(),
                    "score": float(score),
                }
            )
        return detections

    def _format_bbox(self, bbox: Sequence[float]) -> Tuple[str, List[float]]:
        arr = np.asarray(bbox, dtype=float).reshape(-1)
        if arr.size < 4:
            raise ValueError(f"Unsupported bbox dimension: {arr.size}")

        x1, y1, x2, y2 = arr[:4].tolist()
        values = [float(x1), float(y1), float(x2), float(y2)]
        text = f"({int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))})"
        return text, values

    @torch.inference_mode()
    def generate(self, params):
        if "image" not in params:
            txt_e = "Missing required parameter: image"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        image_input = params.get("image")
        box_threshold = float(params.get("box_threshold", self.default_score_thr))
        top1 = bool(params.get("top1", False))
        return_json = bool(params.get("return_json", False))
        override_classes = self._parse_classes_param(params.get("classes"))
        classes = override_classes or self.classes or self.DEFAULT_CLASSES
        if override_classes:
            self._apply_classes_to_model(classes)

        tmp_path = None
        try:
            image_path, tmp_path = self._image_to_path(image_input)
            logger.info(f"Running DenoDetV2 on image={image_path}")
            logger.info(f"box_threshold={box_threshold}, top1={top1}")

            result = inference_detector(self.model, image_path)
            detections = self._normalize_result(result)
            detections = [det for det in detections if det["score"] >= box_threshold]

            if not detections:
                txt_e = "no detections found after thresholding"
                logger.error(txt_e)
                response = {"text": txt_e, "error_code": 4}
                if return_json:
                    response["data"] = []
                return response

            detections.sort(key=lambda item: item["score"], reverse=True)
            if top1:
                detections = detections[:1]

            formatted_lines = []
            formatted_data = []
            for det in detections:
                class_id = int(det["class_id"])
                label = classes[class_id] if 0 <= class_id < len(classes) else "object"
                bbox_text, bbox = self._format_bbox(det["bbox"])
                score = float(det["score"])

                formatted_lines.append(f"{bbox_text}, {label}, score {score:.2f}")
                formatted_data.append(
                    {
                        "class_id": class_id,
                        "class_name": label,
                        "bbox": bbox,
                        "bbox_format": "xyxy",
                        "score": score,
                    }
                )

            response = {"text": "\n".join(formatted_lines), "error_code": 0}
            if return_json:
                response["data"] = formatted_data
            return response
        except Exception as exc:
            txt_e = f"Error in DenoDetV2: {exc}"
            logger.exception(txt_e)
            return {"text": txt_e, "error_code": 1}
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "DenoDetV2",
                "description": (
                    "Detects SAR targets using DenoDetV2 trained on SARDet-100K. "
                    "Returns horizontal boxes for ship, aircraft, car, tank, bridge, and harbor."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Image path on disk, or a base64-encoded image string.",
                        },
                        "box_threshold": {
                            "type": "number",
                            "description": "Score threshold for filtering detections. Default is 0.30.",
                        },
                        "top1": {
                            "type": "boolean",
                            "description": "If true, return only the highest-confidence detection.",
                        },
                        "classes": {
                            "type": "string",
                            "description": (
                                "Optional class-name override. Use dot-separated, comma-separated, "
                                "or space-separated names."
                            ),
                        },
                        "return_json": {
                            "type": "boolean",
                            "description": "If true, include structured detection data under `data`.",
                        },
                    },
                    "required": ["image"],
                },
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20034)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://localhost:20001")
    parser.add_argument("--model-path", type=str,
                        default=env_path("DENO_DET_V2_MODEL_PATH", "${LOCAL_MODEL_ROOT}/cpk/SARDet-100K/best_coco_bbox_mAP_epoch_12.pth"))
    parser.add_argument("--model-config", type=str,
                        default=env_path("DENO_DET_V2_CONFIG_PATH", "${LOCAL_MODEL_ROOT}/cpk/SARDet-100K/gfl_fft_attention_r50_fpn_1x_SARDet_100k.py"))
    parser.add_argument("--denodet-code-dir", type=str,
                        default=env_path("DENO_DET_V2_CODE_DIR", "${LOCAL_MODEL_ROOT}/DenoDetV2"))
    parser.add_argument("--model-name", type=str, default="DenoDetV2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=2)
    parser.add_argument("--default-score-thr", type=float, default=0.30)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = DenoDetV2Worker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        no_register=args.no_register,
        model_path=args.model_path,
        model_config=args.model_config,
        model_name=args.model_name,
        device=args.device,
        limit_model_concurrency=args.limit_model_concurrency,
        host=args.host,
        port=args.port,
        denodet_code_dir=args.denodet_code_dir,
        default_score_thr=args.default_score_thr,
    )
    worker.run()
