"""
A model worker executes the Change Detection model (TEOChat).
"""
import os
import argparse
import re
import tempfile
import traceback
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from tool_server.utils.utils import *
from tool_server.utils.server_utils import *
from tool_server.tool_workers.online_workers.base_tool_worker import BaseToolWorker

# worker id and logger
worker_id = str(uuid.uuid4())[:6]
logger = build_logger(__file__, f"{__file__}_{worker_id}.log")


class ChangeDetectionWorker(BaseToolWorker):
    TARGET_SYNONYMS = {
        "airplane": [
            "airplane", "aircraft", "plane",
            "飞机", "航空器",
        ],
        "bridge": [
            "bridge",
            "桥", "桥梁",
        ],
        "harbor": [
            "harbor", "harbour", "port",
            "港口", "港湾", "港区",
        ],
        "ship": [
            "ship", "boat", "vessel",
            "船", "船只", "舰船",
        ],
        "tank": [
            "tank", "storage tank",
            "储罐", "油罐", "储油罐",
        ],
        "vehicle": [
            "vehicle", "car", "small vehicle", "large vehicle",
            "汽车", "车辆",
        ],
    }

    TARGET_TO_DETECTOR_TEXT = {
        "airplane": "airplane",
        "bridge": "bridge",
        "harbor": "harbor",
        "ship": "ship",
        "tank": "tank",
        "vehicle": "vehicle",
    }

    TARGET_DISPLAY = {
        "airplane": {"zh": ("飞机", "架"), "en": ("airplane", "airplanes")},
        "ship": {"zh": ("船只", "艘"), "en": ("ship", "ships")},
        "tank": {"zh": ("储罐", "个"), "en": ("tank", "tanks")},
        "vehicle": {"zh": ("车辆", "辆"), "en": ("vehicle", "vehicles")},
    }

    TARGET_ACTION_TERMS = [
        "原位", "离开", "还在", "消失", "新增", "出现", "移动", "位置",
        "数量", "增加", "减少", "变化",
        "original position", "still there", "leave", "left", "disappear",
        "disappeared", "appear", "appeared", "new", "newly", "move",
        "moved", "relocate", "relocated", "position", "count", "increase",
        "decrease", "change",
    ]

    SCENE_TERMS = [
        "建筑", "房屋", "设施", "道路", "土地覆盖", "灾害", "损毁", "扩建",
        "整体", "主要变化",
        "building", "buildings", "house", "houses", "facility", "facilities",
        "road", "roads", "land cover", "disaster", "damage", "damaged",
        "destroyed", "expansion", "overall", "major changes",
    ]

    WIDE_CHANGE_TERMS = [
        "有什么变化", "哪些变化", "主要变化", "变化情况",
        "what changed", "what changes", "major changes", "describe changes",
        "summarize changes",
    ]

    REGION_SYNONYMS = [
        ("top_left", ["top left", "upper left", "top left corner", "upper left corner", "左上角", "左上"]),
        ("top_right", ["top right", "upper right", "top right corner", "upper right corner", "右上角", "右上"]),
        ("bottom_left", ["bottom left", "lower left", "bottom left corner", "lower left corner", "左下角", "左下"]),
        ("bottom_right", ["bottom right", "lower right", "bottom right corner", "lower right corner", "右下角", "右下"]),
        ("center", ["center", "centre", "middle", "central", "中间", "中心", "中央"]),
        ("top", ["top", "upper", "top half", "upper half", "上方", "上半部分", "上部"]),
        ("bottom", ["bottom", "lower", "bottom half", "lower half", "下方", "下半部分", "下部"]),
        ("left", ["left", "left side", "left half", "左侧", "左半部分"]),
        ("right", ["right", "right side", "right half", "右侧", "右半部分"]),
    ]

    REGION_DISPLAY = {
        "top_left": {"zh": "左上角区域", "en": "top-left region"},
        "top_right": {"zh": "右上角区域", "en": "top-right region"},
        "bottom_left": {"zh": "左下角区域", "en": "bottom-left region"},
        "bottom_right": {"zh": "右下角区域", "en": "bottom-right region"},
        "top": {"zh": "上方区域", "en": "top region"},
        "bottom": {"zh": "下方区域", "en": "bottom region"},
        "left": {"zh": "左侧区域", "en": "left region"},
        "right": {"zh": "右侧区域", "en": "right region"},
        "center": {"zh": "中心区域", "en": "center region"},
    }

    def __init__(self,
                 controller_addr,
                 worker_addr="auto",
                 worker_id=worker_id,
                 no_register=False,
                 model_name="ChangeDetection",
                 model_base="jirvin16/TEOChat",
                 device="cuda",
                 limit_model_concurrency=5,
                 host="0.0.0.0",
                 port=None,
                 model_semaphore=None,
                 args=None):
        self.tool_addr_cache: Dict[str, str] = {}
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            no_register,
            None,   # model path/name
            model_base,
            model_name,
            False,   # load_8bit handled in load_model
            False,   # load_4bit not used here
            device,
            limit_model_concurrency,
            host,
            port,
            model_semaphore,
            args=args
        )
        self.device = device

    @staticmethod
    def _normalize_text(text: Optional[str]) -> str:
        return " ".join((text or "").strip().lower().replace("_", " ").split())

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))

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
        return ret.json()

    def _resolve_target(self, query_text: Optional[str], target: Optional[str]) -> Optional[str]:
        raw_text = self._normalize_text(f"{target or ''} {query_text or ''}")
        if not raw_text:
            return None

        for canonical, synonyms in self.TARGET_SYNONYMS.items():
            for synonym in synonyms:
                normalized_synonym = self._normalize_text(synonym)
                if normalized_synonym and normalized_synonym in raw_text:
                    return canonical
        return None

    def _resolve_region(self, query_text: Optional[str], region: Optional[str]) -> Optional[str]:
        def normalize_region_text(text: Optional[str]) -> str:
            raw = (text or "").strip().lower()
            raw = raw.replace("_", " ").replace("-", " ").replace("/", " ")
            return " ".join(raw.split())

        def find_region(raw_text: str, allow_ambiguous_single_words: bool) -> Optional[str]:
            ambiguous_single_words = {"top", "upper", "bottom", "lower", "left", "right"}
            if not raw_text:
                return None
            for canonical, synonyms in self.REGION_SYNONYMS:
                if canonical.replace("_", " ") == raw_text:
                    return canonical
                for synonym in synonyms:
                    normalized_synonym = normalize_region_text(synonym)
                    if (
                        not allow_ambiguous_single_words
                        and normalized_synonym in ambiguous_single_words
                    ):
                        continue
                    if normalized_synonym and normalized_synonym in raw_text:
                        return canonical
            return None

        explicit_region = find_region(
            normalize_region_text(region),
            allow_ambiguous_single_words=True,
        )
        if explicit_region:
            return explicit_region

        raw_text = normalize_region_text(query_text)
        return find_region(raw_text, allow_ambiguous_single_words=False)

    def _resolve_mode(self, query_text: str, requested_mode: str, target: Optional[str]) -> str:
        mode = self._normalize_text(requested_mode or "auto")
        if mode in {"scene", "target", "hybrid"}:
            return mode
        if mode != "auto":
            raise ValueError(f"Unsupported mode: {requested_mode}")

        normalized_query = self._normalize_text(query_text)
        canonical_target = self._resolve_target(query_text, target)
        has_target_action = any(term in normalized_query for term in self.TARGET_ACTION_TERMS)
        has_scene_term = any(term in normalized_query for term in self.SCENE_TERMS)
        has_wide_change_term = any(term in normalized_query for term in self.WIDE_CHANGE_TERMS)

        if canonical_target and has_target_action:
            return "target"
        if has_scene_term:
            return "scene"
        if has_wide_change_term:
            return "hybrid"
        if canonical_target:
            return "target"
        return "scene"

    def init_model(self):
        logger.info(f"Initializing ChangeDetection model: {self.model_base}")
        try:
            from videollava.eval.eval import load_model
            self.tokenizer, self.model, self.processor = load_model(
                model_path=self.model_base,
                model_base=None,
                load_8bit=True,
                device=self.device
            )
            print(f"[DEBUG] ChangeDetection model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing ChangeDetection model: {e}")
            logger.error(traceback.format_exc())
            raise

    def _run_scene_flow(self, pre_image: str, post_image: str, query_text: str) -> Dict[str, Any]:
        from videollava.eval.inference import run_inference_single

        query = "These are two satellite images in chronological order: <video> " + query_text
        txt = run_inference_single(
            self.model, self.processor, self.tokenizer, query, [pre_image, post_image]
        )
        return {"text": txt, "error_code": 0, "mode_used": "scene"}

    def _detect_small_objects(
        self,
        image: str,
        canonical_target: str,
        box_threshold: float,
        crop_box: Optional[List[int]] = None,
    ) -> Tuple[List[Dict[str, Any]], Tuple[int, int], Dict[str, Any]]:
        detector_image = image
        cleanup_path: Optional[str] = None
        x_offset = 0
        y_offset = 0
        original_size = self._get_image_size(image)

        if crop_box is not None:
            detector_image, x_offset, y_offset = self._crop_image_to_temp(image, crop_box)
            cleanup_path = detector_image

        detector_text = self.TARGET_TO_DETECTOR_TEXT[canonical_target]
        try:
            resp = self._call_tool(
                "SmallObjectDetection",
                {
                    "image": detector_image,
                    "text": detector_text,
                    "score_threshold": box_threshold,
                },
                timeout=240,
            )
            if resp.get("error_code") != 0:
                raise RuntimeError(f"SmallObjectDetection failed: {resp.get('text', '')}")

            detections = resp.get("detections", [])
            if crop_box is not None:
                detections = self._offset_detections(detections, x_offset, y_offset)
            meta = resp.get("meta", {})
            return detections, original_size, meta
        finally:
            if cleanup_path:
                try:
                    os.unlink(cleanup_path)
                except OSError:
                    logger.warning(f"Failed to remove temporary ROI crop: {cleanup_path}")

    @staticmethod
    def _get_image_size(image_path: str) -> Tuple[int, int]:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size

    @staticmethod
    def _region_to_norm_box(region: str) -> List[float]:
        boxes = {
            "top_left": [0.0, 0.0, 0.5, 0.5],
            "top_right": [0.5, 0.0, 1.0, 0.5],
            "bottom_left": [0.0, 0.5, 0.5, 1.0],
            "bottom_right": [0.5, 0.5, 1.0, 1.0],
            "top": [0.0, 0.0, 1.0, 0.5],
            "bottom": [0.0, 0.5, 1.0, 1.0],
            "left": [0.0, 0.0, 0.5, 1.0],
            "right": [0.5, 0.0, 1.0, 1.0],
            "center": [0.25, 0.25, 0.75, 0.75],
        }
        if region not in boxes:
            raise ValueError(f"Unsupported region: {region}")
        return boxes[region]

    @staticmethod
    def _expand_norm_box(norm_box: List[float], margin_ratio: float) -> List[float]:
        x1, y1, x2, y2 = norm_box
        return [
            max(0.0, x1 - margin_ratio),
            max(0.0, y1 - margin_ratio),
            min(1.0, x2 + margin_ratio),
            min(1.0, y2 + margin_ratio),
        ]

    @staticmethod
    def _norm_box_to_pixel_box(norm_box: List[float], image_size: Tuple[int, int]) -> List[int]:
        width, height = image_size
        x1, y1, x2, y2 = norm_box
        pixel_box = [
            max(0, min(int(round(x1 * width)), width)),
            max(0, min(int(round(y1 * height)), height)),
            max(0, min(int(round(x2 * width)), width)),
            max(0, min(int(round(y2 * height)), height)),
        ]
        if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
            raise ValueError(f"Invalid ROI crop box {pixel_box} for image size {image_size}")
        return pixel_box

    @staticmethod
    def _crop_image_to_temp(image_path: str, crop_box: List[int]) -> Tuple[str, int, int]:
        from PIL import Image

        x1, y1, x2, y2 = crop_box
        with Image.open(image_path) as img:
            crop = img.convert("RGB").crop((x1, y1, x2, y2))
            tmp = tempfile.NamedTemporaryFile(
                prefix="change_detection_roi_",
                suffix=".png",
                delete=False,
            )
            tmp_path = tmp.name
            tmp.close()
            crop.save(tmp_path)
        return tmp_path, x1, y1

    @staticmethod
    def _offset_detections(
        detections: List[Dict[str, Any]],
        x_offset: int,
        y_offset: int,
    ) -> List[Dict[str, Any]]:
        shifted: List[Dict[str, Any]] = []
        for det in detections:
            bbox = det.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            shifted.append(
                {
                    **det,
                    "bbox": [
                        int(x1) + x_offset,
                        int(y1) + y_offset,
                        int(x2) + x_offset,
                        int(y2) + y_offset,
                    ],
                }
            )
        return shifted

    @staticmethod
    def _bbox_center_norm(bbox: List[int], image_size: Tuple[int, int]) -> Tuple[float, float]:
        width, height = image_size
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / (2.0 * max(width, 1)), (y1 + y2) / (2.0 * max(height, 1)))

    @staticmethod
    def _bbox_to_norm_box(bbox: List[int], image_size: Tuple[int, int]) -> List[float]:
        width, height = image_size
        x1, y1, x2, y2 = bbox
        return [
            max(0.0, min(x1 / max(width, 1), 1.0)),
            max(0.0, min(y1 / max(height, 1), 1.0)),
            max(0.0, min(x2 / max(width, 1), 1.0)),
            max(0.0, min(y2 / max(height, 1), 1.0)),
        ]

    @staticmethod
    def _bbox_area_norm(norm_box: List[float]) -> float:
        x1, y1, x2, y2 = norm_box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @classmethod
    def _bbox_iou_norm(cls, left_box: List[float], right_box: List[float]) -> float:
        lx1, ly1, lx2, ly2 = left_box
        rx1, ry1, rx2, ry2 = right_box
        ix1 = max(lx1, rx1)
        iy1 = max(ly1, ry1)
        ix2 = min(lx2, rx2)
        iy2 = min(ly2, ry2)
        inter = cls._bbox_area_norm([ix1, iy1, ix2, iy2])
        union = cls._bbox_area_norm(left_box) + cls._bbox_area_norm(right_box) - inter
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _bbox_to_norm_region(
        bbox: List[int],
        image_size: Tuple[int, int],
        tolerance_ratio: float,
    ) -> List[float]:
        width, height = image_size
        x1, y1, x2, y2 = bbox
        return [
            max(0.0, x1 / max(width, 1) - tolerance_ratio),
            max(0.0, y1 / max(height, 1) - tolerance_ratio),
            min(1.0, x2 / max(width, 1) + tolerance_ratio),
            min(1.0, y2 / max(height, 1) + tolerance_ratio),
        ]

    @staticmethod
    def _norm_region_to_pixels(region: List[float], image_size: Tuple[int, int]) -> List[int]:
        width, height = image_size
        x1, y1, x2, y2 = region
        return [
            int(round(x1 * width)),
            int(round(y1 * height)),
            int(round(x2 * width)),
            int(round(y2 * height)),
        ]

    @staticmethod
    def _center_in_region(center: Tuple[float, float], region: List[float]) -> bool:
        cx, cy = center
        x1, y1, x2, y2 = region
        return x1 <= cx <= x2 and y1 <= cy <= y2

    @staticmethod
    def _grid_cell(center: Tuple[float, float]) -> str:
        cx, cy = center
        col = 0 if cx < 1 / 3 else 1 if cx < 2 / 3 else 2
        row = 0 if cy < 1 / 3 else 1 if cy < 2 / 3 else 2
        names = [
            ["top_left", "top", "top_right"],
            ["left", "center", "right"],
            ["bottom_left", "bottom", "bottom_right"],
        ]
        return names[row][col]

    def _summarize_spatial_distribution(
        self,
        detections: List[Dict[str, Any]],
        image_size: Tuple[int, int],
    ) -> Dict[str, int]:
        distribution: Dict[str, int] = {}
        for det in detections:
            center = self._bbox_center_norm(det["bbox"], image_size)
            cell = self._grid_cell(center)
            distribution[cell] = distribution.get(cell, 0) + 1
        return distribution

    def _evaluate_location_anchored_changes(
        self,
        pre_dets: List[Dict[str, Any]],
        post_dets: List[Dict[str, Any]],
        pre_size: Tuple[int, int],
        post_size: Tuple[int, int],
        tolerance_ratio: float,
        iou_threshold: float,
        min_area_ratio: float,
        max_area_ratio: float,
    ) -> Dict[str, Any]:
        post_centers = [
            self._bbox_center_norm(det["bbox"], post_size)
            for det in post_dets
        ]
        pre_norm_boxes = [
            self._bbox_to_norm_box(det["bbox"], pre_size)
            for det in pre_dets
        ]
        post_norm_boxes = [
            self._bbox_to_norm_box(det["bbox"], post_size)
            for det in post_dets
        ]
        pre_anchor_regions = [
            self._bbox_to_norm_region(det["bbox"], pre_size, tolerance_ratio)
            for det in pre_dets
        ]

        candidates: List[Dict[str, Any]] = []
        for pre_idx, (pre_det, pre_box, anchor_region) in enumerate(
            zip(pre_dets, pre_norm_boxes, pre_anchor_regions)
        ):
            pre_area = self._bbox_area_norm(pre_box)
            for post_idx, (post_det, post_box, post_center) in enumerate(
                zip(post_dets, post_norm_boxes, post_centers)
            ):
                iou = self._bbox_iou_norm(pre_box, post_box)
                within_tolerance = self._center_in_region(post_center, anchor_region)
                if iou <= 0 and not within_tolerance:
                    continue

                post_area = self._bbox_area_norm(post_box)
                area_ratio = post_area / pre_area if pre_area > 0 else 0.0
                area_in_range = min_area_ratio <= area_ratio <= max_area_ratio
                if iou >= iou_threshold and area_in_range:
                    match_class = "stayed"
                    priority = 0
                elif within_tolerance and not area_in_range:
                    match_class = "changed_size_or_replaced"
                    priority = 1
                else:
                    match_class = "moved_or_uncertain"
                    priority = 2

                candidates.append(
                    {
                        "pre_idx": pre_idx,
                        "post_idx": post_idx,
                        "match_class": match_class,
                        "priority": priority,
                        "iou": round(float(iou), 4),
                        "area_ratio": round(float(area_ratio), 4),
                        "within_tolerance": within_tolerance,
                        "area_in_range": area_in_range,
                        "area_deviation": abs(area_ratio - 1.0),
                        "pre_bbox": pre_det["bbox"],
                        "pre_score": pre_det.get("score"),
                        "post_bbox": post_det["bbox"],
                        "post_score": post_det.get("score"),
                        "anchor_region": self._norm_region_to_pixels(anchor_region, pre_size),
                    }
                )

        candidates.sort(
            key=lambda item: (
                item["priority"],
                -item["iou"],
                item["area_deviation"],
                item["pre_idx"],
                item["post_idx"],
            )
        )

        matched_pre = set()
        matched_post = set()
        stayed: List[Dict[str, Any]] = []
        changed_size_or_replaced: List[Dict[str, Any]] = []
        moved_or_uncertain: List[Dict[str, Any]] = []

        for candidate in candidates:
            pre_idx = candidate["pre_idx"]
            post_idx = candidate["post_idx"]
            if pre_idx in matched_pre or post_idx in matched_post:
                continue

            matched_pre.add(pre_idx)
            matched_post.add(post_idx)
            item = {
                "pre_bbox": candidate["pre_bbox"],
                "pre_score": candidate["pre_score"],
                "anchor_region": candidate["anchor_region"],
                "matched_post_bbox": candidate["post_bbox"],
                "post_score": candidate["post_score"],
                "iou": candidate["iou"],
                "area_ratio": candidate["area_ratio"],
                "within_tolerance": candidate["within_tolerance"],
                "match_method": "iou_area_neighborhood",
            }

            if candidate["match_class"] == "stayed":
                stayed.append({**item, "supporting_post_bboxes": [candidate["post_bbox"]]})
            elif candidate["match_class"] == "changed_size_or_replaced":
                changed_size_or_replaced.append(item)
            else:
                moved_or_uncertain.append(item)

        candidates_by_pre: Dict[int, List[Dict[str, Any]]] = {}
        for candidate in candidates:
            candidates_by_pre.setdefault(candidate["pre_idx"], []).append(candidate)

        left: List[Dict[str, Any]] = []
        for pre_idx, (pre_det, anchor_region) in enumerate(zip(pre_dets, pre_anchor_regions)):
            if pre_idx in matched_pre:
                continue
            nearby_candidates = [
                {
                    "post_bbox": cand["post_bbox"],
                    "post_score": cand["post_score"],
                    "iou": cand["iou"],
                    "area_ratio": cand["area_ratio"],
                    "within_tolerance": cand["within_tolerance"],
                }
                for cand in candidates_by_pre.get(pre_idx, [])[:5]
            ]
            item = {
                "pre_bbox": pre_det["bbox"],
                "pre_score": pre_det.get("score"),
                "anchor_region": self._norm_region_to_pixels(anchor_region, pre_size),
            }
            if nearby_candidates:
                item["nearby_candidates"] = nearby_candidates
            left.append(item)

        appeared: List[Dict[str, Any]] = []
        for post_idx, post_det in enumerate(post_dets):
            if post_idx not in matched_post:
                appeared.append(
                    {
                        "post_bbox": post_det["bbox"],
                        "post_score": post_det.get("score"),
                    }
                )

        return {
            "pre_count": len(pre_dets),
            "post_count": len(post_dets),
            "count_delta": len(post_dets) - len(pre_dets),
            "position_tolerance_ratio": tolerance_ratio,
            "iou_threshold": iou_threshold,
            "min_area_ratio": min_area_ratio,
            "max_area_ratio": max_area_ratio,
            "stayed_at_original_position": stayed,
            "changed_size_or_replaced": changed_size_or_replaced,
            "moved_or_uncertain": moved_or_uncertain,
            "left_original_position": left,
            "appeared_at_new_location": appeared,
        }

    @staticmethod
    def _primary_distribution_cell(distribution: Dict[str, int]) -> Optional[str]:
        if not distribution:
            return None
        return max(distribution.items(), key=lambda item: (item[1], item[0]))[0]

    @staticmethod
    def _cell_display(cell: Optional[str], language: str) -> str:
        if cell is None:
            return ""
        zh = {
            "top_left": "左上区域",
            "top": "上方区域",
            "top_right": "右上区域",
            "left": "左侧区域",
            "center": "中心区域",
            "right": "右侧区域",
            "bottom_left": "左下区域",
            "bottom": "下方区域",
            "bottom_right": "右下区域",
        }
        en = {
            "top_left": "top-left area",
            "top": "top area",
            "top_right": "top-right area",
            "left": "left area",
            "center": "center area",
            "right": "right area",
            "bottom_left": "bottom-left area",
            "bottom": "bottom area",
            "bottom_right": "bottom-right area",
        }
        return (zh if language == "zh" else en).get(cell, cell)

    def _target_display(self, canonical_target: str, language: str) -> Tuple[str, str]:
        display = self.TARGET_DISPLAY.get(canonical_target, {}).get(language)
        if display:
            return display
        if language == "zh":
            return canonical_target, "个"
        if canonical_target.endswith("s"):
            return canonical_target, canonical_target
        return canonical_target, f"{canonical_target}s"

    def _format_location_anchored_target_text(
        self,
        canonical_target: str,
        changes: Dict[str, Any],
        pre_distribution: Dict[str, int],
        post_distribution: Dict[str, int],
        query_text: str,
    ) -> str:
        language = "zh" if self._contains_cjk(query_text) else "en"
        display = self._target_display(canonical_target, language)
        region_display = changes.get("region_display")
        if language == "zh":
            target_name, unit = display
            pre_count = changes["pre_count"]
            post_count = changes["post_count"]
            left_count = len(changes["left_original_position"])
            appeared_count = len(changes["appeared_at_new_location"])
            stayed_count = len(changes["stayed_at_original_position"])
            size_changed_count = len(changes.get("changed_size_or_replaced", []))
            uncertain_count = len(changes.get("moved_or_uncertain", []))
            tolerance_percent = changes["position_tolerance_ratio"] * 100
            iou_threshold = changes.get("iou_threshold", 0.0)
            min_area_ratio = changes.get("min_area_ratio", 0.0)
            max_area_ratio = changes.get("max_area_ratio", 0.0)

            if pre_count == 0 and post_count == 0:
                if region_display:
                    return f"在{region_display}内，前图和后图均未检测到{target_name}，无法确认该区域内的{target_name}相关变化。"
                return f"前图和后图均未检测到{target_name}，无法确认{target_name}相关变化。"

            if region_display:
                lines = [f"在{region_display}内，前图检测到 {pre_count} {unit}{target_name}，后图检测到 {post_count} {unit}{target_name}。"]
            else:
                lines = [f"前图检测到 {pre_count} {unit}{target_name}，后图检测到 {post_count} {unit}{target_name}。"]

            if changes["count_delta"] > 0:
                lines.append(f"后图{target_name}数量增加 {changes['count_delta']} {unit}，存在新增{target_name}。")
            elif changes["count_delta"] < 0:
                lines.append(f"后图{target_name}数量减少 {abs(changes['count_delta'])} {unit}，存在{target_name}减少或离开。")
            else:
                lines.append(f"前后{target_name}数量相同。")

            if left_count > 0:
                lines.append(
                    f"有 {left_count} 个前图{target_name}未匹配到满足 IoU、面积比例或位置条件的后图同类目标，"
                    f"因此判断这些位置的{target_name}已离开原位或消失。"
                )
            if size_changed_count > 0:
                lines.append(
                    f"有 {size_changed_count} 个前图{target_name}位置在后图仍有同类目标，"
                    f"但目标框面积比例超出 [{min_area_ratio:.2f}, {max_area_ratio:.2f}]，"
                    f"判断为尺度明显变化或目标替换。"
                )
            if uncertain_count > 0:
                lines.append(
                    f"有 {uncertain_count} 个前后目标存在空间候选关系，但 IoU 或位置证据不足，"
                    f"判断为移动或不确定变化。"
                )
            if appeared_count > 0:
                lines.append(
                    f"后图有 {appeared_count} 个{target_name}未与任何前图{target_name}完成匹配，"
                    f"判断为新位置出现{target_name}。"
                )
            if left_count == 0 and size_changed_count == 0 and uncertain_count == 0 and pre_count > 0:
                lines.append(f"前图 {stayed_count} 个{target_name}均通过 IoU 和面积比例检查，未发现{target_name}离开原位。")

            pre_primary = self._primary_distribution_cell(pre_distribution)
            post_primary = self._primary_distribution_cell(post_distribution)
            if pre_primary and post_primary and pre_primary != post_primary:
                lines.append(
                    f"粗略分布上，前图{target_name}主要位于{self._cell_display(pre_primary, language)}，"
                    f"后图主要位于{self._cell_display(post_primary, language)}。"
                )

            lines.append(
                f"该判断结合 IoU、面积比例和位置邻域：IoU 阈值为 {iou_threshold:.2f}，"
                f"面积比例有效范围为 [{min_area_ratio:.2f}, {max_area_ratio:.2f}]，"
                f"位置邻域容忍约 {tolerance_percent:.1f}% 的图像配准偏移；"
                f"不进行严格实例身份追踪。"
            )
            if region_display:
                lines.append(f"该结论仅针对{region_display}；不判断目标是否移动到了图像其他区域。")
            return "\n".join(lines)

        singular, plural = display
        target_name = singular if max(changes["pre_count"], changes["post_count"]) == 1 else plural
        pre_count = changes["pre_count"]
        post_count = changes["post_count"]
        left_count = len(changes["left_original_position"])
        appeared_count = len(changes["appeared_at_new_location"])
        stayed_count = len(changes["stayed_at_original_position"])
        size_changed_count = len(changes.get("changed_size_or_replaced", []))
        uncertain_count = len(changes.get("moved_or_uncertain", []))
        tolerance_percent = changes["position_tolerance_ratio"] * 100
        iou_threshold = changes.get("iou_threshold", 0.0)
        min_area_ratio = changes.get("min_area_ratio", 0.0)
        max_area_ratio = changes.get("max_area_ratio", 0.0)

        if pre_count == 0 and post_count == 0:
            if region_display:
                return f"No {plural} were detected in the {region_display} in either image, so no {singular}-related change can be confirmed for that region."
            return f"No {plural} were detected in either image, so no {singular}-related change can be confirmed."

        if region_display:
            lines = [f"In the {region_display}, the pre-image contains {pre_count} detected {plural}, and the post-image contains {post_count} detected {plural}."]
        else:
            lines = [f"The pre-image contains {pre_count} detected {plural}, and the post-image contains {post_count} detected {plural}."]

        if changes["count_delta"] > 0:
            lines.append(f"The post-image has {changes['count_delta']} more detected {plural}, indicating newly appearing {plural}.")
        elif changes["count_delta"] < 0:
            lines.append(f"The post-image has {abs(changes['count_delta'])} fewer detected {plural}, indicating reduced or departed {plural}.")
        else:
            lines.append(f"The detected {singular} count is unchanged.")

        if left_count > 0:
            lines.append(
                f"For {left_count} pre-image {target_name} location(s), no post-image same-class target met the IoU, area-ratio, or location criteria; "
                f"these locations are judged as departed from the original position or disappeared."
            )
        if size_changed_count > 0:
            lines.append(
                f"{size_changed_count} pre-image {target_name} location(s) still have same-class detections nearby, "
                f"but the matched box area ratio falls outside [{min_area_ratio:.2f}, {max_area_ratio:.2f}], "
                f"so they are judged as size-changed or replaced targets."
            )
        if uncertain_count > 0:
            lines.append(
                f"{uncertain_count} pre/post target pair(s) have spatial candidate evidence, but insufficient IoU or position evidence; "
                f"they are judged as moved or uncertain changes."
            )
        if appeared_count > 0:
            lines.append(
                f"{appeared_count} post-image {target_name} location(s) were not matched to any pre-image {singular}, "
                f"so they are judged as newly appearing locations."
            )
        if left_count == 0 and size_changed_count == 0 and uncertain_count == 0 and pre_count > 0:
            lines.append(
                f"All {stayed_count} pre-image {singular} location(s) passed the IoU and area-ratio checks."
            )

        pre_primary = self._primary_distribution_cell(pre_distribution)
        post_primary = self._primary_distribution_cell(post_distribution)
        if pre_primary and post_primary and pre_primary != post_primary:
            lines.append(
                f"Coarsely, the pre-image detections are mainly in the {self._cell_display(pre_primary, language)}, "
                f"while the post-image detections are mainly in the {self._cell_display(post_primary, language)}."
            )

        lines.append(
            f"This combines IoU, area ratio, and location-neighborhood evidence: IoU threshold {iou_threshold:.2f}, "
            f"valid area-ratio range [{min_area_ratio:.2f}, {max_area_ratio:.2f}], and about {tolerance_percent:.1f}% image-registration tolerance; "
            f"it does not perform strict instance identity tracking."
        )
        if region_display:
            lines.append(f"This conclusion is limited to the {region_display}; it does not determine whether a target moved elsewhere in the full image.")
        return "\n".join(lines)

    def _run_target_flow(
        self,
        pre_image: str,
        post_image: str,
        query_text: str,
        target: Optional[str],
        box_threshold: float,
        position_tolerance_ratio: float,
        iou_threshold: float,
        min_area_ratio: float,
        max_area_ratio: float,
        region: Optional[str] = None,
        region_margin_ratio: float = 0.05,
    ) -> Dict[str, Any]:
        canonical_target = self._resolve_target(query_text, target)
        if canonical_target is None:
            return {
                "text": f"Unsupported target for target-level change detection: {target or query_text}",
                "error_code": 2,
            }

        resolved_region = self._resolve_region(query_text, region)
        pre_crop_box: Optional[List[int]] = None
        post_crop_box: Optional[List[int]] = None
        if resolved_region:
            base_norm_box = self._region_to_norm_box(resolved_region)
            crop_norm_box = self._expand_norm_box(base_norm_box, region_margin_ratio)
            pre_crop_box = self._norm_box_to_pixel_box(crop_norm_box, self._get_image_size(pre_image))
            post_crop_box = self._norm_box_to_pixel_box(crop_norm_box, self._get_image_size(post_image))

        pre_dets, pre_size, _pre_meta = self._detect_small_objects(
            pre_image,
            canonical_target,
            box_threshold,
            crop_box=pre_crop_box,
        )
        post_dets, post_size, _post_meta = self._detect_small_objects(
            post_image,
            canonical_target,
            box_threshold,
            crop_box=post_crop_box,
        )

        changes = self._evaluate_location_anchored_changes(
            pre_dets=pre_dets,
            post_dets=post_dets,
            pre_size=pre_size,
            post_size=post_size,
            tolerance_ratio=position_tolerance_ratio,
            iou_threshold=iou_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )
        pre_distribution = self._summarize_spatial_distribution(pre_dets, pre_size)
        post_distribution = self._summarize_spatial_distribution(post_dets, post_size)
        changes["pre_distribution"] = pre_distribution
        changes["post_distribution"] = post_distribution
        changes["region"] = resolved_region
        if resolved_region:
            language = "zh" if self._contains_cjk(query_text) else "en"
            changes["region_display"] = self.REGION_DISPLAY[resolved_region][language]
            changes["region_margin_ratio"] = region_margin_ratio
            changes["pre_crop_box"] = pre_crop_box
            changes["post_crop_box"] = post_crop_box
        else:
            changes["region_display"] = None

        text = self._format_location_anchored_target_text(
            canonical_target=canonical_target,
            changes=changes,
            pre_distribution=pre_distribution,
            post_distribution=post_distribution,
            query_text=query_text,
        )

        return {
            "text": text,
            "error_code": 0,
            "mode_used": "target",
            "target": canonical_target,
            "change_level": "location_anchored_target",
            "target_changes": changes,
        }

    def _run_hybrid_flow(
        self,
        pre_image: str,
        post_image: str,
        query_text: str,
        target: Optional[str],
        box_threshold: float,
        position_tolerance_ratio: float,
        iou_threshold: float,
        min_area_ratio: float,
        max_area_ratio: float,
        region: Optional[str] = None,
        region_margin_ratio: float = 0.05,
    ) -> Dict[str, Any]:
        canonical_target = self._resolve_target(query_text, target)
        scene_resp = self._run_scene_flow(pre_image, post_image, query_text)
        if canonical_target is None:
            scene_resp["mode_used"] = "scene"
            return scene_resp

        target_resp = self._run_target_flow(
            pre_image=pre_image,
            post_image=post_image,
            query_text=query_text,
            target=canonical_target,
            box_threshold=box_threshold,
            position_tolerance_ratio=position_tolerance_ratio,
            iou_threshold=iou_threshold,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            region=region,
            region_margin_ratio=region_margin_ratio,
        )
        if target_resp.get("error_code") != 0:
            return target_resp
        if scene_resp.get("error_code") != 0:
            return scene_resp

        language = "zh" if self._contains_cjk(query_text) else "en"
        if language == "zh":
            text = f"目标级结果：\n{target_resp['text']}\n\n场景级补充：\n{scene_resp['text']}"
        else:
            text = f"Target-level result:\n{target_resp['text']}\n\nScene-level supplement:\n{scene_resp['text']}"

        return {
            "text": text,
            "error_code": 0,
            "mode_used": "hybrid",
            "target": target_resp.get("target"),
            "change_level": target_resp.get("change_level"),
            "target_changes": target_resp.get("target_changes"),
            "scene_text": scene_resp.get("text", ""),
        }

    def generate(self, params):
        required_keys = ("pre_image", "post_image", "text")

        if any(k not in params for k in required_keys):
            missing = [k for k in required_keys if k not in params]
            txt_e = f"Missing required parameter(s): {', '.join(missing)}"
            logger.error(txt_e)
            return {"text": txt_e, "error_code": 2}

        pre_image = str(params.get("pre_image")).strip()
        post_image = str(params.get("post_image")).strip()
        query_text = str(params.get("text")).strip()

        for key in ["pre_image", "post_image"]:
            val = locals()[key]
            if not (os.path.exists(val)):
                txt_e = f"Image not found: {val}"
                logger.error(txt_e)
                return {"text": txt_e, "error_code": 3}

        try:
            mode = self._resolve_mode(query_text, params.get("mode", "auto"), params.get("target"))
            box_threshold = float(params.get("box_threshold", 0.25))
            position_tolerance_ratio = float(params.get("position_tolerance_ratio", 0.05))
            iou_threshold = float(params.get("iou_threshold", 0.1))
            min_area_ratio = float(params.get("min_area_ratio", 0.5))
            max_area_ratio = float(params.get("max_area_ratio", 2.0))
            region_margin_ratio = float(params.get("region_margin_ratio", 0.05))

            if box_threshold < 0 or box_threshold > 1:
                return {"text": "box_threshold must be in [0, 1]", "error_code": 2}
            if position_tolerance_ratio < 0 or position_tolerance_ratio > 0.5:
                return {"text": "position_tolerance_ratio must be in [0, 0.5]", "error_code": 2}
            if iou_threshold < 0 or iou_threshold > 1:
                return {"text": "iou_threshold must be in [0, 1]", "error_code": 2}
            if min_area_ratio <= 0:
                return {"text": "min_area_ratio must be > 0", "error_code": 2}
            if max_area_ratio < min_area_ratio:
                return {"text": "max_area_ratio must be >= min_area_ratio", "error_code": 2}
            if region_margin_ratio < 0 or region_margin_ratio > 0.5:
                return {"text": "region_margin_ratio must be in [0, 0.5]", "error_code": 2}

            if mode == "scene":
                return self._run_scene_flow(pre_image, post_image, query_text)
            if mode == "target":
                return self._run_target_flow(
                    pre_image=pre_image,
                    post_image=post_image,
                    query_text=query_text,
                    target=params.get("target"),
                    box_threshold=box_threshold,
                    position_tolerance_ratio=position_tolerance_ratio,
                    iou_threshold=iou_threshold,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    region=params.get("region"),
                    region_margin_ratio=region_margin_ratio,
                )
            if mode == "hybrid":
                return self._run_hybrid_flow(
                    pre_image=pre_image,
                    post_image=post_image,
                    query_text=query_text,
                    target=params.get("target"),
                    box_threshold=box_threshold,
                    position_tolerance_ratio=position_tolerance_ratio,
                    iou_threshold=iou_threshold,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    region=params.get("region"),
                    region_margin_ratio=region_margin_ratio,
                )
            return {"text": f"Unsupported mode: {mode}", "error_code": 2}
        
        except Exception as e:
            txt_e = f"Error in ChangeDetection: {e}"
            logger.exception(txt_e)
            return {"text": txt_e, "error_code": 1}

    def get_tool_instruction(self):
        return {
            "type": "function",
            "function": {
                "name": "ChangeDetection",
                "description": (
                    "Compare pre- and post-event satellite images. Supports scene-level text change "
                    "description and location-anchored target-level change detection for airplane, bridge, "
                    "harbor, ship, tank, and vehicle targets."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": (
                                "Natural language query, e.g., 'Identify the damaged buildings.' "
                                "For target-level change, ask about supported targets leaving, appearing, "
                                "or staying at original positions."
                            )
                        },
                        "pre_image": {
                            "type": "string",
                            "description": "Path of the pre-change image."
                        },
                        "post_image": {
                            "type": "string",
                            "description": "Path of the post-change image."
                        },
                        "mode": {
                            "type": "string",
                            "description": "Optional: auto, scene, target, or hybrid. Default is auto."
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                "Optional target for target/hybrid mode. Supported: airplane, bridge, harbor, ship, "
                                "tank, and vehicle, plus common Chinese/English synonyms."
                            )
                        },
                        "box_threshold": {
                            "type": "number",
                            "description": "SmallObjectDetection score threshold for target mode. Default is 0.25."
                        },
                        "position_tolerance_ratio": {
                            "type": "number",
                            "description": (
                                "Image-size ratio used to expand each pre-image target location when checking "
                                "whether a same-class target remains nearby. Default is 0.05."
                            )
                        },
                        "iou_threshold": {
                            "type": "number",
                            "description": (
                                "Minimum normalized bbox IoU for judging that a same-class target stayed at "
                                "the original position. Default is 0.1."
                            )
                        },
                        "min_area_ratio": {
                            "type": "number",
                            "description": (
                                "Minimum post/pre bbox area ratio accepted as the same-size target. "
                                "Lower values are reported as size-changed or replaced. Default is 0.5."
                            )
                        },
                        "max_area_ratio": {
                            "type": "number",
                            "description": (
                                "Maximum post/pre bbox area ratio accepted as the same-size target. "
                                "Higher values are reported as size-changed or replaced. Default is 2.0."
                            )
                        },
                        "region": {
                            "type": "string",
                            "description": (
                                "Optional ROI for target/hybrid target checking. Supported: top_left, top_right, "
                                "bottom_left, bottom_right, top, bottom, left, right, center. Chinese terms like "
                                "左上角 and 左下角 are also supported. Scene mode still uses the full images."
                            )
                        },
                        "region_margin_ratio": {
                            "type": "number",
                            "description": (
                                "Margin ratio used to expand the ROI crop before SmallObjectDetection. "
                                "Default is 0.05."
                            )
                        },
                    },
                    "required": ["text", "pre_image", "post_image"]
                }
            }
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=20015)
    parser.add_argument("--worker-address", type=str, default="auto")
    parser.add_argument("--controller-address", type=str, default="http://127.0.0.1:20001")
    parser.add_argument("--model_base", type=str, default="jirvin16/TEOChat")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ChangeDetectionWorker(
        controller_addr=args.controller_address,
        worker_addr=args.worker_address,
        worker_id=worker_id,
        limit_model_concurrency=args.limit_model_concurrency,
        model_base=args.model_base,
        host=args.host,
        port=args.port,
        no_register=args.no_register,
        device=args.device,
        args=args,
    )
    worker.run()
