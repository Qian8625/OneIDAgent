import re
from typing import Dict, List, Any, Tuple

from tool_server.tool_workers.skills.registry import ROUTER_RULES, SKILL_REGISTRY


class SkillRouter:
    """
    Lightweight rule-based router for hybrid skills.

    Goals:
      - Recommend top-k candidate skills
      - Be conservative when confidence is low
      - Return [] when query and current inputs do not clearly match any skill
    """

    def __init__(
        self,
        min_score: float = 2.5,
        top_k: int = 3,
        score_margin: float = 0.5,
    ):
        self.min_score = min_score
        self.top_k = top_k
        self.score_margin = score_margin

    # -----------------------------
    # Public API
    # -----------------------------
    def route(
        self,
        user_query: str,
        input_registry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Returns:
            [
              {
                "skill_name": "...",
                "score": 5.5,
                "reasons": ["matched keyword: sar", ...]
              },
              ...
            ]

        Returns [] if no skill is confidently matched.
        """
        query = self._normalize_text(user_query)
        inferred_modalities = self._infer_modalities(query, input_registry)

        scored = []
        for skill_name in SKILL_REGISTRY.keys():
            score, reasons = self._score_skill(
                skill_name=skill_name,
                query=query,
                input_registry=input_registry,
                inferred_modalities=inferred_modalities,
            )
            scored.append(
                {
                    "skill_name": skill_name,
                    "score": score,
                    "reasons": reasons,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Conservative filtering
        if not scored:
            return []

        best_score = scored[0]["score"]
        if best_score < self.min_score:
            return []

        selected = []
        for item in scored:
            if item["score"] < self.min_score:
                continue
            if best_score - item["score"] > self.score_margin and len(selected) >= 1:
                break
            selected.append(item)
            if len(selected) >= self.top_k:
                break

        return selected

    # -----------------------------
    # Scoring
    # -----------------------------
    def _score_skill(
        self,
        skill_name: str,
        query: str,
        input_registry: Dict[str, Any],
        inferred_modalities: List[str],
    ) -> Tuple[float, List[str]]:
        rules = ROUTER_RULES.get(skill_name, {})
        score = 0.0
        reasons = []

        keywords = rules.get("keywords", [])
        negative_keywords = rules.get("negative_keywords", [])
        preferred_inputs = rules.get("preferred_inputs", [])
        required_modalities = rules.get("required_modalities", [])

        # 1) keyword matching
        for kw in keywords:
            if self._contains(query, kw):
                score += 1.2
                reasons.append(f"matched keyword: {kw}")

        # 2) negative keyword penalty
        for nkw in negative_keywords:
            if self._contains(query, nkw):
                score -= 1.5
                reasons.append(f"negative keyword: {nkw}")

        # 3) input preference bonus
        for inp in preferred_inputs:
            if input_registry.get(inp):
                score += 0.5
                reasons.append(f"preferred input available: {inp}")

        # 4) modality constraints / bonuses
        if required_modalities:
            matched_modalities = 0
            for mod in required_modalities:
                if mod in inferred_modalities:
                    matched_modalities += 1
            if matched_modalities == len(required_modalities):
                score += 1.5
                reasons.append(
                    f"required modalities matched: {', '.join(required_modalities)}"
                )
            elif matched_modalities > 0:
                score += 0.5
                reasons.append(
                    f"partial modality match: {', '.join(required_modalities)}"
                )
            else:
                score -= 1.0
                reasons.append(
                    f"missing expected modalities: {', '.join(required_modalities)}"
                )

        # 5) skill-specific structure bonuses
        bonus, bonus_reasons = self._apply_skill_specific_heuristics(
            skill_name, query, input_registry
        )
        score += bonus
        reasons.extend(bonus_reasons)

        return round(score, 3), reasons

    def _apply_skill_specific_heuristics(
        self,
        skill_name: str,
        query: str,
        input_registry: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        bonus = 0.0
        reasons = []

        has_primary = bool(input_registry.get("primary_image"))
        has_t1 = bool(input_registry.get("time1_image"))
        has_t2 = bool(input_registry.get("time2_image"))

        # Final object visualization
        if skill_name == "ObjectVisualOutputSkill":
            if any(self._contains(query, x) for x in ["draw", "box", "mark", "plot", "visualize", "point", "annotate", "highlight", "画框", "标注", "可视化", "画点", "标记", "高亮", "框出"]):
                bonus += 2.0
                reasons.append("final object visualization intent")
            if any(self._contains(query, x) for x in ["measure only", "distance only", "area only", "只测量", "只计算"]):
                bonus -= 1.5
                reasons.append("measure-only intent belongs to another skill")

        # Single-image object geometry measurement
        elif skill_name == "ObjectMeasureSkill":
            if any(self._contains(query, x) for x in ["measure", "distance", "area", "size", "gsd", "center", "centroid", "测量", "距离", "面积", "尺寸", "中心点", "像素距离", "实际距离"]):
                bonus += 2.0
                reasons.append("object geometry measurement intent")
            if any(self._contains(query, x) for x in ["between smallest and largest", "smallest and largest", "最小和最大之间"]):
                bonus += 1.0
                reasons.append("pair measurement selection intent")
            if any(self._contains(query, x) for x in ["draw only", "只画框"]):
                bonus -= 1.5
                reasons.append("draw-only intent belongs to another skill")

        # Single-image object detection / selection
        elif skill_name == "ObjectDetectSelectSkill":
            if any(self._contains(query, x) for x in ["smallest", "largest", "nearest", "farthest", "leftmost", "rightmost", "topmost", "bottommost", "select", "选择", "最小", "最大", "最近", "最远", "最左", "最右", "最上", "最下"]):
                bonus += 2.0
                reasons.append("object detection selection intent")
            elif any(self._contains(query, x) for x in ["detect", "locate", "find", "检测", "定位", "找出"]):
                bonus += 1.5
                reasons.append("object detection intent")
            if any(self._contains(query, x) for x in ["distance", "area", "gsd", "measure", "距离", "面积", "测量"]):
                bonus -= 1.0
                reasons.append("measurement intent belongs to another skill")

        # Target-level optical-SAR matching
        elif skill_name == "CrossModalMatchSkill":
            if any(self._contains(query, x) for x in ["cross-modal match", "rgb sar match", "optical sar match", "match pairs", "deduplicate", "dedup", "target-level match", "跨模态匹配", "光学sar匹配", "匹配", "去重"]):
                bonus += 3.2
                reasons.append("target-level optical-SAR matching intent")
            elif any(self._contains(query, x) for x in ["cross-modal", "rgb sar", "optical sar", "fusion", "跨模态", "光学sar", "融合"]):
                bonus += 2.0
                reasons.append("cross-modal matching modality intent")
            if has_primary and has_t1:
                bonus += 0.6
                reasons.append("two likely modality inputs available")
            if any(self._contains(query, x) for x in ["attribution", "recall", "consistency", "question", "判断", "归因", "补漏", "一致性", "跨模态判断"]):
                bonus -= 1.2
                reasons.append("task-level reasoning belongs to another skill")
            if any(self._contains(query, x) for x in ["whether", "exist", "final conclusion", "final judgment", "cross-validate whether", "supported by sar", "最终结论", "是否存在"]):
                bonus -= 2.0
                reasons.append("existence/final-conclusion task belongs to CrossModalTaskSkill")
            if any(self._contains(query, x) for x in ["poi", "geotiff", "change", "single image only", "兴趣点", "地理", "变化", "单图"]):
                bonus -= 2.0
                reasons.append("non cross-modal match intent")

        # Task-level optical-SAR reasoning
        elif skill_name == "CrossModalTaskSkill":
            if any(self._contains(query, x) for x in ["verify with sar", "confirm with sar", "sar evidence", "optical sar consistency", "sar attribution", "recall candidate", "cross-modal reasoning", "task-level cross-modal", "whether", "exist", "final conclusion", "final judgment", "cross-validate whether", "supported by sar", "consistency conclusion", "用sar确认", "sar证据", "光学sar一致性", "sar归因", "sar补漏", "跨模态判断", "最终结论", "是否存在"]):
                bonus += 3.0
                reasons.append("task-level optical-SAR reasoning intent")
            if any(self._contains(query, x) for x in ["largest", "highest-confidence", "highest confidence", "final position"]):
                bonus += 2.2
                reasons.append("cross-modal target selection task intent")
            elif any(self._contains(query, x) for x in ["rgb", "optical", "sar", "cross-modal", "fusion", "光学", "雷达", "跨模态", "融合"]):
                bonus += 1.4
                reasons.append("cross-modal modality intent")
            if has_primary and has_t1:
                bonus += 0.6
                reasons.append("two likely modality inputs available")
            if any(self._contains(query, x) for x in ["poi", "geotiff", "single-image measure", "draw only", "change", "兴趣点", "地理", "单图测量", "只画框", "变化"]):
                bonus -= 2.0
                reasons.append("non task-level cross-modal intent")

        # SAR-only evidence extraction
        elif skill_name == "SAREvidenceExtractSkill":
            if any(self._contains(query, x) for x in ["sar evidence", "sar response", "scattering", "scatterer", "bright spot", "candidate extraction", "radar evidence", "sar candidate", "雷达证据", "sar证据", "sar响应", "散射", "强散射", "亮斑", "候选目标", "候选框"]):
                bonus += 2.4
                reasons.append("SAR evidence extraction intent")
            elif self._contains(query, "sar") or self._contains(query, "雷达"):
                bonus += 1.0
                reasons.append("SAR modality mentioned")
            if any(self._contains(query, x) for x in ["cross-modal", "confirm", "match", "fusion", "rgb", "optical", "跨模态", "确认", "匹配", "融合", "光学"]):
                bonus -= 1.5
                reasons.append("cross-modal confirmation belongs to another skill")
            if any(self._contains(query, x) for x in ["distance", "area", "measure", "距离", "面积", "测量"]):
                bonus -= 1.0
                reasons.append("measurement intent belongs to another skill")

        return bonus, reasons

    # -----------------------------
    # Modality inference
    # -----------------------------
    def _infer_modalities(
        self,
        query: str,
        input_registry: Dict[str, Any],
    ) -> List[str]:
        modalities = set()

        if self._contains(query, "sar") or self._contains(query, "雷达"):
            modalities.add("sar")
        if self._contains(query, "rgb") or self._contains(query, "optical") or self._contains(query, "光学"):
            modalities.add("rgb")
        if self._contains(query, "geotiff") or self._contains(query, "poi") or self._contains(query, "兴趣点"):
            modalities.add("geotiff")

        primary = input_registry.get("primary_image")
        if primary and self._looks_like_geotiff(primary):
            modalities.add("geotiff")

        if input_registry.get("time1_image") and input_registry.get("time2_image"):
            modalities.add("time_series")

        return sorted(list(modalities))

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _contains(text: str, keyword: str) -> bool:
        return keyword.lower() in text

    @staticmethod
    def _looks_like_geotiff(path: Any) -> bool:
        if not isinstance(path, str):
            return False
        return path.lower().endswith((".tif", ".tiff"))


def route_skills(
    user_query: str,
    input_registry: Dict[str, Any],
    top_k: int = 3,
    min_score: float = 2.5,
    score_margin: float = 0.5,
) -> List[Dict[str, Any]]:
    router = SkillRouter(
        min_score=min_score,
        top_k=top_k,
        score_margin=score_margin,
    )
    return router.route(user_query=user_query, input_registry=input_registry)
