"""Cross-modal matching adapter.

Critically, a legacy ``matched_pair`` is **decomposed** back into its two
constituent observations (one optical, one SAR) rather than ingested as a single
fused identity.  The pair's geometric agreement is preserved only as relation
evidence, so the joint inference — not the greedy matcher — decides whether the
two observations really share an object.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from DS_VI.models.shared_identity.adapters.base import AdapterResult, BaseAdapter, register
from DS_VI.models.shared_identity.observation_builder import (
    build_observation,
    claims_from_detection,
    detection_bbox,
    detection_score,
)
from DS_VI.models.shared_identity.schema import (
    Modality,
    Relation,
    RelationType,
    stable_id,
)


def _as_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _collect(response: Mapping[str, Any], *keys: str) -> List[Dict[str, Any]]:
    """Read a list from the response or its ``structured`` block."""
    for key in keys:
        items = _as_list(response.get(key))
        if items:
            return items
    structured = response.get("structured")
    if isinstance(structured, Mapping):
        for key in keys:
            items = _as_list(structured.get(key))
            if items:
                return items
    return []


@register
class CrossModalAdapter(BaseAdapter):
    """Convert cross-modal matching output into per-modality observations."""

    tool_names = ("CrossModalMatchSkill", "CrossModalInterpret", "CrossModalTaskSkill")
    modality = Modality.CROSS_MODAL.value

    def to_observations(
        self,
        tool_name: str,
        response: Mapping[str, Any],
        context: Optional[Mapping[str, Any]] = None,
    ) -> AdapterResult:
        ctx = dict(context or {})
        reason = self.response_failed(response)
        if reason:
            return AdapterResult(skipped_reason=reason)

        time_key = str(ctx.get("time_key", "t0"))
        rgb_acquisition = ctx.get("optical_image") or ctx.get("rgb_image")
        sar_acquisition = ctx.get("sar_image")
        result = AdapterResult()
        seen: Dict[str, str] = {}

        def add(
            modality: Modality,
            bbox: Optional[List[float]],
            score: Optional[float],
            payload: Mapping[str, Any],
            channel: str,
            index: int,
        ) -> Optional[str]:
            if bbox is None:
                return None
            acquisition = rgb_acquisition if modality is Modality.OPTICAL else sar_acquisition
            observation = build_observation(
                source_tool=str(tool_name),
                modality=modality,
                bbox=bbox,
                time_key=time_key,
                acquisition_id=acquisition,
                detection_confidence=score,
                frame_id=acquisition,
                pixel_size_m=ctx.get("pixel_size_m"),
                input_roots=[a for a in [acquisition] if a],
                context={"cross_modal_index": index, "channel": channel},
                round_id=ctx.get("round_id"),
                index=index,
            )
            if observation.observation_id in seen:
                return observation.observation_id
            seen[observation.observation_id] = channel
            result.observations.append(observation)
            result.claims.extend(
                claims_from_detection(
                    observation,
                    payload,
                    source_channel=(
                        "optical" if modality is Modality.OPTICAL else "sar"
                    ),
                )
            )
            return observation.observation_id

        # Matched pairs are split; the pair itself becomes relation evidence.
        for index, pair in enumerate(_collect(response, "matched_pairs", "matched")):
            rgb_payload = {
                "bbox": pair.get("rgb_bbox"),
                "label": pair.get("rgb_label"),
                "score": pair.get("rgb_score"),
            }
            sar_payload = {
                "bbox": pair.get("sar_bbox"),
                "label": pair.get("sar_label"),
                "score": pair.get("sar_score"),
            }
            rgb_id = add(
                Modality.OPTICAL,
                detection_bbox(rgb_payload, "bbox"),
                detection_score(rgb_payload, "score"),
                rgb_payload,
                "matched_rgb",
                index,
            )
            sar_id = add(
                Modality.SAR,
                detection_bbox(sar_payload, "bbox"),
                detection_score(sar_payload, "score"),
                sar_payload,
                "matched_sar",
                index,
            )
            if rgb_id and sar_id:
                strength = 0.0
                for key in ("match_score", "iou", "center_score"):
                    value = pair.get(key)
                    if value not in (None, ""):
                        try:
                            strength = max(0.0, min(1.0, float(value)))
                        except (TypeError, ValueError):
                            strength = 0.0
                        break
                result.relations.append(
                    Relation(
                        relation_id=stable_id("rel", {"a": rgb_id, "b": sar_id, "kind": "same_scene"}),
                        relation_type=RelationType.SAME_SCENE,
                        source_observation_id=rgb_id,
                        target_observation_id=sar_id,
                        strength=strength,
                        source_tool=str(tool_name),
                        context={
                            "origin": "legacy_matched_pair",
                            "note": "soft prior evidence only; identity resolved by joint inference",
                        },
                    )
                )

        for index, item in enumerate(_collect(response, "rgb_only")):
            payload = {
                "bbox": item.get("rgb_bbox") or item.get("bbox"),
                "label": item.get("rgb_label") or item.get("label"),
                "score": item.get("rgb_score") or item.get("score"),
            }
            add(
                Modality.OPTICAL,
                detection_bbox(payload, "bbox"),
                detection_score(payload, "score"),
                payload,
                "rgb_only",
                index,
            )

        for index, item in enumerate(_collect(response, "sar_only")):
            payload = {
                "bbox": item.get("sar_bbox") or item.get("bbox"),
                "label": item.get("sar_label") or item.get("label"),
                "score": item.get("sar_score") or item.get("score"),
            }
            add(
                Modality.SAR,
                detection_bbox(payload, "bbox"),
                detection_score(payload, "score"),
                payload,
                "sar_only",
                index,
            )

        # Direct observation/claim payloads from a candidate_graph-mode worker.
        for index, item in enumerate(_collect(response, "primitive_observations")):
            modality = Modality.coerce(item.get("modality"))
            add(
                modality if modality is not Modality.UNKNOWN else Modality.OPTICAL,
                detection_bbox(item, "bbox"),
                detection_score(item, "score", "confidence"),
                item,
                "primitive",
                index,
            )

        optical_count = sum(1 for o in result.observations if o.modality is Modality.OPTICAL)
        sar_count = sum(1 for o in result.observations if o.modality is Modality.SAR)
        result.coverage = {
            (Modality.OPTICAL.value, time_key): optical_count,
            (Modality.SAR.value, time_key): sar_count,
        }
        if result.is_empty():
            result.skipped_reason = "no cross-modal evidence"
        return result


__all__ = ["CrossModalAdapter"]
