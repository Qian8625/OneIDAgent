"""SAR evidence adapter."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from DS_VI.models.shared_identity.adapters.base import AdapterResult, BaseAdapter, register
from DS_VI.models.shared_identity.observation_builder import (
    build_observation,
    claims_from_detection,
    detection_bbox,
    detection_score,
    extract_detections,
)
from DS_VI.models.shared_identity.schema import Modality


@register
class SARAdapter(BaseAdapter):
    """Convert SAR candidate responses into observations plus claims."""

    # DenoDetV2 is the SAR target detector (trained on SARDet-100K); it must
    # register as SAR evidence, never as optical.
    tool_names = ("SARToRGB", "SARDetection", "DenoDetV2")
    modality = Modality.SAR.value

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

        detections = extract_detections(response)
        if not detections:
            return AdapterResult(
                skipped_reason="no sar candidates",
                coverage={(self.modality, str(ctx.get("time_key", "t0"))): 0},
            )

        result = AdapterResult()
        time_key = str(ctx.get("time_key", "t0"))
        acquisition_id = ctx.get("acquisition_id") or ctx.get("sar_image")
        input_roots = list(ctx.get("input_roots") or ([acquisition_id] if acquisition_id else []))

        for index, detection in enumerate(detections):
            bbox = detection_bbox(detection, "sar_bbox", "bbox", "box")
            if bbox is None:
                continue
            observation = build_observation(
                source_tool=str(tool_name),
                modality=Modality.SAR,
                bbox=bbox,
                time_key=time_key,
                acquisition_id=acquisition_id,
                detection_confidence=detection_score(detection, "sar_score", "score", "confidence"),
                frame_id=ctx.get("frame_id") or acquisition_id,
                pixel_size_m=ctx.get("pixel_size_m"),
                input_roots=input_roots,
                context={
                    "sar_index": index,
                    "relation_group_id": ctx.get("relation_group_id"),
                    "image_group_id": (
                        ctx.get("image_group_id")
                        or acquisition_id
                        or ctx.get("frame_id")
                    ),
                    "related_image_group_ids": list(
                        ctx.get("related_image_group_ids") or []
                    ),
                },
                round_id=ctx.get("round_id"),
                index=index,
            )
            result.observations.append(observation)
            result.claims.extend(
                claims_from_detection(observation, detection, source_channel="sar")
            )

        result.coverage = {(self.modality, time_key): len(result.observations)}
        return result


__all__ = ["SARAdapter"]
