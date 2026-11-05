"""Temporal change-detection adapter.

The change detector's ``stayed`` / ``left`` / ``appeared`` / ``moved`` verdicts are
ingested as :class:`Relation` evidence and presence/motion claims, never as
deterministic identity links.  A ``stayed`` verdict raises the prior that two
observations share an object; it does not assert it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from DS_VI.models.shared_identity.adapters.base import AdapterResult, BaseAdapter, register
from DS_VI.models.shared_identity.observation_builder import (
    build_claim,
    build_observation,
    detection_bbox,
    detection_score,
)
from DS_VI.models.shared_identity.schema import (
    Modality,
    Relation,
    RelationType,
    stable_id,
)

# Relation type -> (presence value at t1, motion value)
_RELATION_CLAIMS: Dict[RelationType, Dict[str, str]] = {
    RelationType.STAYED: {"presence": "present", "motion": "static"},
    RelationType.MOVED: {"presence": "present", "motion": "moving"},
    RelationType.LEFT: {"presence": "absent"},
    RelationType.APPEARED: {"presence": "present"},
}


def _as_list(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _collect(response: Mapping[str, Any], *keys: str) -> List[Dict[str, Any]]:
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
class TemporalAdapter(BaseAdapter):
    """Convert change-detection output into temporal observations and relations."""

    tool_names = ("ChangeDetection", "ChangeDetectionSkill", "TemporalChange")
    modality = Modality.TEMPORAL.value

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

        before_key = str(ctx.get("before_time_key", "t0"))
        after_key = str(ctx.get("after_time_key", "t1"))
        before_image = ctx.get("before_image") or ctx.get("primary_image")
        after_image = ctx.get("after_image")
        result = AdapterResult()

        records = _collect(
            response, "relations", "temporal_relations", "changes", "change_pairs", "results"
        )

        for index, record in enumerate(records):
            relation_type = RelationType.coerce(
                record.get("relation")
                or record.get("relation_type")
                or record.get("change_type")
                or record.get("status")
            )
            before_bbox = detection_bbox(
                record, "before_bbox", "source_bbox", "bbox_t0", "old_bbox"
            )
            after_bbox = detection_bbox(
                record, "after_bbox", "target_bbox", "bbox_t1", "new_bbox", "bbox"
            )

            before_id: Optional[str] = None
            after_id: Optional[str] = None

            if before_bbox is not None:
                observation = build_observation(
                    source_tool=str(tool_name),
                    modality=Modality.TEMPORAL,
                    bbox=before_bbox,
                    time_key=before_key,
                    acquisition_id=before_image,
                    detection_confidence=detection_score(record, "before_score", "score"),
                    frame_id=before_image,
                    input_roots=[a for a in [before_image] if a],
                    context={
                        "phase": "before",
                        "relation": relation_type.value,
                        "relation_group_id": ctx.get("relation_group_id"),
                        "image_group_id": ctx.get("before_image_group_id") or before_image,
                        "related_image_group_ids": [
                            ctx.get("after_image_group_id") or after_image
                        ],
                    },
                    round_id=ctx.get("round_id"),
                    index=index,
                )
                result.observations.append(observation)
                before_id = observation.observation_id

            if after_bbox is not None:
                observation = build_observation(
                    source_tool=str(tool_name),
                    modality=Modality.TEMPORAL,
                    bbox=after_bbox,
                    time_key=after_key,
                    acquisition_id=after_image,
                    detection_confidence=detection_score(record, "after_score", "score"),
                    frame_id=after_image,
                    input_roots=[a for a in [after_image] if a],
                    context={
                        "phase": "after",
                        "relation": relation_type.value,
                        "relation_group_id": ctx.get("relation_group_id"),
                        "image_group_id": ctx.get("after_image_group_id") or after_image,
                        "related_image_group_ids": [
                            ctx.get("before_image_group_id") or before_image
                        ],
                    },
                    round_id=ctx.get("round_id"),
                    index=index,
                )
                result.observations.append(observation)
                after_id = observation.observation_id

                claim_values = _RELATION_CLAIMS.get(relation_type) or {}
                for attribute, value in claim_values.items():
                    result.claims.append(
                        build_claim(
                            observation,
                            attribute,
                            value,
                            source_channel="temporal",
                            raw_confidence=detection_score(record, "score", "confidence"),
                            time_key=after_key,
                            context={"relation": relation_type.value},
                        )
                    )

            if before_id or after_id:
                strength = 0.0
                for key in ("confidence", "score", "iou"):
                    value = record.get(key)
                    if value not in (None, ""):
                        try:
                            strength = max(0.0, min(1.0, float(value)))
                        except (TypeError, ValueError):
                            strength = 0.0
                        break
                result.relations.append(
                    Relation(
                        relation_id=stable_id(
                            "rel",
                            {
                                "a": before_id,
                                "b": after_id,
                                "kind": relation_type.value,
                                "index": index,
                            },
                        ),
                        relation_type=relation_type,
                        source_observation_id=before_id,
                        target_observation_id=after_id,
                        strength=strength,
                        source_tool=str(tool_name),
                        context={
                            "note": "relation evidence only; does not assert identity",
                        },
                    )
                )

        result.coverage = {
            (self.modality, before_key): sum(
                1 for o in result.observations if o.time_scope.key == before_key
            ),
            (self.modality, after_key): sum(
                1 for o in result.observations if o.time_scope.key == after_key
            ),
        }
        if result.is_empty():
            result.skipped_reason = "no temporal relations"
        return result


__all__ = ["TemporalAdapter"]
