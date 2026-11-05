# CrossModalMatchSkill

`CrossModalMatchSkill` performs target-level optical-SAR matching. It only returns structured matching evidence and does not perform final question answering, attribution, verification, or task-level conclusions.

## Inputs

- `rgb_image`: optical/RGB image path.
- `sar_image`: SAR image path.
- `target`: target class or description.
- `rgb_detections`: optional RGB detections. Aliases include `detections` and `rgb_boxes`.
- `sar_candidates`: optional SAR candidates. Aliases include `sar_detections` and `sar_boxes`.
- `iou_threshold`: IoU threshold for strong matches. Default is `0.2`.
- `center_distance_threshold`: `auto` or numeric pixel distance threshold. Default is `auto`.
- `score_threshold`: minimum detection/candidate score and weak-match score. Default is `0.3`.
- `match_strategy`: `iou`, `center_distance`, `hybrid`, or `global`. Default is `hybrid`.
- `visualize`: whether to emit a visualization artifact. Default is `true`.
- `top1`: keep only the top-scoring candidate from each modality. Default is `false`.
- `max_draw`: maximum number of boxes to draw. Default is `20`.

## Fallbacks

If detections are missing, the skill reuses existing skills instead of loading new models:

- Missing RGB detections: call `ObjectDetectSelectSkill` with `selection_mode=all`.
- Missing SAR candidates: call `SAREvidenceExtractSkill` and consume `sar_candidates`.

## Pairing Backend

Pairing is delegated to `CrossModalInterpret` with `mode=shared_identity`
(D-S pairing -> shared-identity evidence -> cascade assembly). Two consequences
matter to callers:

- `match_score` is the **VI blend probability** (the cascade score), not the
  geometric hybrid score. The old geometric score is preserved as
  `geometry_match_score`, together with `vi_status`, `vi_p_class` and
  `vi_association`, so a downstream threshold can be re-fitted.
- `matched_pairs` only contains pairs the shared-identity layer kept; detections
  the evidence layer rejected appear as modality-only candidates.

The local geometric matcher below stays available as a fallback. It is used when:

- the target is not one of the six canonical classes (`vehicle`, `tank`, `ship`,
  `aircraft`, `bridge`, `harbor`) -- `structured.degrade_reason` is
  `unsupported_target`;
- the interpret worker is unreachable, errors, or returns no mappable object --
  `structured.degrade_reason` is `interpret_unavailable` / `interpret_empty` /
  `interpret_unmapped`.

`structured.mode` is `shared_identity` or `legacy_local`; `structured.pairing`
carries the same information plus `blend_weight` and elapsed time.

## Matching Strategies

- `iou`: accept pairs primarily by IoU.
- `center_distance`: accept pairs by center distance.
- `hybrid`: combine IoU, center distance, area ratio, and confidence.
- `global`: score all candidate pairs and greedily assign one-to-one matches globally.

The matcher prevents one SAR candidate from being reused by multiple RGB detections. High-IoU matches are labeled `strong_match`; low-IoU but center-supported matches are labeled `weak_match`.

## Outputs

- `matched_pairs`: one-to-one RGB/SAR pairs with `match_score`, `match_type`, and `match_reason`.
- `rgb_only`: RGB detections that have no SAR match.
- `sar_only`: SAR candidates that have no RGB match.
- `dedup_detections`: fused matched targets plus modality-only detections for downstream evaluation.
- `match_summary`: counts for `rgb_count`, `sar_count`, `matched_count`, `rgb_only_count`, `sar_only_count`, and `dedup_count`.
- `structured`: stable machine-readable payload for evaluation, including `mode`,
  `degraded`, `degrade_reason` and `pairing` (pairing backend and its metadata).
- `artifacts`: optional visualization artifacts with `type: image`.

## Example

```json
{
  "rgb_image": "/data/optical.png",
  "sar_image": "/data/sar.png",
  "target": "ship",
  "rgb_detections": [{"bbox": [10, 10, 50, 50], "score": 0.9, "label": "ship"}],
  "sar_candidates": [{"bbox": [12, 12, 52, 52], "score": 0.8, "label": "ship"}],
  "match_strategy": "hybrid",
  "visualize": true
}
```
