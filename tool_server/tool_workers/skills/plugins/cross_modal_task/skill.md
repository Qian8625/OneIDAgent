# CrossModalTaskSkill

`CrossModalTaskSkill` performs task-level optical-SAR reasoning. It is not a low-level matcher. It consumes existing cross-modal evidence, or calls `CrossModalMatchSkill` to obtain the underlying `matched_pairs`, `rgb_only`, and `sar_only` evidence before making a final task judgment.

The skill must return both a final task label and the evidence used to support that label. It must not output a positive conclusion without evidence.

## Supported Tasks

- `optical_sar_verify`: decide whether an optical/RGB suspected target is supported by SAR evidence. Labels: `supported` or `unsupported`.
- `sar_optical_attribution`: decide what optical target or region a SAR candidate corresponds to. Labels: `attributed` or `unattributed`.
- `sar_candidate_recall`: decide whether SAR adds a credible candidate missed by optical detection. Labels: `recalled` or `not_recalled`.
- `optical_sar_consistency`: decide whether optical target evidence and SAR response are mutually consistent. Labels: `consistent` or `inconsistent`.

Common aliases such as `verify`, `validate`, `crosscheck exists`, `largest`, `highest-confidence`, and `select_largest` are normalized to `optical_sar_verify`.

## Evidence Sources

The skill prefers provided evidence:

- `matched_pairs`
- `rgb_detections`
- `sar_candidates` or `sar_detections`
- `rgb_only`
- `sar_only`

If `matched_pairs` are missing, the skill calls `CrossModalMatchSkill` and then performs its own task-level decision from the returned evidence.

## Inputs

- `rgb_image`: optical/RGB image path.
- `sar_image`: SAR image path.
- `target`: target class or description.
- `task_type`: one of `optical_sar_verify`, `sar_optical_attribution`, `sar_candidate_recall`, or `optical_sar_consistency`.
- `rgb_detections`: optional RGB detections.
- `sar_candidates` or `sar_detections`: optional SAR candidates.
- `matched_pairs`: optional precomputed optical-SAR matched pairs.
- `rgb_only`, `sar_only`: optional modality-only evidence from previous matching.
- `question`: optional original task question.
- `evidence_required`: whether evidence is required before positive labels. Default is `true`.
- `confidence_threshold`: confidence threshold for task decisions. Default is `0.3`.
- `match_threshold` or `iou_threshold`: optional matching threshold passed to `CrossModalMatchSkill`.
- `visualize`: whether to emit a visualization artifact. Default is `true`.
- `max_draw`: maximum boxes to draw. Default is `20`.

## Output

Every response contains:

- `text`: human-readable task answer.
- `error_code`: `0` for completed reasoning, non-zero for invalid input or runtime failure.
- `task_result`: `{label, confidence, answer}`.
- `evidence`: matched pairs, RGB evidence, SAR evidence, modality-only candidates, and an evidence summary.
- `structured`: stable machine-readable result.
- `artifacts`: optional visualization artifacts with `type: image`.

`structured.status` is one of `success`, `insufficient_evidence`, or `missing_inputs`.

## Examples

Verify an optical candidate with SAR:

```json
{
  "rgb_image": "/data/optical.png",
  "sar_image": "/data/sar.png",
  "target": "ship",
  "task_type": "optical_sar_verify",
  "question": "Is this optical ship supported by SAR evidence?",
  "visualize": true
}
```

Attribute a SAR candidate to an optical target:

```json
{
  "rgb_image": "/data/optical.png",
  "sar_image": "/data/sar.png",
  "target": "aircraft",
  "task_type": "sar_optical_attribution",
  "sar_candidates": [{"bbox": [10, 20, 50, 70], "score": 0.82, "label": "aircraft"}],
  "rgb_detections": [{"bbox": [12, 18, 52, 72], "score": 0.76, "label": "aircraft"}]
}
```

Recall optical-missed SAR candidates:

```json
{
  "rgb_image": "/data/optical.png",
  "sar_image": "/data/sar.png",
  "target": "ship",
  "task_type": "sar_candidate_recall",
  "sar_only": [{"sar_bbox": [100, 80, 140, 118], "sar_score": 0.77, "sar_label": "ship"}]
}
```

Check optical-SAR consistency:

```json
{
  "rgb_image": "/data/optical.png",
  "sar_image": "/data/sar.png",
  "target": "vehicle",
  "task_type": "optical_sar_consistency",
  "matched_pairs": [{"rgb_bbox": [1, 1, 20, 20], "sar_bbox": [80, 80, 100, 100], "match_score": 0.05, "rgb_label": "vehicle", "sar_label": "ship"}]
}
```
