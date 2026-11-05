# SAREvidenceExtractSkill

Extracts SAR-only target evidence candidates from a SAR image. The skill returns candidate boxes, centroids, pixel areas, detector confidence, SAR response statistics, and optional visualization artifacts.

This skill is intentionally evidence-only. It does not compare SAR with optical imagery, make final cross-modal consistency claims, perform distance or area measurement in real-world units, or generate natural-language attribution conclusions.

## Inputs

- `sar_image` (required): path to the SAR image.
- `target` (required): target class or SAR evidence description.
- `preprocess_mode`: `auto`, `direct`, or `sar_to_rgb`. Default is `auto`.
- `confidence_threshold`: DenoDetV2 confidence threshold. Default is `0.25`.
- `response_threshold`: `auto` or numeric response threshold. Values in `[0, 1]` compare against normalized max response; larger values compare against 0-255 max response. Default is `auto`.
- `min_area`, `max_area`: optional candidate area filters in pixels.
- `visualize`: whether to draw candidate boxes. Default is `true`.
- `max_candidates`: maximum candidates to return. Default is `50`.
- `max_draw`: maximum candidate boxes to draw. Default is `50`.
- `detector`: `auto` or `denodetv2`. Default is `auto`.

## Outputs

- `text`: concise extraction summary.
- `error_code`: `0` on success or no SAR response, non-zero on invalid input or runtime errors.
- `sar_candidates`: ranked SAR evidence candidates.
- `structured`: machine-readable result with `status`, thresholds, detector metadata, and candidates.
- `artifacts`: visualization artifacts. When `visualize=true` and candidates exist, includes an item with `type: image`.

`structured.status` is `success` when candidates remain after filtering and `no_sar_response` when no valid SAR candidates are found. The skill never fabricates candidates.

## Example

```json
{
  "sar_image": "/path/to/sar.png",
  "target": "ship",
  "confidence_threshold": 0.25,
  "visualize": true,
  "max_candidates": 20
}
```

## Intended Use

Use this skill before a later cross-modal matching step when SAR-side evidence must be extracted and preserved as structured candidates. For final optical-SAR confirmation, use a dedicated cross-modal matching or confirmation skill instead.
