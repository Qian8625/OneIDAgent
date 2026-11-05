---
description: Detect candidate targets in a training image and select final targets by a selection rule.
---

# When to use
- The task asks to detect, locate, find, or select target objects in one optical training image.
- The task asks for selection by size, position, image-center proximity, or an ordinal candidate.
- The downstream agent or evaluator needs structured candidate and selected bbox output.

# Do not use when
- The task asks to compute distance, area, GSD, pixel size, or any real-world measurement.
- The task asks to segment targets, count pixels, draw the final answer image, or produce a measurement overlay.
- The task is SAR-specific, cross-modal, GeoTIFF/POI, or before/after change analysis.

# Scope
- This skill only detects candidate targets and selects final target detections.
- It must not compute distances, areas beyond bbox pixel area metadata, GSD, or real-world measurements.
- It must not create the final DrawBox answer. If `visualize=true`, it may create a candidate-debug visualization artifact only.

# Required inputs
- image
- target
- selection_mode

# Optional inputs
- top_k
- confidence_threshold
- iou_threshold
- detector
- visualize
- max_draw
- nth

# Selection modes
- all
- smallest
- largest
- nearest
- farthest
- leftmost
- rightmost
- topmost
- bottommost
- nth

# Output contract
- The response must include `structured` for evaluation.
- `structured.status` must be `success` when candidates are detected, or `no_detection` when no target is detected.
- `structured.detections` must contain candidate boxes with `bbox`, `score`, `label`, `area_px`, `centroid`, and `rank`.
- `structured.selected` must contain selected boxes with the same fields plus `selection_reason`.

# Agent examples
- `{"skill_name":"ObjectDetectSelectSkill","image":"/path/to/image.png","target":"storage tank","selection_mode":"smallest","top_k":1}`
- `{"skill_name":"ObjectDetectSelectSkill","image":"/path/to/image.png","target":"airplane","selection_mode":"largest","confidence_threshold":0.25}`
- `{"skill_name":"ObjectDetectSelectSkill","image":"/path/to/image.png","target":"ship","selection_mode":"leftmost","visualize":true,"max_draw":20}`
