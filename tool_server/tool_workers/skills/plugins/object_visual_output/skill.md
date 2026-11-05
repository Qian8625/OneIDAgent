---
description: Final visualization layer for existing detection, selection, and measurement structured outputs.
---

# When to use
- The task asks to draw boxes, plot points, mark candidates, highlight objects, or annotate existing results.
- The task already has bbox, point, detection, selected detection, measurement, or structured output from a previous tool or skill.
- The user needs the final visual answer image for inspection or evaluation.

# Do not use when
- The task asks to detect, select, or measure objects without asking for visualization.
- The task lacks any bbox, point, detection, selected detection, measurement object, or structured visual target.
- The task is SAR-specific, cross-modal, GeoTIFF/POI, or measurement-only.

# Scope
- This is the final output layer skill.
- It does not detect, select, or calculate.
- It only consumes existing visual targets from explicit inputs, previous structured outputs, or raw tool outputs.
- If no bbox or point input exists, it must not fabricate positions and must return `structured.status = missing_visual_target`.

# Supported visual_type
- box
- point
- plot
- box_and_label
- measurement_overlay

# Output contract
- The response includes `image`, `output_path`, `structured`, and `artifacts`.
- `artifacts` must include the visualization image with `type=image` on success.
- For plot or point tasks, bbox inputs are plotted at their centroids, never at arbitrary corners.

# Agent examples
- `{"skill_name":"ObjectVisualOutputSkill","image":"/path/to/image.png","visual_type":"box","bbox":[10,20,60,80]}`
- `{"skill_name":"ObjectVisualOutputSkill","image":"/path/to/image.png","visual_type":"box_and_label","selected_detections":[{"bbox":[10,20,60,80],"label":"tank"}]}`
- `{"skill_name":"ObjectVisualOutputSkill","image":"/path/to/image.png","visual_type":"plot","points":[[35,50]],"labels":["candidate"]}`
- `{"skill_name":"ObjectVisualOutputSkill","image":"/path/to/image.png","visual_type":"measurement_overlay","structured":{"objects_used":[...],"measurement":{"distance_m":25.0}}}`
