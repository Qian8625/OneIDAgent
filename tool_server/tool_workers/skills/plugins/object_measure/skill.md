---
description: Measure object geometry from provided boxes, selected detections, or previous structured outputs.
---

# When to use
- The task asks for geometric measurements such as target area, distance, center point, or bbox size.
- The task provides bbox data, selected detections, detections, or previous structured skill output.
- The task needs GSD conversion from pixel measurements to metric measurements.

# Do not use when
- The task only asks to detect, locate, find, or select target boxes without measuring geometry.
- The task asks for final visual annotation or drawing-only output.
- The task is SAR-specific, cross-modal, GeoTIFF/POI, or before/after change analysis.

# Scope
- This skill only performs object geometry measurement.
- It does not own final visual annotation and should not produce final DrawBox-style answers.
- It does not primarily perform detection. If no bbox-like candidate is available, it may fallback to `ObjectDetectSelectSkill` and must record that fallback in `structured.calculation_trace`.

# Required inputs
- image
- target
- measure_type

# Optional inputs
- `bbox`, `detections`, `selected_detections`, `structured`, or `input_registry` can provide measurement objects.
- `reference_target`, `select_a`, and `select_b` control pair selection.
- `gsd_m_per_pixel` enables metric conversion.
- `unit` is `pixel` or `meter`. Default is `pixel`.
- `visualize` is accepted for compatibility, but this skill does not create final visual annotations.
- `detector` controls fallback detection when no bbox-like input exists.

# Supported measure_type
- area
- distance
- centroid
- bbox_size
- pair_distance

# GSD rules
- Pixel distance multiplied by `gsd_m_per_pixel` gives metric distance: `distance_m = distance_px * gsd_m_per_pixel`.
- Pixel area multiplied by `gsd_m_per_pixel ** 2` gives metric area: `area_m2 = area_px * gsd_m_per_pixel ** 2`.
- By default, the skill returns pixel-unit measurements and sets `structured.unit` to `pixel`.
- If `gsd_m_per_pixel` is provided and `unit=meter`, the skill also returns metric measurements.
- If a metric unit is requested but GSD is missing, `structured.status` is `missing_gsd`.

# Output contract
- The response must include `measurement`, `structured`, and `artifacts`.
- `structured.objects_used` must record every bbox used for the calculation, including roles and selection reasons.
- For smallest/largest pair distance tasks, the two actual bboxes used must be listed explicitly as roles `a` and `b`.
- `artifacts` is normally empty because this skill does not create final visual annotations.

# Agent examples
- `{"skill_name":"ObjectMeasureSkill","image":"/path/to/image.png","target":"storage tank","measure_type":"area","bbox":[10,20,30,60],"gsd_m_per_pixel":0.5}`
- `{"skill_name":"ObjectMeasureSkill","image":"/path/to/image.png","target":"airplane","measure_type":"pair_distance","bbox":[[10,20,30,40],[110,120,130,150]],"gsd_m_per_pixel":0.3}`
- `{"skill_name":"ObjectMeasureSkill","image":"/path/to/image.png","target":"storage tank","measure_type":"pair_distance","select_a":"smallest","select_b":"largest","selected_detections":[...]}`
