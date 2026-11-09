"""Shared helpers for region-probe (bbox) detection: parse/clip a bbox and shift boxes back.

Extracted so DenoDetV2 and SmallObjectDetection cannot drift apart.
"""


def _probe_norm_bbox(bbox):
    """Parse a probe bbox from '(x1,y1,x2,y2)' / [x1,y1,x2,y2]; raises ValueError."""
    if isinstance(bbox, (list, tuple)):
        if len(bbox) != 4:
            raise ValueError(f"expected 4 bbox values, got {bbox}")
        vals = [float(v) for v in bbox]
    elif isinstance(bbox, str):
        import re
        if "<" in bbox or ">" in bbox:
            raise ValueError(f"invalid bbox placeholder: {bbox}")
        nums = re.findall(r'([-+]?\d*\.?\d+)', bbox)
        if len(nums) != 4:
            raise ValueError(f"expected 4 numbers, got {bbox}")
        vals = [float(v) for v in nums]
    else:
        raise ValueError(f"invalid bbox type: {type(bbox).__name__}")
    return vals


def _probe_crop_box(bbox, size):
    """Clip a probe bbox to the image and validate it; returns (x1,y1,x2,y2) ints."""
    W, H = size
    x1, y1, x2, y2 = _probe_norm_bbox(bbox)
    x1 = max(0.0, min(x1, float(W)))
    y1 = max(0.0, min(y1, float(H)))
    x2 = max(0.0, min(x2, float(W)))
    y2 = max(0.0, min(y2, float(H)))
    ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        raise ValueError(f"invalid bbox region {(ix1, iy1, ix2, iy2)} for image size {(W, H)}")
    return ix1, iy1, ix2, iy2


def _probe_offset_detections(detections, x_off, y_off):
    """Shift crop-local boxes back to full-image coordinates (int values stay int)."""
    shifted = []
    for det in detections or []:
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            shifted.append(det)
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]

        def _cast(v):
            f = float(v)
            return int(f) if f.is_integer() else f

        shifted.append({**det, "bbox": [_cast(x1 + x_off), _cast(y1 + y_off),
                                       _cast(x2 + x_off), _cast(y2 + y_off)]})
    return shifted


def _safe_source(image_input, limit=300):
    """Keep probe meta small even when the input was a base64 payload."""
    s = str(image_input)
    return s if len(s) <= limit else s[:limit] + f"...(len={len(s)})"
