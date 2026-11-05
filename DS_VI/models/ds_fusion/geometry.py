"""Geometry for the D-S fusion pipeline: NMS, registration, association.

All pair geometry primitives are imported from
``DS_VI.models.shared_identity.physical_features`` so the experiment cannot
drift from the shared implementation used by the cross-modal workers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from DS_VI.models.shared_identity.physical_features import (
    bbox_area_ratio,
    bbox_center_distance,
    bbox_diagonal,
    bbox_iou,
    center_score_from_distance,
)


def polygon_of(box: Dict[str, Any]):
    import shapely.geometry as sgeom

    points = box.get("polygon")
    if points:
        return sgeom.Polygon(points)
    x1, y1, x2, y2 = [float(v) for v in box["bbox"][:4]]
    return sgeom.box(x1, y1, x2, y2)


def rotated_iou(box_a: Dict[str, Any], box_b: Dict[str, Any]) -> float:
    if not box_a.get("polygon") and not box_b.get("polygon"):
        return bbox_iou(box_a["bbox"], box_b["bbox"])

    poly_a = polygon_of(box_a)
    poly_b = polygon_of(box_b)
    if not poly_a.is_valid or not poly_b.is_valid:
        return bbox_iou(box_a["bbox"], box_b["bbox"])
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def rotated_nms(boxes: Sequence[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Greedy per-class NMS; works for axis-aligned and rotated polygon boxes."""
    ordered = sorted(boxes, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        survivors = []
        for candidate in ordered:
            same_class = str(candidate.get("label")) == str(best.get("label"))
            if same_class and rotated_iou(best, candidate) > float(iou_threshold):
                continue
            survivors.append(candidate)
        ordered = survivors
    return kept


def estimate_affine(
    rgb_image: Optional[str],
    sar_image: Optional[str],
    mode: str = "identity",
    min_inliers: int = 8,
    max_rotation_deg: float = 15.0,
    max_shift_px: float = 200.0,
    max_side: int = 0,
) -> Dict[str, Any]:
    """Estimate the SAR -> RGB affine transform. Always returns a usable matrix.

    ``max_side > 0`` estimates on a downscaled copy (same aspect, longest side at
    most ``max_side``) and rescales the translation back, which keeps ECC/ORB cost
    bounded on large frames. Rotation and scale are unaffected by the resize.
    """
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    wanted = str(mode or "identity").strip().lower()
    info: Dict[str, Any] = {"requested": wanted, "applied": "identity", "matrix": identity.tolist(), "inliers": 0}
    if wanted == "identity" or not rgb_image or not sar_image:
        return info
    rgb_path = Path(str(rgb_image))
    sar_path = Path(str(sar_image))
    if not rgb_path.is_file() or not sar_path.is_file():
        info["reason"] = "image missing"
        return info
    try:
        import cv2
    except ImportError as exc:
        info["reason"] = f"cv2 unavailable: {exc}"
        return info
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
    sar = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
    if rgb is None or sar is None:
        info["reason"] = "image unreadable"
        return info
    resize_factor = 1.0
    longest = max(rgb.shape[0], rgb.shape[1])
    if int(max_side) > 0 and longest > int(max_side):
        resize_factor = float(max_side) / float(longest)
        size = (max(1, int(rgb.shape[1] * resize_factor)), max(1, int(rgb.shape[0] * resize_factor)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        sar = cv2.resize(sar, size, interpolation=cv2.INTER_AREA)
        info["resize_factor"] = round(resize_factor, 6)
    if wanted == "ecc":
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
            correlation, warp = cv2.findTransformECC(rgb, sar, warp, cv2.MOTION_AFFINE, criteria, None, 5)
            # ``correlation`` is the final ECC coefficient in [-1, 1]; 1 means the
            # images align, so ``1 - correlation`` is a bounded mis-registration
            # estimate that adaptive gates can consume without any new knob.
            if resize_factor != 1.0:
                warp = warp.copy()
                warp[0, 2] = warp[0, 2] / resize_factor
                warp[1, 2] = warp[1, 2] / resize_factor
            info.update({
                "applied": "ecc",
                "matrix": warp.tolist(),
                "inliers": -1,
                "correlation": float(correlation),
                "residual": max(0.0, 1.0 - float(correlation)),
            })
            return info
        except cv2.error as exc:
            info["reason"] = f"ecc failed: {exc}"
            return info
    if wanted != "orb":
        info["reason"] = f"unsupported registration mode {wanted!r}"
        return info
    try:
        detector = cv2.ORB_create(nfeatures=2000)
        key_rgb, des_rgb = detector.detectAndCompute(rgb, None)
        key_sar, des_sar = detector.detectAndCompute(sar, None)
        if des_rgb is None or des_sar is None or len(key_rgb) < 4 or len(key_sar) < 4:
            info["reason"] = "too few ORB features"
            return info
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(des_sar, des_rgb), key=lambda m: m.distance)
        keep = matches[: max(4, int(0.2 * len(matches)))]
        if len(keep) < 4:
            info["reason"] = "too few matches"
            return info
        src = np.float32([key_sar[m.queryIdx].pt for m in keep]).reshape(-1, 1, 2)
        dst = np.float32([key_rgb[m.trainIdx].pt for m in keep]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if matrix is None or inliers is None:
            info["reason"] = "RANSAC failed"
            return info
        n_inliers = int(inliers.sum())
        scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
        rotation_deg = abs(math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))))
        shift = float(math.hypot(float(matrix[0, 2]), float(matrix[1, 2])))
        if (
            n_inliers < int(min_inliers)
            or rotation_deg > float(max_rotation_deg)
            or shift > float(max_shift_px)
            or not (0.7 <= scale <= 1.4)
        ):
            info["reason"] = (
                f"rejected estimate: inliers={n_inliers}, rot={rotation_deg:.2f}deg, "
                f"shift={shift:.1f}px, scale={scale:.3f}"
            )
            return info
        if resize_factor != 1.0:
            matrix = matrix.copy()
            matrix[0, 2] = matrix[0, 2] / resize_factor
            matrix[1, 2] = matrix[1, 2] / resize_factor
        inlier_mask = inliers.ravel().astype(bool)
        errors = []
        if inlier_mask.any():
            predicted = cv2.transform(src, matrix)
            deltas = predicted.reshape(-1, 2)[inlier_mask] - dst.reshape(-1, 2)[inlier_mask]
            errors = np.linalg.norm(deltas, axis=1)
        reproj_median = float(np.median(errors)) if len(errors) else None
        height, width = rgb.shape[:2]
        diagonal = float(math.hypot(width, height)) or 1.0
        info.update({
            "applied": "orb",
            "matrix": matrix.tolist(),
            "inliers": n_inliers,
            "rotation_deg": rotation_deg,
            "reproj_median_px": reproj_median,
            "residual": (
                min(1.0, reproj_median / diagonal) if reproj_median is not None else None
            ),
        })
        return info
    except cv2.error as exc:
        info["reason"] = f"orb failed: {exc}"
        return info


def transform_boxes(boxes: Sequence[Dict[str, Any]], matrix: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
    mat = np.array(matrix, dtype=float)
    if np.allclose(mat, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])):
        return list(boxes)
    out: List[Dict[str, Any]] = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box["bbox"][:4]]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        warped = corners @ mat[:, :2].T + mat[:, 2]
        nx1, ny1 = warped[:, 0].min(), warped[:, 1].min()
        nx2, ny2 = warped[:, 0].max(), warped[:, 1].max()
        item = dict(box)
        item["bbox"] = [float(nx1), float(ny1), float(nx2), float(ny2)]
        item.pop("polygon", None)
        out.append(item)
    return out


def affine_registration(
    rgb_boxes: Sequence[Dict[str, Any]],
    sar_boxes: Sequence[Dict[str, Any]],
    rgb_image: Optional[str],
    sar_image: Optional[str],
    mode: str = "identity",
    min_inliers: int = 8,
    max_rotation_deg: float = 15.0,
    max_shift_px: float = 200.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    info = estimate_affine(
        rgb_image,
        sar_image,
        mode=mode,
        min_inliers=min_inliers,
        max_rotation_deg=max_rotation_deg,
        max_shift_px=max_shift_px,
    )
    return list(rgb_boxes), transform_boxes(sar_boxes, info["matrix"]), info


def _normalize_boxes(boxes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for index, box in enumerate(boxes):
        bbox = box.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        if x2 <= x1 or y2 <= y1:
            continue
        item = dict(box)
        item["bbox"] = [x1, y1, x2, y2]
        item.setdefault("index", index)
        out.append(item)
    return out


def pair_geometry(rgb_box: Dict[str, Any], sar_box: Dict[str, Any], weights: Dict[str, float], center_scale: float) -> Dict[str, float]:
    iou = rotated_iou(rgb_box, sar_box)
    distance = float(bbox_center_distance(rgb_box["bbox"], sar_box["bbox"]))
    threshold = max(bbox_diagonal(rgb_box["bbox"]), bbox_diagonal(sar_box["bbox"]), 1.0) * float(center_scale)
    center_score = float(center_score_from_distance(distance, threshold))
    area_ratio = float(bbox_area_ratio(rgb_box["bbox"], sar_box["bbox"]))
    match_score = (
        float(weights.get("w_iou", 0.5)) * iou
        + float(weights.get("w_center", 0.3)) * center_score
        + float(weights.get("w_area", 0.2)) * area_ratio
    )
    return {
        "iou": round(float(iou), 6),
        "center_distance": round(distance, 3),
        "center_threshold": round(threshold, 3),
        "center_score": round(center_score, 6),
        "area_ratio": round(area_ratio, 6),
        "match_score": round(float(match_score), 6),
    }


def hungarian_association(
    rgb_boxes: Sequence[Dict[str, Any]],
    sar_boxes: Sequence[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    center_scale: float = 0.75,
    gate_iou: float = 0.05,
    gate_score: float = 0.35,
    forbid_cost: float = 1e6,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One-to-one RGB/SAR association with the Hungarian algorithm.

    Returns ``(matched, rgb_only, sar_only)`` where ``matched`` items carry
    ``rgb_index``/``sar_index`` plus the geometry record. A pair is admissible
    when ``IoU >= gate_iou`` or ``match_score >= gate_score``.
    """
    from scipy.optimize import linear_sum_assignment

    weights = dict(weights or {"w_iou": 0.5, "w_center": 0.3, "w_area": 0.2})
    rgb_list = _normalize_boxes(rgb_boxes)
    sar_list = _normalize_boxes(sar_boxes)
    if not rgb_list or not sar_list:
        matched: List[Dict[str, Any]] = []
        rgb_only = [{"rgb_index": item["index"], "rgb_bbox": item["bbox"], "rgb_score": item.get("score"), "rgb_label": item.get("label")} for item in rgb_list]
        sar_only = [{"sar_index": item["index"], "sar_bbox": item["bbox"], "sar_score": item.get("score"), "sar_label": item.get("label")} for item in sar_list]
        return matched, rgb_only, sar_only

    cost = np.full((len(rgb_list), len(sar_list)), forbid_cost, dtype=float)
    records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i, rgb_box in enumerate(rgb_list):
        for j, sar_box in enumerate(sar_list):
            record = pair_geometry(rgb_box, sar_box, weights, center_scale)
            allowed = record["iou"] >= float(gate_iou) or record["match_score"] >= float(gate_score)
            if allowed:
                cost[i, j] = 1.0 - record["match_score"]
            records[(i, j)] = record

    rows, cols = linear_sum_assignment(cost)
    matched_pairs: List[Dict[str, Any]] = []
    used_rgb, used_sar = set(), set()
    for i, j in zip(rows.tolist(), cols.tolist(), strict=False):
        if cost[i, j] >= forbid_cost:
            continue
        record = records[(i, j)]
        matched_pairs.append({
            "rgb_index": rgb_list[i]["index"],
            "sar_index": sar_list[j]["index"],
            "rgb_box": rgb_list[i],
            "sar_box": sar_list[j],
            **record,
        })
        used_rgb.add(i)
        used_sar.add(j)
    matched_pairs.sort(key=lambda item: float(item["match_score"]), reverse=True)

    rgb_only = [
        {"rgb_index": item["index"], "rgb_bbox": item["bbox"], "rgb_score": item.get("score"), "rgb_label": item.get("label")}
        for i, item in enumerate(rgb_list)
        if i not in used_rgb
    ]
    sar_only = [
        {"sar_index": item["index"], "sar_bbox": item["bbox"], "sar_score": item.get("score"), "sar_label": item.get("label")}
        for j, item in enumerate(sar_list)
        if j not in used_sar
    ]
    return matched_pairs, rgb_only, sar_only

def _centroid_delta(rgb_box: Mapping[str, Any], sar_box: Mapping[str, Any]) -> Tuple[float, float]:
    """Displacement from the view-A centre to the view-B centre (pixels)."""
    from DS_VI.models.shared_identity.physical_features import bbox_center

    left = bbox_center(rgb_box["bbox"])
    right = bbox_center(sar_box["bbox"])
    if left is None or right is None:
        return (0.0, 0.0)
    return (float(right[0] - left[0]), float(right[1] - left[1]))


def associate_ot(
    rgb_boxes: Sequence[Dict[str, Any]],
    sar_boxes: Sequence[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    center_scale: float = 0.75,
    gate_iou: float = 0.05,
    gate_score: float = 0.35,
    reg: float = 0.10,
    tau_a: Optional[float] = None,
    tau_b: Optional[float] = None,
    dustbin_cost: float = 0.5,
    min_mass: float = 0.5,
    forbid_cost: float = 1.0,
    device: str = "cpu",
    backend: str = "auto",
    displacement_weight: float = 0.0,
    attribute_weight: float = 0.0,
    attributes_a: Optional[Sequence[Mapping[str, Any]]] = None,
    attributes_b: Optional[Sequence[Mapping[str, Any]]] = None,
    attribute_weights: Optional[Mapping[str, float]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """One-to-one association by unbalanced Sinkhorn with an explicit dustbin.

    Same contract as :func:`hungarian_association`, but the assignment is allowed
    to *decline*: a detection whose best counterpart is weak parks its mass on a
    dustbin column instead of being forced onto the least-bad partner, and the KL
    relaxed marginals (``tau_a``/``tau_b``) let the two sides differ in size. This
    targets the failure mode measured in the over-merge analysis (a confirmed wrong
    pair consumes two boxes), which Hungarian + a hard gate cannot express.
    """
    backend_name = str(backend or "auto").strip().lower()
    if backend_name == "auto":
        from DS_VI.models.opt import torch_available
        backend_name = "torch" if torch_available() else "numpy"
    if backend_name == "numpy":
        from DS_VI.models.opt.numpy_ot import log_sinkhorn as solve_np
        from DS_VI.models.opt.torch_ot import assign_from_plan, with_dustbin_square
        solve = lambda cost, **kw: solve_np(cost, **{k: v for k, v in kw.items() if k != "device"})
    else:
        from DS_VI.models.opt.torch_ot import assign_from_plan, log_sinkhorn as solve_torch, with_dustbin_square
        solve = solve_torch

    weights = dict(weights or {"w_iou": 0.5, "w_center": 0.3, "w_area": 0.2})
    rgb_list = _normalize_boxes(rgb_boxes)
    sar_list = _normalize_boxes(sar_boxes)
    if not rgb_list or not sar_list:
        matched: List[Dict[str, Any]] = []
        rgb_only = [{"rgb_index": item["index"], "rgb_bbox": item["bbox"], "rgb_score": item.get("score"), "rgb_label": item.get("label")} for item in rgb_list]
        sar_only = [{"sar_index": item["index"], "sar_bbox": item["bbox"], "sar_score": item.get("score"), "sar_label": item.get("label")} for item in sar_list]
        return matched, rgb_only, sar_only

    rows, cols = len(rgb_list), len(sar_list)
    cost = np.full((rows, cols), float(forbid_cost), dtype=float)
    allowed = np.zeros((rows, cols), dtype=bool)
    records: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for i, rgb_box in enumerate(rgb_list):
        for j, sar_box in enumerate(sar_list):
            record = pair_geometry(rgb_box, sar_box, weights, center_scale)
            records[(i, j)] = record
            if record["iou"] >= float(gate_iou) or record["match_score"] >= float(gate_score):
                cost[i, j] = 1.0 - float(record["match_score"])
                allowed[i, j] = True

    # --- candidate-construction cues -------------------------------------------
    # (1) displacement coherence: a true counterpart shifts like the rest of the
    #     scene; a crossed pairing (A1-B2, A2-B1) produces a displacement that
    #     disagrees with the scene-wide shift. This is the structural cue that a
    #     purely geometric confirmation score lacks.
    # (2) attribute agreement, weighted by measured per-modality informativeness.
    displacement_penalty = np.zeros((rows, cols), dtype=float)
    if float(displacement_weight) > 0.0 and allowed.any():
        shifts = []
        for (i, j) in zip(*np.nonzero(allowed)):
            record = records[(int(i), int(j))]
            shifts.append(_centroid_delta(rgb_list[int(i)], sar_list[int(j)]))
        if shifts:
            array = np.asarray(shifts, dtype=float)
            median = np.median(array, axis=0)
            scale = float(np.median(np.linalg.norm(array - median, axis=1))) + 1e-6
            for (i, j) in zip(*np.nonzero(allowed)):
                delta = np.asarray(_centroid_delta(rgb_list[int(i)], sar_list[int(j)])) - median
                displacement_penalty[int(i), int(j)] = min(
                    1.0, float(np.linalg.norm(delta)) / (3.0 * scale))
    attribute_penalty = np.zeros((rows, cols), dtype=float)
    if (float(attribute_weight) > 0.0 and attributes_a is not None and attributes_b is not None
            and attribute_weights):
        for i in range(rows):
            for j in range(cols):
                if not allowed[i, j]:
                    continue
                total = 0.0
                disagree = 0.0
                left = attributes_a[i] if i < len(attributes_a) else {}
                right = attributes_b[j] if j < len(attributes_b) else {}
                for attribute, weight in attribute_weights.items():
                    if float(weight) <= 0.0:
                        continue
                    value_left = (left or {}).get(attribute)
                    value_right = (right or {}).get(attribute)
                    if not value_left or not value_right:
                        continue
                    total += float(weight)
                    if str(value_left) != str(value_right):
                        disagree += float(weight)
                if total > 0.0:
                    attribute_penalty[i, j] = disagree / total
    if float(displacement_weight) > 0.0 or float(attribute_weight) > 0.0:
        normaliser = 1.0 + float(displacement_weight) + float(attribute_weight)
        for i in range(rows):
            for j in range(cols):
                if not allowed[i, j]:
                    continue
                combined = (cost[i, j]
                            + float(displacement_weight) * displacement_penalty[i, j]
                            + float(attribute_weight) * attribute_penalty[i, j])
                cost[i, j] = combined / normaliser

    # Balanced Sinkhorn on the square dustbin-augmented cost: exact marginals make
    # the solve stable, and each row/column has its own exit so nothing underflows.
    augmented = with_dustbin_square(cost, dustbin_cost=float(dustbin_cost))
    mask = np.ones_like(augmented, dtype=bool)
    mask[:rows, :cols] = allowed
    result = solve(
        augmented, reg=float(reg), mask=mask, device=str(device),
        iterations=1000, tol=1e-12,
        **({} if tau_a is None else {"tau_a": float(tau_a), "tau_b": float(tau_b)}),
    )
    assignment = assign_from_plan(result["plan"][:rows, :cols], min_mass=float(min_mass))

    matched_pairs: List[Dict[str, Any]] = []
    used_rgb: set = set()
    used_sar: set = set()
    for pair in assignment["pairs"]:
        i, j = int(pair["row"]), int(pair["col"])
        matched_pairs.append({
            "rgb_index": rgb_list[i]["index"],
            "sar_index": sar_list[j]["index"],
            "rgb_box": rgb_list[i],
            "sar_box": sar_list[j],
            "ot_mass": round(float(pair["mass"]), 6),
            **records[(i, j)],
        })
        used_rgb.add(i)
        used_sar.add(j)
    matched_pairs.sort(key=lambda item: float(item["match_score"]), reverse=True)

    rgb_only = [
        {"rgb_index": item["index"], "rgb_bbox": item["bbox"], "rgb_score": item.get("score"), "rgb_label": item.get("label")}
        for i, item in enumerate(rgb_list) if i not in used_rgb
    ]
    sar_only = [
        {"sar_index": item["index"], "sar_bbox": item["bbox"], "sar_score": item.get("score"), "sar_label": item.get("label")}
        for j, item in enumerate(sar_list) if j not in used_sar
    ]
    if diagnostics is not None:
        diagnostics.update({
            "method": f"ot_unbalanced:{backend_name}",
            "reg": float(reg),
            "tau_a": None if tau_a is None else float(tau_a),
            "tau_b": None if tau_b is None else float(tau_b),
            "dustbin_cost": float(dustbin_cost),
            "min_mass": float(min_mass),
            "iterations": int(result["iterations"]),
            "converged": bool(result["converged"]),
            "total_mass": float(result["total_mass"]),
            "matched": len(matched_pairs),
            "rows": rows,
            "cols": cols,
        })
    return matched_pairs, rgb_only, sar_only

