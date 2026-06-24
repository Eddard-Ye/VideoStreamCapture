# -*- coding: utf-8 -*-
"""Rotated object size measurement using mask, depth, and camera intrinsics."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from camera_intrinsics import RgbIntrinsics


@dataclass
class RotatedMeasure:
    box_pts: np.ndarray
    length_mm: float
    width_mm: float
    height_mm: float
    z_object_mm: float
    angle_deg: float


def depth_to_mm(depth_raw: np.ndarray, z_unit_mm: float = 1.0) -> np.ndarray:
    depth = depth_raw.astype(np.float32)
    if z_unit_mm != 1.0:
        depth = depth * float(z_unit_mm)
    depth[depth <= 0] = np.nan
    return depth


def align_depth_to_color(
    depth_raw: np.ndarray,
    color_width: int,
    color_height: int,
) -> np.ndarray:
    if depth_raw.shape[1] == color_width and depth_raw.shape[0] == color_height:
        return depth_raw
    return cv2.resize(
        depth_raw,
        (color_width, color_height),
        interpolation=cv2.INTER_NEAREST,
    )


def median_depth_in_roi(
    depth_mm: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    height, width = depth_mm.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    patch = depth_mm[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def plane_depth_in_roi(depth_mm: np.ndarray, x1: int, y1: int, roi_w: int, roi_h: int) -> float:
    """Reference plane depth: prefer center patch, fallback to full ROI median."""
    height, width = depth_mm.shape[:2]
    cx = x1 + roi_w // 2
    cy = y1 + roi_h // 2
    half_w = max(2, min(roi_w // 3, 48))
    half_h = max(2, min(roi_h // 3, 48))
    z_center = median_depth_in_roi(
        depth_mm,
        max(0, cx - half_w),
        max(0, cy - half_h),
        min(width, cx + half_w + 1),
        min(height, cy + half_h + 1),
    )
    if np.isfinite(z_center) and z_center > 0:
        return float(z_center)
    return median_depth_in_roi(depth_mm, x1, y1, x1 + roi_w, y1 + roi_h)


def depth_p90_for_mask(depth_mm: np.ndarray, mask: np.ndarray) -> float:
    """90th percentile of valid depth inside mask (robust object surface depth)."""
    if mask is None or not np.any(mask):
        return float("nan")

    values = depth_mm[mask.astype(bool)]
    valid = values[np.isfinite(values) & (values > 0)]
    if valid.size == 0:
        return float("nan")
    return float(np.percentile(valid, 90))


def object_height_from_plane(z_plane_ref_mm: float, z_object_mm: float) -> float:
    """Physical object height = plane depth - object depth at representative point."""
    if not np.isfinite(z_plane_ref_mm) or not np.isfinite(z_object_mm):
        return float("nan")
    if z_plane_ref_mm <= 0 or z_object_mm <= 0:
        return float("nan")
    return float(max(0.0, z_plane_ref_mm - z_object_mm))


def pixel_edge_len_mm(
    p0: np.ndarray,
    p1: np.ndarray,
    z_mm: float,
    fx: float,
    fy: float,
) -> float:
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    if not np.isfinite(z_mm) or z_mm <= 0 or fx <= 0 or fy <= 0:
        return float("nan")
    mx = dx * z_mm / fx
    my = dy * z_mm / fy
    return float(np.hypot(mx, my))


def largest_contour_from_mask(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def oriented_box_metrics_from_mask(
    mask: np.ndarray,
) -> tuple[np.ndarray, float, float, float] | None:
    """Minimum-area rotated rectangle; length_px is the longer edge."""
    contour = largest_contour_from_mask(mask)
    if contour is None or len(contour) < 3:
        return None

    if len(contour) < 5:
        contour = cv2.convexHull(contour)

    rect = cv2.minAreaRect(contour)
    rw, rh = float(rect[1][0]), float(rect[1][1])
    if rw < 1e-3 or rh < 1e-3:
        return None

    box = np.asarray(cv2.boxPoints(rect), dtype=np.float32)
    length_px = max(rw, rh)
    width_px = min(rw, rh)
    return box, float(length_px), float(width_px), float(rect[2])


def oriented_box_from_mask(mask: np.ndarray) -> np.ndarray | None:
    measured = oriented_box_metrics_from_mask(mask)
    if measured is None:
        return None
    return measured[0]


def min_area_rect_measure_mm(
    contour: np.ndarray,
    z_mm: float,
    fx: float,
    fy: float,
) -> tuple[np.ndarray, float, float, float] | None:
    if contour is None or len(contour) < 4:
        return None

    if len(contour) < 5:
        contour = cv2.convexHull(contour)

    rect = cv2.minAreaRect(contour)
    rw, rh = float(rect[1][0]), float(rect[1][1])
    if rw < 1e-3 or rh < 1e-3:
        return None

    box = np.asarray(cv2.boxPoints(rect), dtype=np.float32)
    edge_a = pixel_edge_len_mm(box[0], box[1], z_mm, fx, fy)
    edge_b = pixel_edge_len_mm(box[1], box[2], z_mm, fx, fy)
    if np.isfinite(edge_a) and np.isfinite(edge_b):
        length_mm, width_mm = (edge_a, edge_b) if edge_a >= edge_b else (edge_b, edge_a)
    else:
        length_mm = width_mm = float("nan")

    angle_deg = float(rect[2])
    return box, float(length_mm), float(width_mm), angle_deg


def measure_mask_mm(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: RgbIntrinsics,
    z_plane_ref_mm: float | None = None,
) -> RotatedMeasure | None:
    contour = largest_contour_from_mask(mask)
    if contour is None:
        return None

    z_object_mm = depth_p90_for_mask(depth_mm, mask)
    measured = min_area_rect_measure_mm(
        contour,
        z_object_mm,
        intrinsics.fx,
        intrinsics.fy,
    )
    if measured is None:
        return None

    box, length_mm, width_mm, angle_deg = measured
    if z_plane_ref_mm is None or not np.isfinite(z_plane_ref_mm):
        height_mm = float("nan")
    else:
        height_mm = object_height_from_plane(z_plane_ref_mm, z_object_mm)

    return RotatedMeasure(
        box_pts=box,
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        z_object_mm=z_object_mm,
        angle_deg=angle_deg,
    )


def format_lwh_mm(length_mm: float, width_mm: float, height_mm: float) -> str:
    length_s = f"{length_mm:.0f}" if np.isfinite(length_mm) else "---"
    width_s = f"{width_mm:.0f}" if np.isfinite(width_mm) else "---"
    height_s = f"{height_mm:.0f}" if np.isfinite(height_mm) else "---"
    return f"{length_s}x{width_s}x{height_s} mm"


def format_lw_label(
    length_mm: float,
    width_mm: float,
    length_px: float,
    width_px: float,
) -> str | None:
    if np.isfinite(length_mm) and np.isfinite(width_mm):
        return f"LxW: {length_mm:.1f}mmx{width_mm:.1f}mm"
    if np.isfinite(length_px) and np.isfinite(width_px):
        return f"LxW: {length_px:.1f}pxx{width_px:.1f}px"
    return None
