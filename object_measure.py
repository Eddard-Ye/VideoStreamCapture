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


def peak_height_points_in_mask(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    z_plane_ref_mm: float,
    *,
    depth_tolerance_mm: float = 0.5,
) -> tuple[list[tuple[int, int]], float]:
    """Return all mask pixels at the maximum height (minimum surface depth).

    Physical height: height = z_plane_ref - depth. Peaks share the smallest depth.
    """
    if not np.any(mask) or not np.isfinite(z_plane_ref_mm) or z_plane_ref_mm <= 0:
        return [], float("nan")

    mask_bool = mask.astype(bool)
    depths = depth_mm.astype(np.float32)
    valid = mask_bool & np.isfinite(depths) & (depths > 0)
    if not np.any(valid):
        return [], float("nan")

    valid_depths = depths[valid]
    min_depth = float(np.min(valid_depths))
    peak_mask = valid & np.isclose(depths, min_depth, rtol=0.0, atol=depth_tolerance_mm)
    ys, xs = np.where(peak_mask)
    if xs.size == 0:
        return [], float("nan")

    height_mm = float(max(0.0, z_plane_ref_mm - min_depth))
    points = [(int(x), int(y)) for x, y in zip(xs.tolist(), ys.tolist())]
    return points, height_mm


def sample_depth_at(
    depth_mm: np.ndarray,
    u: int,
    v: int,
    *,
    radius: int = 2,
) -> float:
    height, width = depth_mm.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return float("nan")

    u0 = max(0, u - radius)
    u1 = min(width, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(height, v + radius + 1)
    patch = depth_mm[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return float("nan")
    return float(np.median(valid))


def obbox_center_axes(
    box_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    box = np.asarray(box_pts, dtype=np.float64).reshape(4, 2)
    center = box.mean(axis=0)
    edge_a = box[1] - box[0]
    edge_b = box[2] - box[1]
    len_a = float(np.linalg.norm(edge_a))
    len_b = float(np.linalg.norm(edge_b))
    if len_a >= len_b:
        u_len = edge_a / max(len_a, 1e-6)
        u_wid = edge_b / max(len_b, 1e-6)
        length_px, width_px = len_a, len_b
    else:
        u_len = edge_b / max(len_b, 1e-6)
        u_wid = edge_a / max(len_a, 1e-6)
        length_px, width_px = len_b, len_a
    return center, u_len, u_wid, length_px, width_px


def plane_sample_uv_points(
    center: np.ndarray,
    u_len: np.ndarray,
    u_wid: np.ndarray,
    length_px: float,
    width_px: float,
    *,
    axis_fraction: float = 0.6,
) -> list[tuple[int, int]]:
    """Four table samples around the object center along OBB axes.

    Local offsets: (±0.6x, 0), (0, ±0.6y) where x/y are OBB length/width in pixels.
    """
    points: list[tuple[int, int]] = []
    for lx, ly in (
        (axis_fraction, 0.0),
        (-axis_fraction, 0.0),
        (0.0, axis_fraction),
        (0.0, -axis_fraction),
    ):
        image_pt = center + lx * length_px * u_len + ly * width_px * u_wid
        points.append((int(round(float(image_pt[0]))), int(round(float(image_pt[1])))))
    return points


def plane_depth_from_obbox_samples(
    depth_mm: np.ndarray,
    box_pts: np.ndarray,
    length_px: float,
    width_px: float,
    *,
    axis_fraction: float = 0.6,
    sample_radius: int = 2,
) -> tuple[float, list[tuple[int, int]]]:
    """Average depth at four axis-offset points around the object OBB center."""
    center, u_len, u_wid, box_length_px, box_width_px = obbox_center_axes(box_pts)
    x = float(length_px) if np.isfinite(length_px) and length_px > 0 else box_length_px
    y = float(width_px) if np.isfinite(width_px) and width_px > 0 else box_width_px
    sample_uv = plane_sample_uv_points(
        center,
        u_len,
        u_wid,
        x,
        y,
        axis_fraction=axis_fraction,
    )

    depths: list[float] = []
    for u, v in sample_uv:
        depth_value = sample_depth_at(depth_mm, u, v, radius=sample_radius)
        if np.isfinite(depth_value) and depth_value > 0:
            depths.append(depth_value)

    if not depths:
        return float("nan"), sample_uv
    return float(np.mean(depths)), sample_uv


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


def format_lxwxh_stream_label(
    length_mm: float,
    width_mm: float,
    height_mm: float,
    length_px: float = float("nan"),
    width_px: float = float("nan"),
) -> str | None:
    if np.isfinite(length_mm) and np.isfinite(width_mm):
        height_s = f"{height_mm:.1f}mm" if np.isfinite(height_mm) else "---mm"
        return f"LxWxH:{length_mm:.1f}mm x {width_mm:.1f}mm x {height_s}"
    if np.isfinite(length_px) and np.isfinite(width_px):
        return f"LxWxH:{length_px:.1f}px x {width_px:.1f}px x ---px"
    return None


def format_plane_depth_stream_label(z_plane_ref_mm: float) -> str | None:
    if np.isfinite(z_plane_ref_mm) and z_plane_ref_mm > 0:
        return f"plane_depth: {z_plane_ref_mm:.1f}mm;"
    return None


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


def instance_height_mm(instance) -> float:
    """Best available object height in mm (peak height preferred)."""
    peak = getattr(instance, "peak_height_mm", float("nan"))
    body = getattr(instance, "height_mm", float("nan"))
    if np.isfinite(peak):
        return float(peak)
    if np.isfinite(body):
        return float(body)
    return float("nan")


def format_instance_height_display(instance) -> str:
    height_mm = instance_height_mm(instance)
    if np.isfinite(height_mm):
        return f"{height_mm:.1f}mm"
    return "---"
