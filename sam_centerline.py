# -*- coding: utf-8 -*-
"""Centerline and water-cut (水切) width analysis for SAM partition masks."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from render_contour_centerline import (
    centerline_dt_ridge_pca,
    centerline_fitodic_voronoi,
    fit_centerline_line_pca,
    longest_mask_intersection_along_line,
    max_width_perpendicular_to_axis,
)
from extract_center_contour import mask_to_contours_xy
from object_measure import depth_p90_for_mask, pixel_edge_len_mm


@dataclass
class WaterCutAnalysis:
    centerline_path: list[tuple[float, float]]
    water_cut_width_px: float
    water_cut_width_mm: float
    width_center: tuple[float, float]
    width_end_a: tuple[float, float]
    width_end_b: tuple[float, float]
    pca_centroid: tuple[float, float]
    pca_axis: tuple[float, float]


def _mask_bbox(mask: np.ndarray, pad: int = 8) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = int(xs.max()) + 1 + pad
    y1 = int(ys.max()) + 1 + pad
    return x0, y0, x1, y1


def _largest_contour_xy(mask_u8: np.ndarray) -> np.ndarray | None:
    contours = mask_to_contours_xy(mask_u8)
    if not contours:
        return None
    return max(contours, key=lambda c: cv2.arcLength(c.astype(np.float32), True))


def _to_global(
    points: list[tuple[float, float]] | np.ndarray,
    ox: int,
    oy: int,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for x, y in points:
        out.append((float(x) + ox, float(y) + oy))
    return out


def _to_global_pt(pt: tuple[float, float], ox: int, oy: int) -> tuple[float, float]:
    return float(pt[0] + ox), float(pt[1] + oy)


def water_cut_width_mm(
    analysis: WaterCutAnalysis,
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    mask: np.ndarray | None = None,
) -> float:
    """Convert pixel water-cut chord to mm using depth at center (fallback: mask p90 depth)."""
    cx, cy = analysis.width_center
    xi = int(round(cx))
    yi = int(round(cy))
    z_mm = float("nan")
    if 0 <= yi < depth_mm.shape[0] and 0 <= xi < depth_mm.shape[1]:
        z_mm = float(depth_mm[yi, xi])
    if not np.isfinite(z_mm) or z_mm <= 0:
        if mask is not None:
            z_mm = depth_p90_for_mask(depth_mm, mask)
    p0 = np.asarray(analysis.width_end_a, dtype=np.float64)
    p1 = np.asarray(analysis.width_end_b, dtype=np.float64)
    return pixel_edge_len_mm(p0, p1, z_mm, fx, fy)


def analyze_water_cut(
    mask: np.ndarray,
    *,
    depth_mm: np.ndarray | None = None,
    fx: float | None = None,
    fy: float | None = None,
    interpolation_distance: float = 0.5,
    pca_width_samples: int = 160,
) -> WaterCutAnalysis | None:
    """
    Compute Voronoi centerline inside mask and max (normal ∩ mask) chord (水切宽度).
    Coordinates are in full-image pixel space. The width segment stays inside the mask.
    """
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return None

    bbox = _mask_bbox(mask_bool)
    if bbox is None:
        return None
    ox, oy, x1, y1 = bbox
    local_mask = mask_bool[oy:y1, ox:x1]
    local_u8 = local_mask.astype(np.uint8) * 255

    poly_loc = _largest_contour_xy(local_u8)
    if poly_loc is None or len(poly_loc) < 3:
        return None

    path_loc: list[tuple[float, float]]
    try:
        path_loc = centerline_fitodic_voronoi(
            poly_loc,
            closed=True,
            interpolation_distance=max(interpolation_distance, 1e-3),
            simplify_tolerance=None,
        )
    except Exception:
        path_loc = []

    if len(path_loc) < 3:
        path_loc = [(float(x), float(y)) for x, y in centerline_dt_ridge_pca(local_mask)]

    if len(path_loc) < 2:
        return None

    path_xy_loc = np.asarray(path_loc, dtype=np.float64)
    c_loc, u_axis = fit_centerline_line_pca(path_xy_loc)
    path_global = _to_global(path_loc, ox, oy)
    c_global = _to_global_pt((float(c_loc[0]), float(c_loc[1])), ox, oy)
    u_tuple = (float(u_axis[0]), float(u_axis[1]))
    # Intersect on the full-image mask (same pixels as the pink overlay).
    width_px, center_pt, end_a, end_b, _idx = max_width_perpendicular_to_axis(
        mask_bool,
        path_global,
        np.asarray(c_global, dtype=np.float64),
        u_axis,
        n_samples=max(32, int(pca_width_samples)),
    )

    analysis = WaterCutAnalysis(
        centerline_path=path_global,
        water_cut_width_px=float(width_px),
        water_cut_width_mm=float("nan"),
        width_center=center_pt,
        width_end_a=end_a,
        width_end_b=end_b,
        pca_centroid=c_global,
        pca_axis=u_tuple,
    )
    if depth_mm is not None and fx is not None and fy is not None:
        analysis.water_cut_width_mm = water_cut_width_mm(analysis, depth_mm, fx, fy, mask_bool)
    return analysis


def pca_axis_segment_in_box(
    centroid: tuple[float, float],
    axis: tuple[float, float],
    box: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Clip the PCA axis line to the span of an oriented box along ``axis``."""
    c = np.asarray(centroid, dtype=np.float64)
    u = np.asarray(axis, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    corners = np.asarray(box, dtype=np.float64).reshape(4, 2)
    t_vals = (corners - c) @ u
    p0 = c + float(t_vals.min()) * u
    p1 = c + float(t_vals.max()) * u
    return (
        (int(round(p0[0])), int(round(p0[1]))),
        (int(round(p1[0])), int(round(p1[1]))),
    )


def _paint_line_inside_mask(
    image_bgr: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    clip_mask: np.ndarray | None,
) -> None:
    """Draw a line; if ``clip_mask`` is set, only paint pixels inside the mask."""
    if clip_mask is None or not np.any(clip_mask):
        cv2.line(image_bgr, p0, p1, color, thickness, cv2.LINE_AA)
        return
    layer = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.line(layer, p0, p1, 255, thickness, cv2.LINE_AA)
    keep = (layer > 0) & clip_mask.astype(bool)
    if not np.any(keep):
        return
    image_bgr[keep] = color


def draw_water_cut_overlay(
    image_bgr: np.ndarray,
    analysis: WaterCutAnalysis,
    *,
    draw_pca_axis: bool = False,
    draw_centerline: bool = True,
    width_line_only: bool = False,
    clip_box: np.ndarray | None = None,
    clip_mask: np.ndarray | None = None,
) -> None:
    """Draw water-cut visualization on ``image_bgr``.

    When ``width_line_only`` is True, only the max-width measurement chord is drawn.
    Otherwise draws the width chord, optional centerline/PCA axis, and a cutWidth label.
    The width chord is the mask intersection. Lines are painted only on ``clip_mask``
    pixels when that mask is provided, so they cannot extend past the pink overlay.
    """
    mask_bool = None if clip_mask is None else clip_mask.astype(bool)
    ea = analysis.width_end_a
    eb = analysis.width_end_b
    ca = analysis.width_center
    if mask_bool is not None and np.any(mask_bool):
        dx = float(eb[0] - ea[0])
        dy = float(eb[1] - ea[1])
        if abs(dx) + abs(dy) < 1e-6:
            dx, dy = float(analysis.pca_axis[1]), float(-analysis.pca_axis[0])
        w_clip, a_clip, b_clip = longest_mask_intersection_along_line(
            mask_bool,
            float(ca[0]),
            float(ca[1]),
            dx,
            dy,
        )
        if w_clip > 0:
            ea, eb = a_clip, b_clip
            ca = (0.5 * (a_clip[0] + b_clip[0]), 0.5 * (a_clip[1] + b_clip[1]))

    ca_i = (int(round(ca[0])), int(round(ca[1])))
    ea_i = (int(round(ea[0])), int(round(ea[1])))
    eb_i = (int(round(eb[0])), int(round(eb[1])))
    _paint_line_inside_mask(image_bgr, ea_i, eb_i, (0, 215, 255), 3, mask_bool)

    if width_line_only:
        return

    if draw_centerline and len(analysis.centerline_path) >= 2:
        pts = np.round(np.asarray(analysis.centerline_path, dtype=np.float32)).astype(np.int32)
        if mask_bool is None:
            cv2.polylines(image_bgr, [pts.reshape(-1, 1, 2)], False, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            for i in range(len(pts) - 1):
                _paint_line_inside_mask(
                    image_bgr,
                    (int(pts[i][0]), int(pts[i][1])),
                    (int(pts[i + 1][0]), int(pts[i + 1][1])),
                    (255, 255, 0),
                    2,
                    mask_bool,
                )

    if mask_bool is None or (
        0 <= ca_i[0] < image_bgr.shape[1]
        and 0 <= ca_i[1] < image_bgr.shape[0]
        and mask_bool[ca_i[1], ca_i[0]]
    ):
        cv2.circle(image_bgr, ca_i, 4, (0, 215, 255), -1, cv2.LINE_AA)

    if draw_pca_axis:
        p0 = p1 = None
        if mask_bool is not None and np.any(mask_bool):
            w_pca, a_pca, b_pca = longest_mask_intersection_along_line(
                mask_bool,
                analysis.pca_centroid[0],
                analysis.pca_centroid[1],
                analysis.pca_axis[0],
                analysis.pca_axis[1],
            )
            if w_pca > 0:
                p0 = (int(round(a_pca[0])), int(round(a_pca[1])))
                p1 = (int(round(b_pca[0])), int(round(b_pca[1])))
        elif clip_box is not None:
            p0, p1 = pca_axis_segment_in_box(analysis.pca_centroid, analysis.pca_axis, clip_box)
        else:
            c = np.asarray(analysis.pca_centroid, dtype=np.float64)
            u = np.asarray(analysis.pca_axis, dtype=np.float64)
            u = u / (np.linalg.norm(u) + 1e-12)
            h, w = image_bgr.shape[:2]
            span = max(w, h) * 2.0
            p0 = (int(round(c[0] - u[0] * span)), int(round(c[1] - u[1] * span)))
            p1 = (int(round(c[0] + u[0] * span)), int(round(c[1] + u[1] * span)))
        if p0 is not None and p1 is not None:
            _paint_line_inside_mask(image_bgr, p0, p1, (20, 120, 255), 2, mask_bool)

    if np.isfinite(analysis.water_cut_width_mm) and analysis.water_cut_width_mm > 0:
        label = f"cutWidth: {analysis.water_cut_width_mm:.1f} mm"
    else:
        label = f"cutWidth: {analysis.water_cut_width_px:.1f} px"
    tx = max(4, ca_i[0] + 8)
    ty = max(24, ca_i[1] - 12)
    cv2.putText(
        image_bgr,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 215, 255),
        2,
        cv2.LINE_AA,
    )
