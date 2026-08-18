# -*- coding: utf-8 -*-
"""Unit tests for water-cut normal ∩ mask intersection."""

from __future__ import annotations

import numpy as np

from render_contour_centerline import (
    longest_mask_intersection_along_line,
    max_width_perpendicular_to_axis,
)
from yolo_sam_refine import build_prompts_from_oriented_box, clip_sam_to_object_interior


def test_intersection_from_outside_start_matches_rect_width() -> None:
    mask = np.zeros((40, 80), dtype=bool)
    mask[10:30, 20:60] = True  # 40px wide, 20px tall
    # Start left of the rectangle on a horizontal line through its middle.
    length, a, b = longest_mask_intersection_along_line(mask, 5.0, 20.0, 1.0, 0.0)
    assert abs(length - 39.0) <= 1.5
    xs = sorted((a[0], b[0]))
    assert xs[0] >= 19.0
    assert xs[1] <= 60.0
    assert mask[int(round(a[1])), int(round(a[0]))]
    assert mask[int(round(b[1])), int(round(b[0]))]


def test_intersection_skips_gap_and_keeps_longest_run() -> None:
    mask = np.zeros((20, 80), dtype=bool)
    mask[8:12, 5:15] = True
    mask[8:12, 20:60] = True  # longer run
    length, a, b = longest_mask_intersection_along_line(mask, 0.0, 10.0, 1.0, 0.0)
    xs = sorted((a[0], b[0]))
    assert xs[0] >= 19.0
    assert xs[1] <= 60.0
    assert abs(length - 39.0) <= 1.5


def test_max_width_uses_intersection_midpoint_inside_mask() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[15:35, 10:40] = True
    path = [(x, 25.0) for x in range(12, 38)]
    c = np.array([25.0, 25.0])
    u = np.array([1.0, 0.0])
    width, center, end_a, end_b, _ = max_width_perpendicular_to_axis(
        mask, path, c, u, n_samples=32
    )
    assert width > 15.0
    assert mask[int(round(center[1])), int(round(center[0]))]
    assert mask[int(round(end_a[1])), int(round(end_a[0]))]
    assert mask[int(round(end_b[1])), int(round(end_b[0]))]
    for pt in (end_a, end_b, center):
        assert 9.0 <= pt[0] <= 40.0
        assert 14.0 <= pt[1] <= 35.0


def test_clip_sam_to_object_interior_drops_crust_leak() -> None:
    obj = np.zeros((80, 80), dtype=bool)
    obj[10:70, 10:70] = True
    sam = np.zeros((80, 80), dtype=bool)
    sam[30:50, 30:50] = True
    sam[10:70, 39:42] = True  # thin leak to the object edge
    clipped = clip_sam_to_object_interior(sam, obj, inset_ratio=0.08, min_inset_px=6)
    assert clipped[40, 40]
    assert not clipped[12, 40]
    assert not clipped[67, 40]


def test_oriented_box_prompts_include_long_side_background() -> None:
    box = np.array([[10.0, 10.0], [90.0, 10.0], [90.0, 40.0], [10.0, 40.0]], dtype=np.float64)
    coords, labels = build_prompts_from_oriented_box(box, length_px=80.0, width_px=30.0)
    assert len(coords) == 7
    assert int(np.sum(labels == 1)) == 5
    assert int(np.sum(labels == 0)) == 2
    bg = coords[labels == 0]
    ys = sorted(float(p[1]) for p in bg)
    # Long-edge midpoints (y=10 and y=40) inset 10% of width (3px) -> y=13 and y=37.
    assert abs(ys[0] - 13.0) <= 0.6
    assert abs(ys[1] - 37.0) <= 0.6
    assert abs(float(np.mean(bg[:, 0])) - 50.0) <= 0.6
