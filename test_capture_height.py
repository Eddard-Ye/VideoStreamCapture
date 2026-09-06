# -*- coding: utf-8 -*-
"""Unit tests for capture height mode / scale / offset."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from object_measure import (
    apply_height_transform,
    instance_height_mm,
    normalize_height_calc_mode,
    normalize_height_percentile,
    percentile_height_in_mask,
    resolve_capture_height_mm,
)
from stream_overlay import _format_lwh_stream_label, format_capture_lw_label
from stream_server import CaptureRequest


def test_normalize_height_calc_mode_defaults_to_peak() -> None:
    assert normalize_height_calc_mode(None) == "peak"
    assert normalize_height_calc_mode("") == "peak"
    assert normalize_height_calc_mode("PEAK") == "peak"
    assert normalize_height_calc_mode("average") == "average"
    assert normalize_height_calc_mode("Average") == "average"
    assert normalize_height_calc_mode("percentile") == "percentile"
    assert normalize_height_calc_mode("PCT") == "percentile"
    assert normalize_height_calc_mode("other") == "peak"


def test_normalize_height_percentile_clamps() -> None:
    assert normalize_height_percentile(None) == 50.0
    assert normalize_height_percentile(-10) == 0.0
    assert normalize_height_percentile(150) == 100.0
    assert normalize_height_percentile(75) == 75.0


def test_instance_height_mode_peak_vs_average() -> None:
    instance = SimpleNamespace(height_mm=10.0, peak_height_mm=20.0)
    assert instance_height_mm(instance) == 20.0
    assert instance_height_mm(instance, calc_mode="peak") == 20.0
    assert instance_height_mm(instance, calc_mode="average") == 10.0


def test_instance_height_mode_percentile() -> None:
    instance = SimpleNamespace(
        height_mm=10.0,
        peak_height_mm=20.0,
        percentile_height_mm=15.0,
        height_percentile=50.0,
    )
    assert instance_height_mm(instance, calc_mode="percentile") == 15.0
    assert resolve_capture_height_mm(
        instance,
        calc_mode="percentile",
        height_percentile=50.0,
        height_scale=2.0,
        height_offset=1.0,
    ) == 31.0


def test_percentile_height_in_mask_0_50_100() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    depth = np.full((20, 20), 500.0, dtype=np.float32)
    # z_plane=500; depth 490 -> h=10, depth 470 -> h=30
    ys, xs = np.where(mask)
    for i, (y, x) in enumerate(zip(ys, xs)):
        depth[y, x] = 490.0 - (i % 21)  # heights 10..30
    z_plane = 500.0
    _, h0 = percentile_height_in_mask(mask, depth, z_plane, 0)
    _, h50 = percentile_height_in_mask(mask, depth, z_plane, 50)
    _, h100 = percentile_height_in_mask(mask, depth, z_plane, 100)
    assert h0 <= h50 <= h100
    assert abs(h100 - 30.0) <= 1.0
    assert abs(h0 - 10.0) <= 1.0


def test_apply_height_transform() -> None:
    assert apply_height_transform(10.0) == 10.0
    assert apply_height_transform(10.0, height_scale=2.0, height_offset=1.0) == 21.0


def test_resolve_capture_height_mm_defaults_match_peak() -> None:
    instance = SimpleNamespace(height_mm=10.0, peak_height_mm=20.0)
    assert resolve_capture_height_mm(instance) == 20.0
    assert resolve_capture_height_mm(
        instance,
        calc_mode="average",
        height_scale=2.0,
        height_offset=1.0,
    ) == 21.0


def test_capture_request_defaults() -> None:
    request = CaptureRequest(
        name="sample",
        height="0.0mm",
        temperature="",
        weight="",
        water_cut=False,
    )
    assert request.height_calc_mode == "peak"
    assert request.height_scale == 1.0
    assert request.height_offset == 0.0
    assert request.height_percentile == 50.0
    assert request.lw_height_mm == 0.0


def test_resolve_lw_depth_mm_subtracts_lw_height() -> None:
    from object_measure import resolve_lw_depth_mm

    assert resolve_lw_depth_mm(500.0, 400.0) == 500.0
    assert resolve_lw_depth_mm(500.0, 400.0, lw_height_mm=30.0) == 470.0
    # Invalid (non-positive) result falls back to plane depth.
    assert resolve_lw_depth_mm(500.0, 400.0, lw_height_mm=500.0) == 500.0
    # No plane: fall back to object depth.
    assert resolve_lw_depth_mm(float("nan"), 400.0, lw_height_mm=30.0) == 400.0


def test_stream_lxwxh_label_uses_capture_height_transform() -> None:
    instance = SimpleNamespace(
        length_mm=100.0,
        width_mm=50.0,
        height_mm=10.0,
        peak_height_mm=20.0,
        length_px=float("nan"),
        width_px=float("nan"),
    )
    # Without transform params: still prefers peak (same as API default).
    assert "x 20.0mm" in (_format_lwh_stream_label(instance) or "")

    # Capture path: average + scale/offset must match format_capture_lw_label / API.
    label = _format_lwh_stream_label(
        instance,
        calc_mode="average",
        height_scale=2.0,
        height_offset=1.0,
    )
    expected = format_capture_lw_label(
        instance,
        calc_mode="average",
        height_scale=2.0,
        height_offset=1.0,
    )
    assert label == expected
