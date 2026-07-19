# -*- coding: utf-8 -*-
"""Unit tests for capture height mode / scale / offset."""

from __future__ import annotations

from types import SimpleNamespace

from object_measure import (
    apply_height_transform,
    instance_height_mm,
    normalize_height_calc_mode,
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
    assert normalize_height_calc_mode("other") == "peak"


def test_instance_height_mode_peak_vs_average() -> None:
    instance = SimpleNamespace(height_mm=10.0, peak_height_mm=20.0)
    assert instance_height_mm(instance) == 20.0
    assert instance_height_mm(instance, calc_mode="peak") == 20.0
    assert instance_height_mm(instance, calc_mode="average") == 10.0


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
    assert "x 21.0mm" in (label or "")
