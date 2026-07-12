# -*- coding: utf-8 -*-
"""Tests for track_smoother."""

import numpy as np

from color_viewer import SegInstance
from track_smoother import TrackSmoother, _align_length_width


def _instance(
    *,
    mask: np.ndarray,
    length_mm: float,
    width_mm: float,
    length_px: float,
    width_px: float,
    height_mm: float = float("nan"),
    peak_height_mm: float = float("nan"),
    confidence: float = 0.9,
) -> SegInstance:
    return SegInstance(
        mask=mask,
        class_id=0,
        class_name="cake",
        confidence=confidence,
        length_mm=length_mm,
        width_mm=width_mm,
        length_px=length_px,
        width_px=width_px,
        height_mm=height_mm,
        peak_height_mm=peak_height_mm,
    )


def test_align_length_width_prefers_previous_assignment():
    length, width = _align_length_width(100.0, 50.0, 52.0, 98.0)
    assert length == 98.0
    assert width == 52.0


def test_smoother_ema_reduces_jitter():
    smoother = TrackSmoother(alpha=0.25, max_miss=3)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True

    first = _instance(
        mask=mask,
        length_mm=100.0,
        width_mm=50.0,
        length_px=200.0,
        width_px=100.0,
    )
    out1 = smoother.update([first])[0]
    assert out1.length_mm == 100.0

    second = _instance(
        mask=mask,
        length_mm=110.0,
        width_mm=55.0,
        length_px=220.0,
        width_px=110.0,
    )
    out2 = smoother.update([second])[0]
    assert out2.length_mm == 102.5
    assert out2.width_mm == 51.25


def test_smoother_ema_reduces_height_jitter():
    smoother = TrackSmoother(alpha=0.25, max_miss=3)
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 30:70] = True

    smoother.update(
        [
            _instance(
                mask=mask,
                length_mm=100.0,
                width_mm=50.0,
                length_px=200.0,
                width_px=100.0,
                peak_height_mm=10.0,
            )
        ]
    )
    out = smoother.update(
        [
            _instance(
                mask=mask,
                length_mm=100.0,
                width_mm=50.0,
                length_px=200.0,
                width_px=100.0,
                peak_height_mm=14.0,
            )
        ]
    )[0]
    assert out.peak_height_mm == 11.0
    assert out.height_mm == 11.0


def test_smoother_keeps_track_across_small_motion():
    smoother = TrackSmoother(alpha=1.0, max_miss=3)
    mask_a = np.zeros((100, 100), dtype=bool)
    mask_a[20:80, 20:60] = True
    mask_b = np.zeros((100, 100), dtype=bool)
    mask_b[22:82, 24:64] = True

    smoother.update(
        [
            _instance(
                mask=mask_a,
                length_mm=80.0,
                width_mm=40.0,
                length_px=160.0,
                width_px=80.0,
            )
        ]
    )
    out = smoother.update(
        [
            _instance(
                mask=mask_b,
                length_mm=81.0,
                width_mm=41.0,
                length_px=161.0,
                width_px=81.0,
            )
        ]
    )
    assert len(out) == 1
    assert out[0].length_mm == 81.0


if __name__ == "__main__":
    test_align_length_width_prefers_previous_assignment()
    test_smoother_ema_reduces_jitter()
    test_smoother_ema_reduces_height_jitter()
    test_smoother_keeps_track_across_small_motion()
    print("ok")
