# -*- coding: utf-8 -*-
"""Shared camera intrinsics types (2D/RGB-D measurement helpers)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RgbIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    calib_width: int
    calib_height: int
    z_unit_mm: float = 1.0

    def scaled(self, image_width: int, image_height: int) -> "RgbIntrinsics":
        if self.calib_width <= 0 or self.calib_height <= 0:
            return self
        sx = image_width / float(self.calib_width)
        sy = image_height / float(self.calib_height)
        return RgbIntrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            calib_width=image_width,
            calib_height=image_height,
            z_unit_mm=self.z_unit_mm,
        )
