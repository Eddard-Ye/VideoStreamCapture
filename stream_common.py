# -*- coding: utf-8 -*-
"""Shared HTTP stream helpers used by both MVS 2D and Orbbec pipelines."""

from __future__ import annotations

import os
import re
from datetime import datetime

import cv2
import numpy as np

from color_viewer import SegInstance
from object_measure import (
    format_lw_label,
    format_lxwxh_stream_label,
    instance_height_mm,
    oriented_box_metrics_from_mask,
)
from stream_overlay import WaterCutOverlay


def attach_oriented_boxes(instances: list[SegInstance]) -> None:
    for instance in instances:
        measured = oriented_box_metrics_from_mask(instance.mask)
        if measured is None:
            continue
        box, length_px, width_px, angle_deg = measured
        instance.box_pts = box
        instance.length_px = length_px
        instance.width_px = width_px
        instance.angle_deg = angle_deg


def encode_jpeg(frame_bgr: np.ndarray, stream_width: int, jpeg_quality: int) -> bytes:
    output = frame_bgr
    if stream_width > 0 and output.shape[1] > stream_width:
        height, width = output.shape[:2]
        new_w = int(stream_width)
        new_h = max(1, int(round(height * new_w / width)))
        output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ok, encoded = cv2.imencode(
        ".jpg",
        output,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 30, 95))],
    )
    if not ok:
        return b""
    return encoded.tobytes()


def build_status(
    *,
    instances: list[SegInstance],
    measured_fps: float,
    water_cut_overlays: list[WaterCutOverlay],
    computing: bool,
) -> dict:
    items = []
    for index, instance in enumerate(instances):
        item = {
            "index": index,
            "class_name": instance.class_name,
            "confidence": round(instance.confidence, 3),
            "length_mm": (
                None if not np.isfinite(instance.length_mm) else round(instance.length_mm, 1)
            ),
            "width_mm": (
                None if not np.isfinite(instance.width_mm) else round(instance.width_mm, 1)
            ),
            "length_px": None if not np.isfinite(instance.length_px) else round(instance.length_px, 1),
            "width_px": None if not np.isfinite(instance.width_px) else round(instance.width_px, 1),
        }
        label = format_lxwxh_stream_label(
            instance.length_mm,
            instance.width_mm,
            instance.peak_height_mm
            if np.isfinite(instance.peak_height_mm)
            else instance.height_mm,
            instance.length_px,
            instance.width_px,
        )
        if label is None:
            label = format_lw_label(
                instance.length_mm,
                instance.width_mm,
                instance.length_px,
                instance.width_px,
            )
        if label:
            item["size_label"] = label
        if np.isfinite(instance.length_mm) and np.isfinite(instance.width_mm):
            item["length"] = round(instance.length_mm, 1)
            item["width"] = round(instance.width_mm, 1)
            item["unit"] = "mm"
        height_mm = instance_height_mm(instance)
        if np.isfinite(height_mm):
            item["height_mm"] = round(float(height_mm), 1)
        if np.isfinite(instance.peak_height_mm):
            item["peak_height_mm"] = round(float(instance.peak_height_mm), 2)
        if instance.peak_height_points:
            item["peak_height_px"] = [[int(u), int(v)] for u, v in instance.peak_height_points]
        if np.isfinite(instance.z_plane_ref_mm):
            item["z_plane_ref_mm"] = round(float(instance.z_plane_ref_mm), 2)
        if instance.plane_sample_points:
            item["plane_sample_px"] = [[int(u), int(v)] for u, v in instance.plane_sample_points]
        items.append(item)

    cuts = []
    for idx, overlay in enumerate(water_cut_overlays):
        wc = overlay.water_cut
        cuts.append(
            {
                "index": idx,
                "water_cut_width_px": round(float(wc.water_cut_width_px), 2),
                "water_cut_width_mm": (
                    None
                    if not np.isfinite(wc.water_cut_width_mm)
                    else round(float(wc.water_cut_width_mm), 2)
                ),
            }
        )

    state = "computing_water_cut" if computing else "running"
    return {
        "state": state,
        "fps": round(measured_fps, 2),
        "instances": len(instances),
        "detections": items,
        "water_cut": cuts,
        "water_cut_active": bool(water_cut_overlays),
    }


def sanitize_capture_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", name.strip(), flags=re.UNICODE)
    return cleaned or "capture"


def save_capture_jpeg(frame_bgr: np.ndarray, output_dir: str, name: str, jpeg_quality: int) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitize_capture_name(name)}_{timestamp}.jpg"
    output_path = os.path.join(output_dir, filename)
    ok = cv2.imwrite(
        output_path,
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 30, 100))],
    )
    if not ok:
        raise RuntimeError(f"Failed to write capture image: {output_path}")
    return output_path
