# -*- coding: utf-8 -*-
"""Depth-based L/W and water-cut metrics for Orbbec RGB-D streams."""

from __future__ import annotations

import os

import numpy as np

from camera_intrinsics import RgbIntrinsics
from stream_common import save_capture_jpeg
from color_viewer import RoiRect, SegInstance
from object_measure import (
    mean_depth_points_in_mask,
    measure_mask_mm,
    oriented_box_from_mask,
    oriented_box_metrics_from_mask,
    peak_height_points_in_mask,
    plane_depth_from_obbox_samples,
    resolve_capture_height_mm,
)
from sam_centerline import analyze_water_cut
from stream_overlay import (
    WaterCutOverlay,
    build_capture_record_info,
    compose_record_frame,
    format_temperature_display,
    LABEL_SIZE_FACTOR,
)
from stream_server import CaptureRequest
from yolo_sam_refine import SamRefiner, prepare_water_cut_box_prompts, run_water_cut_box_sam


def intrinsics_from_orbbec(intrinsic) -> RgbIntrinsics:
    return RgbIntrinsics(
        fx=float(intrinsic.fx),
        fy=float(intrinsic.fy),
        cx=float(intrinsic.cx),
        cy=float(intrinsic.cy),
        calib_width=int(intrinsic.width),
        calib_height=int(intrinsic.height),
    )


def attach_orbbec_instance_metrics(
    instances: list[SegInstance],
    depth_mm: np.ndarray,
    intrinsics: RgbIntrinsics,
) -> None:
    """Convert YOLO mask pixel geometry to mm using aligned depth and RGB intrinsics."""
    for instance in instances:
        if instance.box_pts is None:
            measured_box = oriented_box_metrics_from_mask(instance.mask)
            if measured_box is not None:
                box, length_px, width_px, angle_deg = measured_box
                instance.box_pts = box
                instance.length_px = length_px
                instance.width_px = width_px
                instance.angle_deg = angle_deg

        if instance.box_pts is None:
            continue

        z_plane_ref_mm, plane_sample_points = plane_depth_from_obbox_samples(
            depth_mm,
            instance.box_pts,
            instance.length_px,
            instance.width_px,
        )
        instance.plane_sample_points = plane_sample_points
        instance.z_plane_ref_mm = z_plane_ref_mm

        measured = measure_mask_mm(
            instance.mask,
            depth_mm,
            intrinsics,
            z_plane_ref_mm=z_plane_ref_mm,
        )
        if measured is None:
            instance.peak_height_points = []
            instance.peak_height_mm = float("nan")
            instance.average_height_points = []
            continue

        instance.box_pts = measured.box_pts
        instance.length_mm = measured.length_mm
        instance.width_mm = measured.width_mm
        instance.height_mm = measured.height_mm
        instance.z_object_mm = measured.z_object_mm
        instance.angle_deg = measured.angle_deg
        instance.average_height_points = mean_depth_points_in_mask(instance.mask, depth_mm)

        if np.isfinite(z_plane_ref_mm) and z_plane_ref_mm > 0:
            peak_points, peak_height_mm = peak_height_points_in_mask(
                instance.mask,
                depth_mm,
                z_plane_ref_mm,
            )
            instance.peak_height_points = peak_points
            instance.peak_height_mm = peak_height_mm
        else:
            instance.peak_height_points = []
            instance.peak_height_mm = float("nan")


def compute_orbbec_water_cut_overlays(
    image_bgr: np.ndarray,
    instances: list[SegInstance],
    sam_refiner: SamRefiner,
    *,
    depth_mm: np.ndarray,
    intrinsics: RgbIntrinsics,
) -> list[WaterCutOverlay]:
    overlays: list[WaterCutOverlay] = []
    for instance in instances:
        preview = prepare_water_cut_box_prompts(instance.mask)
        if preview is None:
            continue
        try:
            sam_region = run_water_cut_box_sam(sam_refiner, image_bgr, instance.mask, preview)
        except RuntimeError as exc:
            print(f"  Water-cut SAM failed on {instance.class_name}: {exc}")
            continue
        if sam_region is None:
            print(f"  Water-cut SAM empty on {instance.class_name}.")
            continue

        water_cut = analyze_water_cut(
            sam_region.mask,
            depth_mm=depth_mm,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
        )
        if water_cut is None:
            continue

        box_pts = instance.box_pts
        if box_pts is None:
            box_pts = oriented_box_from_mask(instance.mask)

        overlays.append(
            WaterCutOverlay(
                sam_mask=sam_region.mask.copy(),
                water_cut=water_cut,
                box_pts=None if box_pts is None else box_pts.copy(),
                prompt_coords=(
                    None
                    if sam_region.prompt_coords is None
                    else np.asarray(sam_region.prompt_coords, dtype=np.float32).copy()
                ),
                prompt_labels=(
                    None
                    if sam_region.prompt_labels is None
                    else np.asarray(sam_region.prompt_labels, dtype=np.int32).copy()
                ),
            )
        )
        if np.isfinite(water_cut.water_cut_width_mm):
            print(
                f"  Water-cut {instance.class_name}: "
                f"width={water_cut.water_cut_width_px:.1f}px "
                f"({water_cut.water_cut_width_mm:.2f}mm)"
            )
        else:
            print(
                f"  Water-cut {instance.class_name}: "
                f"width={water_cut.water_cut_width_px:.1f}px"
            )
    return overlays


def process_orbbec_capture_request(
    *,
    request: CaptureRequest,
    image_bgr: np.ndarray,
    instances: list[SegInstance],
    sam_refiner: SamRefiner,
    depth_mm: np.ndarray,
    intrinsics: RgbIntrinsics,
    output_dir: str,
    jpeg_quality: int,
    roi: RoiRect | None = None,
    label_instances: list[SegInstance] | None = None,
    label_size_factor: float = LABEL_SIZE_FACTOR,
) -> dict:
    try:
        labels = label_instances if label_instances is not None else instances
        record_overlays: list[WaterCutOverlay] = []
        if request.water_cut:
            print("Capture requested with water-cut...")
            record_overlays = compute_orbbec_water_cut_overlays(
                image_bgr,
                instances,
                sam_refiner,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
            )

        record_info = build_capture_record_info(
            labels,
            temperature=request.temperature,
            weight=request.weight,
            water_cut_enabled=request.water_cut,
            water_cut_overlays=record_overlays,
            height_calc_mode=request.height_calc_mode,
            height_scale=request.height_scale,
            height_offset=request.height_offset,
        )
        frame = compose_record_frame(
            image_bgr,
            instances,
            record_info,
            water_cut_overlays=record_overlays if request.water_cut else None,
            roi=roi,
            label_instances=labels,
            draw_oriented_boxes=True,
            label_size_factor=label_size_factor,
        )
        output_path = save_capture_jpeg(frame, output_dir, request.name, jpeg_quality)
        file_name = os.path.basename(output_path)
        print(f"Saved capture: {output_path}")

        primary = labels[0] if labels else None
        water_cut_mm = (
            None
            if record_info.water_cut_mm is None or not np.isfinite(record_info.water_cut_mm)
            else round(float(record_info.water_cut_mm), 1)
        )
        primary_height_mm = (
            None
            if primary is None
            else resolve_capture_height_mm(
                primary,
                calc_mode=request.height_calc_mode,
                height_scale=request.height_scale,
                height_offset=request.height_offset,
            )
        )
        return {
            "ok": True,
            "fileName": file_name,
            "name": request.name,
            "water_cut": request.water_cut,
            "record": {
                "lw": record_info.lw_text,
                "height": record_info.height,
                "temperature": format_temperature_display(request.temperature),
                "weight": request.weight,
                "water_cut": record_info.water_cut_line,
            },
            "detections": len(instances),
            "length_mm": (
                None
                if primary is None or not np.isfinite(primary.length_mm)
                else round(float(primary.length_mm), 1)
            ),
            "width_mm": (
                None
                if primary is None or not np.isfinite(primary.width_mm)
                else round(float(primary.width_mm), 1)
            ),
            "height_mm": (
                None
                if primary_height_mm is None or not np.isfinite(primary_height_mm)
                else round(float(primary_height_mm), 1)
            ),
            "water_cut_mm": water_cut_mm if request.water_cut else None,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
