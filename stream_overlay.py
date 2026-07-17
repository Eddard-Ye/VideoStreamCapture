# -*- coding: utf-8 -*-
"""Lightweight overlay compositing for the MJPEG stream."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from color_viewer import RoiRect, SegInstance
from object_measure import (
    DEFAULT_HEIGHT_CALC_MODE,
    DEFAULT_HEIGHT_OFFSET,
    DEFAULT_HEIGHT_SCALE,
    format_height_mm_stream_label,
    format_instance_height_display,
    format_lxw_stream_label,
    format_lxwxh_stream_label,
    format_peak_height_mm_stream_label,
    format_plane_depth_stream_label,
    oriented_box_from_mask,
    resolve_capture_height_mm,
)
from sam_centerline import WaterCutAnalysis, draw_water_cut_overlay
from yolo_sam_refine import SamRefiner, prepare_water_cut_box_prompts

SAM_COLOR_BGR = SamRefiner.SAM_COLOR_BGR
LABEL_COLOR_BGR = (180, 60, 20)  # deep blue — high contrast on light scenes
PLANE_SAMPLE_MARKER_BGR = (0, 0, 255)  # red dots for table plane samples
MIN_HEIGHT_MARKER_BGR = (255, 80, 0)  # bright blue dots for peak-height pixels
AVERAGE_HEIGHT_MARKER_BGR = (0, 255, 120)  # neon green dots for mean-depth pixels
LABEL_OUTLINE_BGR = (255, 255, 255)
YOLO_CONTOUR_BGR = (0, 255, 0)
YOLO_OBB_BGR = (0, 220, 120)
ROI_COLOR_BGR = (0, 0, 255)
RECORD_COLOR_BGR = (255, 255, 255)
RECORD_BG_BGR = (20, 20, 20)


LABEL_SIZE_FACTOR = 2.0


@dataclass
class CaptureRecordInfo:
    height: str
    temperature: str
    weight: str
    lw_text: str
    water_cut_line: str | None = None
    water_cut_mm: float | None = None


def format_water_cut_record_line(
    overlays: list["WaterCutOverlay"] | None,
    *,
    enabled: bool,
) -> tuple[str | None, float | None]:
    if not enabled:
        return None, None
    if not overlays:
        return "water_cut: ---", None

    wc = overlays[0].water_cut
    if np.isfinite(wc.water_cut_width_mm):
        mm = float(wc.water_cut_width_mm)
        return f"water_cut: {mm:.1f}mm", mm
    if np.isfinite(wc.water_cut_width_px):
        return f"water_cut: {wc.water_cut_width_px:.1f}px", None
    return "water_cut: ---", None


def format_capture_lw_label(
    instance: SegInstance,
    *,
    calc_mode: str = DEFAULT_HEIGHT_CALC_MODE,
    height_scale: float = DEFAULT_HEIGHT_SCALE,
    height_offset: float = DEFAULT_HEIGHT_OFFSET,
) -> str:
    height_mm = resolve_capture_height_mm(
        instance,
        calc_mode=calc_mode,
        height_scale=height_scale,
        height_offset=height_offset,
    )
    metric = format_lxwxh_stream_label(
        instance.length_mm,
        instance.width_mm,
        height_mm,
        instance.length_px,
        instance.width_px,
    )
    return metric or "LxWxH: ---"


def format_temperature_display(temperature: str) -> str:
    text = temperature.strip()
    if not text:
        return "---"
    if "°C" in text or "℃" in text:
        return text
    return f"{text}°C"


def build_capture_record_info(
    instances: list[SegInstance],
    *,
    temperature: str,
    weight: str,
    water_cut_enabled: bool = False,
    water_cut_overlays: list["WaterCutOverlay"] | None = None,
    height_calc_mode: str = DEFAULT_HEIGHT_CALC_MODE,
    height_scale: float = DEFAULT_HEIGHT_SCALE,
    height_offset: float = DEFAULT_HEIGHT_OFFSET,
) -> CaptureRecordInfo:
    primary = instances[0] if instances else None
    lw_text = (
        format_capture_lw_label(
            primary,
            calc_mode=height_calc_mode,
            height_scale=height_scale,
            height_offset=height_offset,
        )
        if primary is not None
        else "LxWxH: ---"
    )
    water_cut_line, water_cut_mm = format_water_cut_record_line(
        water_cut_overlays,
        enabled=water_cut_enabled,
    )

    return CaptureRecordInfo(
        height=(
            format_instance_height_display(
                primary,
                calc_mode=height_calc_mode,
                height_scale=height_scale,
                height_offset=height_offset,
            )
            if primary is not None
            else "---"
        ),
        temperature=temperature,
        weight=weight,
        lw_text=lw_text,
        water_cut_line=water_cut_line,
        water_cut_mm=water_cut_mm,
    )


def record_info_lines(info: CaptureRecordInfo) -> list[str]:
    lines = [
        info.lw_text,
        f"height: {info.height}",
        f"temperature: {format_temperature_display(info.temperature)}",
        f"weight: {info.weight}",
    ]
    if info.water_cut_line is not None:
        lines.append(info.water_cut_line)
    return lines


def _record_overlay_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _bgr_to_rgb(color: tuple[int, int, int]) -> tuple[int, int, int]:
    blue, green, red = color
    return red, green, blue


def _draw_record_info_block(
    frame: np.ndarray,
    lines: list[str],
    *,
    label_size_factor: float = LABEL_SIZE_FACTOR,
) -> None:
    """Draw capture summary text; uses Pillow so Unicode such as °C renders correctly."""
    if not lines:
        return

    width_factor = max(0.75, min(1.0, frame.shape[1] / 900.0))
    font_size = max(12, int(round(14 * label_size_factor * width_factor)))
    font = _record_overlay_font(font_size)
    stroke = max(1, int(round(label_size_factor)))
    pad = max(4, int(round(6 * label_size_factor)))
    line_gap = max(2, int(round(4 * label_size_factor)))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)

    line_metrics: list[tuple[int, int, int]] = []
    max_text_w = 0
    total_text_h = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        line_metrics.append((width, height, bbox[1]))
        max_text_w = max(max_text_w, width)
        total_text_h += height + line_gap
    if lines:
        total_text_h -= line_gap

    block_w = max_text_w + pad * 2
    block_h = total_text_h + pad * 2
    x0 = int(round(10 * label_size_factor))
    y0 = frame.shape[0] - block_h - int(round(10 * label_size_factor))

    bg_rgb = _bgr_to_rgb(RECORD_BG_BGR)
    text_rgb = _bgr_to_rgb(RECORD_COLOR_BGR)
    outline_rgb = _bgr_to_rgb(LABEL_OUTLINE_BGR)
    border_rgb = (255, 180, 0)  # BGR (0, 180, 255)

    draw.rectangle((x0, y0, x0 + block_w, y0 + block_h), fill=bg_rgb)
    draw.rectangle(
        (x0, y0, x0 + block_w, y0 + block_h),
        outline=border_rgb,
        width=max(1, int(round(1.5 * label_size_factor))),
    )

    text_x = x0 + pad
    text_y = y0 + pad
    for line, (_width, height, bbox_top) in zip(lines, line_metrics):
        draw.text(
            (text_x, text_y - bbox_top),
            line,
            font=font,
            fill=text_rgb,
            stroke_width=stroke,
            stroke_fill=outline_rgb,
        )
        text_y += height + line_gap

    frame[:, :] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _put_text_outlined(
    image_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 1.0,
    color: tuple[int, int, int] = LABEL_COLOR_BGR,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image_bgr,
        text,
        org,
        font,
        scale,
        LABEL_OUTLINE_BGR,
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(image_bgr, text, org, font, scale, color, thickness, cv2.LINE_AA)


def _format_lwh_stream_label(
    instance: SegInstance,
    *,
    split_height_labels: bool = False,
) -> str | None:
    if split_height_labels:
        return format_lxw_stream_label(
            instance.length_mm,
            instance.width_mm,
            instance.length_px,
            instance.width_px,
        )
    height_mm = (
        instance.peak_height_mm
        if np.isfinite(instance.peak_height_mm)
        else instance.height_mm
    )
    return format_lxwxh_stream_label(
        instance.length_mm,
        instance.width_mm,
        height_mm,
        instance.length_px,
        instance.width_px,
    )


def _draw_plane_sample_markers(frame: np.ndarray, instances: list[SegInstance]) -> None:
    radius = max(4, int(round(5 * LABEL_SIZE_FACTOR * 0.65)))
    outline = max(1, radius // 3)
    for instance in instances:
        for u, v in instance.plane_sample_points:
            center = (int(u), int(v))
            cv2.circle(frame, center, radius + outline, LABEL_OUTLINE_BGR, -1, cv2.LINE_AA)
            cv2.circle(frame, center, radius, PLANE_SAMPLE_MARKER_BGR, -1, cv2.LINE_AA)


def _draw_peak_height_markers(frame: np.ndarray, instances: list[SegInstance]) -> None:
    """Paint all peak-height pixels blue; avoid per-pixel white outlines that merge into white blobs."""
    height, width = frame.shape[:2]
    for instance in instances:
        points = instance.peak_height_points
        if not points:
            continue

        peak_mask = np.zeros((height, width), dtype=np.uint8)
        for u, v in points:
            if 0 <= u < width and 0 <= v < height:
                peak_mask[v, u] = 255

        if not np.any(peak_mask):
            continue

        # Slightly thicken sparse peaks so a single pixel stays visible on stream.
        if len(points) <= 8:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            peak_mask = cv2.dilate(peak_mask, kernel, iterations=1)

        frame[peak_mask > 0] = MIN_HEIGHT_MARKER_BGR


def _draw_average_height_markers(frame: np.ndarray, instances: list[SegInstance]) -> None:
    """Paint pixels closest to mean depth in neon green (ties all shown)."""
    height, width = frame.shape[:2]
    for instance in instances:
        points = getattr(instance, "average_height_points", None) or []
        if not points:
            continue

        avg_mask = np.zeros((height, width), dtype=np.uint8)
        for u, v in points:
            if 0 <= u < width and 0 <= v < height:
                avg_mask[v, u] = 255

        if not np.any(avg_mask):
            continue

        if len(points) <= 8:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            avg_mask = cv2.dilate(avg_mask, kernel, iterations=1)

        frame[avg_mask > 0] = AVERAGE_HEIGHT_MARKER_BGR


def _is_full_frame_roi(roi: RoiRect, width: int, height: int) -> bool:
    return roi.x1 <= 0 and roi.y1 <= 0 and roi.x2 >= width and roi.y2 >= height


def _draw_dashed_line(
    image: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int,
    gap_length: int,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    length = float(np.hypot(x2 - x1, y2 - y1))
    if length <= 0:
        return
    step = dash_length + gap_length
    pos = 0.0
    while pos < length:
        start = pos / length
        end = min(pos + dash_length, length) / length
        sx = int(round(x1 + (x2 - x1) * start))
        sy = int(round(y1 + (y2 - y1) * start))
        ex = int(round(x1 + (x2 - x1) * end))
        ey = int(round(y1 + (y2 - y1) * end))
        cv2.line(image, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        pos += step


def _draw_dashed_rectangle(
    image: np.ndarray,
    roi: RoiRect,
    *,
    color: tuple[int, int, int] = ROI_COLOR_BGR,
    thickness: int = 2,
    dash_length: int = 12,
    gap_length: int = 8,
) -> None:
    x1, y1, x2, y2 = roi.x1, roi.y1, roi.x2, roi.y2
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        _draw_dashed_line(image, start, end, color, thickness, dash_length, gap_length)


def _draw_roi_rect(
    frame: np.ndarray,
    roi: RoiRect | None,
    *,
    label_size_factor: float = LABEL_SIZE_FACTOR,
) -> None:
    if roi is None:
        return
    height, width = frame.shape[:2]
    if _is_full_frame_roi(roi, width, height):
        return

    thickness = max(2, int(round(2 * label_size_factor)))
    dash_length = max(8, int(round(12 * label_size_factor)))
    gap_length = max(6, int(round(8 * label_size_factor)))
    _draw_dashed_rectangle(
        frame,
        roi,
        color=ROI_COLOR_BGR,
        thickness=thickness,
        dash_length=dash_length,
        gap_length=gap_length,
    )


def _draw_instance_contours(frame: np.ndarray, instances: list[SegInstance]) -> None:
    """Draw mask boundary only; mask data is still used upstream for OBB / LxW metrics."""
    thickness = max(2, int(round(2 * LABEL_SIZE_FACTOR)))
    for instance in instances:
        if not np.any(instance.mask):
            continue
        mask_u8 = instance.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(frame, contours, -1, YOLO_CONTOUR_BGR, thickness, cv2.LINE_AA)


def _draw_instance_oriented_boxes(frame: np.ndarray, instances: list[SegInstance]) -> None:
    thickness = max(2, int(round(2 * LABEL_SIZE_FACTOR)))
    for instance in instances:
        box_pts = instance.box_pts
        if box_pts is None:
            box_pts = oriented_box_from_mask(instance.mask)
        if box_pts is None:
            continue
        box_i32 = np.round(box_pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [box_i32], True, YOLO_OBB_BGR, thickness, cv2.LINE_AA)


def _draw_instance_labels(
    frame: np.ndarray,
    instances: list[SegInstance],
    water_cut_overlays: list[WaterCutOverlay] | None,
    *,
    label_size_factor: float = LABEL_SIZE_FACTOR,
    split_height_labels: bool = False,
) -> None:
    label_scale = label_size_factor * max(0.75, min(1.2, frame.shape[1] / 550.0))
    line_height = max(24, int(round(34 * label_scale)))
    x0 = int(round(12 * label_size_factor))
    y0 = max(26, int(round(30 * label_scale)))

    line_index = 0
    thickness = max(2, int(round(2 * label_size_factor)))
    for instance in instances:
        metric = _format_lwh_stream_label(
            instance,
            split_height_labels=split_height_labels,
        )
        if metric is None:
            continue
        text = f"{line_index}: {metric}"
        _put_text_outlined(
            frame,
            text,
            (x0, y0 + line_index * line_height),
            scale=label_scale,
            thickness=thickness,
        )
        line_index += 1

        if split_height_labels:
            for height_label in (
                format_height_mm_stream_label(instance.height_mm),
                format_peak_height_mm_stream_label(instance.peak_height_mm),
            ):
                if height_label is None:
                    continue
                text = f"{line_index}: {height_label}"
                _put_text_outlined(
                    frame,
                    text,
                    (x0, y0 + line_index * line_height),
                    scale=label_scale,
                    thickness=thickness,
                )
                line_index += 1

        plane_label = format_plane_depth_stream_label(instance.z_plane_ref_mm)
        if plane_label is not None:
            text = f"{line_index}: {plane_label}"
            _put_text_outlined(
                frame,
                text,
                (x0, y0 + line_index * line_height),
                scale=label_scale,
                thickness=thickness,
            )
            line_index += 1

    if water_cut_overlays:
        base_y = y0 + line_index * line_height + int(round(8 * label_size_factor))
        for index, item in enumerate(water_cut_overlays):
            wc = item.water_cut
            if np.isfinite(wc.water_cut_width_mm):
                text = f"cut[{index}]: {wc.water_cut_width_mm:.1f}mm"
            else:
                text = f"cut[{index}]: {wc.water_cut_width_px:.0f}px"
            _put_text_outlined(
                frame,
                text,
                (x0, base_y + index * line_height),
                scale=label_scale,
                thickness=max(2, int(round(2 * label_size_factor))),
                color=LABEL_COLOR_BGR,
            )


@dataclass
class WaterCutOverlay:
    sam_mask: np.ndarray
    water_cut: WaterCutAnalysis
    box_pts: np.ndarray | None
    prompt_coords: np.ndarray | None = None
    prompt_labels: np.ndarray | None = None


def _draw_water_cut_sam_prompts(frame: np.ndarray, overlays: list[WaterCutOverlay]) -> None:
    for item in overlays:
        if item.prompt_coords is None or item.prompt_labels is None:
            continue
        labels = np.asarray(item.prompt_labels).reshape(-1)
        coords = np.asarray(item.prompt_coords, dtype=np.float32).reshape(-1, 2)
        fg = labels == 1
        if not np.any(fg):
            continue
        SamRefiner.draw_prompts(frame, coords[fg], labels[fg])


def _draw_live_water_cut_prompts(frame: np.ndarray, instances: list[SegInstance]) -> None:
    """Draw oriented-box SAM foreground prompts for water-cut preview (no SAM run)."""
    for instance in instances:
        preview = prepare_water_cut_box_prompts(instance.mask)
        if (
            preview is None
            or preview.prompt_coords is None
            or preview.prompt_labels is None
            or len(preview.prompt_coords) == 0
        ):
            continue
        SamRefiner.draw_prompts(frame, preview.prompt_coords, preview.prompt_labels)


def compose_record_frame(
    image_bgr: np.ndarray,
    instances: list[SegInstance],
    record_info: CaptureRecordInfo,
    *,
    water_cut_overlays: list[WaterCutOverlay] | None = None,
    roi: RoiRect | None = None,
    label_instances: list[SegInstance] | None = None,
    draw_oriented_boxes: bool = True,
    label_size_factor: float = LABEL_SIZE_FACTOR,
) -> np.ndarray:
    frame = compose_stream_frame(
        image_bgr,
        instances,
        water_cut_overlays=water_cut_overlays,
        roi=roi,
        label_instances=label_instances,
        draw_oriented_boxes=draw_oriented_boxes,
        label_size_factor=label_size_factor,
    )
    _draw_record_info_block(
        frame,
        record_info_lines(record_info),
        label_size_factor=label_size_factor,
    )
    return frame


def compose_stream_frame(
    image_bgr: np.ndarray,
    instances: list[SegInstance],
    *,
    water_cut_overlays: list[WaterCutOverlay] | None = None,
    status_text: str | None = None,
    roi: RoiRect | None = None,
    label_instances: list[SegInstance] | None = None,
    draw_oriented_boxes: bool = False,
    label_size_factor: float = LABEL_SIZE_FACTOR,
    split_height_labels: bool = False,
) -> np.ndarray:
    frame = image_bgr.copy()
    labels = label_instances if label_instances is not None else instances

    _draw_roi_rect(frame, roi, label_size_factor=label_size_factor)
    _draw_instance_contours(frame, instances)
    if draw_oriented_boxes:
        _draw_instance_oriented_boxes(frame, instances)
    _draw_plane_sample_markers(frame, instances)
    _draw_peak_height_markers(frame, instances)
    _draw_average_height_markers(frame, instances)
    _draw_instance_labels(
        frame,
        labels,
        water_cut_overlays,
        label_size_factor=label_size_factor,
        split_height_labels=split_height_labels,
    )

    if water_cut_overlays:
        sam_blend = frame.astype(np.float32)
        sam_color = np.array(SAM_COLOR_BGR, dtype=np.float32)
        for item in water_cut_overlays:
            if np.any(item.sam_mask):
                sam_blend[item.sam_mask] = sam_blend[item.sam_mask] * 0.65 + sam_color * 0.35
        frame = sam_blend.astype(np.uint8)
        for item in water_cut_overlays:
            draw_water_cut_overlay(
                frame,
                item.water_cut,
                draw_pca_axis=True,
                draw_centerline=False,
                clip_box=item.box_pts,
            )
        _draw_water_cut_sam_prompts(frame, water_cut_overlays)
    else:
        _draw_live_water_cut_prompts(frame, instances)

    if status_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        width_factor = max(0.75, min(1.0, frame.shape[1] / 900.0))
        scale = label_size_factor * max(0.55, min(0.95, width_factor))
        thickness = max(1, int(round(scale * 1.5)))
        (tw, th), baseline = cv2.getTextSize(status_text, font, scale, thickness)
        h, w = frame.shape[:2]
        pad_x = max(10, int(round(18 * label_size_factor)))
        pad_y = max(8, int(round(12 * label_size_factor)))
        bx0 = max(0, (w - tw) // 2 - pad_x)
        by0 = max(0, (h - th) // 2 - pad_y)
        bx1 = min(w, bx0 + tw + pad_x * 2)
        by1 = min(h, by0 + th + pad_y * 2 + baseline)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (30, 30, 30), -1)
        cv2.rectangle(
            frame,
            (bx0, by0),
            (bx1, by1),
            (0, 180, 255),
            max(1, int(round(2 * label_size_factor))),
        )
        _put_text_outlined(
            frame,
            status_text,
            (bx0 + pad_x, by0 + pad_y + th),
            font=font,
            scale=scale,
            color=(240, 240, 240),
            thickness=thickness,
        )

    return frame
