# -*- coding: utf-8 -*-
"""Lightweight overlay compositing for the MJPEG stream."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from color_viewer import PALETTE, SegInstance
from object_measure import format_lw_label, oriented_box_from_mask
from sam_centerline import WaterCutAnalysis, draw_water_cut_overlay
from yolo_sam_refine import SamRefiner

SAM_COLOR_BGR = SamRefiner.SAM_COLOR_BGR
LABEL_COLOR_BGR = (0, 255, 255)  # yellow — high contrast on most scenes
LABEL_OUTLINE_BGR = (0, 0, 0)
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


def format_capture_lw_label(instance: SegInstance) -> str:
    if np.isfinite(instance.length_mm) and np.isfinite(instance.width_mm):
        return f"LxW: {instance.length_mm:.1f}mm x {instance.width_mm:.1f}mm"
    if np.isfinite(instance.length_px) and np.isfinite(instance.width_px):
        return f"LxW: {instance.length_px:.1f}px x {instance.width_px:.1f}px"
    return "LxW: ---"


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
    height: str,
    temperature: str,
    weight: str,
    water_cut_enabled: bool = False,
    water_cut_overlays: list["WaterCutOverlay"] | None = None,
) -> CaptureRecordInfo:
    primary = instances[0] if instances else None
    lw_text = format_capture_lw_label(primary) if primary is not None else "LxW: ---"
    water_cut_line, water_cut_mm = format_water_cut_record_line(
        water_cut_overlays,
        enabled=water_cut_enabled,
    )

    return CaptureRecordInfo(
        height=height,
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


def _draw_record_info_block(frame: np.ndarray, lines: list[str]) -> None:
    """Draw capture summary text; uses Pillow so Unicode such as °C renders correctly."""
    if not lines:
        return

    font_size = max(
        18,
        int(round(24 * LABEL_SIZE_FACTOR * max(0.7, min(1.0, frame.shape[1] / 900.0)))),
    )
    font = _record_overlay_font(font_size)
    stroke = max(2, int(round(2 * LABEL_SIZE_FACTOR)))
    pad = int(round(12 * LABEL_SIZE_FACTOR))
    line_gap = int(round(8 * LABEL_SIZE_FACTOR))

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
    x0 = int(round(12 * LABEL_SIZE_FACTOR))
    y0 = frame.shape[0] - block_h - int(round(12 * LABEL_SIZE_FACTOR))

    bg_rgb = _bgr_to_rgb(RECORD_BG_BGR)
    text_rgb = _bgr_to_rgb(RECORD_COLOR_BGR)
    outline_rgb = _bgr_to_rgb(LABEL_OUTLINE_BGR)
    border_rgb = (255, 180, 0)  # BGR (0, 180, 255)

    draw.rectangle((x0, y0, x0 + block_w, y0 + block_h), fill=bg_rgb)
    draw.rectangle((x0, y0, x0 + block_w, y0 + block_h), outline=border_rgb, width=2)

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


def _format_lw_px_label(instance: SegInstance) -> str | None:
    return format_lw_label(
        instance.length_mm,
        instance.width_mm,
        instance.length_px,
        instance.width_px,
    )


def _draw_instance_labels(
    frame: np.ndarray,
    instances: list[SegInstance],
    water_cut_overlays: list[WaterCutOverlay] | None,
) -> None:
    label_scale = LABEL_SIZE_FACTOR * max(0.75, min(1.2, frame.shape[1] / 550.0))
    line_height = max(24, int(round(34 * label_scale)))
    x0 = int(round(12 * LABEL_SIZE_FACTOR))
    y0 = max(26, int(round(30 * label_scale)))

    for index, instance in enumerate(instances):
        metric = _format_lw_px_label(instance)
        if metric is None:
            continue
        text = f"{index}: {instance.class_name} {instance.confidence:.2f} | {metric}"
        _put_text_outlined(
            frame,
            text,
            (x0, y0 + index * line_height),
            scale=label_scale,
            thickness=max(2, int(round(2 * LABEL_SIZE_FACTOR))),
        )

    if water_cut_overlays:
        base_y = y0 + len(instances) * line_height + int(round(8 * LABEL_SIZE_FACTOR))
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
                thickness=max(2, int(round(2 * LABEL_SIZE_FACTOR))),
                color=(0, 200, 255),
            )


@dataclass
class WaterCutOverlay:
    sam_mask: np.ndarray
    water_cut: WaterCutAnalysis
    box_pts: np.ndarray | None


def compose_record_frame(
    image_bgr: np.ndarray,
    instances: list[SegInstance],
    record_info: CaptureRecordInfo,
    *,
    water_cut_overlays: list[WaterCutOverlay] | None = None,
) -> np.ndarray:
    frame = compose_stream_frame(
        image_bgr,
        instances,
        water_cut_overlays=water_cut_overlays,
    )
    _draw_record_info_block(frame, record_info_lines(record_info))
    return frame


def compose_stream_frame(
    image_bgr: np.ndarray,
    instances: list[SegInstance],
    *,
    water_cut_overlays: list[WaterCutOverlay] | None = None,
    status_text: str | None = None,
) -> np.ndarray:
    blend = image_bgr.astype(np.float32)

    for index, instance in enumerate(instances):
        color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        mask = instance.mask
        blend[mask] = blend[mask] * 0.70 + color * 0.30

    frame = blend.astype(np.uint8)

    for index, instance in enumerate(instances):
        box_pts = instance.box_pts
        if box_pts is None:
            box_pts = oriented_box_from_mask(instance.mask)
        if box_pts is not None:
            box_i32 = np.round(box_pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [box_i32], True, (0, 220, 120), 2, cv2.LINE_AA)

    _draw_instance_labels(frame, instances, water_cut_overlays)

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

    if status_text:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = LABEL_SIZE_FACTOR * max(1.2, min(2.2, frame.shape[1] / 500.0))
        thickness = max(3, int(round(scale * 2)))
        (tw, th), baseline = cv2.getTextSize(status_text, font, scale, thickness)
        h, w = frame.shape[:2]
        pad_x = int(round(64 * LABEL_SIZE_FACTOR))
        pad_y = int(round(48 * LABEL_SIZE_FACTOR))
        bx0 = max(0, (w - tw) // 2 - pad_x)
        by0 = max(0, (h - th) // 2 - pad_y)
        bx1 = min(w, bx0 + tw + pad_x * 2)
        by1 = min(h, by0 + th + pad_y * 2 + baseline)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (30, 30, 30), -1)
        cv2.rectangle(frame, (bx0, by0), (bx1, by1), (0, 180, 255), int(round(4 * LABEL_SIZE_FACTOR)))
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
