# -*- coding: utf-8 -*-
"""
Interactive tuner for round-bun water-cut (水切) CV bright-region segmentation.

Three sliders + fast preview (ROI crop). Press M for full width analysis.

Usage:
  python water_cut_cv_tuner.py --image path/to/bread.jpg
  python water_cut_cv_tuner.py --camera

Keys: 1/2/3 views | Y=YOLO | C=camera | M=full width | S/L=save/load | Q=quit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from color_viewer import RoiRect, YoloSegmenter
from extract_center_contour import _otsu_threshold, _rgb_to_lab_l_uint8
from sam_centerline import WaterCutAnalysis, analyze_water_cut, draw_water_cut_overlay

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PARAMS_FILE = PROJECT_ROOT / "config" / "water_cut_cv_params.json"

COLOR_OBJECT = (0, 200, 0)
COLOR_SLIT = (0, 215, 255)
COLOR_TEXT = (40, 40, 40)
COLOR_TEXT_BG = (240, 240, 240)

# Fixed internals (not exposed in UI)
_FIXED = {
    "top_margin": 5,
    "top_relax_ratio": 0.42,
    "center_pick_ratio": 0.15,
    "v_close_kernel_h": 15,
    "open_iterations": 1,
    "close_disk_iterations": 1,
    "min_object_pixels": 100,
    "manual_threshold": 0,
}


@dataclass
class WaterCutCvParams:
    """User-facing params (3 sliders). Legacy JSON keys are mapped in from_dict."""

    brightness: int = 8
    morph_strength: int = 3
    thresh_offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WaterCutCvParams:
        if "brightness" in data:
            return cls(
                brightness=int(data.get("brightness", 8)),
                morph_strength=int(data.get("morph_strength", data.get("v_close_iterations", 3))),
                thresh_offset=int(data.get("thresh_offset", 0)),
            )
        return cls(
            brightness=int(data.get("body_margin", 8)),
            morph_strength=int(data.get("v_close_iterations", 3)),
            thresh_offset=int(data.get("threshold_offset", 0)),
        )


@dataclass
class SlitSegmentResult:
    slit_mask: np.ndarray
    object_mask: np.ndarray
    lab_l: np.ndarray
    otsu_threshold: int
    light_after_morph: np.ndarray
    message: str = ""


def _object_bbox(mask: np.ndarray, pad: int = 48) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        h, w = mask.shape[:2]
        return 0, 0, w, h
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def fast_slit_width_px(slit_mask: np.ndarray) -> float:
    """Quick horizontal span at slit median row (good for vertical round-bun cuts)."""
    ys, xs = np.nonzero(slit_mask)
    if xs.size == 0:
        return float("nan")
    cy = int(np.median(ys))
    row = slit_mask[cy]
    cols = np.flatnonzero(row)
    if cols.size == 0:
        return float("nan")
    return float(cols.max() - cols.min() + 1)


def segment_slit_bright_region(
    rgb: np.ndarray,
    object_mask: np.ndarray,
    params: WaterCutCvParams,
) -> SlitSegmentResult:
    h, w = rgb.shape[:2]
    obj = (object_mask > 0).astype(np.uint8) * 255
    empty_u8 = np.zeros((h, w), dtype=np.uint8)
    empty_b = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(obj > 0)
    min_px = _FIXED["min_object_pixels"]
    if ys.size < min_px:
        return SlitSegmentResult(
            slit_mask=empty_u8,
            object_mask=obj,
            lab_l=empty_u8,
            otsu_threshold=0,
            light_after_morph=empty_b,
            message=f"object too small ({ys.size} px)",
        )

    ox0, ox1 = int(xs.min()), int(xs.max())
    oy0, oy1 = int(ys.min()), int(ys.max())
    obj_w = ox1 - ox0 + 1
    obj_h = oy1 - oy0 + 1
    obj_cx = (ox0 + ox1) / 2.0

    lab_l = _rgb_to_lab_l_uint8(rgb)
    roi = lab_l[obj > 0]
    base_thresh = int(_otsu_threshold(roi)) + int(params.thresh_offset)
    base_thresh = int(np.clip(base_thresh, 1, 254))

    body_margin = int(params.brightness)
    strict_body = min(255, base_thresh + body_margin)
    strict_top = min(255, base_thresh + _FIXED["top_margin"])
    row_threshold = np.full(h, strict_body, dtype=np.float64)
    y_relax_end = min(h, oy0 + int(_FIXED["top_relax_ratio"] * obj_h) + 1)
    row_threshold[:y_relax_end] = strict_top

    light = (lab_l.astype(np.float64) >= row_threshold[:, np.newaxis]) & (obj > 0)

    v_h = max(3, int(_FIXED["v_close_kernel_h"]) | 1)
    vbar = np.zeros((v_h, 3), dtype=bool)
    vbar[:, 1] = True
    v_iters = max(0, int(params.morph_strength))
    if v_iters > 0:
        light = ndimage.binary_closing(light, structure=vbar, iterations=v_iters)

    disk5 = np.zeros((5, 5), dtype=bool)
    yy, xx = np.ogrid[-2:3, -2:3]
    disk5[yy * yy + xx * xx <= 4] = True
    disk3 = np.zeros((3, 3), dtype=bool)
    yy3, xx3 = np.ogrid[-1:2, -1:2]
    disk3[yy3 * yy3 + xx3 * xx3 <= 2] = True
    if _FIXED["open_iterations"] > 0:
        light = ndimage.binary_opening(light, structure=disk3, iterations=_FIXED["open_iterations"])
    if _FIXED["close_disk_iterations"] > 0:
        light = ndimage.binary_closing(light, structure=disk5, iterations=_FIXED["close_disk_iterations"])

    light_after_morph = light.copy()
    labels, n = ndimage.label(light, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return SlitSegmentResult(
            slit_mask=(light.astype(np.uint8) * 255),
            object_mask=obj,
            lab_l=lab_l,
            otsu_threshold=base_thresh,
            light_after_morph=light_after_morph,
            message="no component after morph",
        )

    pick_ratio = _FIXED["center_pick_ratio"]
    mid_x0 = int(obj_cx - pick_ratio * obj_w)
    mid_x1 = int(obj_cx + pick_ratio * obj_w)
    best = 0
    best_area = 0
    for index in range(1, n + 1):
        area = int((labels == index).sum())
        _cy, cx_i = ndimage.center_of_mass(labels == index)
        cx_i = int(cx_i) if not np.isnan(cx_i) else 0
        if mid_x0 <= cx_i <= mid_x1 and area > best_area:
            best_area = area
            best = index
    if best == 0:
        for index in range(1, n + 1):
            area = int((labels == index).sum())
            if area > best_area:
                best_area = area
                best = index

    slit = np.zeros((h, w), dtype=np.uint8)
    if best > 0:
        slit[labels == best] = 255

    return SlitSegmentResult(
        slit_mask=slit,
        object_mask=obj,
        lab_l=lab_l,
        otsu_threshold=base_thresh,
        light_after_morph=light_after_morph,
        message="" if best > 0 else "center component not found",
    )


def _resize_for_display(image: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 0.999:
        return image
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


class WaterCutCvTuner:
    WINDOW = "Water-cut CV Tuner"
    CONTROLS = "Controls"

    def __init__(
        self,
        image_bgr: np.ndarray,
        *,
        object_mask: np.ndarray | None = None,
        params: WaterCutCvParams | None = None,
        segmenter: YoloSegmenter | None = None,
        roi: RoiRect | None = None,
        yolo_imgsz: int = 640,
        fetch_frame_fn=None,
        preview_max_side: int = 1280,
        crop_pad: int = 48,
    ) -> None:
        self.image_bgr = image_bgr
        self.params = params or WaterCutCvParams()
        self.segmenter = segmenter
        self.roi = roi
        self.yolo_imgsz = yolo_imgsz
        self.fetch_frame_fn = fetch_frame_fn
        self.object_mask = object_mask
        self.preview_max_side = preview_max_side
        self.crop_pad = crop_pad
        self.view_mode = 0
        self._trackbars_ready = False
        self._trackbar_binding = False
        self._params_snapshot: tuple[Any, ...] | None = None
        self._last_result: SlitSegmentResult | None = None
        self._last_water_cut: WaterCutAnalysis | None = None
        self._fast_width_px = float("nan")
        self._width_mode = "fast"
        self._last_full_measure = 0.0
        self._recompute_ms = 0.0

    def _full_roi(self) -> RoiRect:
        h, w = self.image_bgr.shape[:2]
        return self.roi if self.roi is not None else RoiRect(0, 0, w, h)

    def run_yolo(self) -> None:
        if self.segmenter is None:
            print("YOLO not configured.")
            return
        instances = self.segmenter.segment_all(
            self.image_bgr, self._full_roi(), imgsz=self.yolo_imgsz
        )
        if not instances:
            print("YOLO: no detections.")
            self.object_mask = None
            return
        primary = max(instances, key=lambda item: float(np.sum(item.mask)))
        self.object_mask = primary.mask.astype(np.uint8) * 255
        print(f"YOLO: {primary.class_name} conf={primary.confidence:.2f}")

    def _recompute(self, *, full_width: bool = False) -> None:
        t0 = time.perf_counter()
        if self.object_mask is None or not np.any(self.object_mask):
            self._last_result = None
            self._last_water_cut = None
            self._fast_width_px = float("nan")
            self._recompute_ms = 0.0
            return

        x0, y0, x1, y1 = _object_bbox(self.object_mask, self.crop_pad)
        rgb = cv2.cvtColor(self.image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        obj_crop = self.object_mask[y0:y1, x0:x1]
        crop_result = segment_slit_bright_region(rgb, obj_crop, self.params)

        h, w = self.image_bgr.shape[:2]
        full_slit = np.zeros((h, w), dtype=np.uint8)
        full_slit[y0:y1, x0:x1] = crop_result.slit_mask
        full_lab = np.zeros((h, w), dtype=np.uint8)
        full_lab[y0:y1, x0:x1] = crop_result.lab_l

        self._last_result = SlitSegmentResult(
            slit_mask=full_slit,
            object_mask=self.object_mask,
            lab_l=full_lab,
            otsu_threshold=crop_result.otsu_threshold,
            light_after_morph=crop_result.light_after_morph,
            message=crop_result.message,
        )

        slit_bool = full_slit > 0
        self._fast_width_px = fast_slit_width_px(slit_bool)

        if full_width and np.any(slit_bool):
            self._last_water_cut = analyze_water_cut(slit_bool)
            self._width_mode = "full"
            self._last_full_measure = time.perf_counter()
        elif not np.any(slit_bool):
            self._last_water_cut = None
            self._width_mode = "fast"
        else:
            self._width_mode = "fast"

        self._recompute_ms = (time.perf_counter() - t0) * 1000.0

    def _draw_overlay(self) -> np.ndarray:
        frame = self.image_bgr
        if self.object_mask is not None and np.any(self.object_mask):
            m = self.object_mask > 0
            frame = frame.copy()
            frame[m] = (
                frame[m].astype(np.float32) * 0.55 + np.array(COLOR_OBJECT, np.float32) * 0.45
            ).astype(np.uint8)

        result = self._last_result
        if result is None:
            return frame

        slit = result.slit_mask > 0
        if np.any(slit):
            frame = frame.copy()
            frame[slit] = (
                frame[slit].astype(np.float32) * 0.45 + np.array(COLOR_SLIT, np.float32) * 0.55
            ).astype(np.uint8)

        if self._width_mode == "full" and self._last_water_cut is not None:
            frame = frame.copy()
            draw_water_cut_overlay(frame, self._last_water_cut, width_line_only=True)
        elif np.isfinite(self._fast_width_px) and np.any(slit):
            frame = frame.copy()
            ys, xs = np.nonzero(slit)
            cy = int(np.median(ys))
            cols = np.flatnonzero(slit[cy])
            if cols.size > 0:
                x1, x2 = int(cols.min()), int(cols.max())
                cv2.line(frame, (x1, cy), (x2, cy), (0, 215, 255), 2, cv2.LINE_AA)

        return frame

    def _draw_l_channel(self) -> np.ndarray:
        if self._last_result is None:
            return self.image_bgr.copy()
        heat = cv2.applyColorMap(self._last_result.lab_l, cv2.COLORMAP_TURBO)
        if self.object_mask is not None:
            heat[self.object_mask == 0] = (heat[self.object_mask == 0] * 0.25).astype(np.uint8)
        return heat

    def _draw_binary_debug(self) -> np.ndarray:
        if self._last_result is None:
            return self.image_bgr.copy()
        after = (self._last_result.light_after_morph.astype(np.uint8) * 255)
        return cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)

    def _compose_main_view(self) -> np.ndarray:
        if self.view_mode == 1:
            base = self._draw_l_channel()
        elif self.view_mode == 2:
            base = self._draw_binary_debug()
        else:
            base = self._draw_overlay()

        banner_h = 88
        banner = np.full((banner_h, base.shape[1], 3), COLOR_TEXT_BG, dtype=np.uint8)
        lines = [
            "3 sliders | 1/2/3 view | Y=YOLO C=cam M=full width S/L=save Q=quit",
        ]
        if self.object_mask is None or not np.any(self.object_mask):
            lines.append("No mask — press Y")
        elif self._last_result is not None:
            r = self._last_result
            slit_px = int((r.slit_mask > 0).sum())
            if self._width_mode == "full" and self._last_water_cut is not None:
                wc = f"{self._last_water_cut.water_cut_width_px:.1f}px (full)"
            else:
                wc = f"{self._fast_width_px:.1f}px (fast, M=full)"
            lines.append(
                f"Otsu={r.otsu_threshold} slit={slit_px}px width={wc} "
                f"{self._recompute_ms:.0f}ms {r.message}"
            )
        y = 24
        for line in lines:
            cv2.putText(banner, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
            y += 22
        return _resize_for_display(np.vstack([banner, base]), self.preview_max_side)

    def _params_snapshot_key(self) -> tuple[Any, ...]:
        return tuple(self.params.to_dict().values())

    def _read_trackbars(self) -> bool:
        if self._trackbar_binding or not self._trackbars_ready:
            return False
        try:
            g = cv2.getTrackbarPos
            w = self.CONTROLS
            self.params.brightness = g("brightness", w)
            self.params.morph_strength = g("morph", w)
            self.params.thresh_offset = g("thresh_shift", w) - 30
        except cv2.error:
            return False
        return True

    def _poll_trackbars(self) -> bool:
        if not self._read_trackbars():
            return False
        key = self._params_snapshot_key()
        if key == self._params_snapshot:
            return False
        self._params_snapshot = key
        return True

    def _sync_trackbars(self) -> None:
        self._trackbar_binding = True
        p = self.params
        s = cv2.setTrackbarPos
        s("brightness", self.CONTROLS, int(p.brightness))
        s("morph", self.CONTROLS, int(p.morph_strength))
        s("thresh_shift", self.CONTROLS, int(p.thresh_offset) + 30)
        self._trackbar_binding = False
        self._params_snapshot = self._params_snapshot_key()

    def _create_trackbars(self) -> None:
        self._trackbars_ready = False
        cv2.namedWindow(self.CONTROLS, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.CONTROLS, 380, 90)
        p = self.params

        def noop(_: int) -> None:
            pass

        cv2.createTrackbar("brightness", self.CONTROLS, int(p.brightness), 40, noop)
        cv2.createTrackbar("morph", self.CONTROLS, int(p.morph_strength), 10, noop)
        cv2.createTrackbar("thresh_shift", self.CONTROLS, int(p.thresh_offset) + 30, 60, noop)
        self._trackbars_ready = True
        self._params_snapshot = self._params_snapshot_key()

    def save_params(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.params.to_dict(), indent=2), encoding="utf-8")
        print(f"Saved: {path}")

    def load_params(self, path: Path) -> None:
        self.params = WaterCutCvParams.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self._sync_trackbars()
        self._recompute(full_width=True)

    def open_image_dialog(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")]
            )
            root.destroy()
        except Exception as exc:
            print(exc)
            return
        if not path:
            return
        img = cv2.imread(path)
        if img is not None:
            self.image_bgr = img
            self.object_mask = None

    def grab_camera_frame(self) -> None:
        if self.fetch_frame_fn is None:
            return
        frame, _ = self.fetch_frame_fn()
        if frame is not None:
            self.image_bgr = frame
            self.object_mask = None

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        self._create_trackbars()
        if self.segmenter is not None and self.object_mask is None:
            self.run_yolo()
        self._recompute(full_width=True)

        idle_full_after = 0.6
        last_change = time.perf_counter()

        while True:
            if self._poll_trackbars():
                self._recompute(full_width=False)
                last_change = time.perf_counter()
            elif time.perf_counter() - last_change > idle_full_after and self._width_mode == "fast":
                self._recompute(full_width=True)

            cv2.imshow(self.WINDOW, self._compose_main_view())
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord("1"):
                self.view_mode = 0
            elif key == ord("2"):
                self.view_mode = 1
            elif key == ord("3"):
                self.view_mode = 2
            elif key in (ord("o"), ord("O")):
                self.open_image_dialog()
                if self.segmenter is not None:
                    self.run_yolo()
                self._recompute(full_width=True)
            elif key in (ord("y"), ord("Y")):
                self.run_yolo()
                self._recompute(full_width=True)
            elif key in (ord("c"), ord("C")):
                self.grab_camera_frame()
                if self.segmenter is not None:
                    self.run_yolo()
                self._recompute(full_width=True)
            elif key in (ord("m"), ord("M")):
                self._recompute(full_width=True)
            elif key in (ord("s"), ord("S")):
                self.save_params(DEFAULT_PARAMS_FILE)
            elif key in (ord("l"), ord("L")):
                if DEFAULT_PARAMS_FILE.is_file():
                    self.load_params(DEFAULT_PARAMS_FILE)
            elif key in (ord("w"), ord("W")):
                out = PROJECT_ROOT / "output" / f"water_cut_tune_{time.strftime('%Y%m%d_%H%M%S')}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), self._compose_main_view())
            elif key in (ord("r"), ord("R")):
                self.params = WaterCutCvParams()
                self._sync_trackbars()
                self._recompute(full_width=True)

        cv2.destroyAllWindows()


def _build_camera_fetch(args: argparse.Namespace):
    from capture_2d import close_camera, fetch_frame, list_devices, open_camera

    devices = list_devices()
    selected = devices[args.device_index]
    camera, payload_size, pixel_format = open_camera(selected["info"], args.pixel_format)

    def fetch():
        try:
            return fetch_frame(camera, payload_size, pixel_format, timeout_ms=args.timeout_ms, warmup_frames=0)
        except RuntimeError as exc:
            print(exc)
            return None, None

    return fetch, lambda: close_camera(camera)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune CV water-cut (3 sliders, fast preview).")
    parser.add_argument("--image", default=None)
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--pixel-format", default=None)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--yolo-model", default="yolov8n-seg.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--mask-npy", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--preview-max-side", type=int, default=1280)
    args = parser.parse_args()

    if not args.image and not args.camera:
        parser.error("Provide --image and/or --camera")

    params_path = Path(args.params) if args.params else DEFAULT_PARAMS_FILE
    params = WaterCutCvParams()
    if params_path.is_file():
        params = WaterCutCvParams.from_dict(json.loads(params_path.read_text(encoding="utf-8")))

    object_mask = None
    if args.mask_npy:
        object_mask = (np.load(args.mask_npy) > 0).astype(np.uint8) * 255

    fetch_fn = close_fn = None
    image_bgr = None
    if args.camera:
        fetch_fn, close_fn = _build_camera_fetch(args)
        image_bgr, _ = fetch_fn()
        if image_bgr is None:
            raise RuntimeError("Camera grab failed.")
    if args.image:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise RuntimeError(f"Cannot read {args.image}")

    segmenter = None if args.no_yolo else YoloSegmenter(args.yolo_model, conf=args.yolo_conf)
    h, w = image_bgr.shape[:2]
    tuner = WaterCutCvTuner(
        image_bgr,
        object_mask=object_mask,
        params=params,
        segmenter=segmenter,
        roi=RoiRect(0, 0, w, h),
        yolo_imgsz=args.yolo_imgsz,
        fetch_frame_fn=fetch_fn,
        preview_max_side=args.preview_max_side,
    )
    try:
        tuner.run()
    finally:
        if close_fn:
            close_fn()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
