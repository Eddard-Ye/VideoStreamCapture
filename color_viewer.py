# -*- coding: utf-8 -*-
"""Interactive color image viewer with zoom/pan and YOLO segmentation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from camera_calib_2d import attach_instance_metrics, water_cut_width_mm_2d
from camera_intrinsics import RgbIntrinsics
from object_measure import (
    align_depth_to_color,
    depth_to_mm,
    format_lw_label,
    format_lwh_mm,
    measure_mask_mm,
    oriented_box_from_mask,
    oriented_box_metrics_from_mask,
    plane_depth_in_roi,
)
from yolo_sam_refine import (
    SamRefiner,
    SamSubRegion,
    prepare_water_cut_box_prompts,
    resolve_inference_device,
    run_water_cut_box_sam,
    yolo_predict_device,
)
from mask_refine import refine_mask_otsu
from sam_centerline import analyze_water_cut, draw_water_cut_overlay

YOLO_MODEL_URLS = {
    "yolov8n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-seg.pt",
    "yolov8s-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-seg.pt",
    "yolov8m-seg.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-seg.pt",
}

PALETTE = [
    (255, 128, 0),
    (0, 200, 255),
    (255, 64, 160),
    (120, 255, 120),
    (255, 220, 0),
    (160, 120, 255),
]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROI_FILE = os.path.join(PROJECT_ROOT, "config", "roi.json")


@dataclass(frozen=True)
class RoiRect:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x < self.x2 and self.y1 <= y < self.y2

    def crop(self, image_bgr: np.ndarray) -> np.ndarray:
        return image_bgr[self.y1 : self.y2, self.x1 : self.x2]

    def embed_mask(self, roi_mask: np.ndarray, full_height: int, full_width: int) -> np.ndarray:
        full_mask = np.zeros((full_height, full_width), dtype=bool)
        roi_h, roi_w = self.height, self.width
        if roi_mask.shape != (roi_h, roi_w):
            roi_mask = cv2.resize(
                roi_mask.astype(np.uint8),
                (roi_w, roi_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        full_mask[self.y1 : self.y2, self.x1 : self.x2] = roi_mask
        return full_mask

    def __str__(self) -> str:
        return f"({self.x1},{self.y1})-({self.x2},{self.y2}) size={self.width}x{self.height}"


def crop_mask_tight(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Crop a boolean mask to its tight axis-aligned bounding box."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return mask[y1:y2, x1:x2].astype(bool), (x1, y1, x2, y2)


def crop_color_by_mask(color_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Crop color image to mask contour; pixels outside mask are transparent (BGRA)."""
    cropped = crop_mask_tight(mask)
    if cropped is None:
        return None

    mask_crop, (x1, y1, x2, y2) = cropped
    color_crop = color_bgr[y1:y2, x1:x2].copy()
    bgra = cv2.cvtColor(color_crop, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask_crop.astype(np.uint8) * 255
    return bgra


def _clamp_roi(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> RoiRect:
    x1 = int(np.clip(x1, 0, width - 2))
    y1 = int(np.clip(y1, 0, height - 2))
    x2 = int(np.clip(x2, x1 + 1, width))
    y2 = int(np.clip(y2, y1 + 1, height))
    if x2 - x1 < 32 or y2 - y1 < 32:
        raise ValueError(f"ROI too small after clamping: ({x1},{y1})-({x2},{y2})")
    return RoiRect(x1, y1, x2, y2)


def parse_roi_spec(spec: str, image_width: int, image_height: int) -> RoiRect:
    parts = [float(part.strip()) for part in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI spec must be 'x1,y1,x2,y2' in pixels or normalized ratios (0-1).")

    if all(0.0 <= part <= 1.0 for part in parts):
        x1 = int(round(parts[0] * image_width))
        y1 = int(round(parts[1] * image_height))
        x2 = int(round(parts[2] * image_width))
        y2 = int(round(parts[3] * image_height))
    else:
        x1, y1, x2, y2 = map(int, parts)

    return _clamp_roi(x1, y1, x2, y2, image_width, image_height)


def load_roi_from_file(roi_file: str, image_width: int, image_height: int) -> RoiRect:
    if not os.path.isfile(roi_file):
        raise FileNotFoundError(f"ROI config not found: {roi_file}")

    with open(roi_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if all(key in data for key in ("x1", "y1", "x2", "y2")):
        return _clamp_roi(int(data["x1"]), int(data["y1"]), int(data["x2"]), int(data["y2"]), image_width, image_height)

    if all(key in data for key in ("x1_ratio", "y1_ratio", "x2_ratio", "y2_ratio")):
        return parse_roi_spec(
            f"{data['x1_ratio']},{data['y1_ratio']},{data['x2_ratio']},{data['y2_ratio']}",
            image_width,
            image_height,
        )

    raise ValueError(f"Invalid ROI config in {roi_file}. Use pixel or ratio keys.")


def save_roi_to_file(roi: RoiRect, roi_file: str, image_width: int, image_height: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(roi_file)), exist_ok=True)
    data = {
        "x1": roi.x1,
        "y1": roi.y1,
        "x2": roi.x2,
        "y2": roi.y2,
        "x1_ratio": round(roi.x1 / image_width, 6),
        "y1_ratio": round(roi.y1 / image_height, 6),
        "x2_ratio": round(roi.x2 / image_width, 6),
        "y2_ratio": round(roi.y2 / image_height, 6),
    }
    with open(roi_file, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    print(f"Saved ROI to {roi_file}: {roi}")


def resolve_roi(
    image_width: int,
    image_height: int,
    roi_spec: Optional[str] = None,
    roi_file: Optional[str] = None,
) -> RoiRect:
    if roi_spec:
        return parse_roi_spec(roi_spec, image_width, image_height)

    candidate_files = []
    if roi_file:
        candidate_files.append(roi_file)
    candidate_files.append(DEFAULT_ROI_FILE)

    for candidate in candidate_files:
        if candidate and os.path.isfile(candidate):
            roi = load_roi_from_file(candidate, image_width, image_height)
            print(f"Loaded ROI from {candidate}: {roi}")
            return roi

    raise ValueError(
        "Fixed ROI is required for YOLO. Pass --roi x1,y1,x2,y2 or create config/roi.json."
    )


def resolve_model_path(model_path: str) -> str:
    project_root = PROJECT_ROOT
    candidates = [
        model_path,
        os.path.join(project_root, model_path),
        os.path.join(project_root, "models", model_path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return model_path


@dataclass
class SegInstance:
    mask: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    box_pts: np.ndarray | None = None
    length_mm: float = float("nan")
    width_mm: float = float("nan")
    length_px: float = float("nan")
    width_px: float = float("nan")
    height_mm: float = float("nan")
    z_object_mm: float = float("nan")
    angle_deg: float = float("nan")
    peak_height_mm: float = float("nan")
    peak_height_points: list[tuple[int, int]] = field(default_factory=list)
    z_plane_ref_mm: float = float("nan")
    plane_sample_points: list[tuple[int, int]] = field(default_factory=list)
    sam_regions: list[SamSubRegion] = field(default_factory=list)
    sam_prompt_coords: list[tuple[float, float]] = field(default_factory=list)
    sam_prompt_labels: list[int] = field(default_factory=list)


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / union


def deduplicate_seg_instances(
    instances: list[SegInstance],
    *,
    iou_threshold: float = 0.5,
) -> list[SegInstance]:
    """Keep highest-confidence detection when multiple masks cover the same object."""
    if len(instances) <= 1:
        return instances

    ranked = sorted(instances, key=lambda item: item.confidence, reverse=True)
    kept: list[SegInstance] = []
    for candidate in ranked:
        if all(_mask_iou(candidate.mask, existing.mask) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def keep_top_confidence_instances(
    instances: list[SegInstance],
    *,
    max_count: int = 1,
) -> list[SegInstance]:
    """Keep up to *max_count* detections with the highest confidence."""
    if max_count <= 0 or not instances:
        return []
    ranked = sorted(instances, key=lambda item: item.confidence, reverse=True)
    return ranked[:max_count]


class YoloSegmenter:
    def __init__(
        self,
        model_path: str = "yolov8n-seg.pt",
        conf: float = 0.25,
        *,
        mask_refine: str = "otsu",
        mask_refine_pad: int = 80,
        force_cpu: bool = False,
    ):
        self.model_path = model_path
        self.conf = conf
        self.mask_refine = mask_refine
        self.mask_refine_pad = max(0, int(mask_refine_pad))
        self._model = None
        self._device = resolve_inference_device(force_cpu=force_cpu)

    def _get_model(self):
        if self._model is None:
            from ultralytics import YOLO

            resolved = resolve_model_path(self.model_path)
            print(f"Loading YOLO model: {resolved} on {self._device} ...")
            try:
                self._model = YOLO(resolved)
            except (ConnectionError, OSError) as exc:
                download_hint = YOLO_MODEL_URLS.get(os.path.basename(self.model_path), "")
                models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                message = (
                    f"Failed to load YOLO model '{self.model_path}'. "
                    f"Place the weights file under '{models_dir}'."
                )
                if download_hint:
                    message += f" Download: {download_hint}"
                raise RuntimeError(message) from exc
            print("YOLO model ready.")
        return self._model

    def _apply_otsu_refine(self, image_bgr: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        return refine_mask_otsu(
            image_bgr,
            mask_bool,
            pad=self.mask_refine_pad,
        )

    def segment_all(
        self,
        image_bgr: np.ndarray,
        roi: RoiRect,
        imgsz: int | None = None,
        conf: float | None = None,
        refine_top_n: int | None = None,
    ) -> list[SegInstance]:
        """Run YOLO segmentation inside *roi*.

        When *refine_top_n* is set with Otsu refinement enabled, Otsu runs only on
        the top-N highest-confidence instances after deduplication (not on every box).
        """
        model = self._get_model()
        roi_image = roi.crop(image_bgr)
        if roi_image.size == 0:
            raise ValueError(f"ROI crop is empty: {roi}")

        predict_kwargs = {
            "source": roi_image,
            "conf": self.conf if conf is None else float(conf),
            "verbose": False,
            "device": yolo_predict_device(self._device),
            "retina_masks": True,
        }
        if imgsz is not None:
            predict_kwargs["imgsz"] = int(imgsz)
        results = model.predict(**predict_kwargs)
        if not results:
            return []

        result = results[0]
        if result.masks is None or result.boxes is None:
            return []

        full_height, full_width = image_bgr.shape[:2]
        roi_height, roi_width = roi_image.shape[:2]
        names = result.names or {}
        instances: list[SegInstance] = []
        defer_otsu = self.mask_refine == "otsu" and refine_top_n is not None

        for index in range(len(result.boxes)):
            mask_tensor = result.masks.data[index]
            mask = mask_tensor.cpu().numpy()
            if mask.shape != (roi_height, roi_width):
                mask = cv2.resize(mask, (roi_width, roi_height), interpolation=cv2.INTER_LINEAR)
            class_id = int(result.boxes.cls[index].item())
            confidence = float(result.boxes.conf[index].item())
            mask_bool = roi.embed_mask(mask > 0.5, full_height, full_width)
            if self.mask_refine == "otsu" and not defer_otsu:
                mask_bool = self._apply_otsu_refine(image_bgr, mask_bool)
            instances.append(
                SegInstance(
                    mask=mask_bool,
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                    confidence=confidence,
                )
            )

        instances = deduplicate_seg_instances(instances)
        if not defer_otsu or not instances:
            return instances

        top_n = max(1, int(refine_top_n))
        ranked = sorted(instances, key=lambda item: item.confidence, reverse=True)
        refine_ids = {id(item) for item in ranked[:top_n]}
        for instance in instances:
            if id(instance) not in refine_ids:
                continue
            instance.mask = self._apply_otsu_refine(image_bgr, instance.mask)
        return instances


class ColorViewer:
    WINDOW_NAME = "MV3D Color Viewer"

    def __init__(
        self,
        image_bgr: np.ndarray,
        yolo_model: str = "yolov8n-seg.pt",
        yolo_conf: float = 0.25,
        roi: Optional[RoiRect] = None,
        roi_file: Optional[str] = None,
        depth_image: Optional[np.ndarray] = None,
        intrinsics: Optional[RgbIntrinsics] = None,
        calib_2d=None,
        output_dir: str = "output",
        fetch_frame_fn: Optional[Callable[[], tuple[Optional[np.ndarray], Optional[np.ndarray]]]] = None,
        live: bool = False,
        yolo_live: bool = False,
        yolo_imgsz: int = 640,
        water_cut: bool = False,
        sam_refine: bool = True,
        sam_checkpoint: Optional[str] = None,
        force_cpu: bool = False,
    ):
        if image_bgr is None:
            raise ValueError("image_bgr is required")

        self.original = image_bgr.copy()
        self.depth_raw = None if depth_image is None else depth_image.copy()
        self.intrinsics = intrinsics
        self.calib_2d = calib_2d
        self.output_dir = output_dir
        self.fetch_frame_fn = fetch_frame_fn
        self.live = live and fetch_frame_fn is not None
        self.yolo_live = yolo_live and self.live
        self.yolo_imgsz = max(32, int(yolo_imgsz))
        self.water_cut = water_cut
        self.sam_checkpoint = sam_checkpoint
        self.frozen = not self.live
        self.water_cut_computing = False

        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.min_scale = 0.1
        self.max_scale = 10.0

        self.dragging = False
        self.drag_anchor = (0, 0)
        self.drag_offset = (0.0, 0.0)

        self.instances: list[SegInstance] = []
        self.selected_index: Optional[int] = None
        self.segmenter = YoloSegmenter(yolo_model, conf=yolo_conf, force_cpu=force_cpu)
        self.sam_refine = sam_refine
        self.sam_refiner: Optional[SamRefiner] = None
        self.sam_prompt_mode = False
        if sam_refine or water_cut:
            self.sam_refiner = SamRefiner(checkpoint=sam_checkpoint, force_cpu=force_cpu)
            if sam_refine:
                print("SAM enabled: press A for prompt mode, left/right click fg/bg, M to run SAM.")
        if self.yolo_live:
            print(f"YOLO live mode enabled (imgsz={self.yolo_imgsz}). Press Space to freeze.")
        if self.water_cut:
            if self.sam_refiner is not None:
                print("Water-cut: SAM region + normal/width lines + cutWidth label.")
                if self.yolo_live:
                    print("  Press Space to freeze the frame and compute water-cut width.")
            else:
                print("Water-cut unavailable (SAM checkpoint missing).")

        image_height, image_width = self.original.shape[:2]
        self.roi_file = roi_file or DEFAULT_ROI_FILE
        self.roi = roi or resolve_roi(image_width, image_height, roi_file=self.roi_file)
        print(f"Using fixed YOLO ROI: {self.roi}")

        self.roi_select_mode = False
        self.roi_drawing = False
        self.roi_draw_start: Optional[tuple[int, int]] = None
        self.roi_draw_current: Optional[tuple[int, int]] = None
        self.z_plane_ref_mm: Optional[float] = None

        self.window_w = min(1280, max(640, self.original.shape[1]))
        self.window_h = min(900, max(480, self.original.shape[0] + 80))

    def _depth_mm(self) -> Optional[np.ndarray]:
        if self.depth_raw is None or self.intrinsics is None:
            return None
        height, width = self.original.shape[:2]
        aligned = align_depth_to_color(self.depth_raw, width, height)
        return depth_to_mm(aligned, self.intrinsics.z_unit_mm)

    def _attach_oriented_boxes(self, instances: list[SegInstance]) -> None:
        for instance in instances:
            measured = oriented_box_metrics_from_mask(instance.mask)
            if measured is None:
                continue
            box, length_px, width_px, angle_deg = measured
            instance.box_pts = box
            instance.length_px = length_px
            instance.width_px = width_px
            instance.angle_deg = angle_deg

    def _attach_physical_metrics(self, instances: list[SegInstance]) -> None:
        depth_mm = self._depth_mm()
        if depth_mm is not None and self.intrinsics is not None:
            for instance in instances:
                measured = measure_mask_mm(
                    instance.mask,
                    depth_mm,
                    self.intrinsics,
                    z_plane_ref_mm=self.z_plane_ref_mm,
                )
                if measured is None:
                    continue
                instance.box_pts = measured.box_pts
                instance.length_mm = measured.length_mm
                instance.width_mm = measured.width_mm
                instance.height_mm = measured.height_mm
                instance.z_object_mm = measured.z_object_mm
                instance.angle_deg = measured.angle_deg
            return

        if self.calib_2d is not None:
            attach_instance_metrics(instances, self.calib_2d)

    def _finalize_water_cut_mm(self, analysis) -> None:
        if analysis is None or self.calib_2d is None:
            return
        if np.isfinite(analysis.water_cut_width_mm) and analysis.water_cut_width_mm > 0:
            return
        analysis.water_cut_width_mm = water_cut_width_mm_2d(analysis, self.calib_2d)

    @staticmethod
    def _size_label_for_mask(
        mask: np.ndarray,
        length_mm: float = float("nan"),
        width_mm: float = float("nan"),
    ) -> str:
        measured = oriented_box_metrics_from_mask(mask)
        if measured is None:
            return format_lw_label(length_mm, width_mm, float("nan"), float("nan")) or "LxW: ---"
        _, length_px, width_px, _ = measured
        return format_lw_label(length_mm, width_mm, length_px, width_px) or "LxW: ---"

    def _apply_water_cut(self, instances: list[SegInstance]) -> None:
        for instance in instances:
            instance.sam_regions.clear()

        if not self.water_cut:
            return

        run_sam = self.sam_refiner is not None and (not self.yolo_live or self.frozen)
        depth_mm = self._depth_mm()
        fx = fy = None
        if self.intrinsics is not None:
            fx = self.intrinsics.fx
            fy = self.intrinsics.fy

        work_items: list[tuple[int, SegInstance, SamSubRegion]] = []
        for index, instance in enumerate(instances):
            preview = prepare_water_cut_box_prompts(instance.mask)
            if preview is None:
                print(f"  Water-cut box prompts empty on [{index}] {instance.class_name}.")
                continue
            work_items.append((index, instance, preview))

        if run_sam and work_items:
            self.water_cut_computing = True
            self._pump_ui()

        try:
            for index, instance, preview in work_items:
                if run_sam:
                    try:
                        sam_region = run_water_cut_box_sam(
                            self.sam_refiner,
                            self.original,
                            instance.mask,
                            preview,
                        )
                    except RuntimeError as exc:
                        print(f"  Water-cut SAM failed on [{index}] {instance.class_name}: {exc}")
                        sam_region = None

                    if sam_region is None:
                        print(f"  Water-cut SAM empty on [{index}] {instance.class_name}.")
                    else:
                        preview = sam_region
                        preview.water_cut = analyze_water_cut(
                            preview.mask,
                            depth_mm=depth_mm,
                            fx=fx,
                            fy=fy,
                        )
                        self._finalize_water_cut_mm(preview.water_cut)

                instance.sam_regions = [preview]
        finally:
            if run_sam and work_items:
                self.water_cut_computing = False

    def _pump_ui(self) -> None:
        canvas = self._render()
        cv2.imshow(self.WINDOW_NAME, canvas)
        cv2.waitKey(1)

    def _apply_segmentation(self, instances: list[SegInstance], verbose: bool = True) -> None:
        self.instances = instances
        self._attach_oriented_boxes(self.instances)
        self._apply_water_cut(self.instances)
        self._attach_physical_metrics(self.instances)
        self.selected_index = None

        if not verbose:
            return

        if not self.instances:
            print("YOLO found no segmented instances.")
            return

        print(f"YOLO found {len(self.instances)} instance(s):")
        for index, instance in enumerate(self.instances):
            size_label = format_lw_label(
                instance.length_mm,
                instance.width_mm,
                instance.length_px,
                instance.width_px,
            )
            size_text = size_label or "size unavailable"
            print(
                f"  [{index}] {instance.class_name} conf={instance.confidence:.2f} {size_text}"
            )
            for region_idx, region in enumerate(instance.sam_regions):
                n_fg = n_bg = 0
                if region.prompt_labels is not None:
                    n_fg = int(np.sum(region.prompt_labels == 1))
                    n_bg = int(np.sum(region.prompt_labels == 0))
                sam_text = ""
                if np.any(region.mask):
                    sam_text = f" sam_area={int(region.mask.sum())} score={region.sam_score:.3f}"
                print(
                    f"    cut[{region_idx}] {region.name} "
                    f"fg={n_fg} bg={n_bg}{sam_text}"
                )
        if self.sam_refine:
            print("Press A to enter SAM prompt mode, then left/right click and M to run SAM.")
        self._save_segmentation_crops()

    def _update_yolo_live(self) -> None:
        try:
            instances = self.segmenter.segment_all(
                self.original,
                self.roi,
                imgsz=self.yolo_imgsz,
            )
        except (RuntimeError, ValueError):
            return
        self._apply_segmentation(instances, verbose=False)

    def _run_sam_manual(self) -> None:
        if self.sam_refiner is None:
            print("SAM is disabled. Restart without --no-sam-refine.")
            return
        if self.selected_index is None:
            print("Select a YOLO instance first (left click outside SAM prompt mode).")
            return

        instance = self.instances[self.selected_index]
        if not instance.sam_prompt_coords:
            print("Add SAM prompt points first: A=mode, left=foreground, right=background.")
            return
        if not any(label == 1 for label in instance.sam_prompt_labels):
            print("Need at least one foreground point (left click).")
            return

        pt_coords = np.array(instance.sam_prompt_coords, dtype=np.float32)
        pt_labels = np.array(instance.sam_prompt_labels, dtype=np.int64)
        try:
            result = self.sam_refiner.predict_with_points(
                self.original,
                instance.mask,
                pt_coords,
                pt_labels,
                region_name="manual",
            )
        except RuntimeError as exc:
            print(f"SAM failed: {exc}")
            return

        if result is None or not result.regions:
            print("SAM returned no partition.")
            return

        instance.sam_regions = result.regions
        region = result.regions[0]
        instance.sam_prompt_coords.clear()
        instance.sam_prompt_labels.clear()
        self.sam_prompt_mode = False

        depth_mm = self._depth_mm()
        fx = fy = None
        if self.intrinsics is not None:
            fx = self.intrinsics.fx
            fy = self.intrinsics.fy
        region.water_cut = analyze_water_cut(
            region.mask,
            depth_mm=depth_mm,
            fx=fx,
            fy=fy,
        )
        self._finalize_water_cut_mm(region.water_cut)
        area = int(region.mask.sum())
        print(
            f"SAM partition on [{self.selected_index}] "
            f"score={region.sam_score:.3f} area={area}px"
        )
        if region.water_cut is not None:
            wc = region.water_cut
            if np.isfinite(wc.water_cut_width_mm) and wc.water_cut_width_mm > 0:
                print(f"  cutWidth={wc.water_cut_width_mm:.1f} mm ({wc.water_cut_width_px:.1f} px)")
            else:
                print(f"  cutWidth={wc.water_cut_width_px:.1f} px (no depth for mm)")
        else:
            print("  cutWidth: centerline analysis failed.")

    def _toggle_sam_prompt_mode(self) -> None:
        if self.sam_refiner is None:
            print("SAM is disabled.")
            return
        if not self.instances:
            print("Run YOLO segmentation first (press S).")
            return
        self.sam_prompt_mode = not self.sam_prompt_mode
        state = "ON (left=fg, right=bg, M=run SAM, X=clear prompts)" if self.sam_prompt_mode else "OFF"
        print(f"SAM prompt mode {state}")
        if self.sam_prompt_mode and self.selected_index is None and self.instances:
            self.selected_index = 0
            print(f"Auto-selected instance [0] {self.instances[0].class_name}")

    def _clear_sam_prompts(self) -> None:
        if self.selected_index is None:
            for instance in self.instances:
                instance.sam_prompt_coords.clear()
                instance.sam_prompt_labels.clear()
                instance.sam_regions.clear()
            print("Cleared SAM prompts and partitions for all instances.")
            return

        instance = self.instances[self.selected_index]
        instance.sam_prompt_coords.clear()
        instance.sam_prompt_labels.clear()
        instance.sam_regions.clear()
        print(f"Cleared SAM prompts and partition for instance [{self.selected_index}].")

    def _add_sam_prompt(self, orig_x: int, orig_y: int, is_foreground: bool) -> None:
        if self.selected_index is None:
            hits = [index for index, inst in enumerate(self.instances) if inst.mask[orig_y, orig_x]]
            if hits:
                self.selected_index = hits[-1]
            else:
                print("Click inside a YOLO instance mask to add SAM prompts.")
                return

        instance = self.instances[self.selected_index]
        if not instance.mask[orig_y, orig_x]:
            print(f"Prompt ({orig_x}, {orig_y}) is outside selected instance mask.")
            return

        label = 1 if is_foreground else 0
        instance.sam_prompt_coords.append((float(orig_x), float(orig_y)))
        instance.sam_prompt_labels.append(label)
        kind = "foreground" if is_foreground else "background"
        print(
            f"Added SAM {kind} point ({orig_x}, {orig_y}) "
            f"on [{self.selected_index}] total={len(instance.sam_prompt_coords)}"
        )

    def _draw_sam_prompts(self, image_bgr: np.ndarray, instance: SegInstance) -> None:
        if instance.sam_regions:
            return
        if not instance.sam_prompt_coords:
            return
        coords = np.array(instance.sam_prompt_coords, dtype=np.float32)
        labels = np.array(instance.sam_prompt_labels, dtype=np.int64)
        SamRefiner.draw_prompts(image_bgr, coords, labels)

    def _set_plane_reference(self) -> None:
        depth_mm = self._depth_mm()
        if depth_mm is None:
            print("No depth frame available for plane reference.")
            return

        z_ref = plane_depth_in_roi(
            depth_mm,
            self.roi.x1,
            self.roi.y1,
            self.roi.width,
            self.roi.height,
        )
        if not np.isfinite(z_ref) or z_ref <= 0:
            print("Failed to capture plane height: no valid depth in ROI.")
            return

        self.z_plane_ref_mm = float(z_ref)
        print(f"Plane reference Z_ref={self.z_plane_ref_mm:.1f} mm (empty ROI, before object)")
        if self.instances:
            self._attach_physical_metrics(self.instances)

    def _clear_plane_reference(self) -> None:
        self.z_plane_ref_mm = None
        print("Plane reference cleared.")
        if self.instances:
            self._attach_physical_metrics(self.instances)

    def _view_to_orig(self, view_x: int, view_y: int) -> tuple[int, int]:
        orig_x = (view_x - self.offset_x) / self.scale
        orig_y = (view_y - self.offset_y) / self.scale
        height, width = self.original.shape[:2]
        return (
            int(np.clip(orig_x, 0, width - 1)),
            int(np.clip(orig_y, 0, height - 1)),
        )

    def _orig_to_view(self, orig_x: int, orig_y: int) -> tuple[int, int]:
        return (
            int(orig_x * self.scale + self.offset_x),
            int(orig_y * self.scale + self.offset_y),
        )

    def _zoom_at(self, view_x: int, view_y: int, factor: float) -> None:
        orig_x, orig_y = self._view_to_orig(view_x, view_y)
        self.scale = float(np.clip(self.scale * factor, self.min_scale, self.max_scale))
        self.offset_x = view_x - orig_x * self.scale
        self.offset_y = view_y - orig_y * self.scale

    def _reset_view(self) -> None:
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def _fit_view(self) -> None:
        height, width = self.original.shape[:2]
        scale_x = self.window_w / width
        scale_y = (self.window_h - 60) / height
        self.scale = min(scale_x, scale_y, 1.0)
        scaled_w = width * self.scale
        scaled_h = height * self.scale
        self.offset_x = (self.window_w - scaled_w) / 2
        self.offset_y = (self.window_h - 60 - scaled_h) / 2

    def _draw_roi(self, image: np.ndarray) -> np.ndarray:
        marked = image.copy()
        cv2.rectangle(marked, (self.roi.x1, self.roi.y1), (self.roi.x2, self.roi.y2), (0, 255, 255), 2)
        label = "YOLO ROI (select mode)" if self.roi_select_mode else "YOLO ROI"
        cv2.putText(
            marked,
            label,
            (self.roi.x1 + 6, max(self.roi.y1 + 22, 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if self.roi_drawing and self.roi_draw_start and self.roi_draw_current:
            x1, y1 = self.roi_draw_start
            x2, y2 = self.roi_draw_current
            left, top = min(x1, x2), min(y1, y2)
            right, bottom = max(x1, x2), max(y1, y2)
            cv2.rectangle(marked, (left, top), (right, bottom), (0, 200, 255), 2)
            cv2.putText(
                marked,
                "Drawing...",
                (left + 6, max(top + 22, 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        return marked

    def _apply_roi_from_points(self, start: tuple[int, int], end: tuple[int, int]) -> None:
        height, width = self.original.shape[:2]
        try:
            self.roi = _clamp_roi(
                min(start[0], end[0]),
                min(start[1], end[1]),
                max(start[0], end[0]),
                max(start[1], end[1]),
                width,
                height,
            )
            self._clear_segmentation()
            print(f"ROI updated: {self.roi}")
        except ValueError as exc:
            print(exc)

    def _save_roi(self) -> None:
        height, width = self.original.shape[:2]
        save_roi_to_file(self.roi, self.roi_file, width, height)

    def _compose_base_image(self) -> np.ndarray:
        image = self._draw_roi(self.original.copy())
        if not self.instances:
            return image

        overlay = image.astype(np.float32)
        for index, instance in enumerate(self.instances):
            color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
            alpha = 0.55 if index == self.selected_index else 0.30
            mask = instance.mask
            overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha

            for region in instance.sam_regions:
                if self.water_cut:
                    if np.any(region.mask):
                        sam_color = np.array(SamRefiner.SAM_COLOR_BGR, dtype=np.float32)
                        sam_alpha = 0.50 if index == self.selected_index else 0.35
                        rmask = region.mask
                        overlay[rmask] = overlay[rmask] * (1.0 - sam_alpha) + sam_color * sam_alpha
                    continue

                if region.rough_cv_mask is not None and np.any(region.rough_cv_mask):
                    cv_color = np.array(SamRefiner.CV_COLOR_BGR, dtype=np.float32)
                    cv_alpha = 0.50 if index == self.selected_index else 0.38
                    cvm = region.rough_cv_mask
                    overlay[cvm] = overlay[cvm] * (1.0 - cv_alpha) + cv_color * cv_alpha

                if np.any(region.mask):
                    sam_color = np.array(SamRefiner.SAM_COLOR_BGR, dtype=np.float32)
                    sam_alpha = 0.50 if index == self.selected_index else 0.35
                    rmask = region.mask
                    overlay[rmask] = overlay[rmask] * (1.0 - sam_alpha) + sam_color * sam_alpha
                    if region.rough_cv_mask is not None and np.any(region.rough_cv_mask):
                        cv_color = np.array(SamRefiner.CV_COLOR_BGR, dtype=np.float32)
                        cv_alpha = 0.22
                        cvm = region.rough_cv_mask & ~rmask
                        overlay[cvm] = overlay[cvm] * (1.0 - cv_alpha) + cv_color * cv_alpha

        overlay = overlay.astype(np.uint8)
        for index, instance in enumerate(self.instances):
            box_pts = instance.box_pts
            if box_pts is None:
                box_pts = oriented_box_from_mask(instance.mask)

            if box_pts is not None:
                box_i32 = np.round(box_pts).astype(np.int32).reshape(-1, 1, 2)
                thickness = 3 if index == self.selected_index else 2
                border_color = (0, 255, 0) if index == self.selected_index else (0, 220, 120)
                cv2.polylines(overlay, [box_i32], True, border_color, thickness, cv2.LINE_AA)
            else:
                contours, _ = cv2.findContours(
                    instance.mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                thickness = 3 if index == self.selected_index else 1
                border_color = (0, 255, 0) if index == self.selected_index else PALETTE[index % len(PALETTE)]
                cv2.drawContours(overlay, contours, -1, border_color, thickness)

            for region in instance.sam_regions:
                if self.water_cut:
                    if region.water_cut is not None:
                        clip_box = instance.box_pts
                        if clip_box is None:
                            clip_box = oriented_box_from_mask(instance.mask)
                        draw_water_cut_overlay(
                            overlay,
                            region.water_cut,
                            draw_pca_axis=True,
                            draw_centerline=False,
                            clip_box=clip_box,
                        )
                    continue

                if (
                    region.prompt_coords is not None
                    and region.prompt_labels is not None
                    and len(region.prompt_coords) > 0
                ):
                    SamRefiner.draw_prompts(overlay, region.prompt_coords, region.prompt_labels)

                if region.water_cut is not None:
                    draw_water_cut_overlay(overlay, region.water_cut)

            if index == self.selected_index:
                self._draw_sam_prompts(overlay, instance)

        return overlay

    def _render(self) -> np.ndarray:
        canvas = np.zeros((self.window_h, self.window_w, 3), dtype=np.uint8)
        canvas[:] = (28, 28, 28)

        base = self._compose_base_image()
        height, width = base.shape[:2]
        scaled_w = max(1, int(round(width * self.scale)))
        scaled_h = max(1, int(round(height * self.scale)))
        interpolation = cv2.INTER_LINEAR if self.scale >= 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(base, (scaled_w, scaled_h), interpolation=interpolation)

        x0 = int(round(self.offset_x))
        y0 = int(round(self.offset_y))
        x1 = x0 + scaled_w
        y1 = y0 + scaled_h

        src_x0 = max(0, -x0)
        src_y0 = max(0, -y0)
        dst_x0 = max(0, x0)
        dst_y0 = max(0, y0)
        dst_x1 = min(self.window_w, x1)
        dst_y1 = min(self.window_h - 60, y1)

        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            src_x1 = src_x0 + (dst_x1 - dst_x0)
            src_y1 = src_y0 + (dst_y1 - dst_y0)
            canvas[dst_y0:dst_y1, dst_x0:dst_x1] = scaled[src_y0:src_y1, src_x0:src_x1]

        if self.water_cut_computing:
            msg = "Calculating water cut width..."
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.9
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(msg, font, scale, thickness)
            if dst_x1 > dst_x0 and dst_y1 > dst_y0:
                cx = (dst_x0 + dst_x1) // 2
                cy = (dst_y0 + dst_y1) // 2
                bx_min_x, bx_min_y = dst_x0, dst_y0
                bx_max_x, bx_max_y = dst_x1, dst_y1
            else:
                cx = self.window_w // 2
                cy = (self.window_h - 60) // 2
                bx_min_x, bx_min_y = 0, 0
                bx_max_x, bx_max_y = self.window_w, self.window_h - 60
            pad_x, pad_y = 24, 16
            bx0 = max(bx_min_x, cx - tw // 2 - pad_x)
            by0 = max(bx_min_y, cy - th // 2 - pad_y)
            bx1 = min(bx_max_x, cx + tw // 2 + pad_x)
            by1 = min(bx_max_y, cy + th // 2 + pad_y + baseline)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (30, 30, 30), -1)
            cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (0, 180, 255), 2)
            cv2.putText(
                canvas,
                msg,
                (bx0 + pad_x, by0 + pad_y + th),
                font,
                scale,
                (240, 240, 240),
                thickness,
                cv2.LINE_AA,
            )

        label_y = 24
        instance_text_color = (255, 0, 0)
        for index, instance in enumerate(self.instances):
            prefix = "> " if index == self.selected_index else "  "
            metric = format_lw_label(
                instance.length_mm,
                instance.width_mm,
                instance.length_px,
                instance.width_px,
            )
            if metric is None:
                metric = format_lwh_mm(instance.length_mm, instance.width_mm, instance.height_mm)
            wc = ""
            for region in instance.sam_regions:
                if region.water_cut is not None:
                    wcut = region.water_cut
                    if np.isfinite(wcut.water_cut_width_mm) and wcut.water_cut_width_mm > 0:
                        wc = f" | cutWidth={wcut.water_cut_width_mm:.0f}mm"
                    else:
                        wc = f" | cutWidth={wcut.water_cut_width_px:.0f}px"
                    break
            text = f"{prefix}{index}: {instance.class_name} {instance.confidence:.2f} | {metric}{wc}"
            cv2.putText(
                canvas,
                text,
                (10, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                instance_text_color,
                1,
                cv2.LINE_AA,
            )
            label_y += 22
            for region_idx, region in enumerate(instance.sam_regions):
                n_fg = n_bg = 0
                if region.prompt_labels is not None:
                    n_fg = int(np.sum(region.prompt_labels == 1))
                    n_bg = int(np.sum(region.prompt_labels == 0))
                cut_line = f"    cut[{region_idx}]: {region.name} fg={n_fg} bg={n_bg}"
                cv2.putText(
                    canvas,
                    cut_line,
                    (10, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    instance_text_color,
                    1,
                    cv2.LINE_AA,
                )
                label_y += 22

        status = "LIVE+YOLO" if self.yolo_live and self.live and not self.frozen else "LIVE" if self.live and not self.frozen else "FROZEN" if self.live else "SNAPSHOT"
        water_status = " | WATER-CUT" if self.water_cut else ""
        mode = " | ROI SELECT" if self.roi_select_mode else ""
        sam_mode = " | SAM PROMPTS" if self.sam_prompt_mode else ""
        if self.z_plane_ref_mm is not None and np.isfinite(self.z_plane_ref_mm):
            plane_status = f"Z_ref={self.z_plane_ref_mm:.0f} mm"
        else:
            plane_status = "Z_ref=unset (empty ROI, press P)"
        help_lines = [
            f"[{status}{water_status}{mode}{sam_mode}] ROI {self.roi} | {plane_status} | Wheel/+/-: zoom | Middle drag: pan",
            "S: YOLO | Space: freeze/resume live | A: SAM prompt mode | L/R click: fg/bg | M: run SAM | X: clear SAM",
            "O: ROI select | G: save ROI | P/U: plane Z_ref | Left click: select | C/V/Q",
        ]
        for idx, line in enumerate(help_lines):
            cv2.putText(
                canvas,
                line,
                (10, self.window_h - 40 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        return canvas

    def _clear_segmentation(self) -> None:
        self.instances.clear()
        self.selected_index = None
        self.sam_prompt_mode = False
        print("Cleared YOLO segmentation results.")

    def _save_segmentation_crops(self) -> list[str]:
        if not self.instances:
            return []

        crop_dir = os.path.join(self.output_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved: list[str] = []

        for index, instance in enumerate(self.instances):
            color_crop = crop_color_by_mask(self.original, instance.mask)
            if color_crop is None:
                continue

            safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in instance.class_name)
            crop_path = os.path.join(
                crop_dir,
                f"seg_crop_{timestamp}_{index:02d}_{safe_name}.png",
            )
            cv2.imwrite(crop_path, color_crop)
            saved.append(crop_path)

            for region_idx, region in enumerate(instance.sam_regions):
                sam_crop = crop_color_by_mask(self.original, region.mask)
                if sam_crop is None:
                    continue
                sam_path = os.path.join(
                    crop_dir,
                    f"seg_sam_{region.name}_{timestamp}_{index:02d}_{region_idx:02d}_{safe_name}.png",
                )
                cv2.imwrite(sam_path, sam_crop)
                saved.append(sam_path)

        if saved:
            print(f"Saved {len(saved)} cropped color image(s) to {crop_dir}:")
            for path in saved:
                print(f"  {path}")
        return saved

    def _run_yolo(self) -> None:
        print(f"Running YOLO segmentation in ROI {self.roi} ...")
        try:
            instances = self.segmenter.segment_all(
                self.original,
                self.roi,
                imgsz=self.yolo_imgsz if self.yolo_live else None,
            )
        except RuntimeError as exc:
            print(exc)
            return
        except ValueError as exc:
            print(exc)
            return

        self._apply_segmentation(instances, verbose=True)

    def _select_at(self, orig_x: int, orig_y: int) -> None:
        if not self.instances:
            print("No segmentation yet. Press S to run YOLO first.")
            return

        if not self.roi.contains(orig_x, orig_y):
            print(f"Click ({orig_x}, {orig_y}) is outside YOLO ROI.")
            return
        hits = [index for index, inst in enumerate(self.instances) if inst.mask[orig_y, orig_x]]
        if not hits:
            print(f"No instance at ({orig_x}, {orig_y}).")
            self.selected_index = None
            return

        self.selected_index = hits[-1]
        selected = self.instances[self.selected_index]
        metric = format_lwh_mm(selected.length_mm, selected.width_mm, selected.height_mm)
        print(
            f"Selected [{self.selected_index}] {selected.class_name} conf={selected.confidence:.2f} "
            f"size={metric}"
        )

    def _save_result(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        color_path = os.path.join(self.output_dir, f"viewer_color_{timestamp}.png")
        cv2.imwrite(color_path, self.original)

        saved = [color_path]
        if self.instances:
            overlay = self._compose_base_image()
            overlay_path = os.path.join(self.output_dir, f"viewer_yolo_{timestamp}.png")
            cv2.imwrite(overlay_path, overlay)
            saved.append(overlay_path)

            measure_path = os.path.join(self.output_dir, f"viewer_measure_{timestamp}.json")
            payload = []
            for index, instance in enumerate(self.instances):
                sam_parts = []
                for region in instance.sam_regions:
                    part = {"name": region.name, "sam_score": region.sam_score}
                    if region.water_cut is not None:
                        wc = region.water_cut
                        part["water_cut_width_px"] = float(wc.water_cut_width_px)
                        part["water_cut_width_mm"] = (
                            None
                            if not np.isfinite(wc.water_cut_width_mm)
                            else float(wc.water_cut_width_mm)
                        )
                        part["water_cut_center"] = [wc.width_center[0], wc.width_center[1]]
                        part["water_cut_end_a"] = [wc.width_end_a[0], wc.width_end_a[1]]
                        part["water_cut_end_b"] = [wc.width_end_b[0], wc.width_end_b[1]]
                        part["centerline_points"] = wc.centerline_path
                    sam_parts.append(part)
                payload.append(
                    {
                        "index": index,
                        "class_name": instance.class_name,
                        "confidence": instance.confidence,
                        "length_mm": None if not np.isfinite(instance.length_mm) else float(instance.length_mm),
                        "width_mm": None if not np.isfinite(instance.width_mm) else float(instance.width_mm),
                        "height_mm": None if not np.isfinite(instance.height_mm) else float(instance.height_mm),
                        "z_object_mm": None
                        if not np.isfinite(instance.z_object_mm)
                        else float(instance.z_object_mm),
                        "z_plane_ref_mm": None
                        if self.z_plane_ref_mm is None or not np.isfinite(self.z_plane_ref_mm)
                        else float(self.z_plane_ref_mm),
                        "angle_deg": None if not np.isfinite(instance.angle_deg) else float(instance.angle_deg),
                        "sam_regions": sam_parts,
                    }
                )
            with open(measure_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            saved.append(measure_path)

            if self.selected_index is not None:
                mask_path = os.path.join(self.output_dir, f"viewer_mask_{timestamp}.png")
                cv2.imwrite(
                    mask_path,
                    self.instances[self.selected_index].mask.astype(np.uint8) * 255,
                )
                saved.append(mask_path)

        print("Saved:", ", ".join(saved))

    def _refresh_frame(self) -> None:
        if self.fetch_frame_fn is None:
            return
        color, depth = self.fetch_frame_fn()
        if color is None:
            print("Failed to fetch a new frame.")
            return
        self.original = color.copy()
        self.depth_raw = None if depth is None else depth.copy()
        self._clear_segmentation()
        print("Color/depth frame updated.")

    def _on_mouse(self, event, x, y, flags, _param) -> None:
        if y >= self.window_h - 60:
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            wheel = (flags >> 16) & 0xFFFF
            if wheel >= 32768:
                wheel -= 65536
            factor = 1.15 if wheel > 0 else 1 / 1.15
            self._zoom_at(x, y, factor)
            return

        if self.roi_select_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_drawing = True
                self.roi_draw_start = self._view_to_orig(x, y)
                self.roi_draw_current = self.roi_draw_start
            elif event == cv2.EVENT_MOUSEMOVE and self.roi_drawing:
                self.roi_draw_current = self._view_to_orig(x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.roi_drawing:
                self.roi_drawing = False
                end = self._view_to_orig(x, y)
                if self.roi_draw_start is not None:
                    self._apply_roi_from_points(self.roi_draw_start, end)
                self.roi_draw_start = None
                self.roi_draw_current = None
            return

        if event == cv2.EVENT_MBUTTONDOWN:
            self.dragging = True
            self.drag_anchor = (x, y)
            self.drag_offset = (self.offset_x, self.offset_y)
        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.drag_anchor[0]
            dy = y - self.drag_anchor[1]
            self.offset_x = self.drag_offset[0] + dx
            self.offset_y = self.drag_offset[1] + dy
        elif event == cv2.EVENT_LBUTTONDOWN:
            orig_x, orig_y = self._view_to_orig(x, y)
            if self.sam_prompt_mode:
                self._add_sam_prompt(orig_x, orig_y, is_foreground=True)
            else:
                self._select_at(orig_x, orig_y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            orig_x, orig_y = self._view_to_orig(x, y)
            if self.sam_prompt_mode:
                self._add_sam_prompt(orig_x, orig_y, is_foreground=False)
            else:
                self._select_at(orig_x, orig_y)

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.window_w, self.window_h)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)
        self._fit_view()

        print("Viewer opened. See on-screen help for controls.")
        print("Press O to enter ROI select mode, then left-drag to draw a new ROI.")
        while True:
            if self.live and not self.frozen and self.fetch_frame_fn is not None:
                color, depth = self.fetch_frame_fn()
                if color is not None:
                    self.original = color.copy()
                    self.depth_raw = None if depth is None else depth.copy()
                    if self.yolo_live:
                        self._update_yolo_live()

            canvas = self._render()
            cv2.imshow(self.WINDOW_NAME, canvas)
            key = cv2.waitKey(30 if self.live and not self.frozen else 20) & 0xFF

            if key in (ord("q"), 27):
                break
            if key in (ord("+"), ord("=")):
                self._zoom_at(self.window_w // 2, (self.window_h - 60) // 2, 1.15)
            elif key in (ord("-"), ord("_")):
                self._zoom_at(self.window_w // 2, (self.window_h - 60) // 2, 1 / 1.15)
            elif key == ord("r"):
                self._reset_view()
            elif key == ord("f"):
                self._fit_view()
            elif key == ord("s"):
                self._run_yolo()
            elif key == ord("a"):
                self._toggle_sam_prompt_mode()
            elif key == ord("m"):
                self._run_sam_manual()
            elif key == ord("x"):
                self._clear_sam_prompts()
            elif key == ord("o"):
                self.roi_select_mode = not self.roi_select_mode
                self.roi_drawing = False
                self.roi_draw_start = None
                self.roi_draw_current = None
                state = "ON (left-drag to draw ROI)" if self.roi_select_mode else "OFF"
                print(f"ROI select mode {state}")
            elif key == ord("p"):
                self._set_plane_reference()
            elif key == ord("u"):
                self._clear_plane_reference()
            elif key == ord("g"):
                self._save_roi()
            elif key == ord("c"):
                self._clear_segmentation()
            elif key == ord("v"):
                self._save_result()
            elif key == ord(" "):
                if self.live:
                    if self.frozen:
                        self.frozen = False
                        print("Live preview resumed.")
                    else:
                        self.frozen = True
                        print("Frame frozen for segmentation.")
                        if self.water_cut and self.yolo_live:
                            self._update_yolo_live()
                else:
                    self._refresh_frame()

        cv2.destroyAllWindows()
