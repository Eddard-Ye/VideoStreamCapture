# -*- coding: utf-8 -*-
"""Refine YOLO object masks with CV center-texture prompts + SAM (ViT-B)."""

from __future__ import annotations

import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from sam_centerline import WaterCutAnalysis

from extract_center_contour import (
    _otsu_threshold,
    _polyline_length,
    _rgb_to_lab_l_uint8,
    mask_to_contours_xy,
    smooth_mask,
)
from object_measure import oriented_box_metrics_from_mask

PROJECT_ROOT = Path(__file__).resolve().parent
SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"


def resolve_inference_device():
    """Pick CUDA when available, else MPS (Apple), else CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def yolo_predict_device(device) -> int | str:
    """Map a torch device to the value Ultralytics ``predict(device=...)`` expects."""
    if device.type == "cuda":
        return device.index if device.index is not None else 0
    if device.type == "mps":
        return "mps"
    return "cpu"


@dataclass
class SamSubRegion:
    name: str
    mask: np.ndarray
    rough_cv_mask: np.ndarray | None = None
    rough_prompt_mask: np.ndarray | None = None
    prompt_coords: np.ndarray | None = None
    prompt_labels: np.ndarray | None = None
    sam_score: float = float("nan")
    water_cut: "WaterCutAnalysis | None" = None


@dataclass
class SamRefineResult:
    regions: list[SamSubRegion] = field(default_factory=list)


def ensure_checkpoint(path: Path, url: str = SAM_VIT_B_URL) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SAM ViT-B checkpoint to {path} ...", flush=True)

    def reporthook(block, bs, total):
        if total > 0 and block % max(1, total // max(bs, 1) // 20 + 1) == 0:
            pct = min(100, int(100 * (block * bs / total)))
            print(f"\r  {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, path, reporthook=reporthook)
        print()
    except OSError as exc:
        raise RuntimeError(
            f"Could not download checkpoint. Save manually from:\n  {url}\n"
            f"to:\n  {path}\nError: {exc}"
        ) from exc


def sam_mask_fill_coverage(mask_uint8: np.ndarray) -> np.ndarray:
    m = (mask_uint8 > 0).astype(bool)
    disk5 = np.zeros((5, 5), dtype=bool)
    yy, xx = np.ogrid[-2:3, -2:3]
    disk5[yy * yy + xx * xx <= 4] = True
    m = ndimage.binary_closing(m, structure=disk5, iterations=1)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_closing(m, structure=disk5, iterations=1)
    return (m.astype(np.uint8)) * 255


def shrink_mask_for_prompts(rough_cv: np.ndarray, iterations: int) -> np.ndarray:
    m = (rough_cv > 0).astype(bool)
    if not m.any():
        return rough_cv
    disk = np.ones((3, 3), dtype=bool)
    it = max(1, int(iterations))
    inner = ndimage.binary_erosion(m, structure=disk, iterations=it)
    if inner.sum() < 48:
        inner = ndimage.binary_erosion(m, structure=disk, iterations=max(1, it // 2))
    if inner.sum() < 24:
        print("Warning: prompt shrink removed almost all pixels; using one-pass erosion.", file=sys.stderr)
        inner = ndimage.binary_erosion(m, structure=disk, iterations=1)
    if inner.sum() < 8:
        print("Warning: prompt shrink failed; using uncropped rough_cv for prompts.", file=sys.stderr)
        return rough_cv.copy()
    return (inner.astype(np.uint8)) * 255


def cap_sam_mask_to_slit_region(
    sam_uint8: np.ndarray,
    rough_cv: np.ndarray,
    dilate_iters: int,
) -> np.ndarray:
    if rough_cv.max() == 0:
        return sam_uint8
    disk = np.ones((3, 3), dtype=bool)
    region = ndimage.binary_dilation((rough_cv > 0), structure=disk, iterations=max(1, dilate_iters))
    capped = (sam_uint8 > 0) & region
    return (capped.astype(np.uint8)) * 255


def sample_polygon_uniform(vertices: np.ndarray, n: int) -> np.ndarray:
    if n < 1:
        return np.zeros((0, 2), dtype=np.float64)
    v = vertices.astype(np.float64)
    if len(v) < 2:
        return np.repeat(v, max(n, 1), axis=0)[:n]
    if len(v) >= 3 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    v_closed = np.vstack([v, v[0]])
    seg = np.linalg.norm(np.diff(v_closed, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-9:
        return np.repeat(v[:1], n, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out: list[np.ndarray] = []
    for i in range(n):
        t = ((i + 0.5) / n) * total
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = int(np.clip(idx, 0, len(seg) - 1))
        sl = seg[idx]
        if sl < 1e-12:
            p = v_closed[idx]
        else:
            local = (t - cum[idx]) / sl
            p = v_closed[idx] + local * (v_closed[idx + 1] - v_closed[idx])
        out.append(p)
    return np.array(out, dtype=np.float64)


def pick_sam_mask_index(
    masks: np.ndarray,
    scores: np.ndarray,
    pt_coords: np.ndarray,
    pt_labels: np.ndarray,
    strategy: str,
) -> tuple[int, list[float]]:
    n = len(scores)
    areas = [float(masks[i].sum()) for i in range(n)]
    if strategy == "score":
        return int(np.argmax(scores)), areas

    fg_ix = np.flatnonzero(pt_labels == 1)
    if fg_ix.size == 0:
        return int(np.argmax(scores)), areas

    candidates: list[tuple[float, int]] = []
    for i in range(n):
        m = masks[i]
        ok = True
        for j in fg_ix:
            x, y = float(pt_coords[j, 0]), float(pt_coords[j, 1])
            xi = int(np.clip(round(x), 0, m.shape[1] - 1))
            yi = int(np.clip(round(y), 0, m.shape[0] - 1))
            if not m[yi, xi]:
                ok = False
                break
        if ok:
            candidates.append((areas[i], i))
    if not candidates:
        print(
            "sam-pick: no mask contains all foreground points; falling back to best score.",
            file=sys.stderr,
        )
        return int(np.argmax(scores)), areas

    if strategy == "smallest":
        return min(candidates)[1], areas
    if strategy == "largest":
        return max(candidates)[1], areas
    return int(np.argmax(scores)), areas


def center_split_mask_in_object(rgb: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """Classical CV center/slit mask inside a YOLO object mask (object-relative band)."""
    obj = (object_mask > 0).astype(np.uint8) * 255
    h, w = rgb.shape[:2]
    ys, xs = np.where(obj > 0)
    if ys.size < 100:
        return np.zeros((h, w), dtype=np.uint8)

    ox0, ox1 = int(xs.min()), int(xs.max())
    oy0, oy1 = int(ys.min()), int(ys.max())
    obj_cx = (ox0 + ox1) / 2.0
    obj_w = ox1 - ox0 + 1

    L = _rgb_to_lab_l_uint8(rgb)
    roi = L[obj > 0]
    thresh = _otsu_threshold(roi)

    body_margin = 8
    top_margin = 5
    strict_body = min(255, thresh + body_margin)
    strict_top = min(255, thresh + top_margin)
    row_thresh = np.full(h, strict_body, dtype=np.float64)
    obj_h = oy1 - oy0 + 1
    y_relax_end = min(h, oy0 + int(0.42 * obj_h) + 1)
    row_thresh[:y_relax_end] = strict_top

    col_half = max(8.0, 0.18 * obj_w)
    column = np.abs(np.arange(w, dtype=np.float32) - obj_cx) < col_half
    column = np.broadcast_to(column, (h, w))
    Lf = L.astype(np.float64)
    light = (Lf >= row_thresh[:, np.newaxis]) & (obj > 0) & column

    vbar = np.zeros((15, 3), dtype=bool)
    vbar[:, 1] = True
    light = ndimage.binary_closing(light, structure=vbar, iterations=3)

    disk5 = np.zeros((5, 5), dtype=bool)
    yy, xx = np.ogrid[-2:3, -2:3]
    disk5[yy * yy + xx * xx <= 4] = True
    disk3 = np.zeros((3, 3), dtype=bool)
    yy3, xx3 = np.ogrid[-1:2, -1:2]
    disk3[yy3 * yy3 + xx3 * xx3 <= 2] = True
    light = ndimage.binary_opening(light, structure=disk3, iterations=1)
    light = ndimage.binary_closing(light, structure=disk5, iterations=1)
    light = light.astype(np.uint8) * 255

    labels, n = ndimage.label(light > 0, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return light

    mid_x0 = int(obj_cx - 0.15 * obj_w)
    mid_x1 = int(obj_cx + 0.15 * obj_w)
    best = 0
    best_area = 0
    for i in range(1, n + 1):
        area = int((labels == i).sum())
        cy_i, cx_i = ndimage.center_of_mass(labels == i)
        cx_i = int(cx_i) if not np.isnan(cx_i) else 0
        if mid_x0 <= cx_i <= mid_x1 and area > best_area:
            best_area = area
            best = i
    if best == 0:
        for i in range(1, n + 1):
            area = int((labels == i).sum())
            if area > best_area:
                best_area = area
                best = i
    if best == 0:
        return light

    out = np.zeros_like(light)
    out[labels == best] = 255
    return out


def build_prompts_from_cv_edges_for_object(
    rough_prompt: np.ndarray,
    object_mask: np.ndarray,
    n_samples: int,
    *,
    rough_for_neg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample SAM foreground points along CV center contour; negatives on object crust."""
    h, w = rough_prompt.shape[:2]
    contours = mask_to_contours_xy(rough_prompt)
    if not contours:
        raise RuntimeError("CV edge prompts: empty center contour inside YOLO mask.")

    verts = max(contours, key=lambda v: _polyline_length(np.vstack((v, v[0]))))
    edge_pts = sample_polygon_uniform(verts, max(4, int(n_samples)))
    pos_arr = edge_pts.astype(np.float32)

    obj = object_mask > 0
    rc_neg = (rough_for_neg if rough_for_neg is not None else rough_prompt) > 0
    ys, xs = np.where(obj)
    if ys.size == 0:
        return pos_arr, np.ones(len(pos_arr), dtype=np.int64)

    cy = float(np.mean(ys))
    mid_y = int(np.clip(cy, 0, h - 1))
    ox0, ox1 = int(xs.min()), int(xs.max())
    span = max(4, ox1 - ox0)
    neg_xs = (
        ox0 + int(0.06 * span),
        ox0 + int(0.12 * span),
        ox1 - int(0.06 * span),
        ox1 - int(0.12 * span),
    )
    neg_list: list[tuple[float, float]] = []
    for x_try in neg_xs:
        x_try = int(np.clip(x_try, 0, w - 1))
        if obj[mid_y, x_try] and not rc_neg[mid_y, x_try]:
            neg_list.append((float(x_try), float(mid_y)))

    if neg_list:
        neg_arr = np.array(neg_list[:4], dtype=np.float32)
        coords = np.vstack([pos_arr, neg_arr])
        labels = np.concatenate(
            [np.ones(len(pos_arr), dtype=np.int64), np.zeros(len(neg_arr), dtype=np.int64)]
        )
        return coords, labels
    return pos_arr, np.ones(len(pos_arr), dtype=np.int64)


def build_prompts_from_oriented_box(
    box: np.ndarray,
    length_px: float,
    width_px: float,
    *,
    fg_samples: int = 16,
    fg_extent_ratio: float = 0.6,
    bg_inward_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build SAM fg/bg prompts from a YOLO mask oriented box.

    Foreground: ``fg_samples`` points on the line through the box center, parallel
    to the long side, spanning ``fg_extent_ratio`` of the long side (symmetric).
    Background: one point per long edge, offset inward from each long-edge midpoint
    by ``bg_inward_ratio`` of the short side.
    """
    if length_px < 1e-3 or width_px < 1e-3 or fg_samples < 1:
        raise RuntimeError("Invalid oriented box dimensions for prompt generation.")

    box = np.asarray(box, dtype=np.float64).reshape(4, 2)
    center = box.mean(axis=0)

    edge_infos: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(4):
        start = box[index]
        end = box[(index + 1) % 4]
        vec = end - start
        edge_len = float(np.linalg.norm(vec))
        if edge_len < 1e-6:
            continue
        edge_infos.append((edge_len, vec / edge_len, start, end))

    if not edge_infos:
        raise RuntimeError("Oriented box has no usable edges.")

    max_edge_len = max(info[0] for info in edge_infos)
    tol = max(1.0, 0.02 * max_edge_len)
    long_edges = [info for info in edge_infos if info[0] >= max_edge_len - tol]
    if len(long_edges) > 2:
        best_pair: tuple[tuple[float, np.ndarray, np.ndarray, np.ndarray], ...] | None = None
        best_dist = -1.0
        for left in range(len(long_edges)):
            for right in range(left + 1, len(long_edges)):
                mid_left = 0.5 * (long_edges[left][2] + long_edges[left][3])
                mid_right = 0.5 * (long_edges[right][2] + long_edges[right][3])
                dist = float(np.linalg.norm(mid_left - mid_right))
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (long_edges[left], long_edges[right])
        long_edges = list(best_pair) if best_pair is not None else long_edges[:2]
    elif len(long_edges) < 2:
        long_edges = sorted(edge_infos, key=lambda item: item[0], reverse=True)[:2]

    long_dir = long_edges[0][1]
    half_extent = 0.5 * fg_extent_ratio * length_px
    line_start = center - long_dir * half_extent
    line_end = center + long_dir * half_extent

    t_values = np.linspace(0.0, 1.0, fg_samples, dtype=np.float64)
    fg_points = line_start[None, :] + t_values[:, None] * (line_end - line_start)[None, :]

    bg_points: list[np.ndarray] = []
    for edge_len, _edge_dir, start, end in long_edges:
        midpoint = 0.5 * (start + end)
        inward = center - midpoint
        inward_len = float(np.linalg.norm(inward))
        if inward_len < 1e-6:
            continue
        inward /= inward_len
        bg_points.append(midpoint + inward * (bg_inward_ratio * width_px))

    if not bg_points:
        raise RuntimeError("Could not place background prompts on oriented box.")

    fg_arr = fg_points.astype(np.float32)
    bg_arr = np.asarray(bg_points, dtype=np.float32)
    coords = np.vstack([fg_arr, bg_arr])
    labels = np.concatenate(
        [
            np.ones(len(fg_arr), dtype=np.int64),
            np.zeros(len(bg_arr), dtype=np.int64),
        ]
    )
    return coords, labels


def resolve_sam_checkpoint(checkpoint: str | Path | None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if path.is_file():
            return path
        candidate = PROJECT_ROOT / checkpoint
        if candidate.is_file():
            return candidate
        return path
    if DEFAULT_CKPT.is_file():
        return DEFAULT_CKPT
    return DEFAULT_CKPT


def prepare_water_cut_box_prompts(
    object_mask: np.ndarray,
    *,
    fg_samples: int = 16,
) -> SamSubRegion | None:
    """Build oriented-box SAM fg/bg prompts for water-cut (no CV edge detection)."""
    object_bool = object_mask.astype(bool)
    if not np.any(object_bool):
        return None

    measured = oriented_box_metrics_from_mask(object_mask)
    if measured is None:
        return None
    box, length_px, width_px, _ = measured

    try:
        pt_coords, pt_labels = build_prompts_from_oriented_box(
            box,
            length_px,
            width_px,
            fg_samples=fg_samples,
        )
    except RuntimeError:
        return None

    height, width = object_mask.shape[:2]
    return SamSubRegion(
        name="box_prompts",
        mask=np.zeros((height, width), dtype=bool),
        prompt_coords=pt_coords,
        prompt_labels=pt_labels,
    )


def run_water_cut_box_sam(
    refiner: SamRefiner,
    image_bgr: np.ndarray,
    object_mask: np.ndarray,
    preview: SamSubRegion,
) -> SamSubRegion | None:
    """Run SAM with oriented-box prompts already stored on ``preview``."""
    object_bool = object_mask.astype(bool)
    if preview.prompt_coords is None or preview.prompt_labels is None:
        return None
    if not np.any(preview.prompt_labels == 1):
        return None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks, scores = refiner._run_predict(rgb, preview.prompt_coords, preview.prompt_labels)
    best, _areas = pick_sam_mask_index(
        masks,
        scores,
        preview.prompt_coords,
        preview.prompt_labels,
        refiner.sam_pick,
    )
    center = (masks[best].astype(np.uint8)) * 255
    center = sam_mask_fill_coverage(center)
    center = smooth_mask(center, sigma=1.2)
    center_bool = (center > 0) & object_bool

    if not np.any(center_bool):
        return None

    preview.mask = center_bool
    preview.sam_score = float(scores[best])
    preview.name = "box_sam"
    return preview


class SamRefiner:
    SAM_COLOR_BGR = (220, 95, 220)
    CV_COLOR_BGR = (190, 175, 0)

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        cv_edge_samples: int = 16,
        prompt_shrink_iters: int = 5,
        sam_pick: str = "smallest",
        sam_cap_dilate: int = 14,
        auto_download: bool = True,
    ):
        self.checkpoint = resolve_sam_checkpoint(checkpoint)
        self.cv_edge_samples = cv_edge_samples
        self.prompt_shrink_iters = prompt_shrink_iters
        self.sam_pick = sam_pick
        self.sam_cap_dilate = sam_cap_dilate
        self.auto_download = auto_download
        self._predictor = None
        self._device = None

    def _ensure_model(self) -> None:
        if self._predictor is not None:
            return

        import torch
        from segment_anything import SamPredictor, sam_model_registry

        if self.auto_download:
            ensure_checkpoint(self.checkpoint)

        if not self.checkpoint.is_file():
            raise RuntimeError(
                f"SAM checkpoint not found: {self.checkpoint}. "
                "Place sam_vit_b_01ec64.pth under checkpoints/ or pass --sam-checkpoint."
            )

        device = resolve_inference_device()

        print(f"Loading SAM ViT-B from {self.checkpoint} on {device} ...")
        sam = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint))
        sam.to(device=device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        self._device = device
        print("SAM model ready.")

    def _run_predict(
        self,
        rgb: np.ndarray,
        pt_coords: np.ndarray,
        pt_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_model()
        self._predictor.set_image(rgb)

        import torch

        def _predict():
            with torch.inference_mode():
                return self._predictor.predict(
                    point_coords=pt_coords,
                    point_labels=pt_labels,
                    multimask_output=True,
                )

        try:
            masks, scores, _ = _predict()
        except RuntimeError as exc:
            if self._device is not None and self._device.type == "mps":
                print(f"  SAM MPS failed ({exc}); retrying on CPU.")
                self._predictor.model.to(torch.device("cpu"))
                self._device = torch.device("cpu")
                masks, scores, _ = _predict()
            else:
                raise
        return masks, scores

    def predict_with_points(
        self,
        image_bgr: np.ndarray,
        object_mask: np.ndarray,
        pt_coords: np.ndarray,
        pt_labels: np.ndarray,
        region_name: str = "manual",
    ) -> SamRefineResult | None:
        object_bool = object_mask.astype(bool)
        if not np.any(object_bool):
            return None
        if pt_coords is None or len(pt_coords) == 0:
            raise RuntimeError("SAM requires at least one prompt point.")
        if not np.any(pt_labels == 1):
            raise RuntimeError("SAM requires at least one foreground point (left click).")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks, scores = self._run_predict(rgb, pt_coords, pt_labels)
        best, _areas = pick_sam_mask_index(masks, scores, pt_coords, pt_labels, self.sam_pick)
        center = (masks[best].astype(np.uint8)) * 255
        center = sam_mask_fill_coverage(center)
        center = smooth_mask(center, sigma=1.2)
        center_bool = (center > 0) & object_bool

        if not np.any(center_bool):
            print("  SAM produced empty mask after clipping to YOLO object.")
            return None

        return SamRefineResult(
            regions=[
                SamSubRegion(
                    name=region_name,
                    mask=center_bool,
                    rough_cv_mask=None,
                    prompt_coords=pt_coords.copy(),
                    prompt_labels=pt_labels.copy(),
                    sam_score=float(scores[best]),
                )
            ]
        )

    def refine_object(self, image_bgr: np.ndarray, object_mask: np.ndarray) -> SamRefineResult | None:
        object_bool = object_mask.astype(bool)
        if not np.any(object_bool):
            return None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        object_u8 = object_bool.astype(np.uint8) * 255

        rough_cv = center_split_mask_in_object(rgb, object_u8)
        rough_cv = smooth_mask(rough_cv, sigma=1.2)
        if rough_cv.max() == 0:
            print("  CV center texture mask empty; skip SAM refine for this object.")
            return None

        rough_prompt = shrink_mask_for_prompts(rough_cv, self.prompt_shrink_iters)
        try:
            pt_coords, pt_labels = build_prompts_from_cv_edges_for_object(
                rough_prompt,
                object_bool,
                self.cv_edge_samples,
                rough_for_neg=rough_cv,
            )
        except RuntimeError as exc:
            print(f"  SAM prompt build failed: {exc}")
            return None

        masks, scores = self._run_predict(rgb, pt_coords, pt_labels)
        best, _areas = pick_sam_mask_index(masks, scores, pt_coords, pt_labels, self.sam_pick)
        center = (masks[best].astype(np.uint8)) * 255
        center = sam_mask_fill_coverage(center)
        center = cap_sam_mask_to_slit_region(center, rough_cv, self.sam_cap_dilate)
        center = smooth_mask(center, sigma=1.2)
        center_bool = (center > 0) & object_bool

        if not np.any(center_bool):
            print("  SAM refine produced empty mask after clipping.")
            return None

        return SamRefineResult(
            regions=[
                SamSubRegion(
                    name="center",
                    mask=center_bool,
                    rough_cv_mask=(rough_cv > 0),
                    prompt_coords=pt_coords,
                    prompt_labels=pt_labels,
                    sam_score=float(scores[best]),
                )
            ]
        )

    @staticmethod
    def draw_prompts(image_bgr: np.ndarray, coords: np.ndarray, labels: np.ndarray) -> None:
        for i in range(len(coords)):
            x, y = int(round(float(coords[i, 0]))), int(round(float(coords[i, 1])))
            is_fg = int(labels[i]) == 1
            color = (90, 220, 30) if is_fg else (70, 70, 255)
            cv2.circle(image_bgr, (x, y), 6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(image_bgr, (x, y), 4, color, -1, cv2.LINE_AA)
