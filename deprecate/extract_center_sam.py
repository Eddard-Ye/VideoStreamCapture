#!/usr/bin/env python3
"""
Bread center (split / crumb) segmentation using Meta Segment Anything (SAM) ViT-B.

The classical `extract_center_contour` mask is used only to place automatic prompts
(positive points inside the rough center, optional negatives on crust), then SAM refines.

Use `--point x,y` (repeatable) to skip classical preprocessing and prompt SAM with
foreground points only (image pixel coordinates, origin top-left).

Use `--cv-edge-prompts` to sample foreground points along the classical CV contour of the
rough center mask, add crust negatives, then SAM + `--sam-pick smallest` (default).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

_cfg = Path(__file__).resolve().parent / ".mplconfig"
_cfg.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cfg))

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from segment_anything import SamPredictor, sam_model_registry

from extract_center_contour import (
    _polyline_length,
    bread_mask,
    center_split_mask,
    draw_overlay_rgb,
    mask_to_contours_xy,
    simplify_xy,
    smooth_mask,
)

SAM_VIT_B_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
DEFAULT_CKPT = _SCRIPT_DIR / "checkpoints" / "sam_vit_b_01ec64.pth"

# Prompt markers: foreground = green, background = red (classical mode with negatives)
_FG_MARKER = (30, 220, 90)
_BG_MARKER = (255, 70, 70)
_MARKER_OUTLINE = (255, 255, 255)
_MARKER_RADIUS = 7

# Visualization: traditional CV vs SAM masks (distinct hues)
_COLOR_MASK_TRADITIONAL = (0, 175, 190)  # 青绿 — 传统 CV
_COLOR_MASK_SAM = (220, 95, 220)  # 洋红 — SAM


def sam_mask_fill_coverage(mask_uint8: np.ndarray) -> np.ndarray:
    """Close small gaps and fill holes; keep closing mild to avoid growing onto crust."""
    m = (mask_uint8 > 0).astype(bool)
    disk5 = np.zeros((5, 5), dtype=bool)
    yy, xx = np.ogrid[-2:3, -2:3]
    disk5[yy * yy + xx * xx <= 4] = True
    m = ndimage.binary_closing(m, structure=disk5, iterations=1)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_closing(m, structure=disk5, iterations=1)
    return (m.astype(np.uint8)) * 255


def shrink_mask_for_prompts(rough_cv: np.ndarray, iterations: int) -> np.ndarray:
    """
    Erode classical slit mask so prompt points (edge samples / centroids) sit deeper inside
    the crack, reducing SAM leaking onto crust.
    """
    m = (rough_cv > 0).astype(bool)
    if not m.any():
        return rough_cv
    disk = np.ones((3, 3), dtype=bool)
    it = max(1, int(iterations))
    inner = ndimage.binary_erosion(m, structure=disk, iterations=it)
    if inner.sum() < 48:
        inner = ndimage.binary_erosion(m, structure=disk, iterations=max(1, it // 2))
    if inner.sum() < 24:
        print(
            "Warning: prompt shrink removed almost all pixels; using one-pass erosion only.",
            file=sys.stderr,
        )
        inner = ndimage.binary_erosion(m, structure=disk, iterations=1)
    if inner.sum() < 8:
        print(
            "Warning: prompt shrink failed; using uncropped rough_cv for prompts.",
            file=sys.stderr,
        )
        return rough_cv.copy()
    return (inner.astype(np.uint8)) * 255


def cap_sam_mask_to_slit_region(
    sam_uint8: np.ndarray,
    rough_cv: np.ndarray,
    dilate_iters: int,
) -> np.ndarray:
    """Clip SAM mask to slightly dilated classical slit (prevents magenta spill on crust)."""
    if rough_cv.max() == 0:
        return sam_uint8
    disk = np.ones((3, 3), dtype=bool)
    region = ndimage.binary_dilation((rough_cv > 0), structure=disk, iterations=max(1, dilate_iters))
    capped = (sam_uint8 > 0) & region
    return (capped.astype(np.uint8)) * 255


def rgba_mask_overlay(
    rgb: np.ndarray,
    mask_uint8: np.ndarray,
    rgb_color: tuple[int, int, int],
    alpha: float = 0.42,
) -> Image.Image:
    """Semi-transparent color fill where mask > 0; returns RGB PIL image."""
    h, w = rgb.shape[:2]
    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    m = (mask_uint8 > 0).astype(np.float32)
    r, g, b = rgb_color
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[:, :, 0] = (m * r).astype(np.uint8)
    layer[:, :, 1] = (m * g).astype(np.uint8)
    layer[:, :, 2] = (m * b).astype(np.uint8)
    layer[:, :, 3] = (m * (255.0 * alpha)).astype(np.uint8)
    top = Image.fromarray(layer, mode="RGBA")
    return Image.alpha_composite(base, top).convert("RGB")


def save_compare_panels_with_legend(
    rgb: np.ndarray,
    rough_cv: np.ndarray,
    sam_mask: np.ndarray,
    path: Path,
) -> None:
    """Side-by-side: traditional overlay | SAM overlay, plus bottom legend strip."""
    left = rgba_mask_overlay(rgb, rough_cv, _COLOR_MASK_TRADITIONAL, 0.44)
    right = rgba_mask_overlay(rgb, sam_mask, _COLOR_MASK_SAM, 0.44)
    w0, h0 = rgb.shape[1], rgb.shape[0]
    gap = 10
    leg_h = 44
    total_w = w0 * 2 + gap
    total_h = h0 + leg_h
    canvas = Image.new("RGB", (total_w, total_h), (22, 28, 36))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w0 + gap, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, h0, total_w, total_h], fill=(16, 22, 28))
    sw, pad = 16, 12
    yb = h0 + (leg_h - sw) // 2
    draw.rectangle([pad, yb, pad + sw, yb + sw], fill=_COLOR_MASK_TRADITIONAL, outline=(240, 240, 240), width=1)
    draw.text((pad + sw + 8, yb - 1), "传统 CV 掩码（青绿）", fill=(230, 235, 240))
    x2 = total_w // 2 + pad
    draw.rectangle([x2, yb, x2 + sw, yb + sw], fill=_COLOR_MASK_SAM, outline=(240, 240, 240), width=1)
    draw.text((x2 + sw + 8, yb - 1), "SAM 掩码（洋红）", fill=(230, 235, 240))
    canvas.save(path)


def traditional_rough_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bread foreground mask + smoothed classical center split mask (uint8 0/255)."""
    bread = bread_mask(rgb)
    rough = center_split_mask(rgb, bread)
    rough = smooth_mask(rough, sigma=1.2)
    return bread, rough


def _draw_prompt_markers(im: Image.Image, pt_coords: np.ndarray, pt_labels: np.ndarray) -> None:
    draw = ImageDraw.Draw(im)
    for i in range(len(pt_coords)):
        x, y = float(pt_coords[i, 0]), float(pt_coords[i, 1])
        is_fg = int(pt_labels[i]) == 1
        fill = _FG_MARKER if is_fg else _BG_MARKER
        r = _MARKER_RADIUS
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=fill,
            outline=_MARKER_OUTLINE,
            width=2,
        )


def overlay_contour_and_prompts(
    rgb: np.ndarray,
    verts_xy: np.ndarray,
    line_color: tuple[int, int, int],
    pt_coords: np.ndarray,
    pt_labels: np.ndarray,
) -> Image.Image:
    """Contour overlay plus SAM prompt points (green=foreground, red=background)."""
    base = draw_overlay_rgb(rgb.copy(), verts_xy, line_color)
    _draw_prompt_markers(base, pt_coords, pt_labels)
    return base


def save_prompts_only_image(
    rgb: np.ndarray,
    pt_coords: np.ndarray,
    pt_labels: np.ndarray,
    path: Path,
) -> None:
    """Original image with prompt markers only (no contour)."""
    im = Image.fromarray(rgb, mode="RGB")
    _draw_prompt_markers(im, pt_coords, pt_labels)
    im.save(path)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    except OSError as e:
        raise RuntimeError(
            f"Could not download checkpoint. Save manually from:\n  {url}\n"
            f"to:\n  {path}\nError: {e}"
        ) from e


def build_prompts_from_rough_mask(
    rough_center: np.ndarray,
    bread: np.ndarray,
    h: int,
    w: int,
    *,
    rough_for_neg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Positive clicks inside rough_center (often eroded slit core).
    Negatives use rough_for_neg when given (full conservative slit) so crust stays label 0.
    """
    rc_pos = rough_center > 0
    rc_neg = (rough_for_neg if rough_for_neg is not None else rough_center) > 0
    br = bread > 0
    ys, xs = np.where(rc_pos)
    if len(xs) == 0:
        cx, cy = w / 2.0, h / 2.0
    else:
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))

    pos: list[tuple[float, float]] = [(cx, cy)]
    if len(ys) > 0:
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        for frac in (0.25, 0.75):
            yy = int(y_min + frac * (y_max - y_min))
            row = rc_pos[yy]
            if row.any():
                xx = np.where(row)[0]
                pos.append((float(np.mean(xx)), float(yy)))

    pos_arr = np.array(pos, dtype=np.float32)

    neg: list[tuple[float, float]] = []
    mid_y = int(np.clip(cy, 0, h - 1))
    for x_try in (
        int(w * 0.22),
        int(w * 0.78),
        int(w * 0.15),
        int(w * 0.85),
    ):
        x_try = int(np.clip(x_try, 0, w - 1))
        if br[mid_y, x_try] and not rc_neg[mid_y, x_try]:
            neg.append((float(x_try), float(mid_y)))

    if neg:
        neg_arr = np.array(neg[:4], dtype=np.float32)
        coords = np.vstack([pos_arr, neg_arr])
        labels = np.array([1] * len(pos_arr) + [0] * len(neg_arr), dtype=np.int64)
    else:
        coords = pos_arr
        labels = np.ones(len(pos_arr), dtype=np.int64)

    return coords, labels


def sample_polygon_uniform(vertices: np.ndarray, n: int) -> np.ndarray:
    """
    Sample n points uniformly by arc length along a closed polygon.
    `vertices` is an open ring (first point may or may not repeat the last).
    """
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


def build_prompts_from_cv_edges(
    rough_prompt: np.ndarray,
    bread: np.ndarray,
    h: int,
    w: int,
    n_samples: int,
    *,
    rough_for_neg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inner slit mask → contour edge samples as SAM foreground (points deep in crack).
    Negatives reference rough_for_neg (full conservative slit) so crust pixels qualify.
    """
    rough = rough_prompt
    contours = mask_to_contours_xy(rough)
    if not contours:
        raise RuntimeError("CV edge prompts: empty contour (try another image or thresholds).")
    verts = max(contours, key=lambda v: _polyline_length(np.vstack((v, v[0]))))
    edge_pts = sample_polygon_uniform(verts, max(4, int(n_samples)))
    pos_arr = edge_pts.astype(np.float32)
    neg_list: list[tuple[float, float]] = []
    rc_neg = (rough_for_neg if rough_for_neg is not None else rough_prompt) > 0
    br = bread > 0
    ys, xs = np.where(rc_neg)
    cy = float(np.mean(ys)) if len(xs) else h / 2.0
    mid_y = int(np.clip(cy, 0, h - 1))
    for x_try in (
        int(w * 0.22),
        int(w * 0.78),
        int(w * 0.15),
        int(w * 0.85),
    ):
        x_try = int(np.clip(x_try, 0, w - 1))
        if br[mid_y, x_try] and not rc_neg[mid_y, x_try]:
            neg_list.append((float(x_try), float(mid_y)))

    if neg_list:
        neg_arr = np.array(neg_list[:4], dtype=np.float32)
        coords = np.vstack([pos_arr, neg_arr])
        labels = np.concatenate(
            [np.ones(len(pos_arr), dtype=np.int64), np.zeros(len(neg_arr), dtype=np.int64)]
        )
        return coords, labels
    return pos_arr, np.ones(len(pos_arr), dtype=np.int64)


def _parse_xy_pair(s: str) -> tuple[float, float]:
    s = s.strip().replace(" ", "")
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected X,Y (comma-separated)")
    return float(parts[0]), float(parts[1])


def build_prompts_foreground_only(
    points: list[tuple[float, float]],
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """SAM prompts: only positive (foreground) clicks; coords clipped to image bounds."""
    coords = []
    for x, y in points:
        xc = float(np.clip(x, 0, w - 1))
        yc = float(np.clip(y, 0, h - 1))
        if (xc, yc) != (x, y):
            print(f"Warning: clamped point ({x:.1f},{y:.1f}) -> ({xc:.1f},{yc:.1f})", file=sys.stderr)
        coords.append((xc, yc))
    arr = np.array(coords, dtype=np.float32)
    labels = np.ones(len(coords), dtype=np.int64)
    return arr, labels


def append_negative_points(
    pt_coords: np.ndarray,
    pt_labels: np.ndarray,
    neg_points: list[tuple[float, float]],
    w: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Append extra background clicks (label=0)."""
    extra_c: list[tuple[float, float]] = []
    for x, y in neg_points:
        xc = float(np.clip(x, 0, w - 1))
        yc = float(np.clip(y, 0, h - 1))
        if (xc, yc) != (x, y):
            print(
                f"Warning: clamped neg point ({x:.1f},{y:.1f}) -> ({xc:.1f},{yc:.1f})",
                file=sys.stderr,
            )
        extra_c.append((xc, yc))
    if not extra_c:
        return pt_coords, pt_labels
    neg_arr = np.array(extra_c, dtype=np.float32)
    neg_lab = np.zeros(len(extra_c), dtype=np.int64)
    return (
        np.vstack([pt_coords, neg_arr]),
        np.concatenate([pt_labels, neg_lab]),
    )


def pick_sam_mask_index(
    masks: np.ndarray,
    scores: np.ndarray,
    pt_coords: np.ndarray,
    pt_labels: np.ndarray,
    strategy: str,
) -> tuple[int, list[float]]:
    """
    multimask_output gives several candidates; 'score' uses SAM's IoU score only.
    'smallest' / 'largest': among masks that contain every foreground prompt, pick min or max area.
    """
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


def main() -> int:
    p = argparse.ArgumentParser(description="Bread center contour via SAM (ViT-B).")
    p.add_argument("image", type=Path, help="Input image path")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("out_bread_contour_sam"),
        help="Output directory",
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="Path to sam_vit_b .pth")
    p.add_argument("--no-download", action="store_true", help="Do not auto-download checkpoint")
    p.add_argument("--json", action="store_true", help="Write simplified contour JSON")
    p.add_argument(
        "--suffix",
        default="_sam",
        help="Suffix before extension in output filenames (default _sam)",
    )
    p.add_argument(
        "--point",
        action="append",
        dest="fg_points",
        metavar="X,Y",
        type=_parse_xy_pair,
        help=(
            "Foreground point in pixel coords (x horizontal, y vertical from top). "
            "Repeatable. If set, skips classical mask and uses only these positive prompts."
        ),
    )
    p.add_argument(
        "--neg-point",
        action="append",
        dest="neg_points",
        metavar="X,Y",
        type=_parse_xy_pair,
        help=(
            "Extra background point (e.g. on crust). Repeatable. "
            "Use with --point to discourage SAM from selecting the whole loaf."
        ),
    )
    p.add_argument(
        "--sam-pick",
        choices=("score", "smallest", "largest"),
        default="smallest",
        help=(
            "Among SAM's 3 masks: 'score' = highest IoU; "
            "'smallest' / 'largest' = min/max area among masks that contain every foreground point. "
            "Use 'largest' for fuller slit coverage; pair with --cv-edge-prompts + post fill (default)."
        ),
    )
    p.add_argument(
        "--no-sam-fill",
        action="store_true",
        help="Disable gap-closing / hole-fill on the SAM mask (see sam_mask_fill_coverage).",
    )
    p.add_argument(
        "--cv-edge-prompts",
        action="store_true",
        help=(
            "Use classical CV rough mask, sample foreground points along its contour edges, "
            "plus crust negatives. Ignores --point. Combines well with --sam-pick smallest (default)."
        ),
    )
    p.add_argument(
        "--cv-edge-samples",
        type=int,
        default=16,
        metavar="N",
        help="Number of edge samples on the CV contour when using --cv-edge-prompts (default: 16).",
    )
    p.add_argument(
        "--prompt-shrink-iters",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Erode classical slit mask by N iterations (3x3) before placing SAM prompts "
            "(edge samples / interior positives). Larger = prompts deeper inside crack (default: 5)."
        ),
    )
    p.add_argument(
        "--no-sam-slit-cap",
        action="store_true",
        help="Do not clip SAM mask to dilated classical slit (may reduce crust bleed).",
    )
    p.add_argument(
        "--sam-cap-dilate",
        type=int,
        default=14,
        metavar="PX",
        help="Pixels (3x3 dilations) to expand classical slit region when capping SAM (default: 14).",
    )
    args = p.parse_args()

    if not args.no_download:
        ensure_checkpoint(args.checkpoint)

    try:
        pil = Image.open(args.image).convert("RGB")
    except OSError as e:
        print(f"Failed to read: {args.image} ({e})", file=sys.stderr)
        return 1

    rgb = np.asarray(pil)
    h, w = rgb.shape[:2]

    try:
        bread, rough_cv = traditional_rough_mask(rgb)
    except RuntimeError as e:
        print(f"Warning: classical rough mask unavailable ({e}); comparison overlays empty.", file=sys.stderr)
        bread = bread_mask(rgb)
        rough_cv = np.zeros((h, w), dtype=np.uint8)

    rough_prompt = shrink_mask_for_prompts(rough_cv, args.prompt_shrink_iters)

    if args.cv_edge_prompts:
        if args.fg_points:
            print(
                "Warning: --cv-edge-prompts ignores manual --point.",
                file=sys.stderr,
            )
        try:
            pt_coords, pt_labels = build_prompts_from_cv_edges(
                rough_prompt,
                bread,
                h,
                w,
                args.cv_edge_samples,
                rough_for_neg=rough_cv,
            )
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        prompt_mode = "cv_edge_prompts"
    elif args.fg_points:
        pt_coords, pt_labels = build_prompts_foreground_only(list(args.fg_points), w, h)
        prompt_mode = "foreground_only"
    else:
        pt_coords, pt_labels = build_prompts_from_rough_mask(
            rough_prompt, bread, h, w, rough_for_neg=rough_cv
        )
        prompt_mode = "classical_auto"

    if args.neg_points:
        pt_coords, pt_labels = append_negative_points(
            pt_coords, pt_labels, list(args.neg_points), w, h
        )

    device = _pick_device()
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)

    def _predict() -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            return predictor.predict(
                point_coords=pt_coords,
                point_labels=pt_labels,
                multimask_output=True,
            )

    try:
        masks, scores, _logits = _predict()
    except RuntimeError as e:
        if device.type == "mps":
            print(f"MPS inference failed ({e}); retrying on CPU.", file=sys.stderr)
            device = torch.device("cpu")
            sam.to(device)
            predictor = SamPredictor(sam)
            predictor.set_image(rgb)
            masks, scores, _logits = _predict()
        else:
            raise

    best, mask_areas = pick_sam_mask_index(
        masks, scores, pt_coords, pt_labels, args.sam_pick
    )
    mask_bool = masks[best]
    center = (mask_bool.astype(np.uint8)) * 255
    if not args.no_sam_fill:
        center = sam_mask_fill_coverage(center)
    if rough_cv.max() > 0 and not args.no_sam_slit_cap:
        center = cap_sam_mask_to_slit_region(center, rough_cv, args.sam_cap_dilate)
    center = smooth_mask(center, sigma=1.2)

    contours = mask_to_contours_xy(center)
    if not contours:
        print("No contour found after SAM.", file=sys.stderr)
        return 1
    verts = max(contours, key=lambda v: _polyline_length(np.vstack((v, v[0]))))
    simp = simplify_xy(verts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.image.stem}{args.suffix}"

    overlay_contour_and_prompts(rgb, verts, (0, 180, 0), pt_coords, pt_labels).save(
        args.out_dir / f"{stem}_contour_full.png"
    )
    overlay_contour_and_prompts(rgb, simp, (200, 60, 200), pt_coords, pt_labels).save(
        args.out_dir / f"{stem}_contour_simplified.png"
    )
    save_prompts_only_image(rgb, pt_coords, pt_labels, args.out_dir / f"{stem}_prompts.png")
    Image.fromarray(center, mode="L").save(args.out_dir / f"{stem}_mask_center.png")
    Image.fromarray(rough_cv, mode="L").save(args.out_dir / f"{stem}_mask_traditional.png")

    rgba_mask_overlay(rgb, rough_cv, _COLOR_MASK_TRADITIONAL, 0.44).save(
        args.out_dir / f"{stem}_overlay_traditional.png"
    )
    rgba_mask_overlay(rgb, center, _COLOR_MASK_SAM, 0.44).save(
        args.out_dir / f"{stem}_overlay_sam.png"
    )
    save_compare_panels_with_legend(
        rgb,
        rough_cv,
        center,
        args.out_dir / f"{stem}_compare_panels.png",
    )

    if args.json:
        pts_full = np.round(verts).astype(int).tolist()
        pts_simplified = np.round(simp).astype(int).tolist()
        meta = {
            "method": "sam_vit_b",
            "prompt_mode": prompt_mode,
            "cv_edge_samples": args.cv_edge_samples if args.cv_edge_prompts else None,
            "sam_pick": args.sam_pick,
            "chosen_mask_index": best,
            "mask_areas_px": mask_areas,
            "checkpoint": str(args.checkpoint.resolve()),
            "device": str(device),
            "sam_scores": [float(s) for s in scores],
            "prompt_points": np.round(pt_coords, 2).tolist(),
            "prompt_labels": pt_labels.tolist(),
            "points": pts_full,
            "points_simplified": pts_simplified,
            "closed": True,
            "legend": {
                "traditional_cv_rgb": list(_COLOR_MASK_TRADITIONAL),
                "sam_rgb": list(_COLOR_MASK_SAM),
            },
            "outputs": {
                "mask_traditional": f"{stem}_mask_traditional.png",
                "mask_sam": f"{stem}_mask_center.png",
                "overlay_traditional": f"{stem}_overlay_traditional.png",
                "overlay_sam": f"{stem}_overlay_sam.png",
                "compare_panels": f"{stem}_compare_panels.png",
            },
        }
        with open(args.out_dir / f"{stem}_contour.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print(f"Prompt mode: {prompt_mode}, sam-pick: {args.sam_pick}, mask index: {best}")
    print(f"Mask areas (px): {[int(a) for a in mask_areas]}, scores: {[float(s) for s in scores]}")
    print(f"Device: {device}, chosen mask score: {float(scores[best]):.4f}")
    print(f"Wrote outputs to {args.out_dir.resolve()}")
    print(
        "Compare: cyan/teal = traditional CV mask; magenta = SAM mask "
        f"({stem}_overlay_*.png, {stem}_compare_panels.png)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
