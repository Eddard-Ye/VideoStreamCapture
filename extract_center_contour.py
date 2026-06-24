#!/usr/bin/env python3
"""
Extract the contour of the light-colored center (split / crumb) of a loaf
on a white background: luminance (approx. CIE L*) inside a non-background mask,
Otsu split, morphology, then largest mid-centered component; contour via OpenCV
findContours (ordered boundary chains; avoids matplotlib contour vertex artifacts).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


def _rgb_to_lab_l_uint8(rgb: np.ndarray) -> np.ndarray:
    """OpenCV-style L channel in 0..255 from sRGB 0..255 (approximate)."""
    r = rgb[..., 0].astype(np.float64) / 255.0
    g = rgb[..., 1].astype(np.float64) / 255.0
    b = rgb[..., 2].astype(np.float64) / 255.0

    def inv_srgb(u: np.ndarray) -> np.ndarray:
        return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)

    r, g, b = inv_srgb(r), inv_srgb(g), inv_srgb(b)
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    y = np.clip(y, 0.0, 1.0)
    eps = (6.0 / 29.0) ** 3
    kappa = 24389.0 / 27.0
    fy = np.where(y > eps, np.cbrt(y), (kappa * y + 16.0) / 116.0)
    l_star = 116.0 * fy - 16.0
    return np.clip(l_star * 255.0 / 100.0, 0, 255).astype(np.uint8)


def bread_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels that are not near-white background."""
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(
        np.float32
    )
    blur = ndimage.gaussian_filter(gray, sigma=1.2)
    bg = (gray > 248) & (blur > 246)
    return (~bg).astype(np.uint8) * 255


def _otsu_threshold(values: np.ndarray) -> int:
    hist = np.bincount(values.flatten(), minlength=256).astype(np.float64)
    hist[0] = 0
    total = hist.sum()
    if total <= 0:
        return 128
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b, w_b, max_var, thresh = 0.0, 0.0, -1.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            thresh = t
    return int(thresh)


def center_split_mask(rgb: np.ndarray, bread: np.ndarray) -> np.ndarray:
    L = _rgb_to_lab_l_uint8(rgb)
    h, w = L.shape
    roi = L[bread > 0]
    if roi.size < 100:
        raise RuntimeError("Bread region too small; check image or thresholds.")
    thresh = _otsu_threshold(roi)
    # Brighter-than-Otsu crumb inside a narrow vertical band. Top of the slit is often
    # darker (shadow, lip, angle) — use a row-wise looser threshold on the upper part of
    # the *bread* bbox so the rough mask still reaches the slit tip for SAM prompts.
    body_margin = 8
    top_margin = 5
    strict_body = min(255, thresh + body_margin)
    strict_top = min(255, thresh + top_margin)
    row_thresh = np.full(h, strict_body, dtype=np.float64)
    bread_rows = np.where(np.any(bread > 0, axis=1))[0]
    if bread_rows.size > 0:
        y0, y1 = int(bread_rows[0]), int(bread_rows[-1])
        bread_h = y1 - y0 + 1
        # Relax threshold for upper ~42% of bread extent (where the score often thins out).
        y_relax_end = min(h, y0 + int(0.42 * bread_h) + 1)
    else:
        y_relax_end = min(h, int(0.38 * h))
    row_thresh[:y_relax_end] = strict_top

    # Narrow band (~36% of image width): central slit only.
    xs = np.abs(np.arange(w, dtype=np.float32) - (w * 0.5)) < (0.18 * w)
    column = np.broadcast_to(xs, (h, w))
    Lf = L.astype(np.float64)
    light = ((Lf >= row_thresh[:, np.newaxis]) & (bread > 0) & column)

    # Vertical closing reconnects the slit if opening/threshold split it into fragments
    # (common at the narrow tip); do this before aggressive opening.
    vbar = np.zeros((15, 3), dtype=bool)
    vbar[:, 1] = True
    light = ndimage.binary_closing(light, structure=vbar, iterations=3)

    disk5 = np.zeros((5, 5), dtype=bool)
    yy, xx = np.ogrid[-2:3, -2:3]
    disk5[yy * yy + xx * xx <= 4] = True
    # Gentler opening preserves thin tips at the top of the slit vs 5×5 disk.
    disk3 = np.zeros((3, 3), dtype=bool)
    yy3, xx3 = np.ogrid[-1:2, -1:2]
    disk3[yy3 * yy3 + xx3 * xx3 <= 2] = True
    light = ndimage.binary_opening(light, structure=disk3, iterations=1)
    light = ndimage.binary_closing(light, structure=disk5, iterations=1)
    light = (light.astype(np.uint8)) * 255

    labels, n = ndimage.label(light > 0, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return light
    h, w = light.shape
    mid_x0, mid_x1 = int(w * 0.35), int(w * 0.65)
    best = 0
    best_area = 0
    for i in range(1, n + 1):
        area = int((labels == i).sum())
        cy, cx = ndimage.center_of_mass(labels == i)
        cx_i = int(cx) if not np.isnan(cx) else 0
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
    # Optional shrink removed: 1× erosion often deleted the narrow upper tip entirely,
    # starving SAM prompts there. Prompt placement uses shrink_mask_for_prompts in SAM.
    return out


def smooth_mask(mask: np.ndarray, sigma: float = 1.8) -> np.ndarray:
    """Soft blur + re-threshold to reduce pixel stair-steps on the contour."""
    m = (mask > 0).astype(np.float64)
    m = ndimage.gaussian_filter(m, sigma=sigma)
    return (m > 0.5).astype(np.uint8) * 255


def mask_to_contours_xy(mask: np.ndarray) -> list[np.ndarray]:
    """Return list of Nx2 float arrays (x, y) in image coordinates.

    Uses ``cv2.findContours`` with ``CHAIN_APPROX_NONE``: vertices follow an ordered
    pixel chain along each boundary. Matplotlib ``Path.vertices`` alone could join
    disconnected subpaths and draw spurious chords across the interior.
    """
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: list[np.ndarray] = []
    for c in contours:
        if len(c) < 3:
            continue
        xy = c.reshape(-1, 2).astype(np.float64)
        out.append(xy)
    return out


def _polyline_length(v: np.ndarray) -> float:
    if len(v) < 2:
        return 0.0
    d = np.diff(v, axis=0)
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker; points Nx2."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    seg = end - start
    seg_len = float(np.hypot(seg[0], seg[1])) or 1.0
    t = ((points - start) * seg).sum(axis=1) / (seg_len * seg_len)
    t = np.clip(t, 0.0, 1.0)
    proj = start + np.outer(t, seg)
    dist = np.sqrt(((points - proj) ** 2).sum(axis=1))
    idx = int(np.argmax(dist))
    if dist[idx] > epsilon:
        left = _rdp(points[: idx + 1], epsilon)
        right = _rdp(points[idx:], epsilon)
        return np.vstack((left[:-1], right))
    return np.vstack((start, end))


def simplify_xy(verts: np.ndarray, epsilon_ratio: float = 0.008) -> np.ndarray:
    peri = _polyline_length(np.vstack((verts, verts[0])))
    eps = max(epsilon_ratio * peri, 1.0)
    closed = np.vstack((verts, verts[0]))
    simp = _rdp(closed, eps)
    if len(simp) > 1 and np.allclose(simp[0], simp[-1]):
        simp = simp[:-1]
    return simp


def draw_overlay_rgb(rgb: np.ndarray, verts_xy: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    pts = [(float(p[0]), float(p[1])) for p in verts_xy]
    if len(pts) >= 2:
        pts_closed = pts + [pts[0]]
        draw.line(pts_closed, fill=color, width=2, joint="curve")
    return im


def main() -> int:
    p = argparse.ArgumentParser(description="Extract bread center split contour.")
    p.add_argument("image", type=Path, help="Input image path")
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("out_bread_contour"),
        help="Output directory",
    )
    p.add_argument("--json", action="store_true", help="Write contour as JSON (xy list)")
    args = p.parse_args()

    try:
        pil = Image.open(args.image).convert("RGB")
    except OSError as e:
        print(f"Failed to read: {args.image} ({e})", file=sys.stderr)
        return 1
    rgb = np.asarray(pil)

    bread = bread_mask(rgb)
    center = center_split_mask(rgb, bread)
    center = smooth_mask(center)
    contours = mask_to_contours_xy(center)
    if not contours:
        print("No contour found.", file=sys.stderr)
        return 1
    verts = max(contours, key=lambda v: _polyline_length(np.vstack((v, v[0]))))
    simp = simplify_xy(verts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    draw_overlay_rgb(rgb.copy(), verts, (0, 200, 0)).save(
        args.out_dir / f"{stem}_contour_full.png"
    )
    draw_overlay_rgb(rgb.copy(), simp, (220, 40, 40)).save(
        args.out_dir / f"{stem}_contour_simplified.png"
    )
    Image.fromarray(center, mode="L").save(args.out_dir / f"{stem}_mask_center.png")

    if args.json:
        pts_full = np.round(verts).astype(int).tolist()
        pts_simplified = np.round(simp).astype(int).tolist()
        with open(args.out_dir / f"{stem}_contour.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "points": pts_full,
                    "points_simplified": pts_simplified,
                    "closed": True,
                },
                f,
                indent=2,
            )

    print(f"Wrote outputs to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
