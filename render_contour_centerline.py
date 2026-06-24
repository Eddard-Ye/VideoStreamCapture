#!/usr/bin/env python3
"""
Load bread_sam_contour.json: draw prompt_points, compute a slit centerline inside the filled
polygon mask, find the maximum chord perpendicular to a **PCA straight-axis fit** of that polyline
(default; avoids tangent jitter on curves). Cyan = curved centerline polyline; orange =
the same PCA axis **drawn with a lateral offset** so it does not paint over the curve
(geometrically the slit axis runs through the ribbon and would otherwise coincide on pixels).
Use ``--fit-line-display-offset 0`` to draw the exact axis (heavy overlap). Yellow width uses
the true axis, not the offset. ``--width-along-curve`` restores local-tangent width scan.

Default ``--centerline voronoi``: fitodic/centerline-style Voronoi medial axis (vendored in
``voronoi_polygon_centerline.py``), without Fiona/GDAL. Uses scipy Voronoi + Shapely ridge
clipping — same GIS workflow as the PyPI ``centerline`` geometry implementation.

Also available: ``ridge`` (distance-transform ridge along PCA), ``skeleton`` (pixel skeleton).

Deps: pip install numpy pillow scipy scikit-image shapely

Example:
  cd scripts/bread_contour/bread_final
  ../.venv/bin/python render_contour_centerline.py \\
    --json ../out/bread_sam_contour.json \\
    --image ../path/to/bread.jpg \\
    -o ./analysis_centerline.png
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge
from skimage.draw import polygon as sk_polygon
from skimage.morphology import skeletonize

_voronoi_spec = importlib.util.spec_from_file_location(
    "voronoi_polygon_centerline",
    Path(__file__).resolve().parent / "voronoi_polygon_centerline.py",
)
assert _voronoi_spec and _voronoi_spec.loader
_voronoi_mod = importlib.util.module_from_spec(_voronoi_spec)
_voronoi_spec.loader.exec_module(_voronoi_mod)
FitodicVoronoiCenterline = _voronoi_mod.FitodicVoronoiCenterline

# Overlay stroke widths (Pillow ``draw.line`` / markers)
W_CONTOUR = 5
W_CENTERLINE = 7
W_MAX_WIDTH = 10
W_FIT_LINE = 8
W_FIT_LINE_HALO = 12
FIT_LINE_COLOR = (255, 120, 20)
# Lateral shift (px) when drawing orange vs true PCA line — avoids raster overlap with cyan.
FIT_LINE_DISPLAY_OFFSET_DEFAULT = 18
PROMPT_R = 9
PROMPT_OUTLINE = 3

# Annotation panel (light; avoids harsh black box + ASCII-only label)
_LABEL_BG = (238, 235, 228)
_LABEL_BORDER = (186, 168, 128)
_LABEL_TEXT = (48, 50, 58)


def _annotation_font(size: int = 22) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def contour_points(data: dict) -> np.ndarray:
    pts = data.get("points") or []
    return np.asarray(pts, dtype=np.float64)


def prompt_points_arr(data: dict) -> np.ndarray:
    pp = data.get("prompt_points") or []
    return np.asarray(pp, dtype=np.float64)


def bbox_points(*arrays: np.ndarray, pad: float = 24.0) -> tuple[int, int, int, int]:
    allp = np.vstack([a for a in arrays if len(a)])
    min_xy = np.floor(allp.min(axis=0) - pad).astype(int)
    max_xy = np.ceil(allp.max(axis=0) + pad).astype(int)
    return int(min_xy[0]), int(min_xy[1]), int(max_xy[0]), int(max_xy[1])


def global_to_local(xy: np.ndarray, ox: int, oy: int) -> np.ndarray:
    out = xy.copy()
    out[:, 0] -= ox
    out[:, 1] -= oy
    return out


def local_to_global(xy: np.ndarray, ox: int, oy: int) -> np.ndarray:
    out = xy.copy()
    out[:, 0] += ox
    out[:, 1] += oy
    return out


def rasterize_polygon(poly_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """poly_xy columns x,y in pixel coords matching mask indexing."""
    if len(poly_xy) < 3:
        raise ValueError("Contour needs at least 3 points.")
    H, W = shape_hw
    rr, cc = sk_polygon(poly_xy[:, 1], poly_xy[:, 0], shape=(H, W))
    m = np.zeros((H, W), dtype=bool)
    m[rr, cc] = True
    return m


def skeleton_longest_geodesic(skel: np.ndarray) -> list[tuple[int, int]]:
    ys, xs = np.where(skel)
    if len(xs) == 0:
        return []
    points = list(zip(xs.tolist(), ys.tolist()))
    idx = {p: i for i, p in enumerate(points)}
    n = len(points)
    adj: list[list[int]] = [[] for _ in range(n)]
    pset = set(points)
    for i, (x, y) in enumerate(points):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                if q in pset:
                    adj[i].append(idx[q])

    def bfs(start: int) -> tuple[list[int], list[int]]:
        dist = [-1] * n
        parent = [-1] * n
        q: deque[int] = deque([start])
        dist[start] = 0
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        far = max(range(n), key=lambda i: dist[i])
        return parent, dist, far

    _, _, u = bfs(0)
    parent, _, v = bfs(u)
    chain_idx: list[int] = []
    cur = v
    while cur >= 0:
        chain_idx.append(cur)
        cur = parent[cur]
    chain_idx.reverse()
    return [points[i] for i in chain_idx]


def centerline_dt_ridge_pca(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Ridge of the EDT inside the mask: slice along the dominant PCA axis of mask pixels and,
    in each slice, take the pixel with maximum distance-to-boundary. Matches symmetry of
    symmetric ribbons better than the longest path on a pixel skeleton graph.
    """
    if not mask.any():
        return []
    dist = distance_transform_edt(mask)
    ys, xs = np.where(mask)
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    c = pts.mean(axis=0)
    xc = pts - c
    cov = (xc.T @ xc) / max(len(pts), 1)
    _, eigvecs = np.linalg.eigh(cov)
    u = eigvecs[:, -1]
    u = u / (np.linalg.norm(u) + 1e-12)
    proj = xc @ u
    t_min = float(proj.min())
    t_max = float(proj.max())
    span = t_max - t_min + 1e-6
    nbins = int(np.clip(span / 1.25, 48, 900))
    edges = np.linspace(t_min, t_max, nbins + 1)

    ridge: list[tuple[int, int]] = []
    for i in range(nbins):
        lo, hi = edges[i], edges[i + 1]
        last = i == nbins - 1
        if last:
            sel = (proj >= lo) & (proj <= hi)
        else:
            sel = (proj >= lo) & (proj < hi)
        inds = np.flatnonzero(sel)
        if inds.size == 0:
            continue
        iy = ys[inds]
        ix = xs[inds]
        local = dist[iy, ix]
        j = inds[int(np.argmax(local))]
        ridge.append((int(xs[j]), int(ys[j])))

    deduped: list[tuple[int, int]] = []
    for p in ridge:
        if not deduped or deduped[-1] != p:
            deduped.append(p)
    return deduped


def _poly_xy_to_shapely(poly_xy: np.ndarray, closed: bool) -> Polygon:
    pts = [(float(poly_xy[i, 0]), float(poly_xy[i, 1])) for i in range(poly_xy.shape[0])]
    if closed and len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        raise ValueError("Contour needs at least 3 unique vertices.")
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        raise ValueError("Contour polygon is empty after repair.")
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.geom_type != "Polygon":
        raise ValueError(f"Unexpected geometry type {poly.geom_type}")
    return poly


def _extract_lines_from_geom(g) -> list[LineString]:
    if g is None or g.is_empty:
        return []
    gt = g.geom_type
    if gt == "LineString":
        return [g]
    if gt == "MultiLineString":
        return list(g.geoms)
    if gt == "GeometryCollection":
        lines: list[LineString] = []
        for sub in g.geoms:
            lines.extend(_extract_lines_from_geom(sub))
        return lines
    return []


def _nk(x: float, y: float) -> tuple[float, float]:
    return (round(float(x), 5), round(float(y), 5))


def _longest_chain_tree_metric(
    adj: dict[tuple[float, float], list[tuple[float, float]]],
    nodes: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    node_set = set(nodes)
    if len(nodes) <= 1:
        return [(float(x), float(y)) for x, y in nodes]

    def bfs_far(start: tuple[float, float]):
        dist = {start: 0}
        parent = {start: None}
        q: deque[tuple[float, float]] = deque([start])
        far = start
        while q:
            u = q.popleft()
            far = u
            for v in adj[u]:
                if v not in node_set:
                    continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        return far, parent

    u, _ = bfs_far(nodes[0])
    v, parent = bfs_far(u)
    chain: list[tuple[float, float]] = []
    cur: tuple[float, float] | None = v
    while cur is not None:
        chain.append(cur)
        cur = parent[cur]
    chain.reverse()
    return chain


def _polyline_euclidean_length(path: Sequence[tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    )


def longest_polyline_from_voronoi_centerline(geom) -> list[tuple[float, float]]:
    merged = linemerge(geom)
    lines = _extract_lines_from_geom(merged)
    if not lines:
        return []

    adj: dict[tuple[float, float], set[tuple[float, float]]] = defaultdict(set)
    for ln in lines:
        coords = list(ln.coords)
        for i in range(len(coords) - 1):
            a = _nk(coords[i][0], coords[i][1])
            b = _nk(coords[i + 1][0], coords[i + 1][1])
            if a == b:
                continue
            adj[a].add(b)
            adj[b].add(a)

    seen_global: set[tuple[float, float]] = set()
    best_path: list[tuple[float, float]] = []
    best_len = -1.0

    for start in adj:
        if start in seen_global:
            continue
        stack = [start]
        comp: list[tuple[float, float]] = []
        local_seen: set[tuple[float, float]] = set()
        while stack:
            u = stack.pop()
            if u in local_seen:
                continue
            local_seen.add(u)
            seen_global.add(u)
            comp.append(u)
            for v in adj[u]:
                if v not in local_seen:
                    stack.append(v)
        comp_set = set(comp)
        adj_sub = {u: [v for v in adj[u] if v in comp_set] for u in comp}
        candidate = _longest_chain_tree_metric(adj_sub, comp)
        ln = _polyline_euclidean_length(candidate)
        if ln > best_len:
            best_len = ln
            best_path = candidate

    return best_path


def centerline_fitodic_voronoi(
    poly_xy_local: np.ndarray,
    closed: bool,
    interpolation_distance: float,
    simplify_tolerance: float | None,
):
    poly = _poly_xy_to_shapely(poly_xy_local, closed)
    if simplify_tolerance is not None and simplify_tolerance > 0:
        sp = poly.simplify(float(simplify_tolerance), preserve_topology=True)
        if not sp.is_empty:
            if sp.geom_type == "MultiPolygon":
                poly = max(sp.geoms, key=lambda g: g.area)
            elif sp.geom_type == "Polygon":
                poly = sp

    vc = FitodicVoronoiCenterline(poly, interpolation_distance=interpolation_distance)
    chain = longest_polyline_from_voronoi_centerline(vc.geometry)
    return [(float(x), float(y)) for x, y in chain]


def tangent_perpendicular(path: Sequence[tuple[float, float]], i: int) -> tuple[float, float, float, float]:
    """Return unit tangent (tx,ty) and perpendicular (nx,ny) at index i."""
    if len(path) < 2:
        return 1.0, 0.0, 0.0, 1.0
    i0 = max(0, i - 2)
    i1 = min(len(path) - 1, i + 2)
    x0, y0 = path[i0]
    x1, y1 = path[i1]
    dx, dy = float(x1 - x0), float(y1 - y0)
    norm = math.hypot(dx, dy) or 1.0
    tx, ty = dx / norm, dy / norm
    nx, ny = -ty, tx
    return tx, ty, nx, ny


def chord_along_normal(
    mask: np.ndarray,
    px: float,
    py: float,
    nx: float,
    ny: float,
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """March from (px,py) along ±n until leaving mask; chord length = Euclidean distance between last inside points."""
    H, W = mask.shape
    nlen = math.hypot(nx, ny) or 1.0
    nx, ny = nx / nlen, ny / nlen

    def march(sign: float) -> tuple[float, float]:
        x, y = float(px), float(py)
        last_x, last_y = x, y
        while True:
            x += sign * nx
            y += sign * ny
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or yi < 0 or xi >= W or yi >= H:
                break
            if not mask[yi, xi]:
                break
            last_x, last_y = x, y
        return last_x, last_y

    ax, ay = march(1.0)
    bx, by = march(-1.0)
    length = math.hypot(ax - bx, ay - by)
    return length, (bx, by), (ax, ay)


def max_perpendicular_width(
    mask: np.ndarray,
    path: Sequence[tuple[float, float]],
    sample_stride: int = 2,
) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], int]:
    """Returns max_width, p_on_centerline, q_boundary_a, q_boundary_b, path_index."""
    if len(path) < 2:
        raise RuntimeError("Centerline too short.")
    best_w = -1.0
    best = None
    for i in range(0, len(path), sample_stride):
        px, py = float(path[i][0]), float(path[i][1])
        _, _, nx, ny = tangent_perpendicular(path, i)
        w, ba, bb = chord_along_normal(mask, px, py, nx, ny)
        if w > best_w:
            best_w = w
            best = (px, py), ba, bb, i
    assert best is not None
    (px, py), ba, bb, idx = best
    return best_w, (px, py), ba, bb, idx


def fit_centerline_line_pca(path_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA axis of centerline points: centroid ``c`` (2,) and unit direction ``u`` along the slit."""
    if len(path_xy) < 2:
        raise ValueError("Need at least two centerline points.")
    xy = np.asarray(path_xy, dtype=np.float64)
    c = xy.mean(axis=0)
    if len(xy) == 2:
        d = xy[1] - xy[0]
        nrm = float(np.linalg.norm(d)) or 1.0
        u = (d / nrm).astype(np.float64)
        return c, u
    xxc = xy - c
    _, _, vt = np.linalg.svd(xxc, full_matrices=False)
    u = vt[0].astype(np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    return c, u


def max_width_perpendicular_to_axis(
    mask: np.ndarray,
    path: Sequence[tuple[float, float]],
    c: np.ndarray,
    u_axis: np.ndarray,
    n_samples: int = 160,
) -> tuple[float, tuple[float, float], tuple[float, float], tuple[float, float], int]:
    """
    Max chord with fixed normal ``n ⟂ u_axis``. Samples stations **on the PCA axis** (projection
    span of the path), not along the cyan polyline — width is strictly vs. the fitted orange line.
    """
    if len(path) < 2:
        raise RuntimeError("Centerline too short.")
    H, W = mask.shape
    u = np.asarray(u_axis, dtype=np.float64).reshape(2)
    u = u / (np.linalg.norm(u) + 1e-12)
    cc = np.asarray(c, dtype=np.float64).reshape(2)
    nx, ny = float(-u[1]), float(u[0])

    xy = np.asarray(path, dtype=np.float64)
    t_proj = (xy - cc) @ u
    t0, t1 = float(t_proj.min()), float(t_proj.max())
    if t1 - t0 < 1e-9:
        raise RuntimeError("Degenerate centerline projection.")

    best_w = -1.0
    best = None
    best_k = 0
    ns = max(16, int(n_samples))
    for k in range(ns):
        t = t0 + (t1 - t0) * (k / max(ns - 1, 1))
        px = float(cc[0] + t * u[0])
        py = float(cc[1] + t * u[1])
        xi, yi = int(round(px)), int(round(py))
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            continue
        w, ba, bb = chord_along_normal(mask, px, py, nx, ny)
        if w > best_w:
            best_w = w
            best = (px, py), ba, bb
            best_k = k
    if best is None:
        raise RuntimeError("No on-axis sample inside mask; try denser centerline.")
    (px, py), ba, bb = best
    return best_w, (px, py), ba, bb, best_k


def clip_pca_line_to_image_global(
    c_loc: np.ndarray,
    u: np.ndarray,
    ox: int,
    oy: int,
    iw: int,
    ih: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip infinite PCA line (local c + t*u) to image [0,iw)×[0,ih) in global pixels."""
    cg = np.asarray(
        local_to_global(np.array([[c_loc[0], c_loc[1]]]), ox, oy)[0],
        dtype=np.float64,
    ).reshape(2)
    uu = np.asarray(u, dtype=np.float64).reshape(2)
    uu = uu / (np.linalg.norm(uu) + 1e-15)
    eps = 1e-9
    ts: list[float] = []

    def consider(t: float) -> None:
        x = cg[0] + t * uu[0]
        y = cg[1] + t * uu[1]
        if -eps <= x <= iw - 1 + eps and -eps <= y <= ih - 1 + eps:
            ts.append(t)

    if abs(uu[0]) > eps:
        for xv in (0.0, float(iw - 1)):
            t = (xv - cg[0]) / uu[0]
            consider(t)
    if abs(uu[1]) > eps:
        for yv in (0.0, float(ih - 1)):
            t = (yv - cg[1]) / uu[1]
            consider(t)

    if len(ts) < 2:
        return None
    t_min, t_max = min(ts), max(ts)
    if t_max - t_min < 1e-9:
        return None
    p0 = cg + t_min * uu
    p1 = cg + t_max * uu
    return ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])))


def shift_segment_parallel_display(
    p0: tuple[float, float],
    p1: tuple[float, float],
    u_axis: np.ndarray,
    delta_px: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Translate segment along unit normal n=(-u_y,u_x). Same line direction, parallel offset."""
    u = np.asarray(u_axis, dtype=np.float64).reshape(2)
    u = u / (np.linalg.norm(u) + 1e-15)
    nx, ny = float(-u[1]), float(u[0])
    dx, dy = delta_px * nx, delta_px * ny
    return (
        (p0[0] + dx, p0[1] + dy),
        (p1[0] + dx, p1[1] + dy),
    )


def draw_visualization(
    bg_rgb: np.ndarray,
    ox: int,
    oy: int,
    contour_xy_global: np.ndarray,
    contour_closed: bool,
    path_global: list[tuple[float, float]],
    max_seg_a: tuple[float, float],
    max_seg_b: tuple[float, float],
    prompt_xy_global: np.ndarray,
    max_width_px: float,
    fit_line_segment: tuple[tuple[float, float], tuple[float, float]] | None = None,
    width_label_pca: bool = False,
    contour_color: tuple[int, int, int] = (220, 95, 220),
    line_center_color: tuple[int, int, int] = (0, 255, 255),
    line_width_color: tuple[int, int, int] = (255, 215, 0),
    fit_line_color: tuple[int, int, int] = FIT_LINE_COLOR,
    prompt_color: tuple[int, int, int] = (50, 205, 50),
) -> Image.Image:
    im = Image.fromarray(bg_rgb.copy(), mode="RGB")
    draw = ImageDraw.Draw(im)
    if len(contour_xy_global) >= 2:
        pl = [(float(p[0]), float(p[1])) for p in contour_xy_global]
        if contour_closed and len(pl) >= 3:
            if pl[0] != pl[-1]:
                pl = pl + [pl[0]]
        draw.line(pl, fill=contour_color, width=W_CONTOUR)
    pr = PROMPT_R
    for x, y in prompt_xy_global:
        draw.ellipse(
            [x - pr, y - pr, x + pr, y + pr],
            fill=prompt_color,
            outline=(255, 255, 255),
            width=PROMPT_OUTLINE,
        )
    # Cyan centerline first, then PCA fit (orange) on top so the straight axis stays visible.
    if len(path_global) >= 2:
        pline = [(float(p[0]), float(p[1])) for p in path_global]
        draw.line(pline, fill=line_center_color, width=W_CENTERLINE)
    if fit_line_segment is not None:
        fl0, fl1 = fit_line_segment
        draw.line([fl0, fl1], fill=(255, 255, 255), width=W_FIT_LINE_HALO)
        draw.line([fl0, fl1], fill=fit_line_color, width=W_FIT_LINE)
    draw.line(
        [max_seg_a, max_seg_b],
        fill=line_width_color,
        width=W_MAX_WIDTH,
    )

    font = _annotation_font(22)
    if width_label_pca:
        label = f"Max width (⊥ true PCA axis): {max_width_px:.1f} px"
    else:
        label = f"Max perpendicular width (yellow): {max_width_px:.1f} px"
    mx = (float(max_seg_a[0]) + float(max_seg_b[0])) * 0.5
    my = (float(max_seg_a[1]) + float(max_seg_b[1])) * 0.5
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = 12
    tx = mx - tw * 0.5
    ty = my - th - pad * 2.5
    iw, ih = im.size
    tx = float(max(pad, min(tx, iw - tw - pad)))
    ty = float(max(pad, min(ty, ih - th - pad)))

    rect = [tx - pad, ty - pad, tx + tw + pad, ty + th + pad]
    draw.rectangle(rect, fill=_LABEL_BG, outline=_LABEL_BORDER, width=2)
    draw.text(
        (tx, ty),
        label,
        fill=_LABEL_TEXT,
        font=font,
        stroke_width=1,
        stroke_fill=(252, 250, 246),
    )
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description="Render prompts + centerline + max width chord.")
    ap.add_argument(
        "--json",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "out" / "bread_sam_contour.json",
        help="Path to bread_sam_contour.json",
    )
    ap.add_argument("--image", type=Path, default=None, help="Background image (RGB).")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_centerline.png",
        help="Output PNG path",
    )
    ap.add_argument("--stride", type=int, default=2, help="Stride along centerline for width scan.")
    ap.add_argument(
        "--centerline",
        choices=("voronoi", "ridge", "skeleton"),
        default="voronoi",
        help="voronoi: fitodic Voronoi-in-polygon (vendored). ridge / skeleton: raster heuristics.",
    )
    ap.add_argument(
        "--interpolation-distance",
        type=float,
        default=12.0,
        help="Voronoi border densification step (pixels along perimeter). Larger is faster/coarser.",
    )
    ap.add_argument(
        "--voronoi-simplify",
        type=float,
        default=2.0,
        help="Douglas–Peucker tolerance before Voronoi (pixels). 0 disables. Ignored with --voronoi-use-json-simplified.",
    )
    ap.add_argument(
        "--voronoi-use-json-simplified",
        action="store_true",
        help="Run Voronoi on JSON points_simplified (fast). Full points still used for mask/contour.",
    )
    ap.add_argument(
        "--width-along-curve",
        action="store_true",
        help="Measure max width using local tangent to the polyline (legacy). Default: PCA straight axis + draw orange fit line.",
    )
    ap.add_argument(
        "--pca-width-samples",
        type=int,
        default=160,
        help="Stations along PCA axis when measuring max width (default mode).",
    )
    ap.add_argument(
        "--fit-line-display-offset",
        type=float,
        default=FIT_LINE_DISPLAY_OFFSET_DEFAULT,
        help=(
            "Draw the orange PCA guide shifted perpendicular by this many pixels so it does not "
            "overlap the cyan curve (slit points lie near the axis). Width math uses the true axis. "
            "Use 0 to draw the exact axis."
        ),
    )
    args = ap.parse_args()

    data = load_json(args.json)
    contour = contour_points(data)
    prompts = prompt_points_arr(data)
    if contour.shape[0] < 3:
        print("JSON missing valid contour points.", file=sys.stderr)
        return 1

    min_x, min_y, max_x, max_y = bbox_points(contour, prompts)
    W = max_x - min_x + 1
    H = max_y - min_y + 1
    ox, oy = min_x, min_y

    poly_loc = global_to_local(contour, ox, oy)
    mask = rasterize_polygon(poly_loc, (H, W))
    contour_closed = bool(data.get("closed", True))

    if args.centerline == "voronoi":
        try:
            voronoi_xy = poly_loc
            if args.voronoi_use_json_simplified:
                simp = data.get("points_simplified")
                if simp is not None and len(simp) >= 3:
                    voronoi_xy = global_to_local(np.asarray(simp, dtype=np.float64), ox, oy)
                else:
                    print(
                        "JSON has no points_simplified; using full contour for Voronoi.",
                        file=sys.stderr,
                    )
            v_simp = args.voronoi_simplify
            path_loc = centerline_fitodic_voronoi(
                voronoi_xy,
                contour_closed,
                interpolation_distance=max(args.interpolation_distance, 1e-3),
                simplify_tolerance=None
                if v_simp <= 0 or args.voronoi_use_json_simplified
                else v_simp,
            )
        except Exception as e:
            print(
                f"Voronoi centerline failed ({e}); try --centerline ridge, "
                "or tune --interpolation-distance / --voronoi-simplify.",
                file=sys.stderr,
            )
            return 1
        if len(path_loc) < 3:
            print(
                "Voronoi centerline too short; try --voronoi-use-json-simplified or --centerline ridge.",
                file=sys.stderr,
            )
            return 1
    elif args.centerline == "ridge":
        path_loc = centerline_dt_ridge_pca(mask)
        if len(path_loc) < 3:
            print(
                "EDT ridge centerline too short (mask empty or degenerate?); "
                "try --centerline skeleton.",
                file=sys.stderr,
            )
            return 1
    else:
        skel = skeletonize(mask)
        path_loc = skeleton_longest_geodesic(skel)
        if len(path_loc) < 4:
            print("Skeleton centerline too short (try a narrower slit mask?).", file=sys.stderr)
            return 1

    path_xy_loc = np.asarray(path_loc, dtype=np.float64)
    c_loc, u_axis = fit_centerline_line_pca(path_xy_loc)

    if args.image and args.image.is_file():
        pil = Image.open(args.image).convert("RGB")
        bg = np.asarray(pil)
        if bg.shape[1] < max_x + 1 or bg.shape[0] < max_y + 1:
            print(
                "Warning: background smaller than contour bbox; padding/cropping not applied — drawing may clip.",
                file=sys.stderr,
            )
    else:
        bg = np.ones((max_y + 1, max_x + 1, 3), dtype=np.uint8) * 255
        if args.image:
            print(f"Image not found: {args.image}; using white canvas sized to bbox.", file=sys.stderr)

    iw, ih = int(bg.shape[1]), int(bg.shape[0])

    if args.width_along_curve:
        width_px, center_pt, end_a, end_b, idx_best = max_perpendicular_width(
            mask, path_loc, sample_stride=max(1, args.stride)
        )
        fit_seg_global = None
        label_pca = False
    else:
        width_px, center_pt, end_a, end_b, idx_best = max_width_perpendicular_to_axis(
            mask,
            path_loc,
            c_loc,
            u_axis,
            n_samples=max(32, args.pca_width_samples),
        )
        fit_seg_global = clip_pca_line_to_image_global(c_loc, u_axis, ox, oy, iw, ih)
        if fit_seg_global is None:
            uu = np.asarray(u_axis, dtype=np.float64).reshape(2)
            uu = uu / (np.linalg.norm(uu) + 1e-15)
            t = (path_xy_loc - c_loc.reshape(2)) @ uu
            span = float(t.max() - t.min()) + 1e-9
            pad = 0.06 * span
            p0 = c_loc + (float(t.min()) - pad) * uu
            p1 = c_loc + (float(t.max()) + pad) * uu
            g0 = tuple(local_to_global(np.array([[p0[0], p0[1]]]), ox, oy)[0])
            g1 = tuple(local_to_global(np.array([[p1[0], p1[1]]]), ox, oy)[0])
            fit_seg_global = (
                (float(g0[0]), float(g0[1])),
                (float(g1[0]), float(g1[1])),
            )
        label_pca = True

    fit_line_draw = fit_seg_global
    if (
        fit_seg_global is not None
        and label_pca
        and abs(float(args.fit_line_display_offset)) > 1e-6
    ):
        fit_line_draw = shift_segment_parallel_display(
            fit_seg_global[0],
            fit_seg_global[1],
            u_axis,
            float(args.fit_line_display_offset),
        )

    path_g = [tuple(local_to_global(np.array([[x, y]]), ox, oy)[0]) for x, y in path_loc]
    end_ag = tuple(local_to_global(np.array([[end_a[0], end_a[1]]]), ox, oy)[0])
    end_bg = tuple(local_to_global(np.array([[end_b[0], end_b[1]]]), ox, oy)[0])

    legend = data.get("legend") or {}
    sam_rgb = legend.get("sam_rgb")
    contour_rgb = (
        tuple(int(c) for c in sam_rgb[:3])
        if isinstance(sam_rgb, (list, tuple)) and len(sam_rgb) >= 3
        else (220, 95, 220)
    )
    out_img = draw_visualization(
        bg,
        ox,
        oy,
        contour,
        contour_closed,
        path_g,
        end_ag,
        end_bg,
        prompts,
        width_px,
        fit_line_segment=fit_line_draw,
        width_label_pca=label_pca,
        contour_color=contour_rgb,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(args.output)

    mode_w = "true PCA axis (yellow ⊥ this axis)" if label_pca else "local tangent along curve"
    print(f"Max width ({mode_w}): {width_px:.2f} px")
    if label_pca and fit_seg_global is not None:
        off = float(args.fit_line_display_offset)
        if abs(off) > 1e-6:
            print(
                f"Orange line: parallel to PCA axis, display offset {off:.1f} px (not used in width).",
                file=sys.stderr,
            )
    idx_note = "PCA-axis station #" if label_pca else "polyline vertex index"
    print(f"Best sample ({idx_note}): {idx_best} ({args.centerline})")
    print(
        "Chord endpoints (global xy): "
        f"({float(end_ag[0]):.2f}, {float(end_ag[1]):.2f}) — ({float(end_bg[0]):.2f}, {float(end_bg[1]):.2f})"
    )
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
