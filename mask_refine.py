# -*- coding: utf-8 -*-
"""Lightweight mask edge refinement (Otsu inside YOLO bbox)."""

from __future__ import annotations

import cv2
import numpy as np


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def refine_mask_otsu(
    image_bgr: np.ndarray,
    yolo_mask: np.ndarray,
    *,
    pad: int = 12,
    min_overlap: int = 8,
) -> np.ndarray:
    """Expand YOLO mask toward high-contrast edges via Otsu inside a padded tight bbox."""
    yolo_bool = yolo_mask.astype(bool)
    if not np.any(yolo_bool):
        return yolo_bool

    bbox = _tight_bbox(yolo_bool)
    if bbox is None:
        return yolo_bool

    x1, y1, x2, y2 = bbox
    height, width = image_bgr.shape[:2]
    pad = max(0, int(pad))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return yolo_bool

    gray = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    yolo_crop = yolo_bool[y1:y2, x1:x2]

    interior = gray[yolo_crop]
    if interior.size == 0:
        return yolo_bool
    exterior = gray[~yolo_crop]
    if exterior.size == 0:
        exterior = gray.reshape(-1)

    if float(interior.mean()) < float(exterior.mean()):
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels = cv2.connectedComponents(otsu)
    if num_labels <= 1:
        return yolo_bool

    best_label = 0
    best_overlap = 0
    for label in range(1, num_labels):
        overlap = int(((labels == label) & yolo_crop).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label

    if best_label == 0 or best_overlap < min_overlap:
        return yolo_bool

    refined_crop = (labels == best_label) | yolo_crop

    full = np.zeros((height, width), dtype=bool)
    full[y1:y2, x1:x2] = refined_crop
    return full
