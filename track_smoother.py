# -*- coding: utf-8 -*-
"""Temporal smoothing for live YOLO instance metrics (track + EMA)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from color_viewer import SegInstance


def _ema(previous: float, current: float, alpha: float) -> float:
    if not np.isfinite(current):
        return previous
    if not np.isfinite(previous):
        return current
    return previous * (1.0 - alpha) + current * alpha


def _align_length_width(
    prev_length: float,
    prev_width: float,
    cur_length: float,
    cur_width: float,
) -> tuple[float, float]:
    """Keep long/short edge assignment stable when minAreaRect flips."""
    if not (np.isfinite(prev_length) and np.isfinite(prev_width)):
        return cur_length, cur_width
    if not (np.isfinite(cur_length) and np.isfinite(cur_width)):
        return cur_length, cur_width

    direct_cost = abs(cur_length - prev_length) + abs(cur_width - prev_width)
    swapped_cost = abs(cur_width - prev_length) + abs(cur_length - prev_width)
    if swapped_cost < direct_cost:
        return cur_width, cur_length
    return cur_length, cur_width


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0, 0, 0, 0)
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + 0.5 * w, y + 0.5 * h)


def _center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(np.hypot(ax - bx, ay - by))


@dataclass
class _TrackState:
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]
    length_mm: float
    width_mm: float
    length_px: float
    width_px: float
    height_mm: float
    peak_height_mm: float
    confidence: float
    box_pts: np.ndarray | None
    miss_count: int = 0


class TrackSmoother:
    """Match detections across frames and EMA-smooth L/W/H readouts for live display."""

    def __init__(
        self,
        *,
        alpha: float = 0.25,
        max_miss: int = 3,
        iou_threshold: float = 0.15,
        max_tracks: int = 0,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.01, 1.0))
        self.max_miss = max(1, int(max_miss))
        self.iou_threshold = float(np.clip(iou_threshold, 0.0, 1.0))
        self.max_tracks = max(0, int(max_tracks))
        self._tracks: list[_TrackState] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def _match_indices(
        self,
        detections: Sequence[SegInstance],
        bboxes: Sequence[tuple[int, int, int, int]],
    ) -> list[tuple[int, int] | None]:
        if not self._tracks or not detections:
            return [None] * len(detections)

        pairs: list[tuple[float, int, int]] = []
        for det_index, bbox in enumerate(bboxes):
            det = detections[det_index]
            for track_index, track in enumerate(self._tracks):
                if det.class_name != track.class_name:
                    continue
                iou = _bbox_iou(bbox, track.bbox)
                if iou < self.iou_threshold:
                    center_dist = _center_distance(bbox, track.bbox)
                    max_dim = max(bbox[2], bbox[3], track.bbox[2], track.bbox[3], 1)
                    if center_dist > max_dim:
                        continue
                    score = iou
                else:
                    score = iou
                pairs.append((score, det_index, track_index))

        pairs.sort(key=lambda item: item[0], reverse=True)
        det_to_track: list[int | None] = [None] * len(detections)
        used_tracks: set[int] = set()
        for score, det_index, track_index in pairs:
            if det_to_track[det_index] is not None or track_index in used_tracks:
                continue
            det_to_track[det_index] = track_index
            used_tracks.add(track_index)
        return det_to_track

    def _update_track(self, track: _TrackState, instance: SegInstance, bbox: tuple[int, int, int, int]) -> None:
        alpha = self.alpha
        cur_length_px, cur_width_px = _align_length_width(
            track.length_px,
            track.width_px,
            instance.length_px,
            instance.width_px,
        )
        cur_length_mm, cur_width_mm = _align_length_width(
            track.length_mm,
            track.width_mm,
            instance.length_mm,
            instance.width_mm,
        )

        track.bbox = bbox
        track.length_px = _ema(track.length_px, cur_length_px, alpha)
        track.width_px = _ema(track.width_px, cur_width_px, alpha)
        track.length_mm = _ema(track.length_mm, cur_length_mm, alpha)
        track.width_mm = _ema(track.width_mm, cur_width_mm, alpha)
        track.height_mm = _ema(track.height_mm, instance.height_mm, alpha)
        track.peak_height_mm = _ema(track.peak_height_mm, instance.peak_height_mm, alpha)
        track.confidence = _ema(track.confidence, instance.confidence, alpha)

        if instance.box_pts is not None:
            current_box = np.asarray(instance.box_pts, dtype=np.float32)
            if track.box_pts is None:
                track.box_pts = current_box.copy()
            else:
                track.box_pts = track.box_pts * (1.0 - alpha) + current_box * alpha
        track.miss_count = 0

    def _new_track(self, instance: SegInstance, bbox: tuple[int, int, int, int]) -> _TrackState:
        box_pts = None if instance.box_pts is None else np.asarray(instance.box_pts, dtype=np.float32).copy()
        track = _TrackState(
            track_id=self._next_track_id,
            class_name=instance.class_name,
            bbox=bbox,
            length_mm=instance.length_mm,
            width_mm=instance.width_mm,
            length_px=instance.length_px,
            width_px=instance.width_px,
            height_mm=instance.height_mm,
            peak_height_mm=instance.peak_height_mm,
            confidence=instance.confidence,
            box_pts=box_pts,
        )
        self._next_track_id += 1
        return track

    def _display_instance(self, instance: SegInstance, track: _TrackState) -> SegInstance:
        box_pts = track.box_pts
        if box_pts is not None:
            box_pts = np.asarray(box_pts, dtype=np.float32)
        return replace(
            instance,
            length_mm=track.length_mm,
            width_mm=track.width_mm,
            length_px=track.length_px,
            width_px=track.width_px,
            height_mm=track.height_mm,
            peak_height_mm=track.peak_height_mm,
            confidence=track.confidence,
            box_pts=box_pts,
        )

    def _update_single_object(self, instances: list[SegInstance]) -> list[SegInstance]:
        """Smooth only the highest-confidence detection; keep at most one track."""
        if not instances:
            if self._tracks:
                self._tracks[0].miss_count += 1
                if self._tracks[0].miss_count > self.max_miss:
                    self._tracks.clear()
            return []

        instance = max(instances, key=lambda item: item.confidence)
        bbox = _mask_bbox(instance.mask)
        if not self._tracks:
            track = self._new_track(instance, bbox)
            self._tracks = [track]
            return [instance]

        track = self._tracks[0]
        self._update_track(track, instance, bbox)
        self._tracks = [track]
        return [self._display_instance(instance, track)]

    def update(self, instances: list[SegInstance]) -> list[SegInstance]:
        """Return display copies with temporally smoothed metrics."""
        if self.max_tracks == 1:
            return self._update_single_object(instances)

        if not instances:
            for track in self._tracks:
                track.miss_count += 1
            self._tracks = [track for track in self._tracks if track.miss_count <= self.max_miss]
            return []

        bboxes = [_mask_bbox(instance.mask) for instance in instances]
        det_to_track = self._match_indices(instances, bboxes)
        matched_track_indices: set[int] = set()
        display: list[SegInstance] = []

        for det_index, instance in enumerate(instances):
            track_index = det_to_track[det_index]
            bbox = bboxes[det_index]
            if track_index is None:
                track = self._new_track(instance, bbox)
                self._tracks.append(track)
                display.append(instance)
                matched_track_indices.add(len(self._tracks) - 1)
                continue

            track = self._tracks[track_index]
            self._update_track(track, instance, bbox)
            display.append(self._display_instance(instance, track))
            matched_track_indices.add(track_index)

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_track_indices:
                track.miss_count += 1

        self._tracks = [track for track in self._tracks if track.miss_count <= self.max_miss]
        return display
