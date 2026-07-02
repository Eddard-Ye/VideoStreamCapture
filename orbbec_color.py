# -*- coding: utf-8 -*-
"""Orbbec Gemini RGB-D frame capture for the streaming pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pyorbbecsdk import (
    AlignFilter,
    Config,
    Context,
    OBError,
    OBFrameAggregateOutputMode,
    OBLogLevel,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

from camera_intrinsics import RgbIntrinsics
from orbbec_camera import (
    CameraSession,
    find_config_path,
    frame_to_bgr_image,
    frame_to_depth_mm,
    load_camera_config,
    try_open_camera_session,
)
from orbbec_metrics import intrinsics_from_orbbec

CaptureMode = Literal["session", "legacy_rgbd", "legacy_color"]

DEFAULT_MIN_DEPTH_MM = 20.0
DEFAULT_MAX_DEPTH_MM = 5000.0


@dataclass
class OrbbecDeviceInfo:
    index: int
    name: str
    serial: str
    firmware: str


@dataclass
class RgbdFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray


@dataclass
class OrbbecRgbdCapture:
    mode: CaptureMode
    session: CameraSession | None = None
    pipeline: Pipeline | None = None
    legacy_config: Config | None = None
    align_filter: AlignFilter | None = None
    config_path: str | None = None
    min_depth_mm: float = DEFAULT_MIN_DEPTH_MM
    max_depth_mm: float = DEFAULT_MAX_DEPTH_MM
    _camera_param: object | None = None

    @property
    def has_depth(self) -> bool:
        return self.mode != "legacy_color"

    def close(self) -> None:
        if self.session is not None:
            self.session.stop()
            self.session = None
        elif self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        self._camera_param = None

    def fetch_rgbd(
        self,
        *,
        timeout_ms: int = 1000,
        warmup_frames: int = 0,
    ) -> RgbdFrame:
        frame: RgbdFrame | None = None
        for _ in range(max(0, warmup_frames) + 1):
            frame = self._fetch_one(timeout_ms=timeout_ms)
            if frame is None:
                raise RuntimeError("Timed out waiting for Orbbec RGB-D frame.")
        assert frame is not None
        return frame

    def fetch_latest_rgbd(
        self,
        *,
        timeout_ms: int = 1000,
        poll_ms: int = 1,
        max_drain: int = 32,
    ) -> RgbdFrame:
        """Return the newest RGB-D frame, draining stale queued frames first."""
        latest = self._fetch_one(timeout_ms=timeout_ms)
        if latest is None:
            raise RuntimeError("Timed out waiting for Orbbec RGB-D frame.")

        drained = 0
        while drained < max_drain:
            nxt = self._fetch_one(timeout_ms=max(1, poll_ms))
            if nxt is None:
                break
            latest = nxt
            drained += 1
        return latest

    def fetch_color(
        self,
        *,
        timeout_ms: int = 1000,
        warmup_frames: int = 0,
    ) -> np.ndarray:
        return self.fetch_rgbd(
            timeout_ms=timeout_ms,
            warmup_frames=warmup_frames,
        ).color_bgr

    def get_rgb_intrinsics(self) -> RgbIntrinsics | None:
        if self.session is not None:
            intrinsic = self.session.get_rgb_intrinsics(roi_adjusted=True)
            if intrinsic is None:
                return None
            return intrinsics_from_orbbec(intrinsic)

        self._ensure_camera_param()
        if self._camera_param is None:
            return None
        return intrinsics_from_orbbec(self._camera_param.rgb_intrinsic)

    def _ensure_camera_param(self) -> None:
        if self.session is not None or self._camera_param is not None:
            return
        if self.pipeline is None:
            return
        for _ in range(30):
            frames = self.pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            try:
                self._camera_param = self.pipeline.get_camera_param()
                return
            except OBError:
                continue

    def _filter_depth_mm(self, depth_mm: np.ndarray) -> np.ndarray:
        return np.where(
            (depth_mm >= self.min_depth_mm) & (depth_mm <= self.max_depth_mm),
            depth_mm,
            0.0,
        )

    def _legacy_fetch_rgbd(self, *, timeout_ms: int) -> RgbdFrame | None:
        assert self.pipeline is not None
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        depth_frame = frames.get_depth_frame()
        if self.align_filter is not None and depth_frame is not None:
            aligned = self.align_filter.process(frames)
            if aligned is not None:
                color_frame = aligned.get_color_frame() or color_frame
                depth_frame = aligned.get_depth_frame() or depth_frame

        color_bgr = frame_to_bgr_image(color_frame)
        if color_bgr is None:
            return None

        if depth_frame is None:
            depth_mm = np.zeros(color_bgr.shape[:2], dtype=np.float32)
        else:
            depth_mm = self._filter_depth_mm(frame_to_depth_mm(depth_frame))
            if color_bgr.shape[:2] != depth_mm.shape[:2]:
                depth_mm = cv2_resize_depth(depth_mm, color_bgr.shape[1], color_bgr.shape[0])

        return RgbdFrame(color_bgr=color_bgr, depth_mm=depth_mm)

    def _fetch_one(self, *, timeout_ms: int) -> RgbdFrame | None:
        if self.session is not None:
            frames = self.session.wait_for_frames(timeout_ms)
            if frames is None:
                return None
            frame_data = self.session.process_frames(frames)
            if frame_data is None:
                return None
            return RgbdFrame(
                color_bgr=frame_data.color_bgr,
                depth_mm=frame_data.depth_mm,
            )

        return self._legacy_fetch_rgbd(timeout_ms=timeout_ms)


def cv2_resize_depth(depth_mm: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return cv2.resize(depth_mm, (width, height), interpolation=cv2.INTER_NEAREST)


def list_devices() -> list[OrbbecDeviceInfo]:
    ctx = Context()
    ctx.set_logger_level(OBLogLevel.WARNING)
    device_list = ctx.query_devices()
    devices: list[OrbbecDeviceInfo] = []
    for index in range(device_list.get_count()):
        info = device_list.get_device_by_index(index).get_device_info()
        devices.append(
            OrbbecDeviceInfo(
                index=index,
                name=info.get_name(),
                serial=info.get_serial_number(),
                firmware=info.get_firmware_version(),
            )
        )
    return devices


def _depth_range_from_config(config_path: str | None) -> tuple[float, float]:
    if config_path is None:
        return DEFAULT_MIN_DEPTH_MM, DEFAULT_MAX_DEPTH_MM
    try:
        cfg = load_camera_config(config_path)
        return cfg.depth.min_depth_mm, cfg.depth.max_depth_mm
    except (OSError, ValueError):
        return DEFAULT_MIN_DEPTH_MM, DEFAULT_MAX_DEPTH_MM


def _build_legacy_rgbd_pipeline() -> tuple[Pipeline, Config, AlignFilter]:
    """Same startup as color_depth_stream.build_legacy_pipeline()."""
    pipeline = Pipeline()
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)
    print(f"Color stream : {color_profile}")

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)
    print(f"Depth stream : {depth_profile}")

    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    try:
        pipeline.enable_frame_sync()
    except OBError as exc:
        print(f"Frame sync warning: {exc}")

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    return pipeline, config, align_filter


def open_rgbd_camera(
    *,
    config_path: str | None = None,
    device_index: int = 0,
    use_config: bool | None = None,
) -> OrbbecRgbdCapture:
    """Open Orbbec color + depth streams using JSON config when available."""
    devices = list_devices()
    if not devices:
        raise RuntimeError("No Orbbec device found. Connect Gemini 215 and retry.")

    if device_index >= len(devices):
        raise RuntimeError(
            f"Invalid device index {device_index}, found {len(devices)} device(s)."
        )

    selected = devices[device_index]
    print(
        f"Orbbec device [{device_index}] "
        f"name={selected.name} serial={selected.serial} fw={selected.firmware}"
    )

    resolved_config: str | None = None
    if config_path:
        resolved_config = os.path.abspath(config_path)
        if not os.path.isfile(resolved_config):
            raise FileNotFoundError(f"Orbbec config not found: {resolved_config}")
    elif use_config is not False:
        found = find_config_path()
        resolved_config = str(found) if found is not None else None

    if use_config is False:
        resolved_config = None

    min_depth_mm, max_depth_mm = _depth_range_from_config(resolved_config)

    session: CameraSession | None = None
    if resolved_config is not None:
        print(f"Using Orbbec camera config: {resolved_config}")
        try:
            session = try_open_camera_session(resolved_config)
        except OBError as exc:
            print(f"[warn] Configured pipeline failed ({exc}); falling back to legacy startup.")
            session = None

    if session is not None:
        return OrbbecRgbdCapture(
            mode="session",
            session=session,
            config_path=resolved_config,
            min_depth_mm=min_depth_mm,
            max_depth_mm=max_depth_mm,
        )

    if resolved_config is not None:
        print("Starting legacy Orbbec pipeline (default device streams).\n")
    else:
        print("No orbbec_camera.json — starting legacy RGB-D pipeline.\n")

    pipeline, legacy_config, align_filter = _build_legacy_rgbd_pipeline()
    pipeline.start(legacy_config)
    capture = OrbbecRgbdCapture(
        mode="legacy_rgbd",
        pipeline=pipeline,
        legacy_config=legacy_config,
        align_filter=align_filter,
        config_path=resolved_config,
        min_depth_mm=min_depth_mm,
        max_depth_mm=max_depth_mm,
    )
    capture._ensure_camera_param()
    return capture


def open_color_camera(
    *,
    config_path: str | None = None,
    device_index: int = 0,
    use_config: bool | None = None,
) -> OrbbecRgbdCapture:
    """Backward-compatible alias for :func:`open_rgbd_camera`."""
    return open_rgbd_camera(
        config_path=config_path,
        device_index=device_index,
        use_config=use_config,
    )


def close_rgbd_camera(capture: OrbbecRgbdCapture | None) -> None:
    if capture is not None:
        capture.close()


def close_color_camera(capture: OrbbecRgbdCapture | None) -> None:
    close_rgbd_camera(capture)
