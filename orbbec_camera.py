#!/usr/bin/env python3
"""
Orbbec camera configuration loader and pipeline session.

If ``orbbec_camera.json`` (or ``config/orbbec_camera.json``) exists in the
project, import this module and call :func:`find_config_path` /
:func:`load_camera_config` / :class:`CameraSession` to start the camera with
fixed stream, alignment, depth and ROI settings.

Designed for downstream tasks such as segmentation-based object size estimation.

Lifecycle (strict):
  1. ``build()`` — select stream profiles only
  2. ``_prepare_before_stream()`` — apply hardware + freeze software runtime
  3. ``start()`` — lock configuration, then ``pipeline.start()``
  4. ``process_frames()`` — use frozen runtime only; no config changes
  5. ``stop()`` — unlock; call ``build()`` / ``start()`` again to reconfigure
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBAlignMode,
    OBDepthPrecisionLevel,
    OBError,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBPermissionType,
    OBPropertyID,
    OBSensorType,
    OBStreamType,
    Pipeline,
)

AlignModeName = Literal["hw_d2c", "sw_d2c", "none"]
DisparityToDepthMode = Literal["hardware", "software", "off"]
RoiModeName = Literal["center", "absolute"]
DepthColormapName = Literal["jet", "turbo", "viridis", "magma", "inferno", "bone", "ocean"]
DepthPreprocessName = Literal["dynamic", "fixed"]
FrameAggregateName = Literal["FULL_FRAME_REQUIRE", "COLOR_FRAME_REQUIRE", "ANY_SITUATION"]

_PROJECT_ROOT = Path(__file__).resolve().parent
_CONFIG_CANDIDATES = (
    _PROJECT_ROOT / "orbbec_camera.json",
    _PROJECT_ROOT / "config" / "orbbec_camera.json",
)

_ALIGN_MODE_MAP = {
    "hw_d2c": OBAlignMode.HW_MODE,
    "sw_d2c": OBAlignMode.SW_MODE,
    "none": OBAlignMode.DISABLE,
}

_FRAME_AGGREGATE_MAP = {
    "FULL_FRAME_REQUIRE": OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE,
    "COLOR_FRAME_REQUIRE": OBFrameAggregateOutputMode.COLOR_FRAME_REQUIRE,
    "ANY_SITUATION": OBFrameAggregateOutputMode.ANY_SITUATION,
}

_DEPTH_PRECISION_BY_MM: dict[float, OBDepthPrecisionLevel] = {
    1.0: OBDepthPrecisionLevel.ONE_MM,
    0.8: OBDepthPrecisionLevel.ZERO_POINT_EIGHT_MM,
    0.4: OBDepthPrecisionLevel.ZERO_POINT_FOUR_MM,
    0.2: OBDepthPrecisionLevel.ZERO_POINT_TWO_MM,
    0.1: OBDepthPrecisionLevel.ZERO_POINT_ONE_MM,
}

_DEPTH_COLORMAP_MAP: dict[str, int] = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "magma": cv2.COLORMAP_MAGMA,
    "inferno": cv2.COLORMAP_INFERNO,
    "bone": cv2.COLORMAP_BONE,
    "ocean": cv2.COLORMAP_OCEAN,
}

_COLOR_FORMAT_MAP = {
    "RGB": OBFormat.RGB,
    "BGR": OBFormat.BGR,
    "MJPG": OBFormat.MJPG,
    "YUYV": OBFormat.YUYV,
    "UYVY": OBFormat.UYVY,
}


def _resolve_color_format(name: str | None) -> OBFormat | None:
    if not name or name.upper() in ("AUTO", "DEFAULT", "ANY"):
        return None
    return _COLOR_FORMAT_MAP.get(name.upper())


@dataclass
class StreamConfig:
    color_width: int | None = None
    color_height: int | None = None
    depth_width: int | None = None
    depth_height: int | None = None
    color_fps: int | None = None
    depth_fps: int | None = None
    color_format: str = "auto"


SettingStatus = Literal[
    "applied",
    "verified",
    "mismatch",
    "read_na",
    "software",
    "skipped",
    "unsupported",
    "pending",
]


@dataclass
class SettingResult:
    name: str
    requested: Any
    status: SettingStatus
    detail: str = ""
    actual: Any | None = None


@dataclass
class AlignmentConfig:
    mode: AlignModeName = "sw_d2c"
    fallback_to_software: bool = True


@dataclass
class DepthConfig:
    disparity_to_depth_mode: DisparityToDepthMode = "hardware"
    depth_precision_mm: float = 0.4
    min_depth_mm: float = 20.0
    max_depth_mm: float = 5000.0
    hole_filter: bool = False


@dataclass
class VisualizationConfig:
    """Orbbec Viewer 'Visualization' panel — software depth preview only, not device properties."""

    depth_colormap: DepthColormapName = "jet"
    preprocess: DepthPreprocessName = "dynamic"
    relief_3d: bool = False
    histogram_equalization: bool = True
    visual_range_min_mm: float = 100.0
    visual_range_max_mm: float | None = None


@dataclass
class RoiRect:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)


@dataclass
class StreamLayout:
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    color_fps: int
    depth_fps: int


@dataclass
class FrozenRuntime:
    """Snapshot of software processing parameters, fixed before the first frame."""

    depth_min_mm: float
    depth_max_mm: float
    probe_radius: int
    roi_rect: RoiRect | None
    apply_software_roi: bool
    depth_colormap: int
    preprocess: DepthPreprocessName
    relief_3d: bool
    histogram_equalization: bool
    visual_range_min_mm: float
    visual_range_max_mm: float


@dataclass
class RoiConfig:
    enabled: bool = False
    mode: RoiModeName = "center"
    width: int = 640
    height: int = 480
    x0: int = 0
    y0: int = 0
    x1: int = 640
    y1: int = 480


@dataclass
class SyncConfig:
    frame_sync: bool = True
    frame_aggregate: FrameAggregateName = "FULL_FRAME_REQUIRE"


@dataclass
class MeasurementConfig:
    depth_sample_radius: int = 2
    units: str = "mm"


@dataclass
class DeviceToggles:
    apply_hardware_settings: bool = False
    laser_enable: bool | None = None


@dataclass
class CameraConfig:
    streams: StreamConfig = field(default_factory=StreamConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    device: DeviceToggles = field(default_factory=DeviceToggles)

    @property
    def uses_hardware_d2c(self) -> bool:
        return self.alignment.mode == "hw_d2c"

    @property
    def uses_software_align(self) -> bool:
        return self.alignment.mode == "sw_d2c"


def find_config_path() -> Path | None:
    """Return camera config path.

    Lookup order:
      1. ``ORBBEC_CAMERA_CONFIG=/path/to/file.json``
      2. ``orbbec_camera.json`` in the project (unless ``ORBBEC_USE_CAMERA_CONFIG=0``)
    """
    env_path = os.environ.get("ORBBEC_CAMERA_CONFIG")
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"ORBBEC_CAMERA_CONFIG not found: {env_path}")

    disable = os.environ.get("ORBBEC_USE_CAMERA_CONFIG", "").strip().lower()
    if disable in ("0", "false", "no", "off"):
        return None

    for candidate in _CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _strip_meta_keys(data: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in data.items():
        if str(key).startswith("_"):
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_meta_keys(value)
        else:
            cleaned[key] = value
    return cleaned


def _merge_dict(defaults: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(defaults)
    for key, value in _strip_meta_keys(overrides).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_depth_colormap(name: str) -> int:
    key = name.strip().lower()
    if key not in _DEPTH_COLORMAP_MAP:
        supported = ", ".join(sorted(_DEPTH_COLORMAP_MAP))
        raise ValueError(f"Unsupported depth colormap '{name}'. Supported: {supported}")
    return _DEPTH_COLORMAP_MAP[key]


def resolve_depth_precision_level(depth_precision_mm: float) -> OBDepthPrecisionLevel:
    value = round(float(depth_precision_mm), 2)
    precision = _DEPTH_PRECISION_BY_MM.get(value)
    if precision is None:
        supported = ", ".join(str(mm) for mm in sorted(_DEPTH_PRECISION_BY_MM))
        raise ValueError(
            f"Unsupported depth_precision_mm={depth_precision_mm}. "
            f"Supported on Gemini 2/215: {supported}"
        )
    return precision


def _visual_range_max_mm(cfg: CameraConfig) -> float:
    if cfg.visualization.visual_range_max_mm is not None:
        return float(cfg.visualization.visual_range_max_mm)
    return float(cfg.depth.max_depth_mm)


def render_depth_preview(depth_mm: np.ndarray, runtime: FrozenRuntime) -> np.ndarray:
    """Render a colorized depth preview (matches Orbbec Viewer Visualization panel)."""
    vmin = runtime.visual_range_min_mm
    vmax = runtime.visual_range_max_mm
    valid = (depth_mm > 0) & (depth_mm >= vmin) & (depth_mm <= vmax)

    if runtime.preprocess == "dynamic":
        depth_8bit = np.zeros(depth_mm.shape, dtype=np.uint8)
        if np.any(valid):
            lo = float(depth_mm[valid].min())
            hi = float(depth_mm[valid].max())
            if hi <= lo:
                hi = lo + 1.0
            depth_8bit[valid] = ((depth_mm[valid] - lo) / (hi - lo) * 255.0).astype(np.uint8)
    else:
        span = max(vmax - vmin, 1e-6)
        clipped = np.clip(depth_mm, vmin, vmax)
        depth_8bit = np.where(
            clipped > vmin,
            ((clipped - vmin) / span * 255.0).astype(np.uint8),
            np.uint8(0),
        )

    if runtime.histogram_equalization and np.any(depth_8bit > 0):
        depth_8bit = cv2.equalizeHist(depth_8bit)

    if runtime.relief_3d:
        depth_gamma = np.power(depth_8bit.astype(np.float32) / 255.0, 0.8)
        depth_8bit = (depth_gamma * 255.0).astype(np.uint8)
        grad_x = cv2.Scharr(depth_8bit, cv2.CV_32F, 1, 0)
        grad_y = cv2.Scharr(depth_8bit, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(grad_x, grad_y) + 1.0
        lighting = -0.707 * (grad_x + grad_y) / mag
        lighting = np.clip(lighting * 0.15 + 0.85, 0.7, 1.0)
        depth_colored = cv2.applyColorMap(depth_8bit, runtime.depth_colormap)
        return (depth_colored * lighting[..., np.newaxis]).astype(np.uint8)

    return cv2.applyColorMap(depth_8bit, runtime.depth_colormap)


def _normalize_depth_section(depth_data: dict[str, Any]) -> dict[str, Any]:
    depth = dict(depth_data)
    if "disparity_to_depth_mode" not in depth:
        if depth.pop("sdk_disparity_to_depth", False):
            depth["disparity_to_depth_mode"] = "software"
        elif depth.pop("disparity_to_depth", False):
            depth["disparity_to_depth_mode"] = "hardware"
        else:
            depth["disparity_to_depth_mode"] = "off"
    else:
        depth.pop("disparity_to_depth", None)
        depth.pop("sdk_disparity_to_depth", None)
    return depth


def uses_default_streams(cfg: CameraConfig) -> bool:
    streams = cfg.streams
    return (
        streams.color_width is None
        and streams.color_height is None
        and streams.depth_width is None
        and streams.depth_height is None
        and streams.color_fps is None
        and streams.depth_fps is None
        and _resolve_color_format(streams.color_format) is None
    )


def print_config_summary(cfg: CameraConfig, config_path: Path | None, align_mode: AlignModeName) -> None:
    print("\n" + "=" * 60)
    print("Orbbec camera config")
    if config_path is not None:
        print(f"  file      : {config_path}")
    print(f"  alignment : {align_mode} (requested: {cfg.alignment.mode})")
    print(
        f"  streams   : color={cfg.streams.color_width or 'default'}x"
        f"{cfg.streams.color_height or 'default'}@"
        f"{cfg.streams.color_fps or 'default'}fps fmt={cfg.streams.color_format} | "
        f"depth={cfg.streams.depth_width or 'default'}x"
        f"{cfg.streams.depth_height or 'default'}@"
        f"{cfg.streams.depth_fps or 'default'}fps"
    )
    print(
        f"  depth     : d2d={cfg.depth.disparity_to_depth_mode} | "
        f"unit={cfg.depth.depth_precision_mm}mm | "
        f"valid range {cfg.depth.min_depth_mm}-{cfg.depth.max_depth_mm} mm (software filter)"
    )
    print(
        f"  visualize : colormap={cfg.visualization.depth_colormap} | "
        f"preprocess={cfg.visualization.preprocess} | "
        f"histogram_eq={cfg.visualization.histogram_equalization} | "
        f"relief_3d={cfg.visualization.relief_3d} | "
        f"range={cfg.visualization.visual_range_min_mm}-"
        f"{cfg.visualization.visual_range_max_mm or cfg.depth.max_depth_mm}mm (Viewer panel, software only)"
    )
    print(
        f"  roi       : {'on' if cfg.roi.enabled else 'off'}"
        f"{f' ({cfg.roi.mode})' if cfg.roi.enabled else ''}"
    )
    print(
        f"  device    : hardware_override={cfg.device.apply_hardware_settings} | "
        f"laser={cfg.device.laser_enable}"
    )
    print("=" * 60 + "\n")


def print_setting_report(results: list[SettingResult]) -> None:
    if not results:
        return
    print("\n[orbbec_camera] Device / software setting report:")
    marks = {
        "verified": "OK ",
        "applied": "OK ",
        "mismatch": "FAIL",
        "read_na": "WRN ",
        "software": "SW ",
        "skipped": "SKIP",
        "unsupported": "N/A",
        "pending": "...",
    }
    counts = {"verified": 0, "mismatch": 0, "read_na": 0, "skipped": 0, "unsupported": 0}
    for item in results:
        mark = marks[item.status]
        if item.status in counts:
            counts[item.status] += 1
        detail = f" — {item.detail}" if item.detail else ""
        actual = ""
        if item.actual is not None and item.status in ("verified", "mismatch", "read_na"):
            actual = f" | actual={item.actual!r}"
        print(f"  [{mark}] {item.name}: requested={item.requested!r}{actual}{detail}")

    verified = counts["verified"]
    mismatches = counts["mismatch"]
    read_na = counts["read_na"]
    if verified or mismatches or read_na:
        print(
            f"\n[orbbec_camera] Hardware readback: "
            f"{verified} verified, {mismatches} mismatch, {read_na} write-only/unreadable"
        )
    print()


def _default_dict() -> dict[str, Any]:
    return {
        "streams": StreamConfig().__dict__,
        "alignment": AlignmentConfig().__dict__,
        "depth": DepthConfig().__dict__,
        "roi": RoiConfig().__dict__,
        "sync": SyncConfig().__dict__,
        "measurement": MeasurementConfig().__dict__,
        "visualization": VisualizationConfig().__dict__,
        "device": DeviceToggles().__dict__,
    }


def load_camera_config(path: str | Path | None = None) -> CameraConfig:
    """Load :class:`CameraConfig` from JSON. Uses :func:`find_config_path` when *path* is omitted."""
    config_path = Path(path).expanduser().resolve() if path else find_config_path()
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError("Orbbec camera config not found")

    with open(config_path, encoding="utf-8") as fh:
        raw = json.load(fh)

    data = _merge_dict(_default_dict(), raw)
    data["depth"] = _normalize_depth_section(data["depth"])
    return CameraConfig(
        streams=StreamConfig(**data["streams"]),
        alignment=AlignmentConfig(**data["alignment"]),
        depth=DepthConfig(**data["depth"]),
        roi=RoiConfig(**data["roi"]),
        sync=SyncConfig(**data["sync"]),
        measurement=MeasurementConfig(**data["measurement"]),
        visualization=VisualizationConfig(**data["visualization"]),
        device=DeviceToggles(**data["device"]),
    )


def _find_video_profile(
    profiles,
    width: int | None,
    height: int | None,
    fmt=None,
    fps: int | None = None,
):
    for index in range(len(profiles)):
        profile = profiles[index]
        if width is not None and profile.get_width() != width:
            continue
        if height is not None and profile.get_height() != height:
            continue
        if fmt is not None and profile.get_format() != fmt:
            continue
        if fps is not None and profile.get_fps() != fps:
            continue
        return profile
    return None


def _select_video_profile(
    profiles,
    width: int | None,
    height: int | None,
    fmt=None,
    fps: int | None = None,
):
    """Pick a stream profile by resolution/format/fps; falls back to device default."""
    if width is not None or height is not None or fmt is not None or fps is not None:
        try:
            return profiles.get_video_stream_profile(
                width or 0,
                height or 0,
                fmt if fmt is not None else OBFormat.UNKNOWN_FORMAT,
                fps or 0,
            )
        except OBError:
            profile = _find_video_profile(profiles, width, height, fmt, fps)
            if profile is not None:
                return profile
    return profiles.get_default_video_stream_profile()


def _layout_from_profiles(color_profile, depth_profile) -> StreamLayout:
    return StreamLayout(
        color_width=color_profile.get_width(),
        color_height=color_profile.get_height(),
        depth_width=depth_profile.get_width(),
        depth_height=depth_profile.get_height(),
        color_fps=color_profile.get_fps(),
        depth_fps=depth_profile.get_fps(),
    )


def _build_hw_d2c_config(pipeline: Pipeline, cfg: CameraConfig) -> tuple[Config, StreamLayout] | None:
    streams = cfg.streams
    preferred_fmt = _resolve_color_format(streams.color_format)
    format_candidates = [preferred_fmt] if preferred_fmt is not None else [
        OBFormat.RGB,
        OBFormat.MJPG,
        OBFormat.YUYV,
        OBFormat.UYVY,
        OBFormat.BGR,
        None,
    ]
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    seen: set[tuple[int, int, int]] = set()

    for color_fmt in format_candidates:
        for index in range(len(color_profiles)):
            color_profile = color_profiles[index]
            if color_fmt is not None and color_profile.get_format() != color_fmt:
                continue
            if streams.color_width and color_profile.get_width() != streams.color_width:
                continue
            if streams.color_height and color_profile.get_height() != streams.color_height:
                continue
            if streams.color_fps and color_profile.get_fps() != streams.color_fps:
                continue

            key = (
                color_profile.get_width(),
                color_profile.get_height(),
                int(color_profile.get_format()),
            )
            if key in seen:
                continue
            seen.add(key)

            hw_depth_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
            if len(hw_depth_list) == 0:
                continue

            depth_profile = hw_depth_list[0]
            if streams.depth_width and depth_profile.get_width() != streams.depth_width:
                continue
            if streams.depth_height and depth_profile.get_height() != streams.depth_height:
                continue
            if streams.depth_fps and depth_profile.get_fps() != streams.depth_fps:
                continue

            config.enable_stream(depth_profile)
            config.enable_stream(color_profile)
            config.set_align_mode(OBAlignMode.HW_MODE)
            print(
                f"[orbbec_camera] HW D2C color {color_profile.get_width()}x"
                f"{color_profile.get_height()}@{color_profile.get_fps()} "
                f"fmt={color_profile.get_format()}"
            )
            print(
                f"[orbbec_camera] HW D2C depth {depth_profile.get_width()}x"
                f"{depth_profile.get_height()}@{depth_profile.get_fps()}"
            )
    for color_fmt in format_candidates:
        for index in range(len(color_profiles)):
            color_profile = color_profiles[index]
            if color_fmt is not None and color_profile.get_format() != color_fmt:
                continue
            if streams.color_width and color_profile.get_width() != streams.color_width:
                continue
            if streams.color_height and color_profile.get_height() != streams.color_height:
                continue
            if streams.color_fps and color_profile.get_fps() != streams.color_fps:
                continue
            hw_depth_list = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
            if len(hw_depth_list) == 0:
                continue
            depth_profile = None
            for depth_index in range(len(hw_depth_list)):
                candidate = hw_depth_list[depth_index]
                if streams.depth_fps and candidate.get_fps() != streams.depth_fps:
                    continue
                depth_profile = candidate
                break
            if depth_profile is None:
                continue
            config.enable_stream(depth_profile)
            config.enable_stream(color_profile)
            config.set_align_mode(OBAlignMode.HW_MODE)
            print(
                f"[orbbec_camera] HW D2C color {color_profile.get_width()}x"
                f"{color_profile.get_height()}@{color_profile.get_fps()} "
                f"fmt={color_profile.get_format()}"
            )
            print(
                f"[orbbec_camera] HW D2C depth {depth_profile.get_width()}x"
                f"{depth_profile.get_height()}@{depth_profile.get_fps()} (relaxed match)"
            )
            return config, _layout_from_profiles(color_profile, depth_profile)

    return None


def _build_legacy_default_config(pipeline: Pipeline) -> tuple[Config, StreamLayout]:
    """Identical stream selection to color_depth_stream.build_pipeline()."""
    config = Config()
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profiles.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    print(f"[orbbec_camera] Color stream : {color_profile}")
    print(f"[orbbec_camera] Depth stream : {depth_profile}")
    return config, _layout_from_profiles(color_profile, depth_profile)


def _build_software_align_config(pipeline: Pipeline, cfg: CameraConfig) -> tuple[Config, StreamLayout]:
    """Default color+depth streams; AlignFilter handles D2C (no pipeline align mode)."""
    if uses_default_streams(cfg):
        return _build_legacy_default_config(pipeline)

    streams = cfg.streams
    color_fmt = _resolve_color_format(streams.color_format)
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    color_profile = _select_video_profile(
        color_profiles,
        streams.color_width,
        streams.color_height,
        color_fmt,
        streams.color_fps,
    )

    depth_profile = _select_video_profile(
        depth_profiles,
        streams.depth_width,
        streams.depth_height,
        None,
        streams.depth_fps,
    )

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    # Match color_depth_stream.py: do not call set_align_mode; use AlignFilter instead.
    print(
        f"[orbbec_camera] Color {color_profile.get_width()}x{color_profile.get_height()}@"
        f"{color_profile.get_fps()} fmt={color_profile.get_format()}"
    )
    print(
        f"[orbbec_camera] Depth {depth_profile.get_width()}x{depth_profile.get_height()}@"
        f"{depth_profile.get_fps()} (software AlignFilter)"
    )
    return config, _layout_from_profiles(color_profile, depth_profile)


def _build_standard_config(
    pipeline: Pipeline, cfg: CameraConfig, align_mode: OBAlignMode
) -> tuple[Config, StreamLayout]:
    streams = cfg.streams
    color_fmt = _resolve_color_format(streams.color_format)
    config = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

    color_profile = _select_video_profile(
        color_profiles,
        streams.color_width,
        streams.color_height,
        color_fmt,
        streams.color_fps,
    )

    depth_profile = _select_video_profile(
        depth_profiles,
        streams.depth_width,
        streams.depth_height,
        None,
        streams.depth_fps,
    )

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_align_mode(align_mode)
    print(
        f"[orbbec_camera] Color {color_profile.get_width()}x{color_profile.get_height()}@"
        f"{color_profile.get_fps()} fmt={color_profile.get_format()}"
    )
    print(
        f"[orbbec_camera] Depth {depth_profile.get_width()}x{depth_profile.get_height()}@"
        f"{depth_profile.get_fps()} align={align_mode}"
    )
    return config, _layout_from_profiles(color_profile, depth_profile)


def build_pipeline_config(
    pipeline: Pipeline, cfg: CameraConfig
) -> tuple[Config, AlignModeName, StreamLayout]:
    """Build a :class:`Config` and return the alignment mode actually selected."""
    aggregate = _FRAME_AGGREGATE_MAP.get(cfg.sync.frame_aggregate)
    if aggregate is None:
        raise ValueError(f"Unsupported frame_aggregate: {cfg.sync.frame_aggregate}")

    selected_mode: AlignModeName = cfg.alignment.mode

    if cfg.alignment.mode == "hw_d2c":
        built = _build_hw_d2c_config(pipeline, cfg)
        if built is None:
            if cfg.alignment.fallback_to_software:
                print("[orbbec_camera] Hardware D2C unavailable, falling back to software alignment.")
                selected_mode = "sw_d2c"
                config, layout = _build_software_align_config(pipeline, cfg)
            else:
                raise RuntimeError("Hardware D2C is not supported for the requested stream profiles.")
        else:
            config, layout = built
    elif cfg.alignment.mode == "sw_d2c":
        config, layout = _build_software_align_config(pipeline, cfg)
    else:
        config, layout = _build_standard_config(pipeline, cfg, OBAlignMode.DISABLE)
        selected_mode = "none"

    config.set_frame_aggregate_output_mode(aggregate)
    return config, selected_mode, layout


def resolve_roi_rect(image_width: int, image_height: int, roi: RoiConfig) -> RoiRect:
    if roi.mode == "center":
        crop_w = min(roi.width, image_width)
        crop_h = min(roi.height, image_height)
        x0 = (image_width - crop_w) // 2
        y0 = (image_height - crop_h) // 2
        return RoiRect(x0=x0, y0=y0, x1=x0 + crop_w, y1=y0 + crop_h)

    x0 = max(0, min(roi.x0, image_width))
    y0 = max(0, min(roi.y0, image_height))
    x1 = max(x0, min(roi.x1, image_width))
    y1 = max(y0, min(roi.y1, image_height))
    return RoiRect(x0=x0, y0=y0, x1=x1, y1=y1)


def apply_roi_to_array(image: np.ndarray, roi_rect: RoiRect) -> np.ndarray:
    y_slice, x_slice = roi_rect.as_slices()
    return image[y_slice, x_slice]


def adjust_intrinsics_for_roi(intrinsic, roi_rect: RoiRect):
    """Return a shallow copy of intrinsics with ROI crop offsets applied."""
    adjusted = copy.copy(intrinsic)
    adjusted.cx = intrinsic.cx - roi_rect.x0
    adjusted.cy = intrinsic.cy - roi_rect.y0
    adjusted.width = roi_rect.width
    adjusted.height = roi_rect.height
    return adjusted


def pixel_extent_to_mm(
    width_px: float,
    height_px: float,
    depth_mm: float,
    fx: float,
    fy: float,
) -> tuple[float, float]:
    """Convert pixel extents at a given depth to physical width/height in millimeters."""
    if depth_mm <= 0 or fx <= 0 or fy <= 0:
        return 0.0, 0.0
    width_mm = width_px * depth_mm / fx
    height_mm = height_px * depth_mm / fy
    return width_mm, height_mm


def frame_to_depth_mm(depth_frame) -> np.ndarray:
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    return depth_raw.astype(np.float32) * scale


def frame_to_bgr_image(color_frame) -> np.ndarray | None:
    width = color_frame.get_width()
    height = color_frame.get_height()
    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        return np.resize(data, (height, width, 3))
    if color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    print(f"[orbbec_camera] Unsupported color format: {color_format}")
    return None


def sample_depth_mm(
    depth_mm: np.ndarray,
    u: int,
    v: int,
    radius: int,
    min_depth_mm: float,
    max_depth_mm: float,
) -> float | None:
    height, width = depth_mm.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return None

    u0 = max(0, u - radius)
    u1 = min(width, u + radius + 1)
    v0 = max(0, v - radius)
    v1 = min(height, v + radius + 1)
    patch = depth_mm[v0:v1, u0:u1]
    valid = patch[(patch >= min_depth_mm) & (patch <= max_depth_mm)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _can_read_property(device, prop: OBPropertyID) -> bool:
    return device.is_property_supported(prop, OBPermissionType.PERMISSION_READ) or device.is_property_supported(
        prop, OBPermissionType.PERMISSION_READ_WRITE
    )


def _write_bool_with_verify(
    device,
    prop: OBPropertyID,
    value: bool,
    label: str,
    results: list[SettingResult],
) -> None:
    if not device.is_property_supported(prop, OBPermissionType.PERMISSION_WRITE) and not device.is_property_supported(
        prop, OBPermissionType.PERMISSION_READ_WRITE
    ):
        results.append(
            SettingResult(label, value, "unsupported", "device property not writable on this model")
        )
        return

    try:
        device.set_bool_property(prop, value)
    except OBError as exc:
        results.append(SettingResult(label, value, "mismatch", f"write failed: {exc}"))
        return

    if not _can_read_property(device, prop):
        results.append(
            SettingResult(
                label,
                value,
                "read_na",
                "write succeeded; property is not readable on this device",
            )
        )
        return

    try:
        actual = bool(device.get_bool_property(prop))
    except OBError as exc:
        results.append(
            SettingResult(label, value, "read_na", f"write ok, readback failed: {exc}")
        )
        return

    if actual == value:
        results.append(
            SettingResult(label, value, "verified", "hardware readback matches", actual=actual)
        )
    else:
        results.append(
            SettingResult(
                label,
                value,
                "mismatch",
                "hardware readback differs from requested value",
                actual=actual,
            )
        )


def _write_int_with_verify(
    device,
    prop: OBPropertyID,
    value: int,
    label: str,
    results: list[SettingResult],
) -> None:
    if not device.is_property_supported(prop, OBPermissionType.PERMISSION_WRITE) and not device.is_property_supported(
        prop, OBPermissionType.PERMISSION_READ_WRITE
    ):
        results.append(
            SettingResult(label, value, "unsupported", "device property not writable on this model")
        )
        return

    try:
        device.set_int_property(prop, value)
    except OBError as exc:
        results.append(SettingResult(label, value, "mismatch", f"write failed: {exc}"))
        return

    if not _can_read_property(device, prop):
        results.append(
            SettingResult(
                label,
                value,
                "read_na",
                "write succeeded; property is not readable on this device",
            )
        )
        return

    try:
        actual = int(device.get_int_property(prop))
    except OBError as exc:
        results.append(
            SettingResult(label, value, "read_na", f"write ok, readback failed: {exc}")
        )
        return

    if actual == value:
        results.append(
            SettingResult(label, value, "verified", "hardware readback matches", actual=actual)
        )
    else:
        results.append(
            SettingResult(
                label,
                value,
                "mismatch",
                "hardware readback differs from requested value",
                actual=actual,
            )
        )


def verify_pipeline_streams(config: Config, layout: StreamLayout) -> list[SettingResult]:
    """Verify enabled pipeline stream profiles match the layout selected at build time."""
    results: list[SettingResult] = []
    try:
        profiles = config.get_enabled_stream_profile_list()
        count = profiles.get_count()
    except OBError as exc:
        results.append(
            SettingResult(
                "pipeline_streams",
                layout,
                "read_na",
                f"could not read enabled stream profiles: {exc}",
            )
        )
        return results

    color_actual: tuple[int, int, int] | None = None
    depth_actual: tuple[int, int, int] | None = None
    for index in range(count):
        profile = profiles.get_stream_profile_by_index(index)
        if not profile.is_video_stream_profile():
            continue
        video = profile.as_video_stream_profile()
        spec = (video.get_width(), video.get_height(), video.get_fps())
        stream_type = profile.get_type()
        if stream_type == OBStreamType.COLOR_STREAM:
            color_actual = spec
        elif stream_type == OBStreamType.DEPTH_STREAM:
            depth_actual = spec

    expected_color = (layout.color_width, layout.color_height, layout.color_fps)
    expected_depth = (layout.depth_width, layout.depth_height, layout.depth_fps)

    if color_actual == expected_color:
        results.append(
            SettingResult(
                "pipeline.color_stream",
                expected_color,
                "verified",
                "enabled color profile matches build layout (wxh@fps)",
                actual=color_actual,
            )
        )
    else:
        results.append(
            SettingResult(
                "pipeline.color_stream",
                expected_color,
                "mismatch",
                "enabled color profile differs from build layout (wxh@fps)",
                actual=color_actual,
            )
        )

    if depth_actual == expected_depth:
        results.append(
            SettingResult(
                "pipeline.depth_stream",
                expected_depth,
                "verified",
                "enabled depth profile matches build layout (wxh@fps)",
                actual=depth_actual,
            )
        )
    else:
        results.append(
            SettingResult(
                "pipeline.depth_stream",
                expected_depth,
                "mismatch",
                "enabled depth profile differs from build layout (wxh@fps)",
                actual=depth_actual,
            )
        )
    return results


def software_setting_results(cfg: CameraConfig, roi_rect: RoiRect | None = None) -> list[SettingResult]:
    """Software parameters frozen before streaming starts."""
    results = [
        SettingResult(
            "depth_valid_range_mm",
            (cfg.depth.min_depth_mm, cfg.depth.max_depth_mm),
            "software",
            "frozen before stream; applied to each received frame",
        ),
        SettingResult(
            "alignment",
            cfg.alignment.mode,
            "software",
            "AlignFilter prepared before stream when mode is sw_d2c",
        ),
        SettingResult(
            "measurement.depth_sample_radius",
            cfg.measurement.depth_sample_radius,
            "software",
            "frozen before stream for click / mask depth sampling",
        ),
        SettingResult(
            "visualization.depth_colormap",
            cfg.visualization.depth_colormap,
            "software",
            "OpenCV colormap for depth preview (Orbbec Viewer Visualization)",
        ),
        SettingResult(
            "visualization.preprocess",
            cfg.visualization.preprocess,
            "software",
            "dynamic=per-frame auto range; fixed=use visual_range_min/max (Viewer Preprocess)",
        ),
        SettingResult(
            "visualization.histogram_equalization",
            cfg.visualization.histogram_equalization,
            "software",
            "Orbbec Viewer Histogram Equalization toggle",
        ),
        SettingResult(
            "visualization.relief_3d",
            cfg.visualization.relief_3d,
            "software",
            "Orbbec Viewer 3D Relief Effect toggle",
        ),
        SettingResult(
            "visualization.visual_range_mm",
            (cfg.visualization.visual_range_min_mm, _visual_range_max_mm(cfg)),
            "software",
            "Orbbec Viewer Visual Range sliders (used when preprocess=fixed)",
        ),
    ]
    if cfg.roi.enabled:
        results.append(
            SettingResult(
                "roi_crop",
                roi_rect or cfg.roi,
                "software",
                "resolved and frozen before stream",
            )
        )
    return results


def apply_device_settings(
    device, cfg: CameraConfig, roi_rect: RoiRect | None = None
) -> list[SettingResult]:
    """Write hardware properties. Must run only before pipeline.start()."""
    depth = cfg.depth
    toggles = cfg.device
    results: list[SettingResult] = list(software_setting_results(cfg, roi_rect))

    if not toggles.apply_hardware_settings:
        results.append(
            SettingResult(
                "hardware_device_properties",
                False,
                "skipped",
                "firmware defaults kept (enabling mid-stream can freeze depth on some models)",
            )
        )
        return results

    d2d_mode = depth.disparity_to_depth_mode
    if d2d_mode == "hardware":
        _write_bool_with_verify(
            device,
            OBPropertyID.OB_PROP_DISPARITY_TO_DEPTH_BOOL,
            True,
            "hardware_disparity_to_depth",
            results,
        )
        _write_bool_with_verify(
            device,
            OBPropertyID.OB_PROP_SDK_DISPARITY_TO_DEPTH_BOOL,
            False,
            "sdk_disparity_to_depth",
            results,
        )
    elif d2d_mode == "software":
        _write_bool_with_verify(
            device,
            OBPropertyID.OB_PROP_DISPARITY_TO_DEPTH_BOOL,
            False,
            "hardware_disparity_to_depth",
            results,
        )
        _write_bool_with_verify(
            device,
            OBPropertyID.OB_PROP_SDK_DISPARITY_TO_DEPTH_BOOL,
            True,
            "sdk_disparity_to_depth",
            results,
        )
    else:
        _write_bool_with_verify(
            device,
            OBPropertyID.OB_PROP_DISPARITY_TO_DEPTH_BOOL,
            False,
            "hardware_disparity_to_depth",
            results,
        )
        _write_bool_with_verify(
            device,
            OBPropertyID.OB_PROP_SDK_DISPARITY_TO_DEPTH_BOOL,
            False,
            "sdk_disparity_to_depth",
            results,
        )

    precision = resolve_depth_precision_level(depth.depth_precision_mm)
    _write_int_with_verify(
        device,
        OBPropertyID.OB_PROP_DEPTH_PRECISION_LEVEL_INT,
        int(precision),
        f"depth_precision_level ({depth.depth_precision_mm}mm)",
        results,
    )

    if depth.hole_filter:
        _write_bool_with_verify(
            device,
            OBPropertyID.OB_PROP_DEPTH_HOLEFILTER_BOOL,
            True,
            "depth_hole_filter",
            results,
        )

    if toggles.laser_enable is not None:
        _write_bool_with_verify(
            device, OBPropertyID.OB_PROP_LASER_BOOL, toggles.laser_enable, "laser", results
        )
    else:
        results.append(SettingResult("laser", None, "skipped", "not set in config"))

    return results


@dataclass
class FrameData:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    color_frame: Any
    depth_frame: Any
    roi_rect: RoiRect | None = None


class CameraSession:
    """Configured Orbbec RGB-D session with optional ROI and alignment helpers."""

    def __init__(
        self,
        cfg: CameraConfig,
        pipeline: Pipeline | None = None,
        config_path: Path | None = None,
    ):
        self.cfg = cfg
        self.config_path = config_path
        self.pipeline = pipeline or Pipeline()
        self.config: Config | None = None
        self.align_filter: AlignFilter | None = None
        self.align_mode: AlignModeName = cfg.alignment.mode
        self._stream_layout: StreamLayout | None = None
        self._runtime: FrozenRuntime | None = None
        self._roi_rect: RoiRect | None = None
        self._camera_param = None
        self._started = False
        self._config_locked = False
        self.setting_results: list[SettingResult] = []

    def _assert_config_mutable(self, action: str) -> None:
        if self._config_locked:
            raise RuntimeError(
                f"Cannot {action} while streaming. Call stop() first, then reconfigure and start()."
            )

    def build(
        self,
        align_mode: AlignModeName | None = None,
        use_device_defaults: bool = False,
    ) -> None:
        self._assert_config_mutable("rebuild pipeline")
        cfg = copy.deepcopy(self.cfg)
        if align_mode is not None:
            cfg.alignment.mode = align_mode
        if use_device_defaults:
            cfg.streams.color_format = "auto"
            cfg.streams.color_width = None
            cfg.streams.color_height = None
            cfg.streams.depth_width = None
            cfg.streams.depth_height = None
        self.config, self.align_mode, self._stream_layout = build_pipeline_config(self.pipeline, cfg)
        self._runtime = None
        if self.align_mode == "sw_d2c":
            self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        else:
            self.align_filter = None

        if self.cfg.sync.frame_sync:
            try:
                self.pipeline.enable_frame_sync()
            except OBError as exc:
                print(f"[orbbec_camera] Frame sync warning: {exc}")

        print_config_summary(self.cfg, self.config_path, self.align_mode)

    def _restart_pipeline(self) -> None:
        import time

        if self._started:
            self.pipeline.stop()
            self._started = False
            self._config_locked = False
            self._runtime = None
            time.sleep(0.5)

    def _fallback_to_software(self) -> None:
        import time

        print("[orbbec_camera] Switching to software alignment (device default streams).")
        self._restart_pipeline()
        time.sleep(1.0)
        self.pipeline = Pipeline()
        self.config = None
        self._stream_layout = None
        self.build(align_mode="sw_d2c", use_device_defaults=True)
        assert self.config is not None
        self._prepare_before_stream()
        aggregate = _FRAME_AGGREGATE_MAP.get(self.cfg.sync.frame_aggregate)
        if aggregate is not None:
            self.config.set_frame_aggregate_output_mode(aggregate)
        self.pipeline.start(self.config)
        self._started = True
        self._config_locked = True
        self.align_mode = "sw_d2c"
        self._prime_camera_param()

    def _resolve_roi_rect(self) -> RoiRect | None:
        if not self.cfg.roi.enabled or self._stream_layout is None:
            return None

        layout = self._stream_layout
        return resolve_roi_rect(layout.color_width, layout.color_height, self.cfg.roi)

    def _prepare_before_stream(self) -> None:
        """Apply hardware settings and freeze software runtime before the first frame."""
        if self._runtime is not None:
            return

        roi_rect = self._resolve_roi_rect()
        self._roi_rect = roi_rect

        if self.cfg.device.apply_hardware_settings:
            device = self.pipeline.get_device()
            self.setting_results = apply_device_settings(device, self.cfg, roi_rect)
        else:
            self.setting_results = software_setting_results(self.cfg, roi_rect)
            self.setting_results.append(
                SettingResult(
                    "hardware_device_properties",
                    False,
                    "skipped",
                    "firmware defaults kept; only pre-stream software settings are active",
                )
            )

        self._runtime = FrozenRuntime(
            depth_min_mm=self.cfg.depth.min_depth_mm,
            depth_max_mm=self.cfg.depth.max_depth_mm,
            probe_radius=self.cfg.measurement.depth_sample_radius,
            roi_rect=roi_rect,
            apply_software_roi=self.cfg.roi.enabled,
            depth_colormap=resolve_depth_colormap(self.cfg.visualization.depth_colormap),
            preprocess=self.cfg.visualization.preprocess,
            relief_3d=self.cfg.visualization.relief_3d,
            histogram_equalization=self.cfg.visualization.histogram_equalization,
            visual_range_min_mm=self.cfg.visualization.visual_range_min_mm,
            visual_range_max_mm=_visual_range_max_mm(self.cfg),
        )

        if self.config is not None and self._stream_layout is not None:
            self.setting_results.extend(verify_pipeline_streams(self.config, self._stream_layout))

        print_setting_report(self.setting_results)
        print("[orbbec_camera] Configuration frozen. Starting stream.\n")

    def start(self) -> None:
        if self._config_locked:
            raise RuntimeError("Session already streaming.")

        if self.config is None:
            self.build()

        assert self.config is not None
        self._prepare_before_stream()
        self.pipeline.start(self.config)
        self._started = True
        self._config_locked = True

        if self.cfg.alignment.mode == "hw_d2c":
            frames = self.pipeline.wait_for_frames(3000)
            ok = (
                frames is not None
                and frames.get_color_frame() is not None
                and frames.get_depth_frame() is not None
            )
            if not ok and self.cfg.alignment.fallback_to_software:
                self._fallback_to_software()
                return

        self._prime_camera_param()

    def _filter_depth_mm(self, depth_mm: np.ndarray) -> np.ndarray:
        runtime = self._runtime
        assert runtime is not None
        return np.where(
            (depth_mm >= runtime.depth_min_mm) & (depth_mm <= runtime.depth_max_mm),
            depth_mm,
            0.0,
        )

    def stop(self) -> None:
        if self._started:
            self.pipeline.stop()
        self._started = False
        self._config_locked = False
        self._runtime = None

    def _prime_camera_param(self) -> None:
        for _ in range(30):
            frames = self.pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            try:
                self._camera_param = self.pipeline.get_camera_param()
                return
            except OBError:
                continue

    def get_camera_param(self):
        if self._camera_param is None:
            self._prime_camera_param()
        return self._camera_param

    def get_rgb_intrinsics(self, roi_adjusted: bool = True):
        param = self.get_camera_param()
        if param is None:
            return None
        intrinsic = param.rgb_intrinsic
        if roi_adjusted and self._roi_rect is not None and self.cfg.roi.enabled:
            return adjust_intrinsics_for_roi(intrinsic, self._roi_rect)
        return intrinsic

    def wait_for_frames(self, timeout_ms: int = 1000):
        return self.pipeline.wait_for_frames(timeout_ms)

    def process_frames(self, frames) -> FrameData | None:
        if not self._config_locked or self._runtime is None:
            raise RuntimeError("process_frames() called before start(); configuration is not frozen yet.")

        if frames is None:
            return None

        runtime = self._runtime
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None

        if self.align_filter is not None:
            aligned = self.align_filter.process(frames)
            if aligned is None:
                return None
            color_frame = aligned.get_color_frame() or color_frame
            depth_frame = aligned.get_depth_frame() or depth_frame

        color_bgr = frame_to_bgr_image(color_frame)
        if color_bgr is None:
            return None

        depth_mm = self._filter_depth_mm(frame_to_depth_mm(depth_frame))
        if color_bgr.shape[:2] != depth_mm.shape[:2]:
            depth_mm = cv2.resize(
                depth_mm,
                (color_bgr.shape[1], color_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if runtime.apply_software_roi and runtime.roi_rect is not None:
            color_bgr = apply_roi_to_array(color_bgr, runtime.roi_rect)
            depth_mm = apply_roi_to_array(depth_mm, runtime.roi_rect)

        return FrameData(
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            color_frame=color_frame,
            depth_frame=depth_frame,
            roi_rect=runtime.roi_rect,
        )

    def sample_depth(self, depth_mm: np.ndarray, u: int, v: int) -> float | None:
        runtime = self._runtime
        if runtime is None:
            return None
        return sample_depth_mm(
            depth_mm,
            u,
            v,
            runtime.probe_radius,
            runtime.depth_min_mm,
            runtime.depth_max_mm,
        )

    def estimate_mask_size_mm(
        self,
        mask: np.ndarray,
        depth_mm: np.ndarray,
        depth_at: Literal["median", "center"] = "median",
    ) -> dict[str, float] | None:
        """Estimate physical size of a binary mask using RGB intrinsics."""
        if not np.any(mask):
            return None

        intrinsic = self.get_rgb_intrinsics(roi_adjusted=True)
        if intrinsic is None:
            return None

        ys, xs = np.where(mask)
        width_px = float(xs.max() - xs.min() + 1)
        height_px = float(ys.max() - ys.min() + 1)

        if depth_at == "center":
            cy = int((ys.min() + ys.max()) / 2)
            cx = int((xs.min() + xs.max()) / 2)
            runtime = self._runtime
            if runtime is None:
                return None
            depth_value = sample_depth_mm(
                depth_mm,
                cx,
                cy,
                runtime.probe_radius,
                runtime.depth_min_mm,
                runtime.depth_max_mm,
            )
        else:
            runtime = self._runtime
            if runtime is None:
                return None
            values = depth_mm[mask]
            valid = values[
                (values >= runtime.depth_min_mm) & (values <= runtime.depth_max_mm)
            ]
            if valid.size == 0:
                return None
            depth_value = float(np.median(valid))

        if depth_value is None or depth_value <= 0:
            return None

        width_mm, height_mm = pixel_extent_to_mm(
            width_px, height_px, depth_value, intrinsic.fx, intrinsic.fy
        )
        return {
            "width_mm": width_mm,
            "height_mm": height_mm,
            "depth_mm": depth_value,
            "width_px": width_px,
            "height_px": height_px,
        }


def try_open_camera_session(
    config_path: str | Path | None = None,
) -> CameraSession | None:
    """Load config and start a session. Returns ``None`` if no config file exists."""
    path = Path(config_path).expanduser().resolve() if config_path else find_config_path()
    if path is None or not path.is_file():
        return None

    cfg = load_camera_config(path)
    session = CameraSession(cfg, config_path=path)
    session.build()
    session.start()
    return session


def create_session_from_discovered_config(pipeline: Pipeline | None = None) -> CameraSession | None:
    """Create a :class:`CameraSession` when a config file exists; otherwise return ``None``."""
    path = find_config_path()
    if path is None:
        return None
    cfg = load_camera_config(path)
    session = CameraSession(cfg, pipeline=pipeline, config_path=path)
    session.build()
    return session
