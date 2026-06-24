# -*- coding: utf-8 -*-
"""Chessboard calibration for Hikvision MVS 2D cameras and pixel-to-mm conversion."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np

from object_measure import (
    min_area_rect_measure_mm,
    pixel_edge_len_mm,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CALIB_FILE = os.path.join(PROJECT_ROOT, "config", "camera_calib_2d.json")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FAST_CHECK
)


def _as_float(value) -> float:
    """Convert OpenCV/numpy scalar or 1-element array to Python float."""
    return float(np.asarray(value).reshape(-1)[0])


@dataclass
class CameraCalib2D:
    """Pinhole intrinsics plus working-plane depth for monocular mm conversion."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float]
    calib_width: int
    calib_height: int
    plane_z_mm: float
    pattern_cols: int = 11
    pattern_rows: int = 8
    square_size_mm: float = 1.5
    rms_reprojection_error: float = float("nan")
    image_count: int = 0

    @property
    def pattern_size(self) -> tuple[int, int]:
        return int(self.pattern_cols), int(self.pattern_rows)

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist_coeffs(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=np.float64).reshape(-1, 1)

    def scaled(self, image_width: int, image_height: int) -> "CameraCalib2D":
        if self.calib_width <= 0 or self.calib_height <= 0:
            return self
        sx = image_width / float(self.calib_width)
        sy = image_height / float(self.calib_height)
        return CameraCalib2D(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            dist=list(self.dist),
            calib_width=image_width,
            calib_height=image_height,
            plane_z_mm=self.plane_z_mm,
            pattern_cols=self.pattern_cols,
            pattern_rows=self.pattern_rows,
            square_size_mm=self.square_size_mm,
            rms_reprojection_error=self.rms_reprojection_error,
            image_count=self.image_count,
        )

    def pixel_edge_len_mm(
        self,
        p0: Sequence[float],
        p1: Sequence[float],
        z_mm: float | None = None,
    ) -> float:
        depth = self.plane_z_mm if z_mm is None else z_mm
        return pixel_edge_len_mm(
            np.asarray(p0, dtype=np.float64),
            np.asarray(p1, dtype=np.float64),
            float(depth),
            self.fx,
            self.fy,
        )

    def undistort(self, image_bgr: np.ndarray) -> np.ndarray:
        return cv2.undistort(
            image_bgr,
            self.camera_matrix,
            self.dist_coeffs,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CameraCalib2D":
        dist = payload.get("dist", [])
        if isinstance(dist, dict):
            dist = [float(dist[key]) for key in sorted(dist.keys(), key=lambda k: int(k))]
        return cls(
            fx=float(payload["fx"]),
            fy=float(payload["fy"]),
            cx=float(payload["cx"]),
            cy=float(payload["cy"]),
            dist=[float(v) for v in dist],
            calib_width=int(payload["calib_width"]),
            calib_height=int(payload["calib_height"]),
            plane_z_mm=float(payload["plane_z_mm"]),
            pattern_cols=int(payload.get("pattern_cols", 11)),
            pattern_rows=int(payload.get("pattern_rows", 8)),
            square_size_mm=float(payload.get("square_size_mm", 1.5)),
            rms_reprojection_error=float(payload.get("rms_reprojection_error", float("nan"))),
            image_count=int(payload.get("image_count", 0)),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = self.to_dict()
        payload["version"] = 1
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"Saved calibration: {path}")

    @classmethod
    def load(cls, path: str) -> "CameraCalib2D":
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)


def load_calib_or_none(path: str | None, *, required: bool = False) -> CameraCalib2D | None:
    if not path:
        return None
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(f"Calibration file not found: {path}")
        print(f"Warning: calibration file not found: {path} (mm labels disabled)")
        return None
    calib = CameraCalib2D.load(path)
    print(
        f"Loaded 2D calibration: fx={calib.fx:.2f} fy={calib.fy:.2f} "
        f"plane_z={calib.plane_z_mm:.1f}mm image={calib.calib_width}x{calib.calib_height}"
    )
    return calib


def resolve_calib_path(path: str | None) -> str:
    if path:
        return path
    return DEFAULT_CALIB_FILE


def list_calibration_images(images_dir: str) -> list[str]:
    paths: list[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
    return sorted(set(paths))


def _read_image_bgr(path: str) -> np.ndarray | None:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"  Skip unreadable image: {path}")
    return image


def _object_points(pattern_size: tuple[int, int], square_size_mm: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp[:, :2] = grid * float(square_size_mm)
    return objp


def find_chessboard_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    found, corners = cv2.findChessboardCorners(gray, pattern_size, CHESSBOARD_FLAGS)
    if not found or corners is None:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=criteria,
    )
    return True, corners


def detect_corners_in_image(
    image_bgr: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return find_chessboard_corners(gray, pattern_size)


def estimate_plane_z_mm(
    image_bgr: np.ndarray,
    calib: CameraCalib2D,
    pattern_size: tuple[int, int] | None = None,
    square_size_mm: float | None = None,
) -> float:
    pattern = pattern_size or calib.pattern_size
    square_mm = calib.square_size_mm if square_size_mm is None else float(square_size_mm)
    found, corners = detect_corners_in_image(image_bgr, pattern)
    if not found or corners is None:
        raise RuntimeError("Could not detect chessboard corners in plane reference image.")

    objp = _object_points(pattern, square_mm)
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        calib.camera_matrix,
        calib.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed for plane reference image.")
    return _as_float(abs(tvec[2]))


@dataclass
class CalibrationResult:
    calib: CameraCalib2D
    used_images: list[str] = field(default_factory=list)
    skipped_images: list[str] = field(default_factory=list)


def calibrate_from_images(
    image_paths: Iterable[str],
    *,
    pattern_cols: int,
    pattern_rows: int,
    square_size_mm: float,
    plane_z_mm: float | None = None,
    plane_image: str | None = None,
) -> CalibrationResult:
    pattern_size = (int(pattern_cols), int(pattern_rows))
    objp = _object_points(pattern_size, square_size_mm)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[str] = []
    skipped_images: list[str] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = _read_image_bgr(path)
        if image is None:
            skipped_images.append(path)
            continue

        height, width = image.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            print(f"  Skip size mismatch {width}x{height}: {path}")
            skipped_images.append(path)
            continue

        found, corners = detect_corners_in_image(image, pattern_size)
        if not found or corners is None:
            print(f"  No chessboard: {path}")
            skipped_images.append(path)
            continue

        object_points.append(objp.copy())
        image_points.append(corners)
        used_images.append(path)
        print(f"  Accepted: {os.path.basename(path)}")

    if len(used_images) < 3:
        raise RuntimeError(
            f"Need at least 3 valid calibration images, found {len(used_images)}."
        )
    if image_size is None:
        raise RuntimeError("No valid calibration images.")

    width, height = image_size
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        (width, height),
        None,
        None,
    )

    dist = [_as_float(v) for v in dist_coeffs.reshape(-1)]
    calib = CameraCalib2D(
        fx=_as_float(camera_matrix[0, 0]),
        fy=_as_float(camera_matrix[1, 1]),
        cx=_as_float(camera_matrix[0, 2]),
        cy=_as_float(camera_matrix[1, 2]),
        dist=dist,
        calib_width=width,
        calib_height=height,
        plane_z_mm=float("nan"),
        pattern_cols=pattern_cols,
        pattern_rows=pattern_rows,
        square_size_mm=float(square_size_mm),
        rms_reprojection_error=_as_float(rms),
        image_count=len(used_images),
    )

    if plane_z_mm is not None and plane_z_mm > 0:
        calib.plane_z_mm = float(plane_z_mm)
    elif plane_image:
        plane = _read_image_bgr(plane_image)
        if plane is None:
            raise RuntimeError(f"Could not read plane reference image: {plane_image}")
        calib.plane_z_mm = estimate_plane_z_mm(plane, calib, pattern_size, square_size_mm)
    else:
        z_values: list[float] = []
        for corners in image_points:
            ok, _rvec, tvec = cv2.solvePnP(
                objp,
                corners,
                calib.camera_matrix,
                calib.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                z_values.append(_as_float(abs(tvec[2])))
        if z_values:
            calib.plane_z_mm = float(np.median(z_values))
            print(
                "Warning: plane_z_mm estimated from calibration poses (median). "
                "For production, place the board on the measurement plane and pass --plane-image."
            )

    if not np.isfinite(calib.plane_z_mm) or calib.plane_z_mm <= 0:
        raise RuntimeError(
            "plane_z_mm is missing. Pass --plane-z-mm or --plane-image with the board on the work plane."
        )

    print(
        f"Calibration done: rms={calib.rms_reprojection_error:.4f}px, "
        f"fx={calib.fx:.2f}, fy={calib.fy:.2f}, plane_z={calib.plane_z_mm:.1f}mm, "
        f"images={calib.image_count}"
    )
    return CalibrationResult(calib=calib, used_images=used_images, skipped_images=skipped_images)


def measure_mask_mm_2d(
    mask: np.ndarray,
    calib: CameraCalib2D,
    z_mm: float | None = None,
):
    from object_measure import RotatedMeasure, largest_contour_from_mask

    contour = largest_contour_from_mask(mask)
    if contour is None:
        return None

    depth = calib.plane_z_mm if z_mm is None else float(z_mm)
    measured = min_area_rect_measure_mm(contour, depth, calib.fx, calib.fy)
    if measured is None:
        return None

    box, length_mm, width_mm, angle_deg = measured
    return RotatedMeasure(
        box_pts=box,
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=float("nan"),
        z_object_mm=depth,
        angle_deg=angle_deg,
    )


def attach_instance_metrics(instances, calib: CameraCalib2D, z_mm: float | None = None) -> None:
    for instance in instances:
        measured = measure_mask_mm_2d(instance.mask, calib, z_mm=z_mm)
        if measured is None:
            continue
        instance.box_pts = measured.box_pts
        instance.length_mm = measured.length_mm
        instance.width_mm = measured.width_mm
        instance.z_object_mm = measured.z_object_mm
        instance.angle_deg = measured.angle_deg


def water_cut_width_mm_2d(analysis, calib: CameraCalib2D, z_mm: float | None = None) -> float:
    return calib.pixel_edge_len_mm(analysis.width_end_a, analysis.width_end_b, z_mm=z_mm)


def capture_calibration_images(
    *,
    device_index: int,
    output_dir: str,
    count: int,
    interval_s: float,
    pixel_format: str | None,
    timeout_ms: int,
) -> list[str]:
    from capture_2d import close_camera, fetch_frame, list_devices, open_camera

    os.makedirs(output_dir, exist_ok=True)
    devices = list_devices()
    if device_index >= len(devices):
        raise RuntimeError(f"Invalid device index {device_index}, found {len(devices)} device(s).")

    camera = None
    saved: list[str] = []
    try:
        camera, payload_size, pixel_format_value = open_camera(
            devices[device_index]["info"],
            pixel_format,
        )
        print(f"Capturing {count} calibration frames to {output_dir} ...")
        for index in range(count):
            image, frame_info = fetch_frame(
                camera,
                payload_size,
                pixel_format_value,
                timeout_ms=timeout_ms,
                warmup_frames=1,
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(output_dir, f"calib_{index + 1:03d}_{timestamp}.png")
            cv2.imwrite(path, image)
            saved.append(path)
            print(
                f"  [{index + 1}/{count}] saved {os.path.basename(path)} "
                f"({frame_info.nWidth}x{frame_info.nHeight})"
            )
            if index + 1 < count and interval_s > 0:
                time.sleep(interval_s)
    finally:
        close_camera(camera)
    return saved


def validate_calibration(calib: CameraCalib2D, image_path: str, output_path: str | None) -> None:
    image = _read_image_bgr(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    found, corners = detect_corners_in_image(image, calib.pattern_size)
    if not found or corners is None:
        raise RuntimeError("Chessboard not detected in validation image.")

    undistorted = calib.undistort(image)
    vis = undistorted.copy()
    cv2.drawChessboardCorners(vis, calib.pattern_size, corners, found)

    objp = _object_points(calib.pattern_size, calib.square_size_mm)
    ok, _rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        calib.camera_matrix,
        calib.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed during validation.")

    plane_z = _as_float(abs(tvec[2]))
    cols, rows = calib.pattern_size
    p0 = corners[0, 0]
    p1 = corners[cols - 1, 0]
    edge_px = float(np.linalg.norm(p1 - p0))
    edge_mm = calib.pixel_edge_len_mm(p0, p1, z_mm=plane_z)
    expected_mm = (cols - 1) * calib.square_size_mm

    print(f"Validation image: {image_path}")
    print(f"  solvePnP plane_z={plane_z:.2f} mm (calib plane_z={calib.plane_z_mm:.2f} mm)")
    print(f"  top row: {edge_px:.1f}px -> {edge_mm:.2f} mm (expected {expected_mm:.2f} mm)")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        cv2.imwrite(output_path, vis)
        print(f"  saved overlay: {output_path}")


def _add_common_chessboard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pattern-cols", type=int, default=11, help="Inner corner columns.")
    parser.add_argument("--pattern-rows", type=int, default=8, help="Inner corner rows.")
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=1.5,
        help="Physical square size in millimeters.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate Hikvision 2D camera with chessboard images and save mm conversion params."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Run chessboard calibration from an image folder.",
    )
    calibrate_parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing chessboard calibration images.",
    )
    calibrate_parser.add_argument(
        "--output",
        default=DEFAULT_CALIB_FILE,
        help=f"Output calibration JSON (default: {DEFAULT_CALIB_FILE}).",
    )
    calibrate_parser.add_argument(
        "--plane-image",
        default=None,
        help="Optional image with the board flat on the measurement plane (for plane_z_mm).",
    )
    calibrate_parser.add_argument(
        "--plane-z-mm",
        type=float,
        default=None,
        help="Optional manual camera-to-work-plane distance in mm.",
    )
    _add_common_chessboard_args(calibrate_parser)

    capture_parser = subparsers.add_parser(
        "capture",
        help="Capture calibration images from a Hikvision MVS 2D camera.",
    )
    capture_parser.add_argument("--device-index", type=int, default=0)
    capture_parser.add_argument("--output-dir", default="calib_images")
    capture_parser.add_argument("--count", type=int, default=20)
    capture_parser.add_argument("--interval-s", type=float, default=1.0)
    capture_parser.add_argument("--timeout-ms", type=int, default=1000)
    capture_parser.add_argument("--pixel-format", default=None)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a calibration file against one chessboard image.",
    )
    validate_parser.add_argument("--calib", default=DEFAULT_CALIB_FILE)
    validate_parser.add_argument("--image", required=True)
    validate_parser.add_argument("--output", default=None, help="Optional annotated output image.")

    args = parser.parse_args()

    if args.command == "capture":
        saved = capture_calibration_images(
            device_index=args.device_index,
            output_dir=args.output_dir,
            count=max(1, int(args.count)),
            interval_s=max(0.0, float(args.interval_s)),
            pixel_format=args.pixel_format,
            timeout_ms=int(args.timeout_ms),
        )
        print(f"Captured {len(saved)} image(s). Next:")
        print(
            f"  python camera_calib_2d.py calibrate --images-dir {args.output_dir} "
            f"--pattern-cols 11 --pattern-rows 8 --square-size-mm 1.5"
        )
        return 0

    if args.command == "calibrate":
        image_paths = list_calibration_images(args.images_dir)
        if args.plane_image:
            plane_abs = os.path.abspath(args.plane_image)
            image_paths = [
                path
                for path in image_paths
                if os.path.abspath(path) != plane_abs
            ]
        if not image_paths:
            raise RuntimeError(f"No images found in {args.images_dir}")
        print(f"Found {len(image_paths)} image(s) in {args.images_dir}")
        result = calibrate_from_images(
            image_paths,
            pattern_cols=args.pattern_cols,
            pattern_rows=args.pattern_rows,
            square_size_mm=args.square_size_mm,
            plane_z_mm=args.plane_z_mm,
            plane_image=args.plane_image,
        )
        result.calib.save(args.output)
        if result.skipped_images:
            print(f"Skipped {len(result.skipped_images)} image(s).")
        return 0

    if args.command == "validate":
        calib = CameraCalib2D.load(args.calib)
        validate_calibration(calib, args.image, args.output)
        return 0

    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
