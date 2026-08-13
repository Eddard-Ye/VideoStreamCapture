# -*- coding: utf-8 -*-
"""Stream Orbbec Gemini RGB-D frames over HTTP with live YOLO and remote water-cut trigger."""

from __future__ import annotations

import argparse
import copy
import sys
import time

import numpy as np

from stream_common import (
    attach_oriented_boxes,
    build_status,
    encode_jpeg,
)
from color_viewer import RoiRect, YoloSegmenter, keep_top_confidence_instances, resolve_roi
from orbbec_color import close_rgbd_camera, list_devices, open_rgbd_camera
from orbbec_metrics import (
    attach_orbbec_instance_metrics,
    compute_orbbec_water_cut_overlays,
    process_orbbec_capture_request,
)
from stream_overlay import compose_stream_frame
from stream_server import StreamHub, start_stream_server
from track_smoother import TrackSmoother
from yolo_sam_refine import SamRefiner

# Default stream overlay uses LABEL_SIZE_FACTOR=2.0; Orbbec live view uses smaller blue labels.
ORBBEC_LABEL_SIZE_FACTOR = 0.9


def _resolve_stream_roi(
    width: int,
    height: int,
    *,
    roi_spec: str | None,
    roi_file: str | None,
) -> RoiRect:
    if roi_spec or roi_file:
        return resolve_roi(width, height, roi_spec=roi_spec, roi_file=roi_file)
    try:
        return resolve_roi(width, height)
    except (ValueError, FileNotFoundError):
        return RoiRect(0, 0, width, height)


def _fetch_rgbd(camera, args: argparse.Namespace):
    if args.latest_frame:
        return camera.fetch_latest_rgbd(timeout_ms=args.timeout_ms)
    return camera.fetch_rgbd(timeout_ms=args.timeout_ms, warmup_frames=0)


def _yolo_detect(
    segmenter: YoloSegmenter,
    image: np.ndarray,
    roi: RoiRect,
    *,
    imgsz: int,
) -> list:
    refine_top_n = 1 if segmenter.mask_refine == "otsu" else None
    return segmenter.segment_all(
        image,
        roi,
        imgsz=imgsz,
        refine_top_n=refine_top_n,
    )


def _finalize_output_instances(instances: list) -> list:
    """Keep a single highest-confidence detection for metrics and display."""
    return keep_top_confidence_instances(instances, max_count=1)


def _prepare_output_instances(
    detected: list,
    depth_mm: np.ndarray,
    intrinsics,
    smoother: TrackSmoother | None,
) -> tuple[list, list]:
    """Select top-1, attach depth metrics, then temporal smooth that output only."""
    raw_instances = _finalize_output_instances(detected)
    _attach_metrics(raw_instances, depth_mm, intrinsics)
    display_instances = (
        smoother.update(raw_instances) if smoother is not None else raw_instances
    )
    return raw_instances, display_instances


def _print_height_metrics(instances: list) -> None:
    for index, instance in enumerate(instances):
        height_s = (
            f"{instance.height_mm:.1f}mm"
            if np.isfinite(instance.height_mm)
            else "---"
        )
        peak_s = (
            f"{instance.peak_height_mm:.1f}mm"
            if np.isfinite(instance.peak_height_mm)
            else "---"
        )
        print(f"  [{index}] height_mm={height_s}  peak_height_mm={peak_s}")


def _attach_metrics(
    instances: list,
    depth_mm: np.ndarray,
    intrinsics,
) -> None:
    attach_oriented_boxes(instances)
    attach_orbbec_instance_metrics(
        instances,
        depth_mm,
        intrinsics,
    )


def run_stream(args: argparse.Namespace) -> int:
    print("Scanning for Orbbec devices...")
    devices = list_devices()
    if not devices:
        raise RuntimeError("No Orbbec device found. Connect Gemini 215 and retry.")
    for device in devices:
        print(
            f"  [{device.index}] {device.name}  "
            f"SN={device.serial}  FW={device.firmware}"
        )

    if args.device_index >= len(devices):
        raise RuntimeError(f"Invalid device index {args.device_index}, found {len(devices)} device(s).")

    camera = None
    server = None
    try:
        camera = open_rgbd_camera(
            config_path=args.orbbec_config,
            device_index=args.device_index,
            use_config=None if args.no_orbbec_config else True,
        )
        print(f"Orbbec capture mode: {camera.mode}")

        intrinsics = camera.get_rgb_intrinsics()
        if intrinsics is None:
            raise RuntimeError("Failed to read Orbbec RGB intrinsics from device.")

        rgbd = camera.fetch_rgbd(
            timeout_ms=args.timeout_ms,
            warmup_frames=args.warmup_frames,
        )
        image = rgbd.color_bgr
        height, width = image.shape[:2]
        intrinsics = intrinsics.scaled(width, height)
        roi = _resolve_stream_roi(width, height, roi_spec=args.roi, roi_file=args.roi_file)
        print(
            f"Camera frame: {width}x{height}, ROI: {roi} | "
            f"fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} (depth-based mm sizing)"
        )

        segmenter = YoloSegmenter(
            args.yolo_model,
            conf=args.yolo_conf,
            mask_refine="otsu" if args.mask_refine else "off",
            mask_refine_pad=args.mask_refine_pad,
            force_cpu=args.cpu,
        )
        print("Preloading YOLO model...")
        segmenter._get_model()
        _yolo_detect(segmenter, image, roi, imgsz=args.yolo_imgsz)
        print("YOLO warmup done.")

        sam_refiner = SamRefiner(checkpoint=args.sam_checkpoint, force_cpu=args.cpu)
        smoother = None if args.no_smooth else TrackSmoother(
            alpha=args.smooth_alpha,
            max_miss=args.smooth_max_miss,
            max_tracks=1,
        )
        if smoother is not None:
            print(
                f"Live metric smoothing enabled: alpha={args.smooth_alpha}, "
                f"max_miss={args.smooth_max_miss}, single-object track"
            )

        perf_notes = []
        if args.latest_frame:
            perf_notes.append("latest-frame drain")
        if args.infer_every > 1:
            perf_notes.append(f"YOLO every {args.infer_every} frames")
        if not args.mask_refine:
            perf_notes.append("mask refine off")
        else:
            perf_notes.append("Otsu on top-1 detection only")
        if perf_notes:
            print("Performance: " + ", ".join(perf_notes))

        hub = StreamHub(target_fps=args.fps)
        server = start_stream_server(
            hub,
            args.host,
            args.port,
            capture_output_dir=args.capture_output_dir,
        )
        base_url = (
            f"http://{args.host}:{args.port}"
            if args.host not in ("0.0.0.0", "")
            else f"http://127.0.0.1:{args.port}"
        )
        print(f"Stream server listening on {args.host}:{args.port}")
        print(f"  MJPEG:    {base_url}/video")
        print(f"  Status:   {base_url}/status")
        print(f"  Snapshot: {base_url}/snapshot.jpg")
        print(f"  Trigger:  POST {base_url}/water-cut")
        print(f"  Clear:    POST {base_url}/water-cut/clear")
        print(f"  Capture:  POST {base_url}/capture")
        print(f"  Captures: GET {base_url}/captures/{{fileName}}")
        print(f"Capture output dir: {args.capture_output_dir}")
        print(f"Target stream fps={args.fps}, width={args.stream_width}, jpeg_q={args.jpeg_quality}")

        interval = 1.0 / max(0.1, float(args.fps))
        water_cut_overlays = []
        fps_value = 0.0
        fps_alpha = 0.2
        frame_index = 0
        cached_detected: list = []

        while True:
            loop_start = time.perf_counter()
            frame_index += 1

            if hub.consume_clear_water_cut_request():
                water_cut_overlays.clear()
                print("Water-cut overlay cleared.")

            rgbd = _fetch_rgbd(camera, args)
            image = rgbd.color_bgr
            depth_mm = rgbd.depth_mm

            run_yolo = (
                args.infer_every <= 1
                or not cached_detected
                or frame_index % args.infer_every == 1
            )
            if run_yolo:
                try:
                    detected = _yolo_detect(segmenter, image, roi, imgsz=args.yolo_imgsz)
                except (RuntimeError, ValueError) as exc:
                    print(f"YOLO failed: {exc}")
                    detected = []
                cached_detected = copy.deepcopy(detected)
            else:
                detected = copy.deepcopy(cached_detected)

            raw_instances, display_instances = _prepare_output_instances(
                detected,
                depth_mm,
                intrinsics,
                smoother,
            )
            _print_height_metrics(display_instances)

            capture_req = hub.consume_capture_request()
            if capture_req is not None:
                if not run_yolo and cached_detected:
                    try:
                        detected = _yolo_detect(segmenter, image, roi, imgsz=args.yolo_imgsz)
                        cached_detected = copy.deepcopy(detected)
                        raw_instances, display_instances = _prepare_output_instances(
                            detected,
                            depth_mm,
                            intrinsics,
                            smoother,
                        )
                    except (RuntimeError, ValueError) as exc:
                        print(f"YOLO failed on capture: {exc}")

                hub.computing_water_cut = capture_req.water_cut
                if capture_req.water_cut:
                    status_text = "calculating water cut for capture..."
                    preview = compose_stream_frame(
                        image,
                        display_instances,
                        water_cut_overlays=water_cut_overlays or None,
                        status_text=status_text,
                        roi=roi,
                        label_size_factor=ORBBEC_LABEL_SIZE_FACTOR,
                        split_height_labels=True,
                        draw_metric_labels=False,
                    )
                    hub.set_frame(
                        encode_jpeg(preview, args.stream_width, args.jpeg_quality),
                        build_status(
                            instances=display_instances,
                            measured_fps=fps_value,
                            water_cut_overlays=water_cut_overlays,
                            computing=True,
                        ),
                    )
                capture_req.result = process_orbbec_capture_request(
                    request=capture_req,
                    image_bgr=image,
                    instances=raw_instances,
                    sam_refiner=sam_refiner,
                    depth_mm=depth_mm,
                    intrinsics=intrinsics,
                    output_dir=args.capture_output_dir,
                    jpeg_quality=args.jpeg_quality,
                    roi=roi,
                    label_instances=display_instances,
                    label_size_factor=ORBBEC_LABEL_SIZE_FACTOR,
                )
                hub.computing_water_cut = False
                hub.finish_capture(capture_req)

            status_text = None
            if hub.consume_water_cut_request():
                if not run_yolo and cached_detected:
                    try:
                        detected = _yolo_detect(segmenter, image, roi, imgsz=args.yolo_imgsz)
                        cached_detected = copy.deepcopy(detected)
                        raw_instances, display_instances = _prepare_output_instances(
                            detected,
                            depth_mm,
                            intrinsics,
                            smoother,
                        )
                    except (RuntimeError, ValueError) as exc:
                        print(f"YOLO failed on water-cut: {exc}")

                hub.computing_water_cut = True
                status_text = "calculating water cut width..."
                preview = compose_stream_frame(
                    image,
                    display_instances,
                    water_cut_overlays=water_cut_overlays or None,
                    status_text=status_text,
                    roi=roi,
                    label_size_factor=ORBBEC_LABEL_SIZE_FACTOR,
                    split_height_labels=True,
                    draw_metric_labels=False,
                )
                hub.set_frame(
                    encode_jpeg(preview, args.stream_width, args.jpeg_quality),
                    build_status(
                        instances=display_instances,
                        measured_fps=fps_value,
                        water_cut_overlays=water_cut_overlays,
                        computing=True,
                    ),
                )
                print("Water-cut requested...")
                water_cut_overlays = compute_orbbec_water_cut_overlays(
                    image,
                    raw_instances,
                    sam_refiner,
                    depth_mm=depth_mm,
                    intrinsics=intrinsics,
                )
                hub.computing_water_cut = False
                status_text = None

            frame = compose_stream_frame(
                image,
                display_instances,
                water_cut_overlays=water_cut_overlays or None,
                roi=roi,
                label_size_factor=ORBBEC_LABEL_SIZE_FACTOR,
                split_height_labels=True,
                draw_metric_labels=False,
            )
            jpeg = encode_jpeg(frame, args.stream_width, args.jpeg_quality)

            elapsed = max(1e-6, time.perf_counter() - loop_start)
            instant_fps = 1.0 / elapsed
            fps_value = instant_fps if fps_value <= 0 else fps_value * (1.0 - fps_alpha) + instant_fps * fps_alpha

            hub.set_frame(
                jpeg,
                build_status(
                    instances=display_instances,
                    measured_fps=fps_value,
                    water_cut_overlays=water_cut_overlays,
                    computing=hub.computing_water_cut,
                ),
            )

            sleep_s = interval - (time.perf_counter() - loop_start)
            if sleep_s > 0:
                time.sleep(sleep_s)

    finally:
        if server is not None:
            server.shutdown()
        close_rgbd_camera(camera)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Stream Orbbec Gemini RGB-D camera with depth-based YOLO sizing "
            "and HTTP water-cut trigger."
        )
    )
    parser.add_argument("--device-index", type=int, default=0, help="Orbbec device index.")
    parser.add_argument(
        "--orbbec-config",
        default=None,
        help="Path to orbbec_camera.json (default: config/orbbec_camera.json or project root).",
    )
    parser.add_argument(
        "--no-orbbec-config",
        action="store_true",
        help="Ignore orbbec_camera.json and use legacy default RGB-D streams.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port for /video and control API.")
    parser.add_argument("--fps", type=float, default=10.0, help="Target stream frame rate.")
    parser.add_argument("--stream-width", type=int, default=720, help="JPEG width (0 = full resolution).")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality 30-95.")
    parser.add_argument("--timeout-ms", type=int, default=1000, help="Orbbec frame wait timeout.")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Discard this many frames after open.")
    parser.add_argument(
        "--latest-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drain SDK queue and process only the newest frame (default: on).",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Run YOLO every N frames; reuse mask in between (default: 1 = every frame).",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="YOLO ROI as x1,y1,x2,y2 pixels or x1,y1,x2,y2 ratios (0-1).",
    )
    parser.add_argument(
        "--roi-file",
        default=None,
        help="ROI JSON path (default: auto-load config/roi.json when present).",
    )
    parser.add_argument("--yolo-model", default="yolov8m-seg.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run YOLO and SAM inference on CPU only (ignore CUDA/MPS).",
    )
    parser.add_argument(
        "--mask-refine",
        action="store_true",
        help="Enable Otsu edge refinement after YOLO (slower, default: off).",
    )
    parser.add_argument(
        "--mask-refine-pad",
        type=int,
        default=80,
        help="Padding (px) around YOLO bbox for Otsu refinement.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Path to sam_vit_b_01ec64.pth (default: checkpoints/sam_vit_b_01ec64.pth).",
    )
    parser.add_argument(
        "--capture-output-dir",
        default="output/captures",
        help="Directory for POST /capture saved JPEG files.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.25,
        help="EMA weight for new live L/W samples (0.01-1, lower = smoother).",
    )
    parser.add_argument(
        "--smooth-max-miss",
        type=int,
        default=3,
        help="Drop a track after this many consecutive unmatched frames.",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable temporal smoothing of live L/W/H readouts.",
    )
    args = parser.parse_args()
    if args.infer_every < 1:
        parser.error("--infer-every must be >= 1")

    try:
        return run_stream(args)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
