# -*- coding: utf-8 -*-
"""Stream MVS 2D camera frames over HTTP with live YOLO and remote water-cut trigger."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from camera_calib_2d import (
    DEFAULT_CALIB_FILE,
    attach_instance_metrics,
    load_calib_or_none,
    water_cut_width_mm_2d,
)
from capture_2d import close_camera, fetch_frame, list_devices, open_camera
from color_viewer import RoiRect, SegInstance, YoloSegmenter
from object_measure import instance_height_mm, oriented_box_from_mask
from sam_centerline import analyze_water_cut
from stream_common import (
    attach_oriented_boxes,
    build_status,
    encode_jpeg,
    sanitize_capture_name,
    save_capture_jpeg,
)
from stream_overlay import (
    WaterCutOverlay,
    build_capture_record_info,
    compose_record_frame,
    compose_stream_frame,
    format_temperature_display,
)
from stream_server import CaptureRequest, StreamHub, start_stream_server
from track_smoother import TrackSmoother
from yolo_sam_refine import SamRefiner, prepare_water_cut_box_prompts, run_water_cut_box_sam


def compute_water_cut_overlays(
    image_bgr: np.ndarray,
    instances: list[SegInstance],
    sam_refiner: SamRefiner,
    calib=None,
) -> list[WaterCutOverlay]:
    overlays: list[WaterCutOverlay] = []
    for instance in instances:
        preview = prepare_water_cut_box_prompts(instance.mask)
        if preview is None:
            continue
        try:
            sam_region = run_water_cut_box_sam(sam_refiner, image_bgr, instance.mask, preview)
        except RuntimeError as exc:
            print(f"  Water-cut SAM failed on {instance.class_name}: {exc}")
            continue
        if sam_region is None:
            print(f"  Water-cut SAM empty on {instance.class_name}.")
            continue

        water_cut = analyze_water_cut(sam_region.mask)
        if water_cut is None:
            continue
        if calib is not None:
            water_cut.water_cut_width_mm = water_cut_width_mm_2d(water_cut, calib)

        box_pts = instance.box_pts
        if box_pts is None:
            box_pts = oriented_box_from_mask(instance.mask)

        overlays.append(
            WaterCutOverlay(
                sam_mask=sam_region.mask.copy(),
                water_cut=water_cut,
                box_pts=None if box_pts is None else box_pts.copy(),
                prompt_coords=(
                    None
                    if sam_region.prompt_coords is None
                    else np.asarray(sam_region.prompt_coords, dtype=np.float32).copy()
                ),
                prompt_labels=(
                    None
                    if sam_region.prompt_labels is None
                    else np.asarray(sam_region.prompt_labels, dtype=np.int32).copy()
                ),
            )
        )
        if calib is not None and np.isfinite(water_cut.water_cut_width_mm):
            print(
                f"  Water-cut {instance.class_name}: "
                f"width={water_cut.water_cut_width_px:.1f}px "
                f"({water_cut.water_cut_width_mm:.2f}mm)"
            )
        else:
            print(
                f"  Water-cut {instance.class_name}: "
                f"width={water_cut.water_cut_width_px:.1f}px"
            )
    return overlays


def process_capture_request(
    *,
    request: CaptureRequest,
    image_bgr: np.ndarray,
    instances: list[SegInstance],
    sam_refiner: SamRefiner,
    calib,
    output_dir: str,
    jpeg_quality: int,
) -> dict:
    try:
        record_overlays: list[WaterCutOverlay] = []
        if request.water_cut:
            print("Capture requested with water-cut...")
            record_overlays = compute_water_cut_overlays(
                image_bgr,
                instances,
                sam_refiner,
                calib=calib,
            )

        record_info = build_capture_record_info(
            instances,
            temperature=request.temperature,
            weight=request.weight,
            water_cut_enabled=request.water_cut,
            water_cut_overlays=record_overlays,
        )
        frame = compose_record_frame(
            image_bgr,
            instances,
            record_info,
            water_cut_overlays=record_overlays if request.water_cut else None,
        )
        output_path = save_capture_jpeg(frame, output_dir, request.name, jpeg_quality)
        file_name = os.path.basename(output_path)
        print(f"Saved capture: {output_path}")

        primary = instances[0] if instances else None
        water_cut_mm = (
            None
            if record_info.water_cut_mm is None or not np.isfinite(record_info.water_cut_mm)
            else round(float(record_info.water_cut_mm), 1)
        )
        primary_height_mm = None if primary is None else instance_height_mm(primary)
        return {
            "ok": True,
            "fileName": file_name,
            "name": request.name,
            "water_cut": request.water_cut,
            "record": {
                "lw": record_info.lw_text,
                "height": record_info.height,
                "temperature": format_temperature_display(request.temperature),
                "weight": request.weight,
                "water_cut": record_info.water_cut_line,
            },
            "detections": len(instances),
            "length_mm": (
                None
                if primary is None or not np.isfinite(primary.length_mm)
                else round(float(primary.length_mm), 1)
            ),
            "width_mm": (
                None
                if primary is None or not np.isfinite(primary.width_mm)
                else round(float(primary.width_mm), 1)
            ),
            "height_mm": (
                None
                if primary_height_mm is None or not np.isfinite(primary_height_mm)
                else round(float(primary_height_mm), 1)
            ),
            "water_cut_mm": water_cut_mm if request.water_cut else None,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def run_stream(args: argparse.Namespace) -> int:
    print(f"Scanning for MVS 2D devices...")
    devices = list_devices()
    if args.device_index >= len(devices):
        raise RuntimeError(f"Invalid device index {args.device_index}, found {len(devices)} device(s).")

    selected = devices[args.device_index]
    print(
        f"Opening device [{args.device_index}] "
        f"transport={selected['transport']} serial={selected['serial']} ..."
    )

    camera = None
    server = None
    try:
        camera, payload_size, pixel_format = open_camera(selected["info"], args.pixel_format)
        image, _frame_info = fetch_frame(
            camera,
            payload_size,
            pixel_format,
            timeout_ms=args.timeout_ms,
            warmup_frames=args.warmup_frames,
        )
        height, width = image.shape[:2]
        roi = RoiRect(0, 0, width, height)
        print(f"Camera frame: {width}x{height}, ROI: {roi}")

        calib_path = None if args.no_calib else args.calib_file
        calib = load_calib_or_none(calib_path)
        if calib is not None:
            calib = calib.scaled(width, height)
        elif not args.no_calib:
            print("Running without mm calibration; overlay will show pixels only.")

        segmenter = YoloSegmenter(
            args.yolo_model,
            conf=args.yolo_conf,
            mask_refine="off" if args.no_mask_refine else "otsu",
            mask_refine_pad=args.mask_refine_pad,
            force_cpu=args.cpu,
        )
        sam_refiner = SamRefiner(checkpoint=args.sam_checkpoint, force_cpu=args.cpu)
        smoother = None if args.no_smooth else TrackSmoother(
            alpha=args.smooth_alpha,
            max_miss=args.smooth_max_miss,
        )
        if smoother is not None:
            print(
                f"Live metric smoothing enabled: alpha={args.smooth_alpha}, "
                f"max_miss={args.smooth_max_miss}"
            )

        hub = StreamHub(target_fps=args.fps)
        server = start_stream_server(
            hub,
            args.host,
            args.port,
            capture_output_dir=args.capture_output_dir,
        )
        base_url = f"http://{args.host}:{args.port}" if args.host not in ("0.0.0.0", "") else f"http://127.0.0.1:{args.port}"
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
        water_cut_overlays: list[WaterCutOverlay] = []
        fps_value = 0.0
        fps_alpha = 0.2

        while True:
            loop_start = time.perf_counter()

            if hub.consume_clear_water_cut_request():
                water_cut_overlays.clear()
                print("Water-cut overlay cleared.")

            image, _frame_info = fetch_frame(
                camera,
                payload_size,
                pixel_format,
                timeout_ms=args.timeout_ms,
                warmup_frames=0,
            )

            try:
                instances = segmenter.segment_all(
                    image,
                    roi,
                    imgsz=args.yolo_imgsz,
                )
            except (RuntimeError, ValueError) as exc:
                print(f"YOLO failed: {exc}")
                instances = []

            attach_oriented_boxes(instances)
            if calib is not None:
                attach_instance_metrics(instances, calib)

            raw_instances = instances
            display_instances = (
                smoother.update(raw_instances) if smoother is not None else raw_instances
            )

            capture_req = hub.consume_capture_request()
            if capture_req is not None:
                hub.computing_water_cut = capture_req.water_cut
                if capture_req.water_cut:
                    status_text = "calculating water cut for capture..."
                    preview = compose_stream_frame(
                        image,
                        display_instances,
                        water_cut_overlays=water_cut_overlays or None,
                        status_text=status_text,
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
                capture_req.result = process_capture_request(
                    request=capture_req,
                    image_bgr=image,
                    instances=raw_instances,
                    sam_refiner=sam_refiner,
                    calib=calib,
                    output_dir=args.capture_output_dir,
                    jpeg_quality=args.jpeg_quality,
                )
                hub.computing_water_cut = False
                hub.finish_capture(capture_req)

            status_text = None
            if hub.consume_water_cut_request():
                hub.computing_water_cut = True
                status_text = "calculating water cut width..."
                preview = compose_stream_frame(
                    image,
                    display_instances,
                    water_cut_overlays=water_cut_overlays or None,
                    status_text=status_text,
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
                water_cut_overlays = compute_water_cut_overlays(
                    image,
                    raw_instances,
                    sam_refiner,
                    calib=calib,
                )
                hub.computing_water_cut = False
                status_text = None

            frame = compose_stream_frame(
                image,
                display_instances,
                water_cut_overlays=water_cut_overlays or None,
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
        close_camera(camera)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream MVS 2D camera with YOLO overlay and HTTP water-cut trigger."
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address.")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port for /video and control API.")
    parser.add_argument("--fps", type=float, default=10.0, help="Target stream frame rate.")
    parser.add_argument("--stream-width", type=int, default=720, help="JPEG width (0 = full resolution).")
    parser.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality 30-95.")
    parser.add_argument("--timeout-ms", type=int, default=1000)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--yolo-model", default="yolov8n-seg.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run YOLO and SAM inference on CPU only (ignore CUDA/MPS).",
    )
    parser.add_argument(
        "--no-mask-refine",
        action="store_true",
        help="Disable Otsu edge refinement after YOLO segmentation.",
    )
    parser.add_argument(
        "--mask-refine-pad",
        type=int,
        default=100,
        help="Padding (px) around YOLO bbox for Otsu refinement (default: 80).",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Path to sam_vit_b_01ec64.pth (default: checkpoints/sam_vit_b_01ec64.pth).",
    )
    parser.add_argument("--pixel-format", default=None)
    parser.add_argument(
        "--calib-file",
        default=DEFAULT_CALIB_FILE,
        help=f"2D camera calibration JSON (default: {DEFAULT_CALIB_FILE}).",
    )
    parser.add_argument(
        "--no-calib",
        action="store_true",
        help="Disable mm conversion and show pixel sizes only.",
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
        help="Disable temporal smoothing of live L/W readouts.",
    )
    args = parser.parse_args()

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
