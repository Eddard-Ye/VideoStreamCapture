# -*- coding: utf-8 -*-
"""Capture frames from a Hikvision MVS 2D industrial camera."""

import argparse
import os
import sys
import time
from ctypes import (
    POINTER,
    byref,
    cast,
    c_ubyte,
    memset,
    sizeof,
)

_MVIMPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
if _MVIMPORT_DIR not in sys.path:
    sys.path.insert(0, _MVIMPORT_DIR)

import cv2
import numpy as np

from CameraParams_const import MV_ACCESS_Exclusive, MV_GIGE_DEVICE, MV_USB_DEVICE
from CameraParams_header import (
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_CC_PIXEL_CONVERT_PARAM,
    MV_FRAME_OUT_INFO_EX,
    MV_TRIGGER_MODE_OFF,
    MVCC_ENUMVALUE,
    MVCC_INTVALUE,
)
from MvCameraControl_class import MvCamera
from PixelType_header import (
    PixelType_Gvsp_BayerBG8,
    PixelType_Gvsp_BayerGB8,
    PixelType_Gvsp_BayerGR8,
    PixelType_Gvsp_BayerRG8,
    PixelType_Gvsp_BGR8_Packed,
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_RGB8_Packed,
)
from camera_calib_2d import load_calib_or_none
from color_viewer import ColorViewer, RoiRect

DEVICE_TYPES = MV_GIGE_DEVICE | MV_USB_DEVICE
BAYER8_FORMATS = frozenset(
    {
        PixelType_Gvsp_BayerRG8,
        PixelType_Gvsp_BayerGB8,
        PixelType_Gvsp_BayerGR8,
        PixelType_Gvsp_BayerBG8,
    }
)
# GenICam/Hikvision Bayer labels differ from OpenCV; map to the paired pattern so
# demosaiced colors match MVS (e.g. camera BayerRG8 -> OpenCV BayerBG2BGR).
BAYER8_OPENCV = {
    PixelType_Gvsp_BayerRG8: cv2.COLOR_BayerBG2BGR,
    PixelType_Gvsp_BayerBG8: cv2.COLOR_BayerRG2BGR,
    PixelType_Gvsp_BayerGR8: cv2.COLOR_BayerGB2BGR,
    PixelType_Gvsp_BayerGB8: cv2.COLOR_BayerGR2BGR,
}
KNOWN_PIXEL_TYPES = frozenset(
    {
        PixelType_Gvsp_BGR8_Packed,
        PixelType_Gvsp_RGB8_Packed,
        PixelType_Gvsp_Mono8,
        *BAYER8_FORMATS,
    }
)


def _decode_c_string(raw_bytes):
    return bytes(bytearray(raw_bytes)).decode("ascii", errors="ignore").strip("\x00")


def _ip_from_int(ip_value):
    return (
        f"{(ip_value >> 24) & 0xff}."
        f"{(ip_value >> 16) & 0xff}."
        f"{(ip_value >> 8) & 0xff}."
        f"{ip_value & 0xff}"
    )


def _device_summary(device_info):
    if device_info.nTLayerType == MV_GIGE_DEVICE:
        gige = device_info.SpecialInfo.stGigEInfo
        return {
            "transport": "GigE",
            "model": _decode_c_string(gige.chModelName),
            "serial": _decode_c_string(gige.chSerialNumber),
            "ip": _ip_from_int(gige.nCurrentIp),
        }

    if device_info.nTLayerType == MV_USB_DEVICE:
        usb = device_info.SpecialInfo.stUsb3VInfo
        return {
            "transport": "USB3",
            "model": _decode_c_string(usb.chModelName),
            "serial": _decode_c_string(usb.chSerialNumber),
            "ip": None,
        }

    return {
        "transport": f"0x{device_info.nTLayerType:x}",
        "model": "unknown",
        "serial": "unknown",
        "ip": None,
    }


def list_devices():
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(DEVICE_TYPES, device_list)
    if ret != 0:
        raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:x}")

    if device_list.nDeviceNum == 0:
        raise RuntimeError("No MVS 2D camera found.")

    devices = []
    for index in range(device_list.nDeviceNum):
        device_info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        summary = _device_summary(device_info)
        devices.append(
            {
                "index": index,
                "info": device_info,
                **summary,
            }
        )
        ip_text = f", ip={summary['ip']}" if summary["ip"] else ""
        print(
            f"[{index}] transport={summary['transport']}, "
            f"model={summary['model']}, serial={summary['serial']}{ip_text}"
        )
    return devices


def _configure_gige_packet_size(camera, device_info):
    if device_info.nTLayerType != MV_GIGE_DEVICE:
        return

    packet_size = camera.MV_CC_GetOptimalPacketSize()
    if int(packet_size) <= 0:
        print(f"Warning: MV_CC_GetOptimalPacketSize failed: 0x{packet_size:x}")
        return

    ret = camera.MV_CC_SetIntValue("GevSCPSPacketSize", int(packet_size))
    if ret != 0:
        print(f"Warning: set GevSCPSPacketSize failed: 0x{ret:x}")


def _get_camera_pixel_format(camera):
    pixel_format = MVCC_ENUMVALUE()
    memset(byref(pixel_format), 0, sizeof(pixel_format))
    ret = camera.MV_CC_GetEnumValue("PixelFormat", pixel_format)
    if ret != 0:
        raise RuntimeError(f"get PixelFormat failed: 0x{ret:x}")
    return int(pixel_format.nCurValue)


def _effective_frame_len(frame_len, pixel_format, width, height, payload_size):
    if frame_len > 0:
        return int(frame_len)
    if pixel_format in BAYER8_FORMATS or pixel_format == PixelType_Gvsp_Mono8:
        return width * height
    if pixel_format in (PixelType_Gvsp_BGR8_Packed, PixelType_Gvsp_RGB8_Packed):
        return width * height * 3
    return int(payload_size)


def _frame_pixel_type(frame_info, pixel_format: int) -> int:
    """Prefer configured PixelFormat; frame_info.enPixelType is unreliable with some SDK headers."""
    src = int(frame_info.enPixelType)
    if src in KNOWN_PIXEL_TYPES and int(frame_info.nFrameLen) > 0:
        return src
    return int(pixel_format)


def _opencv_bayer_to_bgr(raw: np.ndarray, width: int, height: int, pixel_format: int) -> np.ndarray:
    bayer_code = BAYER8_OPENCV.get(pixel_format)
    if bayer_code is None:
        raise RuntimeError(f"Unsupported Bayer pixel format: 0x{pixel_format:x}")
    bayer = raw.reshape(height, width)
    return cv2.cvtColor(bayer, bayer_code)


def open_camera(device_info, pixel_format_name=None):
    camera = MvCamera()

    ret = camera.MV_CC_CreateHandle(device_info)
    if ret != 0:
        raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:x}")

    try:
        ret = camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:x}")

        _configure_gige_packet_size(camera, device_info)

        ret = camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            raise RuntimeError(f"set TriggerMode failed: 0x{ret:x}")

        if pixel_format_name:
            ret = camera.MV_CC_SetEnumValueByString("PixelFormat", pixel_format_name)
            if ret != 0:
                raise RuntimeError(f"set PixelFormat={pixel_format_name} failed: 0x{ret:x}")

        pixel_format = _get_camera_pixel_format(camera)
        print(f"PixelFormat: 0x{pixel_format:x}")

        payload = MVCC_INTVALUE()
        memset(byref(payload), 0, sizeof(payload))
        ret = camera.MV_CC_GetIntValue("PayloadSize", payload)
        if ret != 0:
            raise RuntimeError(f"get PayloadSize failed: 0x{ret:x}")

        ret = camera.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:x}")

        return camera, int(payload.nCurValue), pixel_format
    except Exception:
        camera.MV_CC_DestroyHandle()
        raise


def close_camera(camera):
    if camera is None:
        return

    camera.MV_CC_StopGrabbing()
    camera.MV_CC_CloseDevice()
    camera.MV_CC_DestroyHandle()


def _sdk_convert_to_bgr(
    camera,
    data_buf,
    width: int,
    height: int,
    src_pixel_type: int,
    src_len: int,
) -> np.ndarray:
    """Convert raw camera buffer to BGR via Hikvision SDK (matches MVS ISP pipeline)."""
    dst_size = width * height * 3 + 2048
    dst_buf = (c_ubyte * dst_size)()
    convert_param = MV_CC_PIXEL_CONVERT_PARAM()
    memset(byref(convert_param), 0, sizeof(convert_param))
    convert_param.nWidth = width
    convert_param.nHeight = height
    convert_param.enSrcPixelType = src_pixel_type
    convert_param.pSrcData = cast(data_buf, POINTER(c_ubyte))
    convert_param.nSrcDataLen = src_len
    convert_param.enDstPixelType = PixelType_Gvsp_BGR8_Packed
    convert_param.pDstBuffer = dst_buf
    convert_param.nDstBufferSize = dst_size

    ret = camera.MV_CC_ConvertPixelType(convert_param)
    if ret != 0:
        raise RuntimeError(
            f"MV_CC_ConvertPixelType failed: 0x{ret:x}, "
            f"pixel_format=0x{src_pixel_type:x}, src_len={src_len}"
        )

    out_len = int(convert_param.nDstLen) if int(convert_param.nDstLen) > 0 else width * height * 3
    raw = np.frombuffer(dst_buf, dtype=np.uint8, count=out_len)
    return raw.reshape(height, width, 3)


def _bayer_to_bgr(
    camera,
    data_buf,
    width: int,
    height: int,
    src_pixel_type: int,
    src_len: int,
) -> np.ndarray:
    # MV_CC_ConvertPixelType(Bayer->BGR) is unsupported on some models (e.g. MV-CU060).
    try:
        return _sdk_convert_to_bgr(camera, data_buf, width, height, src_pixel_type, src_len)
    except RuntimeError:
        raw = np.frombuffer(data_buf, dtype=np.uint8, count=src_len)
        return _opencv_bayer_to_bgr(raw, width, height, src_pixel_type)


def frame_to_bgr(camera, data_buf, frame_info, pixel_format, payload_size):
    width = int(frame_info.nWidth)
    height = int(frame_info.nHeight)
    src_pixel_type = _frame_pixel_type(frame_info, pixel_format)
    frame_len = _effective_frame_len(
        int(frame_info.nFrameLen),
        src_pixel_type,
        width,
        height,
        payload_size,
    )

    if src_pixel_type == PixelType_Gvsp_BGR8_Packed:
        raw = np.frombuffer(data_buf, dtype=np.uint8, count=frame_len)
        return raw.reshape(height, width, 3)

    if src_pixel_type == PixelType_Gvsp_RGB8_Packed:
        raw = np.frombuffer(data_buf, dtype=np.uint8, count=frame_len)
        rgb = raw.reshape(height, width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if src_pixel_type == PixelType_Gvsp_Mono8:
        raw = np.frombuffer(data_buf, dtype=np.uint8, count=frame_len)
        gray = raw.reshape(height, width)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if src_pixel_type in BAYER8_FORMATS:
        return _bayer_to_bgr(camera, data_buf, width, height, src_pixel_type, frame_len)

    return _sdk_convert_to_bgr(camera, data_buf, width, height, src_pixel_type, frame_len)


def fetch_frame(camera, payload_size, pixel_format, timeout_ms=1000, warmup_frames=0):
    data_buf = (c_ubyte * payload_size)()
    frame_info = MV_FRAME_OUT_INFO_EX()
    attempts = max(1, warmup_frames + 1)
    image = None

    for _ in range(attempts):
        memset(byref(frame_info), 0, sizeof(frame_info))
        ret = camera.MV_CC_GetOneFrameTimeout(data_buf, payload_size, frame_info, timeout_ms)
        if ret != 0:
            raise RuntimeError(f"MV_CC_GetOneFrameTimeout failed: 0x{ret:x}")

        image = frame_to_bgr(camera, data_buf, frame_info, pixel_format, payload_size)

    return image, frame_info


def save_output(image, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"frame_{timestamp}.png")
    cv2.imwrite(output_path, image)
    print(f"Saved image: {output_path}")
    return output_path


def run_gui(camera, payload_size, pixel_format, args):
    calib_2d = load_calib_or_none(args.calib_file)

    def fetch_frame_fn():
        try:
            image, _frame_info = fetch_frame(
                camera,
                payload_size,
                pixel_format,
                timeout_ms=args.timeout_ms,
                warmup_frames=0,
            )
            return image, None
        except RuntimeError as exc:
            print(exc)
            return None, None

    if args.live:
        print("Live preview mode. Press Space to freeze frame for YOLO.")
        color_image, _depth_image = fetch_frame_fn()
        if color_image is None:
            raise RuntimeError("No color frame available for viewer.")
    else:
        color_image, _frame_info = fetch_frame(
            camera,
            payload_size,
            pixel_format,
            timeout_ms=args.timeout_ms,
            warmup_frames=args.warmup_frames,
        )

    height, width = color_image.shape[:2]
    full_roi = RoiRect(0, 0, width, height)
    if calib_2d is not None:
        calib_2d = calib_2d.scaled(width, height)

    viewer = ColorViewer(
        color_image,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        roi=full_roi,
        output_dir=args.output_dir,
        fetch_frame_fn=fetch_frame_fn,
        live=args.live,
        yolo_live=args.yolo_live,
        yolo_imgsz=args.yolo_imgsz,
        water_cut=args.water_cut,
        sam_refine=False,
        sam_checkpoint=args.sam_checkpoint,
        calib_2d=calib_2d,
        force_cpu=args.cpu,
    )
    viewer.run()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Capture frames from Hikvision MVS 2D camera.")
    parser.add_argument("--device-index", type=int, default=0, help="Device index from list_devices().")
    parser.add_argument("--output-dir", default="output", help="Directory for saved images.")
    parser.add_argument("--timeout-ms", type=int, default=1000, help="GetOneFrame timeout in ms.")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Discard this many frames before capture.")
    parser.add_argument("--list-only", action="store_true", help="Only enumerate devices and exit.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open interactive viewer with zoom and YOLO segmentation.",
    )
    parser.add_argument("--live", action="store_true", help="With --gui, show live preview until frozen.")
    parser.add_argument(
        "--yolo-live",
        action="store_true",
        help="With --gui --live, run YOLO on every camera frame and overlay results.",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference size (smaller is faster).")
    parser.add_argument("--yolo-model", default="yolov8n-seg.pt", help="Ultralytics YOLO seg weights.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run YOLO and SAM inference on CPU only (ignore CUDA/MPS).",
    )
    parser.add_argument(
        "--water-cut",
        action="store_true",
        help="Preview CV texture region and SAM fg/bg prompts inside each YOLO detection (no SAM run).",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Path to sam_vit_b_01ec64.pth (default: checkpoints/sam_vit_b_01ec64.pth).",
    )
    parser.add_argument(
        "--pixel-format",
        default=None,
        help='Optional PixelFormat string, e.g. "RGB8Packed" or "BayerRG8".',
    )
    parser.add_argument(
        "--calib-file",
        default=None,
        help="2D camera calibration JSON from camera_calib_2d.py (enables mm labels).",
    )
    args = parser.parse_args()

    if args.live and not args.gui:
        parser.error("--live requires --gui")
    if args.yolo_live and not args.gui:
        parser.error("--yolo-live requires --gui")
    if args.yolo_live:
        args.live = True

    print(f"MVS SDK version: 0x{MvCamera.MV_CC_GetSDKVersion():x}")
    print("Scanning for MVS 2D devices...")
    devices = list_devices()
    if args.list_only:
        return 0

    if args.device_index >= len(devices):
        raise RuntimeError(f"Invalid device index {args.device_index}, found {len(devices)} device(s).")

    selected = devices[args.device_index]
    print(
        f"Opening device [{args.device_index}] "
        f"transport={selected['transport']} serial={selected['serial']} ..."
    )

    camera = None
    try:
        camera, payload_size, pixel_format = open_camera(selected["info"], args.pixel_format)
        if args.gui:
            return run_gui(camera, payload_size, pixel_format, args)

        image, frame_info = fetch_frame(
            camera,
            payload_size,
            pixel_format,
            timeout_ms=args.timeout_ms,
            warmup_frames=args.warmup_frames,
        )
        print(
            f"Captured frame: {frame_info.nWidth}x{frame_info.nHeight}, "
            f"pixel_format=0x{pixel_format:x}, frame={frame_info.nFrameNum}"
        )
        save_output(image, args.output_dir)
        print("Capture completed.")
        return 0
    finally:
        close_camera(camera)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
