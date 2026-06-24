# -*- coding: utf-8 -*-
"""Capture color and depth frames from a Hikvision MV3D RGB-D camera."""

import argparse
import os
import sys
import time
from ctypes import byref, pointer, string_at

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEPRECATE = os.path.dirname(os.path.abspath(__file__))
for _path in (_ROOT, _DEPRECATE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import cv2
import numpy as np

from camera_intrinsics import RgbIntrinsics
from color_viewer import ColorViewer, DEFAULT_ROI_FILE, resolve_roi
from rgbd_intrinsics import get_rgb_intrinsics
from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_DEVICE_INFO_LIST,
    MV3D_RGBD_FRAME_DATA,
    MV3D_RGBD_OK,
    DeviceType_Ethernet,
    DeviceType_Ethernet_Vir,
    DeviceType_USB,
    DeviceType_USB_Vir,
    ImageType_Depth,
    ImageType_Mono8,
    ImageType_RGB8_Planar,
    ImageType_YUV420SP_NV12,
    ImageType_YUV420SP_NV21,
    ImageType_YUV422,
)

DEVICE_TYPES = (
    DeviceType_Ethernet
    | DeviceType_USB
    | DeviceType_Ethernet_Vir
    | DeviceType_USB_Vir
)

COLOR_IMAGE_TYPES = {
    ImageType_RGB8_Planar,
    ImageType_YUV420SP_NV12,
    ImageType_YUV420SP_NV21,
    ImageType_YUV422,
}


def _decode_c_string(raw_bytes):
    return bytes(bytearray(raw_bytes)).decode("ascii", errors="ignore").strip("\x00")


def list_devices():
    device_count = __import__("ctypes").c_uint(0)
    device_count_ref = byref(device_count)
    ret = Mv3dRgbd.MV3D_RGBD_GetDeviceNumber(DEVICE_TYPES, device_count_ref)
    if ret != MV3D_RGBD_OK:
        raise RuntimeError(f"MV3D_RGBD_GetDeviceNumber failed: 0x{ret:x}")

    if device_count.value == 0:
        raise RuntimeError("No MV3D RGB-D device found.")

    device_list = MV3D_RGBD_DEVICE_INFO_LIST()
    ret = Mv3dRgbd.MV3D_RGBD_GetDeviceList(
        DEVICE_TYPES,
        pointer(device_list.DeviceInfo[0]),
        20,
        device_count_ref,
    )
    if ret != MV3D_RGBD_OK:
        raise RuntimeError(f"MV3D_RGBD_GetDeviceList failed: 0x{ret:x}")

    devices = []
    for index in range(device_count.value):
        info = device_list.DeviceInfo[index]
        devices.append(
            {
                "index": index,
                "model": _decode_c_string(info.chModelName),
                "serial": _decode_c_string(info.chSerialNumber),
                "info": info,
            }
        )
        print(f"[{index}] model={devices[-1]['model']}, serial={devices[-1]['serial']}")
    return devices


def _image_to_numpy(image_data):
    width = image_data.nWidth
    height = image_data.nHeight
    raw = string_at(image_data.pData, image_data.nDataLen)
    image_type = image_data.enImageType

    if image_type == ImageType_Depth:
        depth = np.frombuffer(raw, dtype=np.uint16).reshape(height, width)
        return "depth", depth

    if image_type == ImageType_RGB8_Planar:
        planar = np.frombuffer(raw, dtype=np.uint8).reshape(3, height * width)
        rgb = np.stack(
            [
                planar[0].reshape(height, width),
                planar[1].reshape(height, width),
                planar[2].reshape(height, width),
            ],
            axis=-1,
        )
        return "color", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if image_type == ImageType_YUV420SP_NV12:
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
        return "color", cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    if image_type == ImageType_YUV420SP_NV21:
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
        return "color", cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)

    if image_type == ImageType_YUV422:
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 2)
        return "color", cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)

    if image_type == ImageType_Mono8:
        mono = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        return "mono", mono

    return "unknown", None


def fetch_rgbd_frame(camera, timeout_ms=5000, warmup_frames=0):
    """Fetch synchronized color and depth frames."""
    frame_data = MV3D_RGBD_FRAME_DATA()
    attempts = max(1, warmup_frames + 1)

    color_image = None
    depth_image = None
    for _ in range(attempts):
        ret = camera.MV3D_RGBD_FetchFrame(pointer(frame_data), timeout_ms)
        if ret != MV3D_RGBD_OK:
            raise RuntimeError(f"MV3D_RGBD_FetchFrame failed: 0x{ret:x}")

        color_image = None
        depth_image = None
        for index in range(frame_data.nImageCount):
            kind, array = _image_to_numpy(frame_data.stImageData[index])
            if kind == "color" and color_image is None:
                color_image = array
            elif kind == "depth" and depth_image is None:
                depth_image = array

    if color_image is None:
        raise RuntimeError("No color image found in fetched frame.")
    return color_image, depth_image


def fetch_color_frame(camera, timeout_ms=5000, warmup_frames=0):
    color_image, _depth_image = fetch_rgbd_frame(
        camera,
        timeout_ms=timeout_ms,
        warmup_frames=warmup_frames,
    )
    return color_image


def fetch_rgbd(camera, timeout_ms=5000, warmup_frames=3):
    frame_data = MV3D_RGBD_FRAME_DATA()
    color_image = None
    depth_image = None

    for _ in range(warmup_frames):
        ret = camera.MV3D_RGBD_FetchFrame(pointer(frame_data), timeout_ms)
        if ret != MV3D_RGBD_OK:
            raise RuntimeError(f"MV3D_RGBD_FetchFrame failed: 0x{ret:x}")

    ret = camera.MV3D_RGBD_FetchFrame(pointer(frame_data), timeout_ms)
    if ret != MV3D_RGBD_OK:
        raise RuntimeError(f"MV3D_RGBD_FetchFrame failed: 0x{ret:x}")

    print(f"Fetched frame with {frame_data.nImageCount} image(s).")
    for index in range(frame_data.nImageCount):
        image_data = frame_data.stImageData[index]
        kind, array = _image_to_numpy(image_data)
        print(
            f"  image[{index}] type=0x{image_data.enImageType:x}, "
            f"size={image_data.nWidth}x{image_data.nHeight}, "
            f"frame={image_data.nFrameNum}, decoded={kind}"
        )
        if kind == "color" and color_image is None:
            color_image = array
        elif kind == "depth" and depth_image is None:
            depth_image = array

    return color_image, depth_image


def save_outputs(color_image, depth_image, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saved = []

    if color_image is not None:
        color_path = os.path.join(output_dir, f"color_{timestamp}.png")
        cv2.imwrite(color_path, color_image)
        saved.append(color_path)
        print(f"Saved color image: {color_path}")

    if depth_image is not None:
        depth_raw_path = os.path.join(output_dir, f"depth_{timestamp}.png")
        depth_vis_path = os.path.join(output_dir, f"depth_vis_{timestamp}.png")
        cv2.imwrite(depth_raw_path, depth_image)
        valid = depth_image > 0
        if np.any(valid):
            depth_vis = np.zeros_like(depth_image, dtype=np.uint8)
            min_depth = depth_image[valid].min()
            max_depth = depth_image[valid].max()
            depth_vis[valid] = (
                (depth_image[valid].astype(np.float32) - min_depth)
                / max(1.0, max_depth - min_depth)
                * 255
            ).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            depth_vis[~valid] = 0
        else:
            depth_vis = np.zeros((*depth_image.shape, 3), dtype=np.uint8)
        cv2.imwrite(depth_vis_path, depth_vis)
        saved.extend([depth_raw_path, depth_vis_path])
        print(f"Saved depth image: {depth_raw_path}")
        print(f"Saved depth visualization: {depth_vis_path}")

    return saved


def run_gui(camera, args):
    intrinsics = get_rgb_intrinsics(camera)
    print(
        f"RGB intrinsics: fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
        f"cx={intrinsics.cx:.2f} cy={intrinsics.cy:.2f} "
        f"calib={intrinsics.calib_width}x{intrinsics.calib_height} z_unit={intrinsics.z_unit_mm}"
    )

    def fetch_frame():
        try:
            return fetch_rgbd_frame(camera, timeout_ms=args.timeout_ms, warmup_frames=0)
        except RuntimeError as exc:
            print(exc)
            return None, None

    if args.live:
        print("Live preview mode. Press Space to freeze frame for segmentation.")
        color_image, depth_image = fetch_frame()
    else:
        color_image, depth_image = fetch_rgbd(camera, timeout_ms=args.timeout_ms)
        if color_image is None:
            raise RuntimeError("No color image available for viewer.")

    height, width = color_image.shape[:2]
    roi = resolve_roi(width, height, roi_spec=args.roi, roi_file=args.roi_file)
    scaled_intrinsics = intrinsics.scaled(width, height)

    viewer = ColorViewer(
        color_image,
        depth_image=depth_image,
        intrinsics=scaled_intrinsics,
        yolo_model=args.yolo_model,
        yolo_conf=args.yolo_conf,
        roi=roi,
        roi_file=args.roi_file or DEFAULT_ROI_FILE,
        output_dir=args.output_dir,
        fetch_frame_fn=fetch_frame,
        live=args.live,
        sam_refine=not args.no_sam_refine,
        sam_checkpoint=args.sam_checkpoint,
    )
    viewer.run()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Capture color/depth from Hikvision MV3D camera.")
    parser.add_argument("--device-index", type=int, default=0, help="Device index from list_devices().")
    parser.add_argument("--output-dir", default="output", help="Directory for saved images.")
    parser.add_argument("--timeout-ms", type=int, default=5000, help="FetchFrame timeout in ms.")
    parser.add_argument("--list-only", action="store_true", help="Only list devices and exit.")
    parser.add_argument("--gui", action="store_true", help="Open interactive color viewer with zoom and YOLO segmentation.")
    parser.add_argument("--live", action="store_true", help="With --gui, show live color preview until frozen.")
    parser.add_argument("--yolo-model", default="yolov8n-seg.pt", help="Ultralytics YOLO seg weights.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--roi",
        default=None,
        help="Fixed YOLO ROI as x1,y1,x2,y2 pixels or normalized ratios (0-1).",
    )
    parser.add_argument(
        "--roi-file",
        default=None,
        help="JSON ROI config path. Defaults to config/roi.json when --roi is omitted.",
    )
    parser.add_argument(
        "--no-sam-refine",
        action="store_true",
        help="Disable SAM refine after YOLO (CV center texture prompts -> SAM partition).",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default=None,
        help="Path to sam_vit_b_01ec64.pth (default: checkpoints/sam_vit_b_01ec64.pth).",
    )
    args = parser.parse_args()

    if args.live and not args.gui:
        parser.error("--live requires --gui")

    ret = Mv3dRgbd.MV3D_RGBD_Initialize()
    if ret != MV3D_RGBD_OK:
        raise RuntimeError(f"MV3D_RGBD_Initialize failed: 0x{ret:x}")

    try:
        print("Scanning for MV3D RGB-D devices...")
        devices = list_devices()
        if args.list_only:
            return 0

        if args.device_index >= len(devices):
            raise RuntimeError(f"Invalid device index {args.device_index}, found {len(devices)} device(s).")

        camera = Mv3dRgbd()
        selected = devices[args.device_index]
        print(f"Opening device [{args.device_index}] serial={selected['serial']} ...")

        ret = camera.MV3D_RGBD_OpenDevice(pointer(selected["info"]))
        if ret != MV3D_RGBD_OK:
            raise RuntimeError(f"MV3D_RGBD_OpenDevice failed: 0x{ret:x}")

        try:
            ret = camera.MV3D_RGBD_Start()
            if ret != MV3D_RGBD_OK:
                raise RuntimeError(f"MV3D_RGBD_Start failed: 0x{ret:x}")

            if args.gui:
                return run_gui(camera, args)

            color_image, depth_image = fetch_rgbd(camera, timeout_ms=args.timeout_ms)
            if color_image is None and depth_image is None:
                raise RuntimeError("Frame received, but no color or depth image was decoded.")

            save_outputs(color_image, depth_image, args.output_dir)
            print("Capture completed.")
            return 0
        finally:
            camera.MV3D_RGBD_Stop()
            camera.MV3D_RGBD_CloseDevice()
    finally:
        Mv3dRgbd.MV3D_RGBD_Release()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
