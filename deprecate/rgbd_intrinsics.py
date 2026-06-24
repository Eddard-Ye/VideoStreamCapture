# -*- coding: utf-8 -*-
"""Load Hikvision MV3D RGB camera intrinsics via RGB-D SDK."""

from __future__ import annotations

from ctypes import pointer

import numpy as np

from camera_intrinsics import RgbIntrinsics
from Mv3dRgbdImport.Mv3dRgbdApi import Mv3dRgbd
from Mv3dRgbdImport.Mv3dRgbdDefine import (
    MV3D_RGBD_CAMERA_PARAM,
    MV3D_RGBD_FLOAT_Z_UNIT,
    MV3D_RGBD_OK,
    MV3D_RGBD_PARAM,
    ParamType_Float,
)


def get_rgb_intrinsics(camera: Mv3dRgbd) -> RgbIntrinsics:
    camera_param = MV3D_RGBD_CAMERA_PARAM()
    ret = camera.MV3D_RGBD_GetCameraParam(pointer(camera_param))
    if ret != MV3D_RGBD_OK:
        raise RuntimeError(f"MV3D_RGBD_GetCameraParam failed: 0x{ret:x}")

    intrinsic = camera_param.stRgbCalibInfo.stIntrinsic.fData
    fx = float(intrinsic[0])
    fy = float(intrinsic[4])
    cx = float(intrinsic[2])
    cy = float(intrinsic[5])
    calib_w = int(camera_param.stRgbCalibInfo.nWidth)
    calib_h = int(camera_param.stRgbCalibInfo.nHeight)

    z_unit = 1.0
    param = MV3D_RGBD_PARAM()
    param.enParamType = ParamType_Float
    ret = camera.MV3D_RGBD_GetParam(MV3D_RGBD_FLOAT_Z_UNIT, pointer(param))
    if ret == MV3D_RGBD_OK:
        z_unit = float(param.ParamInfo.stFloatParam.fCurValue)
        if not np.isfinite(z_unit) or z_unit <= 0:
            z_unit = 1.0

    return RgbIntrinsics(fx, fy, cx, cy, calib_w, calib_h, z_unit)
