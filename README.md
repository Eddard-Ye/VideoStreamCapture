# BreadTestingBackend

海康威视 **2D 工业相机** + YOLO 分割 + HTTP 推流与远程采集的后端服务，用于面包尺寸检测与水切宽度分析。

## 功能

- 实时 MJPEG 推流（YOLO 实例分割 + 旋转框 L×W 标注）
- 2D 棋盘格标定，像素尺寸转毫米
- HTTP 控制 API：状态查询、快照、水切计算、带摘要信息的截图保存
- 可选 SAM（Segment Anything）精修水切区域

## 环境要求

| 依赖 | 说明 |
|------|------|
| Python 3.10+ | 推荐 Conda 环境 |
| [海康 MVS SDK](https://www.hikrobotics.com/cn/machinevision/service/download) | 2D 相机 `MvCameraControl.dll`，安装后需能正常枚举 GigE/USB 设备 |
| CUDA（可选） | 加速 YOLO / SAM；无 GPU 时使用 CPU |

## 安装

```powershell
conda create -n ScanSize python=3.11 -y
conda activate ScanSize
pip install -r requirements.txt
```

安装 PyTorch 时若需 GPU，请按 [pytorch.org](https://pytorch.org) 选择对应 CUDA 版本后再装其余依赖。

## 模型权重（不入库，需本地准备）

首次运行会自动下载 YOLO；SAM 在水切功能首次触发时下载（需联网）。

| 文件 | 用途 | 获取方式 |
|------|------|----------|
| `yolov8n-seg.pt` | 默认分割模型 | Ultralytics 首次运行自动下载 |
| `checkpoints/sam_vit_b_01ec64.pth` | 水切 SAM 模型 (~358 MB) | 首次水切自动下载，或手动放置 |

SAM 手动下载：<https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>

## 快速开始

### 1. 启动推流服务

```powershell
conda activate ScanSize
python capture_2d_stream.py
```

默认监听 `0.0.0.0:8080`。局域网访问请使用本机 IP（非 `127.0.0.1`）。

常用参数：

```powershell
python capture_2d_stream.py --stream-width 720 --fps 10
python capture_2d_stream.py --no-calib          # 仅显示像素，不用标定
python capture_2d_stream.py --calib-file config/camera_calib_2d.json
```

### 2. 浏览器 / 客户端

- 视频流：`http://<host>:8080/video`
- 状态 JSON：`http://<host>:8080/status`
- 快照：`http://<host>:8080/snapshot.jpg`

### 3. CLI 控制（可选）

```powershell
python stream_control.py status
python stream_control.py water-cut
python stream_control.py snapshot --output snap.jpg
python stream_control.py capture --name bread01 --height 10.0mm --temperature 27.3 --weight 14.2g --water-cut
```

### 4. 2D 相机标定

```powershell
# 采集标定图到 calib_images/
python camera_calib_2d.py capture --device-index 0

# 标定并写入 config/camera_calib_2d.json
python camera_calib_2d.py calibrate --pattern-cols 8 --pattern-rows 11 --square-size 6.0

# 验证
python camera_calib_2d.py validate
```

### 5. 本地 GUI 预览（调试）

```powershell
python capture_2d.py --gui
```

## HTTP API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/video` | MJPEG 实时流 |
| GET | `/snapshot.jpg` | 当前帧 JPEG |
| GET | `/status` | 检测实例、FPS、水切状态 |
| POST | `/water-cut` | 对当前帧计算水切并叠加 |
| POST | `/water-cut/clear` | 清除水切叠加 |
| POST | `/capture` | 保存带摘要的截图 |
| GET | `/captures/{fileName}` | 读取已保存截图 |

**POST `/capture` 请求体示例：**

```json
{
  "name": "bread01",
  "height": "10.0mm",
  "temperature": "27.3",
  "weight": "14.2g",
  "water_cut": true
}
```

响应含 `fileName`、`length_mm`、`width_mm`、`water_cut_mm` 等字段。文件保存在 `output/captures/`。

## 项目结构

```
BreadTestingBackend/
├── capture_2d_stream.py   # 主入口：相机 + YOLO + HTTP
├── stream_server.py       # HTTP 服务
├── stream_overlay.py      # 画面叠加与截图摘要
├── stream_control.py      # HTTP API 命令行客户端
├── capture_2d.py          # 2D 相机 SDK 封装 / GUI
├── camera_calib_2d.py     # 棋盘格标定
├── color_viewer.py        # YOLO 分割器、GUI 组件
├── object_measure.py      # 旋转框与尺寸计算
├── yolo_sam_refine.py     # SAM 水切精修
├── sam_centerline.py      # 水切宽度分析
├── MvImport/              # 海康 MVS 2D SDK Python 绑定
├── config/
│   ├── camera_calib_2d.json
│   └── roi.json
├── checkpoints/           # SAM 权重（.gitignore）
├── output/captures/       # 截图输出（.gitignore）
└── deprecate/             # 旧版 RGB-D、外设脚本
```

## deprecate 目录

以下代码**不参与** `capture_2d_stream` 主流程，保留作参考：

- `capture_rgbd.py` — MV3D RGB-D 相机
- `Mv3dRgbdImport/` — RGB-D SDK 绑定
- `rgbd_intrinsics.py` — RGB-D 出厂内参
- `rs485_temperature.py`、`km11_weight.py`、`ld_bg_laser_distance.py` — 外设
- `extract_center_sam.py` — 旧版 SAM 中心线脚本

## 许可证说明

- 本项目代码：按仓库约定使用
- 海康 SDK、`MvImport/`：遵循海康机器人 SDK 许可
- YOLO（Ultralytics）、SAM（Meta）：遵循各自开源协议
