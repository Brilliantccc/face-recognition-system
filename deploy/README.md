# C++ 部署版本

高性能 C++ 人脸识别门禁系统，基于 ONNX Runtime GPU 加速。

## 环境依赖

| 软件 | 说明 |
|------|------|
| Visual Studio 2022 | C++ 编译器 |
| CMake 3.20+ | 构建工具 |
| vcpkg | C++ 包管理器 |
| Python 3.8+ | 导出脚本 |

### GPU 依赖（可选）

| 软件 | 说明 |
|------|------|
| NVIDIA 驱动 | GPU 驱动 |
| CUDA Toolkit | GPU 计算 |
| cuDNN | 深度学习加速 |

## 快速开始

### 1. 安装 vcpkg

```bash
cd D:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

### 2. 安装依赖库

```bash
.\vcpkg install opencv4:x64-windows
.\vcpkg install onnxruntime-gpu:x64-windows
```

### 3. 编译

```bash
cd deploy
.\build.bat
```

### 4. 运行

```bash
双击 启动门禁系统.bat
```

## 注册新人员

C++ 版本只负责识别，注册仍在 Python 系统中：

1. 在 Python 人事管理系统中注册人员
2. 生成编码：`python admin/generate_encodings.py --model mobilenet`
3. 导出编码：`python deploy/export_database.py`
4. 重启 C++ 门禁系统

## 配置

编辑 `config/config.json`：

```json
{
    "detector_model": "models/yolov11l-face.onnx",
    "recognizer_model": "models/mobilefacenet.onnx",
    "database_path": "models/embeddings_mbn.bin",
    "recognition_threshold": 0.55,
    "use_gpu": true,
    "camera_id": 0,
    "input_width": 640,
    "input_height": 480
}
```

## 模型文件

```
deploy/models/
├── yolov11l-face.onnx       # YOLO 人脸检测模型（97MB）
├── mobilefacenet.onnx        # MobileFaceNet 识别模型（3.8MB）
├── embeddings_mbn.bin        # MobileFaceNet 人脸编码（128维）
├── embeddings_if.bin         # InsightFace 人脸编码（512维，可选）
└── buffalo_l/                # InsightFace 预训练模型（可选）
    ├── w600k_r100.onnx       # ArcFace-R100 识别模型
    └── ...
```

## 性能

- 检测延迟: ~8-10ms (GPU)
- 识别延迟: ~5-8ms (GPU)
- 总延迟: ~15-20ms
- FPS: 50-60
