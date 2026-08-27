# Face Recognition Gate System (C++ Deployment)

高性能 C++ 人脸识别门禁系统，基于 ONNX Runtime GPU 加速。

## 功能特性

- 实时人脸检测 (YOLOv8n)
- 人脸识别 (MobileFaceNet)
- GPU 加速推理 (ONNX Runtime CUDA)
- 人脸数据库匹配
- 门禁控制（识别成功后开门信号 + 冷却期）

## 环境依赖

### 必需软件

| 软件 | 版本 | 说明 | 下载地址 |
|------|------|------|---------|
| Windows | 10/11 | 操作系统 | - |
| Visual Studio | 2022 | C++ 编译器 | https://visualstudio.microsoft.com/ |
| CMake | 3.20+ | 构建工具 | https://cmake.org/download/ |
| vcpkg | 最新 | C++ 包管理器 | https://github.com/microsoft/vcpkg |
| Python | 3.8+ | 导出脚本 | https://www.python.org/ |

### GPU 依赖（可选，用于 GPU 加速）

| 软件 | 版本 | 说明 | 下载地址 |
|------|------|------|---------|
| NVIDIA 驱动 | 最新 | GPU 驱动 | https://www.nvidia.com/drivers |
| CUDA Toolkit | 11.8 或 12.x | GPU 计算 | https://developer.nvidia.com/cuda-downloads |
| cuDNN | 8.x 或 9.x | 深度学习加速 | https://developer.nvidia.com/cudnn |

> ⚠️ 不安装 GPU 依赖也可运行，程序会自动回退到 CPU 模式。

### C++ 依赖库（通过 vcpkg 安装）

| 库 | 用途 | 安装命令 |
|----|------|---------|
| OpenCV 4.x | 图像处理、摄像头 | `vcpkg install opencv4:x64-windows` |
| ONNX Runtime GPU | 模型推理 | `vcpkg install onnxruntime-gpu:x64-windows` |

### 依赖关系图

```
                    ┌─────────────────┐
                    │   face_engine   │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│    OpenCV     │   │ ONNX Runtime  │   │   Windows     │
│  (图像处理)   │   │  (模型推理)   │   │   API         │
└───────────────┘   └───────┬───────┘   └───────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                    ▼               ▼
            ┌───────────────┐ ┌───────────────┐
            │     CUDA      │ │    cuDNN      │
            │  (GPU 计算)   │ │ (深度学习)    │
            └───────────────┘ └───────────────┘
```

## 快速开始

### 1. 安装环境

#### 1.1 安装 Visual Studio 2022

下载并安装 [Visual Studio 2022](https://visualstudio.microsoft.com/)，确保勾选 **"使用 C++ 的桌面开发"** 工作负载。

#### 1.2 安装 CMake

下载并安装 [CMake 3.20+](https://cmake.org/download/)，安装时勾选 "Add CMake to the system PATH"。

#### 1.3 安装 vcpkg

```bash
cd D:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
```

#### 1.4 安装 C++ 依赖库

```bash
# OpenCV（图像处理）
.\vcpkg install opencv4:x64-windows

# ONNX Runtime GPU（模型推理，支持 GPU 加速）
.\vcpkg install onnxruntime-gpu:x64-windows
```

#### 1.5 安装 GPU 依赖（可选）

如果需要 GPU 加速：

1. 安装 [NVIDIA 驱动](https://www.nvidia.com/drivers)
2. 安装 [CUDA Toolkit 11.8 或 12.x](https://developer.nvidia.com/cuda-downloads)
3. 安装 [cuDNN](https://developer.nvidia.com/cudnn)

> 检查 GPU 是否可用：`nvidia-smi`

### 2. 编译项目

```bash
cd deploy
.\build.bat
```

### 3. 运行

```bash
# 方式1: 双击启动脚本
启动门禁系统.bat

# 方式2: 命令行
cd deploy
bin\Release\face_engine.exe
```

## 项目结构

```
deploy/
├── CMakeLists.txt          # CMake 配置
├── build.bat               # 构建脚本
├── README.md               # 文档
├── gate_onnx.py            # Python ONNX 版本（备用）
├── 启动门禁系统.bat          # 启动脚本
├── export_database.py      # 导出数据库脚本
├── config/
│   └── config.json         # 配置文件
├── models/                 # 模型文件（不上传）
│   ├── mobilefacenet.onnx  # 识别模型
│   ├── yolov8n.onnx        # 检测模型
│   └── embeddings.bin      # 人脸数据库
├── include/
│   ├── face_engine.h       # 主接口
│   ├── detector.h          # 检测器
│   ├── recognizer.h        # 识别器
│   ├── database.h          # 数据库
│   └── preprocess.h        # 预处理
└── src/
    ├── main.cpp            # 主程序
    ├── face_engine.cpp     # 引擎实现
    ├── detector.cpp        # 检测器实现
    ├── recognizer.cpp      # 识别器实现
    ├── database.cpp        # 数据库实现
    └── preprocess.cpp      # 预处理实现
```

## 模型说明

模型文件需要单独准备，不包含在代码中：

| 文件 | 说明 | 来源 |
|------|------|------|
| mobilefacenet.onnx | 人脸识别模型 | 从 PyTorch 模型导出 |
| yolov8n.onnx | 人脸检测模型 | YOLOv8 预训练 |
| embeddings.bin | 人脸数据库 | 从 SQLite 导出 |

### 导出模型

```bash
# 导出识别模型
python deploy/export_recognizer.py

# 导出检测模型
python deploy/export_detector.py

# 导出人脸数据库
python deploy/export_database.py
```

## 门禁控制逻辑

1. 识别到已注册人员后，发送开门信号
2. 进入 5 秒冷却期，期间不进行检测
3. 冷却期结束后恢复检测

```cpp
// 门禁控制代码 (src/main.cpp)
if (r.is_known && r.similarity > config.recognition_threshold) {
    std::cout << "[GATE] Access granted: " << r.name << std::endl;
    std::cout << "[SIGNAL] >>> OPEN GATE <<<" << std::endl;
    gate_opened = true;
    gate_open_time = std::chrono::steady_clock::now();
}
```

## 注册新人员

1. 在 Python 系统中注册人员
2. 导出新的 embeddings.bin：
   ```bash
   python deploy/export_database.py
   ```
3. 重启 C++ 门禁系统

## 配置说明

编辑 `config/config.json`：

```json
{
    "detector_model": "models/yolov8n.onnx",
    "recognizer_model": "models/mobilefacenet.onnx",
    "database_path": "models/embeddings.bin",
    "recognition_threshold": 0.6,
    "use_gpu": true,
    "camera_id": 0,
    "input_width": 640,
    "input_height": 480
}
```

## 性能指标

- 检测延迟: ~8-10ms (GPU)
- 识别延迟: ~5-8ms (GPU)
- 总延迟: ~15-20ms
- FPS: 50-60

## 常见问题

### GPU 不可用

```bash
nvidia-smi  # 检查 GPU 驱动
```

### 编译错误

1. 确认 vcpkg 安装在 `D:\vcpkg`
2. 检查 CMake 版本: `cmake --version`
3. 确保安装了 VS 2022 C++ 工作负载

### FPS 降低

1. 降低输入分辨率
2. 检查 GPU 占用: `nvidia-smi`
3. 尝试 CPU 模式

## License

Internal use only.
