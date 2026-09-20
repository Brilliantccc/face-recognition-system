# 人脸识别门禁系统

<div align="center">

**基于 YOLO 检测 + MobileFaceNet/InsightFace 识别的模块化人脸识别门禁系统**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16+-005CED.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 功能特性

- **模块化架构**：检测器 / 识别器 / 数据库可插拔，自由组合
- **三种识别模型**：MobileFaceNet（自训练）、InsightFace（预训练高精度）、face_recognition（兜底）
- **三种检测引擎**：YOLO（快速）、face_recognition（精确）、Haar Cascade（轻量）
- **人事管理 GUI**：员工管理、工作照上传裁剪、编码生成、模型管理
- **门禁 GUI**：Kiosk 沉浸式风格、实时识别、门禁开关控制
- **模型管理**：一键下载 InsightFace buffalo_l，断点续传，暂停/取消
- **首次运行向导**：自动检测缺失组件并引导配置
- **多平台部署**：PyQt5 GUI / Python ONNX / C++ ONNX

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 首次启动

```bash
python admin/main.py
```

首次启动会自动检测缺失的模型和依赖，弹出配置向导：

| 缺失项 | 向导操作 |
|--------|---------|
| PyTorch / ultralytics | 自动 pip install |
| YOLO 检测模型 | 自动下载 |
| InsightFace 识别模型 | 自动下载 buffalo_l (~300MB) |

### 3. 注册人员

启动人事管理系统后：

- **快速注册**：点击「快速注册」 → 填写信息 → 摄像头拍照（至少 5 张）
- **批量导入**：准备文件夹 `照片文件夹/姓名/照片.jpg`，点击「批量导入」
- **导入 uploads**：将照片放入 `data/uploads/姓名/`，点击「导入 uploads」

### 4. 生成编码

注册完成后，点击「生成编码」，选择模型：

| 模型 | 维度 | 特点 | 获取方式 |
|------|------|------|---------|
| **InsightFace** | 512 维 | 预训练高精度（推荐） | GUI 内一键下载 |
| **MobileFaceNet** | 128 维 | 自训练模型，轻量级 | 自训练生成 |
| face_recognition | 128 维 | 兜底方案，基于 dlib | pip install |

不同模型的编码独立存储，互不干扰。

### 5. 启动门禁

```bash
python gate/main.py
```

门禁界面右上角设置按钮可切换识别模型和检测引擎。

---

## 模型权重下载

| 模型 | 下载地址 | 说明 |
|------|---------|------|
| MobileFaceNet (v2) | [ModelScope](https://www.modelscope.cn/models/Brilliantccc/facenet-optimal-v2) | 自训练模型，10575 类，128 维 |
| InsightFace buffalo_l | [GitHub Releases](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) | 预训练 ArcFace-R100，512 维 |
| YOLOv11l-face | [Ultralytics](https://github.com/ultralytics/assets/releases) | 人脸检测模型 |

### 下载后放置路径

```
deploy/models/
├── buffalo_l/                    # InsightFace
│   ├── w600k_r100.onnx
│   └── w600k_r50.onnx
├── mobilefacenet.onnx            # MobileFaceNet ONNX
└── yolov11l-face.onnx            # YOLO 检测 ONNX

trainer/models/
├── yolov11l-face.pt              # YOLO 检测 PyTorch
└── saved/                        # 训练产出
    └── optimal_v2/
        ├── best_model.pth
        └── inference_model.pth
```

---

## 模块化配置

编辑 `common/config.py` 切换检测器 / 识别器：

```python
# 检测器: "yolo", "face_recognition", "haar", "auto"
DETECTION_BACKEND = "auto"

# 识别器: "mobilenet", "insightface", "face_recognition", "auto"
RECOGNITION_BACKEND = "auto"
```

| 组合 | 检测器 | 识别器 | 说明 |
|------|--------|--------|------|
| 推荐 | yolo | insightface | 最高精度，需下载 buffalo_l |
| 默认 | yolo | mobilenet | 自训练模型，开箱即用 |
| 回退 | face_recognition | face_recognition | 无需 GPU，基于 dlib |
| 自动 | auto | auto | 按优先级自动选择 |

---

## 部署方式

| 方式 | 启动命令 | 适用场景 |
|------|----------|---------|
| PyQt5 GUI | `python gate/main.py` | 开发调试，GUI 切换模型 |
| Python ONNX | `python deploy/gate_onnx.py` | 高性能部署 |
| C++ ONNX | `deploy/启动门禁系统.bat` | 无 Python 环境，最高性能 |

### C++ 编译

```bash
# 需要 Visual Studio 2022 + CMake + vcpkg
cd deploy
.\build.bat
.\启动门禁系统.bat
```

---

## 训练模型（可选）

```bash
# 1. 准备数据（下载 CASIA-WebFace 解压到 trainer/data/raw/）
python trainer/preprocess.py

# 2. 训练（RTX 3060 优化版）
python trainer/train_optimal.py --epochs 100

# 3. 测试
python trainer/inference.py
```

详见 [trainer/README.md](trainer/README.md)

---

## 项目结构

```
人脸识别/
├── common/                      # 公共模块（可插拔架构）
│   ├── detectors/               # 人脸检测器：yolo / face_recognition / haar
│   ├── recognizers/             # 人脸识别器：mobilenet / insightface / face_recognition
│   ├── databases/               # 人脸数据库：embeddings_bin / sqlite
│   ├── config.py                # 系统配置
│   ├── database.py              # SQLite 数据库（按模型区分编码）
│   └── torch_utils.py           # PyTorch 环境管理
├── admin/                       # 人事管理系统
│   ├── gui/admin_window.py      # GUI 界面（自适应缩放）
│   ├── setup_wizard.py          # 首次运行配置向导
│   ├── generate_encodings.py    # 编码生成（多模型支持）
│   └── batch_register.py        # 批量注册
├── gate/                        # 门禁系统
│   ├── gui/gate_window.py       # GUI 界面（Kiosk 沉浸式）
│   └── access_control.py        # 核心控制逻辑
├── trainer/                     # 模型训练
│   ├── models/facenet.py        # MobileFaceNet 模型定义
│   ├── train_optimal.py         # 主力训练脚本（RTX 3060 优化）
│   └── preprocess.py            # 数据预处理
├── deploy/                      # 部署（ONNX / C++）
│   ├── gate_onnx.py             # ONNX 独立门禁
│   ├── config/config.json       # ONNX 门禁配置
│   ├── models/                  # ONNX 模型 + embeddings
│   └── src/ + include/          # C++ 源码
├── data/                        # 运行数据（.gitignore）
│   ├── db/face_access.db        # 主数据库
│   └── faces/                   # 用户人脸照片
└── docs/                        # 模型分析文档
```

---

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 运行环境 |
| PyTorch | 2.0+ | 模型推理（可选 GPU） |
| PyQt5 | 5.15+ | GUI 界面 |
| ultralytics | 8.0+ | YOLO 检测 |
| OpenCV | 4.0+ | 图像处理 |
| ONNX Runtime | 1.16+ | 高性能推理（可选） |

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 摄像头打不开 | 检查摄像头连接，修改 `config.py` 中 `CAMERA_INDEX` |
| 识别不准 | 增加注册照片（建议每人 20 张+），或切换到 insightface 模型 |
| YOLO 加载失败 | 确认 `trainer/models/yolov11l-face.pt` 存在，或安装 ultralytics |
| InsightFace 不可用 | 通过「模型管理」一键下载，或手动下载到 `deploy/models/buffalo_l/` |
| 编码维度不匹配 | 确保门禁识别器与编码模型一致（mobilenet=128维, insightface=512维） |
| 首次启动无模型 | 系统会弹出配置向导，按提示下载即可 |
| ONNX 门禁找不到编码 | 检查 `deploy/config/config.json` 中 `database_path` |

---

## 相关链接

- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ModelScope 模型库](https://www.modelscope.cn/models/Brilliantccc/facenet-optimal-v2)

---

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

仅供个人学习和研究使用，商用需获得作者书面授权。
