# YOLO 人脸检测集成说明

## 概述

本项目已成功集成 YOLOv8 进行人脸检测，相比传统的 `face_recognition` 库，YOLO 提供更快的检测速度和更高的准确率。

## 优势对比

| 特性 | face_recognition | YOLOv8 |
|------|------------------|--------|
| 检测速度 | 较慢 | 快 (3-5x) |
| 准确率 | 高 | 高 |
| CPU 性能 | 一般 | 优秀 |
| GPU 加速 | 支持 | 支持 (CUDA) |
| 多人脸检测 | 一般 | 优秀 |
| 复杂场景 | 一般 | 优秀 |

## 安装依赖

```bash
pip install ultralytics>=8.0.0
```

## 配置说明

在 `common/config.py` 中可以配置 YOLO 相关参数：

```python
# YOLO 人脸检测配置
USE_YOLO_DETECTION = True  # 是否使用 YOLO 进行人脸检测
YOLO_MODEL_SIZE = "n"      # 模型大小: "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (xlarge)
YOLO_CONFIDENCE = 0.5      # 置信度阈值
```

### 模型大小选择

| 模型 | 参数量 | 速度 | 准确率 | 推荐场景 |
|------|--------|------|--------|----------|
| YOLOv8n | 3.2M | 最快 | 较高 | 实时门禁、嵌入式设备 |
| YOLOv8s | 11.2M | 快 | 高 | 一般场景 |
| YOLOv8m | 25.9M | 中等 | 很高 | 高精度需求 |
| YOLOv8l | 43.7M | 较慢 | 极高 | 离线分析 |
| YOLOv8x | 68.2M | 最慢 | 最高 | 研究/竞赛 |

**推荐**: 门禁系统使用 `YOLOv8n` (nano)，平衡速度和准确率。

## 使用方法

### 1. 自动启用 (推荐)

修改 `config.py` 即可全局启用：

```python
USE_YOLO_DETECTION = True
```

### 2. 编程方式控制

```python
from gate.access_control import AccessControl

# 创建门禁控制器时指定
access_control = AccessControl(
    user_manager=user_manager,
    use_yolo=True,
    yolo_model_size="n",
    yolo_confidence=0.5
)

# 动态启用/禁用
access_control.set_yolo_enabled(True)

# 获取当前检测方法
method = access_control.get_detection_method()  # "YOLO" 或 "face_recognition"
```

### 3. 仅使用 YOLO 检测器

```python
from common.yolo_detector import YOLOFaceDetector

# 创建检测器
detector = YOLOFaceDetector(model_size="n", confidence=0.5)

# 检测人脸
faces = detector.detect(image)  # 返回 [(x1, y1, x2, y2, confidence), ...]

# 提取人脸区域
for x1, y1, x2, y2, conf in faces:
    face_img = detector.extract_face(image, (x1, y1, x2, y2))
```

## 文件结构

```
人脸识别/
├── common/
│   ├── config.py              # 配置文件 (包含 YOLO 配置)
│   └── yolo_detector.py       # YOLO 检测器模块 (新增)
├── gate/
│   └── access_control.py      # 门禁控制 (已集成 YOLO)
├── trainer/
│   └── inference.py           # 推理模块 (已集成 YOLO)
├── test_yolo.py               # YOLO 测试脚本 (新增)
├── YOLO_INTEGRATION.md        # 本文档 (新增)
└── requirements.txt           # 已添加 ultralytics
```

## 测试

运行 YOLO 集成测试：

```bash
python test_yolo.py
```

测试内容：
1. YOLO 检测器初始化
2. 实时人脸检测
3. FPS 性能测试
4. 门禁系统集成测试

## 性能优化建议

### 1. GPU 加速

如果有 NVIDIA GPU，可以启用 CUDA 加速：

```python
# 在 yolo_detector.py 中
self.device = "cuda"  # 改为 "cuda"
```

### 2. 分辨率调整

降低输入分辨率可以提高速度：

```python
# 在 _detect_faces_yolo 方法中
small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # 可调整为 0.33
```

### 3. 处理间隔调整

调整 `access_control.py` 中的处理间隔：

```python
self.process_interval = 0.3  # 从 0.5 改为 0.3，提高响应速度
```

## 故障排除

### 问题1: 导入错误

```
ImportError:请安装 ultralytics: pip install ultralytics
```

**解决**: 运行 `pip install ultralytics>=8.0.0`

### 问题2: 模型下载失败

```
YOLO initialization failed
```

**解决**:
1. 检查网络连接
2. 手动下载模型到 `models/` 目录
3. 或使用国内镜像源

### 问题3: 检测不到人脸

**可能原因**:
- 置信度阈值过高 → 降低 `YOLO_CONFIDENCE`
- 光线不足 → 改善光照条件
- 人脸角度过大 → 调整摄像头角度

## 回退方案

如果 YOLO 工作不正常，可以快速回退到 `face_recognition`：

```python
# config.py
USE_YOLO_DETECTION = False  # 禁用 YOLO
```

或在运行时动态切换：

```python
access_control.set_yolo_enabled(False)  # 切换到 face_recognition
```

## 更新日志

- **2026-07-14**: 初始集成 YOLOv8 人脸检测
  - 添加 `common/yolo_detector.py`
  - 更新 `gate/access_control.py`
  - 更新 `trainer/inference.py`
  - 添加 `test_yolo.py` 测试脚本
  - 更新 `requirements.txt`
