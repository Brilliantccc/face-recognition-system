# 人脸识别门禁系统 v3.2

基于 face_recognition (dlib) 和 MobileFaceNet + ArcFace Loss 的人脸识别门禁系统，支持双模型切换、YOLO 检测。

## 系统架构

```
人脸识别/
├── common/                        # 公共模块
│   ├── database.py               # 数据库操作
│   ├── user_manager.py           # 用户管理
│   ├── config.py                 # 系统配置
│   ├── yolo_detector.py          # YOLO 人脸检测
│   └── torch_utils.py            # PyTorch 工具
│
├── admin/                         # 人事管理模块
│   ├── main.py                   # 管理程序入口
│   ├── generate_encodings.py     # 人脸编码生成（支持双模型）
│   ├── batch_register.py         # 批量导入注册
│   └── gui/admin_window.py       # 管理界面
│
├── gate/                          # 门禁系统模块
│   ├── main.py                   # 门禁程序入口
│   ├── access_control.py         # 门禁控制逻辑
│   └── gui/gate_window.py        # 门禁界面（沉浸式 Kiosk 风格）
│
├── trainer/                       # 训练模块（命令行）
│   ├── train_optimal.py          # 训练脚本（推荐）
│   ├── train.py                  # 训练脚本（通用）
│   ├── train_simple.py           # 训练脚本（简化版）
│   ├── preprocess.py             # 数据预处理
│   ├── inference.py              # 模型推理
│   ├── test_model.py             # 模型测试
│   └── models/facenet.py         # MobileFaceNet + ArcFaceLoss
│
├── data/                          # 运行时数据（不提交）
├── 启动系统.bat                   # 综合启动器
├── 启动人事管理系统.bat
├── 启动门禁系统.bat
├── requirements.txt
└── yolov8n.pt
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速上手

### 第一步：注册人员

启动人事管理系统（双击 `启动人事管理系统.bat` 或命令行）：

```bash
python admin/main.py
```

三种注册方式：

| 方式 | 操作 |
|------|------|
| **快速注册** | 点击"⚡ 快速注册"→ 填写信息 → 摄像头拍照 → 一键注册 |
| **批量导入** | 点击"📂 批量导入"→ 选择文件夹（格式：`文件夹/姓名/照片`） |
| **导入 uploads** | 将照片放入 `data/uploads/姓名/照片.jpg` → 点击"📥 导入 uploads" |

> 💡 批量导入和导入 uploads 只录入姓名。如需补充工号、部门、电话，在列表中选中员工后点击"✏️ 编辑选中员工"。

### 第二步：生成编码

注册人员后，需要为人脸生成编码（特征向量）：

1. 在人事管理系统中点击 **"🧠 生成编码"**
2. 选择编码模型：
   - **face_recognition**（推荐）：基于 dlib，128维，适合门禁验证
   - **训练模型**：MobileFaceNet，256维，适合分类任务
3. 等待生成完成

> ⚠️ 切换模型时会清除旧编码并重新生成。同一模型重复生成会跳过（除非强制）。

### 第三步：启动门禁

启动门禁系统（双击 `启动门禁系统.bat` 或命令行）：

```bash
python gate/main.py
```

- 门禁启动后自动加载已注册用户
- 注册用户自动通过，未注册人员自动拒绝
- 可在设置面板切换人脸检测方式（face_recognition / YOLO）

> 🔒 严格安全模式：只允许数据库中已激活的用户通过，即使模型识别到相似人脸，若不在数据库中也会被拒绝。

## 人脸检测方式

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| **face_recognition** | 基于 dlib，CPU 运行 | 默认方式，稳定可靠 |
| **YOLO** | 基于 YOLOv8，支持 GPU 加速 | 需要更快检测速度时 |

在门禁界面右上角设置面板中切换检测方式。

## 人脸编码模型

| 模型 | 维度 | 说明 |
|------|------|------|
| **face_recognition** | 128维 | 基于 dlib，专为验证任务设计，推荐用于门禁 |
| **训练模型** | 256维 | MobileFaceNet + ArcFace，为分类任务设计 |

> ⚠️ 两种模型的编码维度不同，切换模型时需要重新生成编码。

### 预训练模型下载

如果你不想自己训练，可以直接下载预训练模型：

| 模型 | 说明 | 下载地址 |
|------|------|---------|
| facenet-optimal-v2 | MobileFaceNet + ArcFace，10575类 | [ModelScope](https://www.modelscope.cn/models/Brilliantccc/facenet-optimal-v2) |

下载后将模型文件放到 `trainer/models/saved/optimal_v2/` 目录下即可。

！！！注意！！！作者训练出的模型更适合人脸分类，可能不太适合本项目，使用后的实际效果也不好，所以本项目下的训练方式也不适合，慎用！！

## 自己训练模型（慎用）

> 门禁系统支持两种识别方式，可在下拉框自由切换：
> - **face_recognition**（内置，无需训练，开箱即用）
> - **MobileFaceNet**（你训练的模型，精度更高）
>
> 有训练模型时默认用 MobileFaceNet，没有时自动使用 face_recognition。

### 置信度说明

| 识别方式 | 置信度含义 | 通过条件 |
|---------|-----------|---------|
| 训练模型 | 余弦相似度（0~1） | 相似度 ≥ 训练时的最佳阈值 |
| face_recognition | 欧氏距离 | 距离 ≤ 0.6 |

训练模型的阈值会在训练时自动计算并保存到 `model_info.json`，显示在门禁状态栏中。

### 前置条件

- 已安装 PyTorch
- 有 GPU 可用（推荐，CPU 也能训练但很慢）

### 第一步：准备训练数据

训练模型需要人脸数据集。推荐使用公开数据集，也可使用自己的照片。

#### 方式一：下载公开数据集（推荐）

**推荐数据集：**

| 数据集 | 人数 | 图片数 | 下载地址 |
|--------|------|--------|---------|
| CASIA-WebFace | 10,575 | ~500K | https://blog.csdn.net/sscc_learning/article/details/87003874 |
| LFW | 5,749 | ~13K | http://vis-www.cs.umass.edu/lfw/ |
| MegaFace | 672K | ~4.7M | https://megaface.cs.washington.edu/ |

以 **CASIA-WebFace** 为例（最常用）：

```bash
# 1. 下载并解压 CASIA-WebFace 数据集
#    解压后目录结构为：CASIA-WebFace/0000001/001.jpg, 002.jpg, ...

# 2. 将解压后的文件夹放到 trainer/data/raw/ 下
#    trainer/data/raw/CASIA-WebFace/0000001/001.jpg, ...

# 3. 运行预处理脚本（自动检测人脸、裁剪对齐、划分训练集/验证集）
python trainer/preprocess.py
```

预处理脚本参数：

```bash
python trainer/preprocess.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--raw-dir` | `trainer/data/raw/CASIA-WebFace` | 原始数据目录 |
| `--output-dir` | `trainer/data/aligned` | 输出目录 |
| `--size` | 112 | 人脸图片尺寸 |
| `--val-ratio` | 0.1 | 验证集比例（10%） |
| `--method` | yolo | 人脸检测方法（yolo 或 face_recognition） |

预处理后的目录结构：

```
trainer/data/aligned/
├── train/
│   ├── 0000001/
│   │   ├── 001.jpg   (112×112 对齐后)
│   │   └── ...
│   └── ...
└── val/
    ├── 0000001/
    └── ...
```

### 第二步：开始训练

```bash
# 推荐：GPU 训练（自动使用 AMP 加速，默认读取 trainer/data/aligned/）
python trainer/train_optimal.py --epochs 100 --batch-size 128 --lr 0.01

# CPU 训练（较慢）
python trainer/train_simple.py --epochs 50 --batch-size 32 --lr 0.001
```

> 训练脚本默认从 `trainer/data/aligned/` 读取数据。如果使用自定义目录，用 `--data-dir` 参数指定。

**训练参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数，数据少时可设 50~100，数据多时可设 100~300 |
| `--batch-size` | 128 | 批次大小，显存不够时改小（如 64、32） |
| `--lr` | 0.01 | 学习率，一般用默认值即可 |
| `--resume` | False | 从上次中断处继续训练 |

**训练脚本选择：**

| 脚本 | 适用场景 |
|------|---------|
| `train_optimal.py` | 有 GPU（推荐，支持 AMP 加速） |
| `train.py` | 有 GPU 但 AMP 有问题时 |
| `train_simple.py` | 只有 CPU，或调试用 |

**断点续训：**

训练中断（关闭窗口、断电等）不会丢失进度，恢复命令：

```bash
python trainer/train_optimal.py --resume
```

### 第三步：测试模型

训练完成后，测试模型识别效果：

```bash
python trainer/test_model.py --model-dir trainer/models/saved/最优模型目录
```

- 自动在验证集上计算识别准确率
- 准确率 80% 以上为正常，90% 以上为优秀

### 第四步：在门禁中使用

1. 启动门禁系统：`python gate/main.py`
2. 在界面右上角 **"识别模型"** 下拉框中选择你训练的模型
3. 门禁自动加载模型，开始使用新模型识别

> 模型保存在 `trainer/models/saved/` 目录下，门禁会自动扫描该目录下所有可用模型。

### 训练常见问题

| 问题 | 解决方案 |
|------|---------|
| `PyTorch 未安装` | 运行 `pip install torch torchvision` |
| `CUDA out of memory` | 减小 batch-size，如 `--batch-size 32` |
| `没有找到训练数据` | 先运行 `preprocess.py` 处理数据集，或检查目录结构是否正确 |
| `准确率很低` | 增加每人照片数量（建议 30+ 张），增加训练轮数 |
| 训练中断了 | 运行 `python trainer/train_optimal.py --resume` 继续 |
| `preprocess.py 报错` | 检查数据集是否解压到 `trainer/data/raw/` 下，目录名是否正确 |

## 配置说明

配置文件位于 `common/config.py`：

```python
# 人脸识别配置
FACE_RECOGNITION_TOLERANCE = 0.6    # 识别容差阈值（face_recognition 推荐 0.6）
FACE_DETECTION_MODEL = "hog"        # 人脸检测模型: "hog" (CPU快速) 或 "cnn" (GPU高精度)

# YOLO 人脸检测配置
USE_YOLO_DETECTION = False          # 是否使用 YOLO 进行人脸检测
YOLO_MODEL_SIZE = "n"               # YOLO 模型大小: n/s/m/l/x
YOLO_CONFIDENCE = 0.5               # YOLO 置信度阈值

# 识别方式选择
USE_TRAINED_MODEL = False           # 是否使用训练模型（False = 使用 face_recognition）

# 门禁安全配置
STRICT_REGISTRATION_ONLY = True     # 只允许数据库注册用户通过
MIN_FACE_PHOTOS = 5                 # 快速注册最少照片数
```

## 注意事项

1. 摄像头需正确连接
2. 拍摄人脸时光线充足、正对摄像头
3. 训练需要 PyTorch，推荐 GPU 加速
4. 门禁严格模式下，只有通过人事管理系统注册的用户才能通过
5. 切换编码模型后需要重新生成编码

## 许可证

本项目仅供个人学习和研究使用。

**商用需获得作者书面授权**，未经授权用于商业用途视为侵权。

联系方式：请通过 GitHub 仓库提交 Issue 联系作者。

## 更新日志

- **v3.2** — 双模型编码选择、YOLO+face_recognition 联合检测、沉浸式门禁界面、人事管理界面重构
- **v3.1** — 快速注册、批量导入、严格安全模式、移除训练 GUI、修复中文路径
- **v3.0** — 门禁多模型切换、embedding 余弦相似度识别
- **v2.0** — 训练 GUI、断点续训
- **v1.0** — 初始版本

## 贡献指南

欢迎提交 Issue 和 Pull Request。请确保：

1. 代码符合项目风格
2. 添加必要的注释
3. 更新相关文档
