# 人脸识别训练模块

基于 PyTorch 的 MobileFaceNet + ArcFace Loss 人脸识别训练系统。

## 项目结构

```
trainer/
├── models/
│   ├── facenet.py           # MobileFaceNet 网络 + ArcFaceLoss 定义
│   ├── yolov11l-face.pt     # YOLO 人脸检测模型
│   └── saved/               # 训练好的模型
│       ├── optimal_v1/
│       └── optimal_v2/
├── data/
│   ├── raw/                 # 原始数据集（CASIA-WebFace 解压到这里）
│   └── aligned/             # 预处理后的数据集（脚本自动生成）
│       ├── train/
│       └── val/
├── train_optimal.py         # ★ 主力训练脚本（RTX 3060 优化）
├── finetune_local.py        # 微调 + 阈值校准
├── preprocess.py            # CASIA-WebFace 预处理脚本
├── collect_data.py          # 数据收集工具（摄像头/文件夹）
├── inference.py             # 实时识别测试
└── view_logs.py             # 门禁日志查看工具
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 CASIA-WebFace 数据集

CASIA-WebFace 是中科院公开的人脸数据集，包含 **10,575 人、494,414 张**图片。

**下载方式：**
- 官方地址：http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html
- 或在 Kaggle 搜索 "CASIA-WebFace"

**下载后解压到：**
```
trainer/data/raw/CASIA-WebFace/
├── 0000001/
│   ├── 001.jpg
│   └── ...
├── 0000002/
└── ...
```

### 3. 预处理数据集

使用 YOLO 检测人脸并裁剪对齐到 112×112：

```bash
# 检查数据集
python trainer/preprocess.py --check

# 开始预处理（默认使用 YOLO）
python trainer/preprocess.py

# 使用 face_recognition 检测（如果 YOLO 不可用）
python trainer/preprocess.py --method face_recognition

# 自定义参数
python trainer/preprocess.py --raw-dir trainer/data/raw/CASIA-WebFace \
                             --output-dir trainer/data/aligned \
                             --size 112 --val-ratio 0.1
```

### 4. 训练模型

```bash
# RTX 3060 优化版（推荐）
python trainer/train_optimal.py --epochs 100 --batch-size 128 --lr 0.01

# 从断点续训
python trainer/train_optimal.py --resume

# CPU 模式
python trainer/train_optimal.py --cpu --epochs 50 --batch-size 32
```

**训练参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --epochs | 100 | 训练轮数 |
| --batch-size | 128 | 批次大小（RTX 3060 6GB 推荐 128） |
| --lr | 0.01 | 峰值学习率（OneCycleLR 调度） |
| --cpu | - | 强制使用 CPU |
| --resume | - | 从上次断点继续训练 |

### 5. 测试识别

```bash
# 打开摄像头实时测试
python trainer/inference.py

# 指定模型目录
python trainer/inference.py --model-dir trainer/models/saved/optimal_v2
```

## 模型结构

**MobileFaceNet** — 轻量级人脸识别网络：
- 输入: 112×112 RGB 图片
- 输出: 128 维 L2 归一化 embedding
- 参数量: ~1M
- 基于 MobileNetV2 倒残差结构

**ArcFace Loss** — 角度间隔损失函数：
- 增大类间距离，缩小类内距离
- s=32.0（缩放因子），m=0.20（角度间隔）

## 训练结果

训练完成后，模型保存在 `trainer/models/saved/optimal_vN/`：

| 文件 | 大小 | 说明 |
|------|------|------|
| `best_model.pth` | ~9.4MB | 完整模型（含 ArcFace 分类头 + class_to_idx） |
| `inference_model.pth` | ~4.2MB | 推理模型（仅特征提取层，无分类头） |
| `arcface_weight.pth` | ~5.2MB | ArcFace 分类层权重（续训/微调用） |
| `model_info.json` | - | 类别映射 + 训练最佳阈值 |

## 微调模型

用门禁实际数据微调预训练模型，提升特定人员的识别准确率：

```bash
# 微调
python trainer/finetune_local.py

# 仅校准阈值（不重新训练）
python trainer/finetune_local.py --calibrate
```

## 数据收集

如果只想用少量人员数据训练：

```bash
# 从摄像头收集（每人 50 张）
python trainer/collect_data.py --mode camera --name "张三" --num 50

# 从文件夹收集
python trainer/collect_data.py --mode folder --name "张三" --source "path/to/photos"

# 划分训练/测试集
python trainer/collect_data.py --mode split
```

## 日志查看

```bash
# 查看最近 7 天门禁日志
python trainer/view_logs.py

# 查看特定用户
python trainer/view_logs.py --user "张三"

# 导出为 CSV
python trainer/view_logs.py --export access_logs.csv --days 30
```
