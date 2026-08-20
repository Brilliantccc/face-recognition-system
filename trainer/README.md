# 人脸识别训练模块

基于 PyTorch 的 MobileFaceNet + ArcFace Loss 人脸识别训练系统。

## 项目结构

```
trainer/
├── models/
│   ├── __init__.py
│   └── facenet.py          # MobileFaceNet 网络 + ArcFaceLoss 定义
├── gui/
│   ├── __init__.py
│   └── trainer_window.py    # 训练 GUI 平台
├── data/
│   ├── raw/                 # 原始数据集（CASIA-WebFace 解压到这里）
│   │   └── CASIA-WebFace/
│   │       ├── 0000001/
│   │       └── ...
│   ├── aligned/             # 预处理后的数据集（脚本自动生成）
│   │   ├── train/
│   │   └── val/
│   └── register/            # 门禁注册数据
├── models/saved/            # 训练好的模型
├── collect_data.py          # 数据收集工具
├── preprocess.py            # CASIA-WebFace 预处理脚本
├── train.py                 # 训练脚本
├── inference.py             # 推理模块
└── main.py                  # 训练平台入口
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

预处理完成后，数据在 `trainer/data/aligned/` 中：
```
trainer/data/aligned/
├── train/          # 训练集（90%）
│   ├── 0000001/    # 112×112 对齐后的人脸
│   └── ...
└── val/            # 验证集（10%）
    ├── 0000001/
    └── ...
```

### 4. 训练模型

```bash
# 命令行训练
python trainer/train.py --data trainer/data/aligned --epochs 50 --batch-size 64

# 或使用 GUI 训练平台（推荐）
python 启动_训练平台.py
```

**训练参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 50 | 训练轮数 |
| batch-size | 32 | 批次大小 |
| lr | 0.001 | 学习率 |
| --cpu | - | 强制使用 CPU |

### 5. 测试识别

```bash
# 打开摄像头测试
python trainer/inference.py
```

## 模型结构

**MobileFaceNet** — 轻量级人脸识别网络：
- 输入: 112×112 RGB 图片
- 输出: 128 维嵌入向量
- 参数量: ~1M
- 基于 MobileNetV2 倒残差结构

**ArcFace Loss** — 角度间隔损失函数：
- 增大类间距离，缩小类内距离
- s=30.0（缩放因子），m=0.50（角度间隔）

## 训练结果

训练完成后，模型保存在 `trainer/models/saved/`：
- `best_model.pth`: 最佳模型（含分类头）
- `inference_model.pth`: 推理模型（无分类头）
- `model_info.json`: 类别信息
- `training_history.json`: 训练历史

## 集成到门禁系统

训练完成后，门禁系统会自动加载 `trainer/models/saved/` 中的模型。
在 `gate/access_control.py` 中，系统优先使用训练模型进行识别，
置信度不够时回退到 face_recognition 库。

## 数据收集（可选）

如果只想用少量人员数据训练：

```bash
# 从摄像头收集（每人 50 张）
python trainer/collect_data.py --mode camera --name "张三" --num 50

# 从文件夹收集
python trainer/collect_data.py --mode folder --name "张三" --source "path/to/photos"
```
