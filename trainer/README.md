# 人脸识别训练模块

基于 PyTorch 的自定义人脸识别训练系统，使用 MobileFaceNet + ArcFace Loss。

## 功能

- **数据收集**: 从摄像头或文件夹收集人脸数据
- **模型训练**: 使用 MobileFaceNet 进行训练
- **模型推理**: 使用训练好的模型进行识别

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision opencv-python pillow numpy
```

### 2. 收集数据

```bash
# 从摄像头收集（每人 50 张）
python trainer/collect_data.py --mode camera --name "张三" --num 50

# 从文件夹收集
python trainer/collect_data.py --mode folder --name "张三" --source "path/to/photos"

# 划分训练集和测试集
python trainer/collect_data.py --mode split

# 查看数据集信息
python trainer/collect_data.py --mode info
```

### 3. 训练模型

```bash
# 默认训练 50 轮
python trainer/train.py

# 自定义参数
python trainer/train.py --epochs 100 --batch-size 64 --lr 0.001
```

### 4. 测试识别

```bash
# 打开摄像头测试
python trainer/inference.py
```

## 数据目录结构

```
trainer/data/
├── 张三/
│   ├── 0000.jpg
│   ├── 0001.jpg
│   └── ...
├── 李四/
│   ├── 0000.jpg
│   └── ...
├── train/
│   ├── 张三/
│   └── 李四/
├── test/
│   ├── 张三/
│   └── 李四/
└── dataset_info.json
```

## 模型结构

使用 MobileFaceNet（轻量级人脸识别网络）：
- 输入: 112x112 RGB 图片
- 输出: 128 维嵌入向量
- 参数量: ~1M（适合移动端部署）

## 损失函数

使用 ArcFace Loss（角度间隔损失）：
- 增大类间距离
- 缩小类内距离
- 提高识别准确率

## 训练结果

训练完成后，模型保存在 `trainer/models/saved/`：
- `best_model.pth`: 最佳模型（含分类头）
- `inference_model.pth`: 推理模型（无分类头）
- `model_info.json`: 类别信息
- `training_history.json`: 训练历史

## 集成到门禁系统

训练完成后，修改 `common/config.py` 中的配置即可使用自训练模型。
