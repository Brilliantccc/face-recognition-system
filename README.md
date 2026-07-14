# 人脸识别门禁系统 v2.0

基于人脸识别的门禁管理系统，支持自训练模型，分为人事管理、门禁系统和训练模块。

## 系统架构

```
人脸识别/
├── common/                    # 公共模块
│   ├── database.py           # 数据库操作
│   ├── user_manager.py       # 用户管理
│   └── config.py             # 系统配置
│
├── admin/                     # 人事管理模块
│   ├── main.py               # 管理程序入口
│   └── gui/admin_window.py   # 管理界面
│
├── gate/                      # 门禁系统模块
│   ├── main.py               # 门禁程序入口
│   ├── access_control.py     # 门禁控制逻辑
│   └── gui/gate_window.py    # 门禁界面
│
├── trainer/                   # 训练模块
│   ├── collect_data.py       # 数据收集工具
│   ├── train.py              # 模型训练
│   ├── inference.py          # 模型推理
│   └── models/facenet.py     # MobileFaceNet 模型
│
└── data/                      # 数据目录
    ├── db/                   # 数据库
    └── faces/                # 人脸照片
```

## 功能说明

### 人事管理模块 (admin/)
- 员工信息录入（姓名、工号、部门、电话）
- 人脸照片上传（支持拍照和本地上传）
- 员工信息查询和管理
- 人脸数据查看和删除

### 门禁系统模块 (gate/)
- 实时摄像头监控
- 人脸识别和比对
- 通行状态显示（通过/拒绝）
- 通行日志记录

### 训练模块 (trainer/)
- 人脸数据收集（摄像头/文件夹）
- MobileFaceNet 模型训练
- ArcFace Loss 角度间隔损失
- 模型推理测试

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 方式一：使用内置人脸识别（快速上手）

```bash
# 启动人事管理系统
python admin/main.py

# 启动门禁系统
python gate/main.py
```

### 方式二：训练自定义模型（推荐）

#### 1. 收集数据

```bash
# 从摄像头收集（每人 50 张）
python trainer/collect_data.py --mode camera --name "张三" --num 50

# 从文件夹收集
python trainer/collect_data.py --mode folder --name "张三" --source "D:\photos"

# 查看数据集信息
python trainer/collect_data.py --mode info
```

#### 2. 划分训练集/测试集

```bash
python trainer/collect_data.py --mode split
```

数据结构：
```
trainer/data/
├── train/          # 80% 用于训练
│   ├── 张三/
│   └── 李四/
└── test/           # 20% 用于测试
    ├── 张三/
    └── 李四/
```

#### 3. 训练模型

```bash
# 默认训练 50 轮
python trainer/train.py

# 自定义参数
python trainer/train.py --epochs 100 --batch-size 64 --lr 0.001
```

训练完成后模型保存在 `trainer/models/saved/`

#### 4. 测试识别

```bash
python trainer/inference.py
```

## 训练数据要求

| 要求 | 说明 |
|------|------|
| 每人照片数 | 最少 20 张，建议 30-50 张 |
| 图片格式 | jpg, jpeg, png, bmp |
| 图片大小 | 建议 112x112 以上 |
| 人脸要求 | 清晰、光线充足、无遮挡 |

### 数据组织方式

按照人名创建文件夹，文件夹名即为标签：

```
trainer/data/
├── 张三/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── 李四/
│   └── ...
└── 王五/
    └── ...
```

## 配置说明

配置文件位于 `common/config.py`：

```python
# 人脸识别容差（越小越严格）
FACE_RECOGNITION_TOLERANCE = 0.6

# 摄像头索引
CAMERA_INDEX = 0

# 数据库路径
DATABASE_PATH = "data/db/face_access.db"
```

## 使用流程

### 快速上手

1. 启动人事管理系统 → 添加员工 → 录入人脸
2. 启动门禁系统 → 自动识别

### 自定义训练

1. 收集数据 → `python trainer/collect_data.py`
2. 划分数据 → `python trainer/collect_data.py --mode split`
3. 训练模型 → `python trainer/train.py`
4. 测试效果 → `python trainer/inference.py`
5. 部署使用

## 注意事项

1. 确保摄像头已正确连接
2. 拍摄人脸照片时保持光线充足
3. 人脸应正对摄像头，避免遮挡
4. 建议为每个员工拍摄 30-50 张不同角度的照片
5. 训练时需要 PyTorch（支持 GPU 加速）
