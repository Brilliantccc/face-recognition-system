# 人脸识别门禁系统

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 注册人员

双击 `启动人事管理系统.bat` 或运行：

```bash
python admin/main.py
```

注册方式：
- **快速注册**：点击"⚡ 快速注册" → 填写信息 → 摄像头拍照
- **批量导入**：准备文件夹 `照片文件夹/姓名/照片.jpg`，点击"📂 批量导入"
- **导入 uploads**：将照片放入 `data/uploads/姓名/`，点击"📥 导入 uploads"

### 3. 生成编码

注册完成后，点击 **"🧠 生成编码"**

### 4. 启动门禁

双击 `启动门禁系统.bat` 或运行：

```bash
python gate/main.py
```

---

## 模块化配置

编辑 `common/config.py` 切换检测器/识别器/数据库：

```python
# 检测器: "yolo", "face_recognition", "haar", "auto"
DETECTION_BACKEND = "yolo"

# 识别器: "mobilenet", "face_recognition", "auto"
RECOGNITION_BACKEND = "mobilenet"

# 数据库: "bin", "db", "auto"
DATABASE_BACKEND = "bin"
```

| 组合 | 检测器 | 识别器 | 数据库 | 说明 |
|------|--------|--------|--------|------|
| 默认 | yolo | mobilenet | bin | 推荐，速度快 |
| 回退 | face_recognition | face_recognition | db | 稳定，无需 GPU |
| 自动 | auto | auto | auto | 按优先级自动选择 |

---

## 三种启动方式

| 方式 | 启动命令 | 模型格式 | 适用场景 |
|------|----------|----------|----------|
| PyQt5 | `python gate/main.py` | .pt | 开发调试 |
| Python ONNX | `python deploy/gate_onnx.py` | .onnx | 部署 |
| C++ ONNX | `deploy/启动门禁系统.bat` | .onnx | 高性能 |

---

## 训练模型（可选）

### 准备数据

```bash
# 下载 CASIA-WebFace 数据集，解压到 trainer/data/raw/
python trainer/preprocess.py
```

### 开始训练

```bash
python trainer/train_optimal.py --epochs 100
```

### 测试模型

```bash
python trainer/test_model.py --model-dir trainer/models/saved/模型目录
```

### 使用模型

1. 模型保存在 `trainer/models/saved/`
2. 门禁界面右上角设置面板选择模型
3. 点击"🧠 生成编码"重新生成编码

---

## C++ 部署

详见 [deploy/README.md](deploy/README.md)

```bash
cd deploy
.\build.bat
.\启动门禁系统.bat
```

---

## 目录结构

```
人脸识别/
├── common/              # 公共模块
│   ├── detectors/       # 人脸检测器（可插拔）
│   ├── recognizers/     # 人脸识别器（可插拔）
│   ├── databases/       # 人脸数据库（可插拔）
│   └── config.py        # 系统配置
├── admin/               # 人事管理
├── gate/                # 门禁系统
├── trainer/             # 模型训练
├── deploy/              # C++ 部署
└── data/                # 运行数据
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 摄像头打不开 | 检查摄像头连接，尝试更换摄像头索引 |
| 识别不准 | 重新生成编码，或训练新模型 |
| 内存不足 | 减小 batch-size，或使用 CPU 训练 |
| 编码生成失败 | 确保已安装 face_recognition 或 PyTorch |

---

## 许可证

仅供个人学习和研究使用，商用需获得作者书面授权。
