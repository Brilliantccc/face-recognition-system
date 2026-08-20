"""
系统配置模块
存储系统配置参数
"""

import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 数据库配置
DATABASE_PATH = os.path.join(DATA_DIR, "db", "face_access.db")

# 人脸数据目录
FACES_DIR = os.path.join(DATA_DIR, "faces")

# 工作照片目录
WORK_PHOTOS_DIR = os.path.join(DATA_DIR, "work_photos")

# 人脸识别配置
FACE_RECOGNITION_TOLERANCE = 0.45  # 识别容差阈值，越小越严格
FACE_DETECTION_MODEL = "hog"  # 人脸检测模型: "hog" (CPU快速) 或 "cnn" (GPU高精度)
FACE_INPUT_SIZE = 112  # 人脸输入尺寸 (112x112)，MobileFaceNet标准尺寸

# YOLO 人脸检测配置
USE_YOLO_DETECTION = False  # 是否使用 YOLO 进行人脸检测 (False 则使用 face_recognition)
YOLO_MODEL_SIZE = "n"  # YOLO 模型大小: "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (xlarge)
YOLO_CONFIDENCE = 0.5  # YOLO 置信度阈值

# PyTorch / CUDA 配置
PREFER_GPU = True  # 是否优先使用 GPU（如果可用）
TRAINING_DEVICE = "auto"  # 训练设备: "auto" (自动选择), "cuda" (强制GPU), "cpu" (强制CPU)
CUDA_VISIBLE_DEVICES = "0"  # 可见的 CUDA 设备（多GPU时使用）

# 摄像头配置
CAMERA_INDEX = 0  # 摄像头索引

# GUI配置
ADMIN_WINDOW_TITLE = "人脸识别人事管理系统"
ADMIN_WINDOW_WIDTH = 1200
ADMIN_WINDOW_HEIGHT = 800

GATE_WINDOW_TITLE = "人脸识别门禁系统"
GATE_WINDOW_WIDTH = 1200
GATE_WINDOW_HEIGHT = 800

# 日志配置
LOG_MAX_ENTRIES = 100  # 最大日志条数

# 安全配置
MAX_LOGIN_ATTEMPTS = 5  # 最大登录尝试次数
LOCKOUT_DURATION = 300  # 锁定时间（秒）

# 门禁安全配置
STRICT_REGISTRATION_ONLY = True  # 门禁只允许数据库注册用户通过（严格模式）
MIN_FACE_PHOTOS = 5  # 快速注册最少照片数
