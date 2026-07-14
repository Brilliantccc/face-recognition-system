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

# 人脸识别配置
FACE_RECOGNITION_TOLERANCE = 0.6  # 识别容差阈值，越小越严格
FACE_DETECTION_MODEL = "hog"  # 人脸检测模型: "hog" (CPU) 或 "cnn" (GPU)

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
