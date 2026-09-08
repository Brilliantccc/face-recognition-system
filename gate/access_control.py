"""
门禁控制模块
实现实时人脸识别和门禁控制逻辑
使用模块化架构：检测器 + 识别器 + 数据库 可自由组合
"""

import sys
import os

# 确保项目根目录在最前面
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import threading
import time
from typing import Tuple, Optional, List

from common.user_manager import UserManager
from common.config import (
    FACE_RECOGNITION_THRESHOLD, DETECTION_BACKEND, RECOGNITION_BACKEND,
    DATABASE_BACKEND, DETECTION_PRIORITY, RECOGNITION_PRIORITY, DATABASE_PRIORITY
)

# 门禁控制参数
DOOR_OPEN_DURATION = 5.0       # 开门持续时间（秒）
DOOR_COOLDOWN = 3.0            # 关门后冷却时间（秒），防止重复触发


def imread_safe(filepath):
    """读取图片，支持中文路径（Windows）"""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class AccessControl:
    def __init__(self, user_manager: UserManager, tolerance: float = None, gate_name: str = "Main Gate",
                 detection_backend: str = None, recognition_backend: str = None, database_backend: str = None):
        """
        初始化门禁控制器
        :param user_manager: 用户管理器实例
        :param tolerance: 人脸识别容差阈值
        :param gate_name: 门禁点名称
        :param detection_backend: 检测器后端 ("yolo", "face_recognition", "haar", "auto")
        :param recognition_backend: 识别器后端 ("mobilenet", "face_recognition", "auto")
        :param database_backend: 数据库后端 ("bin", "db", "auto")
        """
        self.user_manager = user_manager
        self.gate_name = gate_name

        # 模块化组件
        self.detector = None
        self.recognizer = None
        self.database = None

        # 初始化检测器
        self._init_detector(detection_backend or DETECTION_BACKEND)

        # 初始化识别器
        self._init_recognizer(recognition_backend or RECOGNITION_BACKEND)

        # 初始化数据库
        self._init_database(database_backend or DATABASE_BACKEND)

        # 加载数据库到识别器
        self._load_database_to_recognizer()

        # 用于兼容旧版的属性
        self.use_yolo = self.detector is not None and hasattr(self.detector, 'model')
        self.yolo_detector = self.detector
        self.use_trained_model = self.recognizer is not None
        self.trained_model = self.recognizer

        self.current_status = "Standby"
        self.current_user = None
        self.confidence = 0.0

        # 多线程处理相关
        self.processing = False
        self.latest_frame = None
        self.result_frame = None
        self.result_status = "Standby"
        self.result_user = None
        self.result_confidence = 0.0
        self.frame_lock = threading.Lock()
        self.model_lock = threading.Lock()  # 模型加载/切换锁
        self.last_process_time = 0
        self.process_interval = 0.3  # 处理间隔

        # 缓存 PIL 字体（避免每次加载）
        self._pil_font = None
        self._pil_font_small = None

        # 门禁开关状态
        self.door_open = False
        self.door_open_time = 0.0        # 开门时刻
        self.door_cooldown_until = 0.0   # 冷却期截止时刻
        self.on_door_open = None         # 回调：开门时触发
        self.on_door_close = None        # 回调：关门时触发

        # 兼容旧版
        self.registered_embeddings = {}

    def _init_detector(self, backend: str):
        """初始化检测器"""
        from common.detectors import DetectorFactory

        try:
            if backend == "auto":
                self.detector = DetectorFactory.create("auto")
            else:
                self.detector = DetectorFactory.create(backend)
            print(f"Detector initialized: {type(self.detector).__name__}")
        except Exception as e:
            print(f"Warning: Failed to initialize detector: {e}")
            self.detector = None

    def _init_recognizer(self, backend: str):
        """初始化识别器"""
        from common.recognizers import RecognizerFactory

        try:
            if backend == "auto":
                self.recognizer = RecognizerFactory.create("auto")
            else:
                self.recognizer = RecognizerFactory.create(backend)
            print(f"Recognizer initialized: {type(self.recognizer).__name__}")
        except Exception as e:
            print(f"Warning: Failed to initialize recognizer: {e}")
            self.recognizer = None

    def _init_database(self, backend: str):
        """初始化数据库"""
        from common.databases import DatabaseFactory

        try:
            if backend == "auto":
                self.database = DatabaseFactory.create("auto")
            else:
                self.database = DatabaseFactory.create(backend)
            print(f"Database initialized: {type(self.database).__name__}")
        except Exception as e:
            print(f"Warning: Failed to initialize database: {e}")
            self.database = None

    def _load_database_to_recognizer(self):
        """加载数据库到识别器"""
        if self.database is None or self.recognizer is None:
            return

        try:
            embeddings = self.database.load()
            if embeddings:
                self.recognizer.load_database(embeddings)
                print(f"Loaded {len(embeddings)} users into recognizer")
        except Exception as e:
            print(f"Warning: Failed to load database to recognizer: {e}")

    def _detect_faces(self, image: np.ndarray):
        """
        检测人脸
        :param image: BGR 格式的图片
        :return: 人脸检测结果列表
        """
        if self.detector is None:
            return []

        try:
            return self.detector.detect(image)
        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def _crop_face_with_margin(self, image: np.ndarray, face_rect, margin_ratio=0.2):
        """
        裁剪人脸区域并扩展 margin
        :param image: BGR 图片
        :param face_rect: (x1, y1, x2, y2)
        :param margin_ratio: 边距比例（默认 20%）
        :return: 裁剪后的人脸图片 (BGR)
        """
        if hasattr(face_rect, 'x1'):
            x1, y1, x2, y2 = face_rect.x1, face_rect.y1, face_rect.x2, face_rect.y2
        else:
            x1, y1, x2, y2 = face_rect

        h, w = image.shape[:2]
        face_w = x2 - x1
        face_h = y2 - y1
        margin_x = int(face_w * margin_ratio)
        margin_y = int(face_h * margin_ratio)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        return image[y1:y2, x1:x2]

    def _recognize_face(self, face_img: np.ndarray):
        """
        识别人脸
        :param face_img: 人脸图片 (BGR, 已裁剪)
        :return: (姓名, 置信度)
        """
        if self.recognizer is None:
            return "Unknown", 0.0

        try:
            return self.recognizer.recognize(face_img)
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0.0

    def _open_door(self, user_name: str):
        """开门"""
        now = time.time()
        if self.door_open or now < self.door_cooldown_until:
            return  # 已开门或在冷却期，不重复触发

        self.door_open = True
        self.door_open_time = now
        self.current_user = user_name
        print(f"Door OPEN - {user_name} (auto-close in {DOOR_OPEN_DURATION}s)")

        # 触发开门回调（可连接 GPIO/继电器等）
        if self.on_door_open:
            try:
                self.on_door_open(user_name)
            except Exception as e:
                print(f"Door open callback error: {e}")

    def _update_door_state(self, current_time: float):
        """检查是否需要自动关门"""
        if not self.door_open:
            return

        elapsed = current_time - self.door_open_time
        if elapsed >= DOOR_OPEN_DURATION:
            self.door_open = False
            self.door_cooldown_until = current_time + DOOR_COOLDOWN
            print(f"Door CLOSE (was open for {elapsed:.1f}s, cooldown {DOOR_COOLDOWN}s)")

            # 触发关门回调
            if self.on_door_close:
                try:
                    self.on_door_close()
                except Exception as e:
                    print(f"Door close callback error: {e}")

    def _process_in_thread(self):
        """在后台线程中处理人脸识别"""
        # 导入 torch_utils 用于 PIL 绘图
        from common import torch_utils
        torch = torch_utils.torch
        TORCH_AVAILABLE = torch_utils.TORCH_AVAILABLE

        while self.processing:
            current_time = time.time()
            if current_time - self.last_process_time < self.process_interval:
                time.sleep(0.05)
                continue

            # 门禁开关状态管理
            self._update_door_state(current_time)

            # 门已打开或在冷却期，跳过识别（节省算力）
            if self.door_open or current_time < self.door_cooldown_until:
                with self.frame_lock:
                    if self.latest_frame is None:
                        time.sleep(0.05)
                        continue
                    # 只更新画面，不做人脸识别
                    self.result_frame = self.latest_frame.copy()
                    self.result_status = "Pass" if self.door_open else "Standby"
                    self.result_user = self.current_user
                    self.result_confidence = 0.0
                self.last_process_time = current_time
                time.sleep(0.1)
                continue

            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self.latest_frame.copy()

            # 检测人脸
            detections = self._detect_faces(frame)

            status = "Standby"
            user_name = None
            confidence = 0.0
            processed_frame = frame.copy()

            if len(detections) > 0:
                # 取置信度最高的人脸
                best_detection = max(detections, key=lambda d: d.confidence)

                # 裁剪人脸并扩展 margin
                face_img = self._crop_face_with_margin(frame, best_detection)

                if face_img is not None and face_img.size > 0:
                    # 识别用户
                    user_name, confidence = self._recognize_face(face_img)

                    if user_name != "Unknown":
                        status = "Pass"
                        color = (0, 255, 0)  # 绿色
                        text = "Welcome"
                        detail_text = f"{confidence:.2%}"
                        # 触发开门
                        self._open_door(user_name)
                    else:
                        status = "Reject"
                        color = (0, 0, 255)  # 红色
                        text = "Unauthorized"
                        detail_text = ""

                    # 获取边界框坐标
                    if hasattr(best_detection, 'x1'):
                        left, top, right, bottom = best_detection.x1, best_detection.y1, best_detection.x2, best_detection.y2
                    else:
                        left, top, right, bottom = best_detection

                    # 绘制边框
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 3)

                    # 绘制标签背景 (主文字)
                    cv2.rectangle(processed_frame, (left, bottom - 50), (right, bottom), color, cv2.FILLED)

                    # 使用 PIL 绘制中文文字
                    if TORCH_AVAILABLE and torch_utils.Image is not None:
                        from PIL import ImageDraw
                        pil_img = torch_utils.Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                        pil_draw = ImageDraw.Draw(pil_img)

                        # 使用缓存的字体
                        font = self._get_pil_font(24)

                        pil_draw.text((left + 10, bottom - 45), text, fill=(255, 255, 255), font=font)

                        # 绘制置信度
                        if detail_text:
                            pil_draw.text((left + 10, bottom - 18), detail_text, fill=(255, 255, 255), font=font)

                        processed_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    else:
                        # Fallback: 使用 OpenCV
                        cv2.putText(processed_frame, text, (left + 10, bottom - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        if detail_text:
                            cv2.putText(processed_frame, detail_text, (left + 10, bottom - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 记录门禁日志
                    user_id = -1
                    if user_name != "Unknown":
                        user = self.user_manager.db.get_user_by_name(user_name)
                        if user:
                            user_id = user['id']

                    if status == "Pass":
                        self.user_manager.db.log_access(
                            user_id, user_name, "pass", confidence, self.gate_name
                        )
                    else:
                        self.user_manager.db.log_access(
                            -1, "Unauthorized", "reject", 0.0, self.gate_name
                        )

            # 更新结果
            self.result_frame = processed_frame
            self.result_status = status
            self.result_user = user_name
            self.result_confidence = confidence
            self.last_process_time = current_time

    def _get_pil_font(self, size=24):
        """获取缓存的 PIL 字体"""
        from common import torch_utils

        if size == 24 and self._pil_font:
            return self._pil_font
        if size == 16 and self._pil_font_small:
            return self._pil_font_small

        if torch_utils.TORCH_AVAILABLE and torch_utils.Image is not None:
            try:
                from PIL import ImageFont
                font = ImageFont.truetype("msyh.ttc", size)
            except:
                font = ImageFont.load_default()

            if size == 24:
                self._pil_font = font
            else:
                self._pil_font_small = font
            return font
        return None

    def update_frame(self, frame: np.ndarray):
        """更新待处理的帧"""
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def start_processing(self):
        """启动后台处理线程"""
        if not self.processing:
            self.processing = True
            self.thread = threading.Thread(target=self._process_in_thread, daemon=True)
            self.thread.start()

    def stop_processing(self):
        """停止后台处理线程"""
        self.processing = False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, Optional[str], float]:
        """
        处理帧 - 更新最新帧并返回最近的识别结果
        :param frame: 视频帧
        :return: (处理后的帧, 状态, 用户名, 置信度)
        """
        self.update_frame(frame)

        if self.result_frame is not None:
            return self.result_frame, self.result_status, self.result_user, self.result_confidence
        else:
            return frame, "Standby", None, 0.0

    def get_current_status(self) -> Tuple[str, Optional[str], float]:
        """
        获取当前门禁状态
        :return: (状态, 用户名, 置信度)
        """
        return self.current_status, self.current_user, self.confidence

    def get_known_faces_count(self) -> int:
        """
        获取已知人脸数量
        :return: 已知人脸数量
        """
        if self.database:
            return len(self.database.list_users())
        return 0

    def get_active_users_count(self) -> int:
        """获取数据库中已激活的注册用户数量"""
        users = self.user_manager.get_all_users(active_only=True)
        return len(users)

    def is_strict_mode(self) -> bool:
        """是否为严格模式（只允许数据库注册用户通过）"""
        from common.config import STRICT_REGISTRATION_ONLY
        return STRICT_REGISTRATION_ONLY

    def get_detection_method(self) -> str:
        """获取当前使用的检测方法"""
        if self.detector:
            return type(self.detector).__name__
        return "None"

    def get_recognition_method(self) -> str:
        """获取当前使用的识别方法"""
        if self.recognizer:
            return type(self.recognizer).__name__
        return "None"

    # ====== 兼容旧版 API ======

    def reload_known_faces(self):
        """重新加载已知人脸特征"""
        self._load_database_to_recognizer()

    def get_available_models(self) -> list:
        """获取所有可用的模型列表（兼容旧版 GUI）"""
        models = []

        # 添加 face_recognition 选项（兜底方案）
        models.append({
            'name': 'face_recognition',
            'path': 'face_recognition',
            'num_classes': None,
            'is_builtin': True
        })

        # 添加训练模型
        models_dir = "trainer/models/saved"
        if os.path.exists(models_dir):
            import json
            for name in sorted(os.listdir(models_dir)):
                model_dir = os.path.join(models_dir, name)
                if os.path.isdir(model_dir):
                    has_inference = os.path.exists(os.path.join(model_dir, "inference_model.pth"))
                    has_best = os.path.exists(os.path.join(model_dir, "best_model.pth"))
                    if has_inference or has_best:
                        info_path = os.path.join(model_dir, "model_info.json")
                        num_classes = None
                        if os.path.exists(info_path):
                            with open(info_path, 'r', encoding='utf-8') as f:
                                info = json.load(f)
                                num_classes = info.get('num_classes')
                        models.append({
                            'name': name,
                            'path': model_dir,
                            'num_classes': num_classes,
                            'is_builtin': False
                        })

        return models

    def load_model_from_dir(self, model_dir: str) -> bool:
        """从指定目录加载模型（兼容旧版 GUI）"""
        try:
            from common.recognizers import MobileFaceNetRecognizer
            self.recognizer = MobileFaceNetRecognizer(model_dir=model_dir)
            self.use_trained_model = True
            self.trained_model = self.recognizer
            self._load_database_to_recognizer()
            return True
        except Exception as e:
            print(f"Failed to load model from {model_dir}: {e}")
            return False

    def set_yolo_enabled(self, enabled: bool):
        """动态启用/禁用 YOLO 检测（兼容旧版 GUI）"""
        if enabled:
            try:
                from common.detectors import YOLODetector
                self.detector = YOLODetector()
                self.use_yolo = True
            except Exception as e:
                print(f"Failed to enable YOLO: {e}")
                self.use_yolo = False
        else:
            try:
                from common.detectors import FaceRecognitionDetector
                self.detector = FaceRecognitionDetector()
                self.use_yolo = False
            except Exception as e:
                print(f"Failed to disable YOLO: {e}")
