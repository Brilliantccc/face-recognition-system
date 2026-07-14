"""
门禁控制模块
实现实时人脸识别和门禁控制逻辑
支持 YOLO 人脸检测 (更快速)
"""

import cv2
import numpy as np
import face_recognition
import threading
import time
from typing import Tuple, Optional, List

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.user_manager import UserManager
from common.config import FACE_RECOGNITION_TOLERANCE, USE_YOLO_DETECTION, YOLO_MODEL_SIZE, YOLO_CONFIDENCE


class AccessControl:
    def __init__(self, user_manager: UserManager, tolerance: float = None, gate_name: str = "Main Gate",
                 use_yolo: bool = None, yolo_model_size: str = None, yolo_confidence: float = None):
        """
        初始化门禁控制器
        :param user_manager: 用户管理器实例
        :param tolerance: 人脸识别容差阈值
        :param gate_name: 门禁点名称
        :param use_yolo: 是否使用 YOLO 进行人脸检测
        :param yolo_model_size: YOLO 模型大小
        :param yolo_confidence: YOLO 置信度阈值
        """
        self.user_manager = user_manager
        self.tolerance = tolerance or FACE_RECOGNITION_TOLERANCE
        self.gate_name = gate_name

        # YOLO 配置
        self.use_yolo = use_yolo if use_yolo is not None else USE_YOLO_DETECTION
        self.yolo_detector = None

        if self.use_yolo:
            self._init_yolo_detector(yolo_model_size, yolo_confidence)

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
        self.last_process_time = 0
        self.process_interval = 0.5  # 每0.5秒处理一次人脸识别

        # 加载已知人脸特征
        self.load_known_faces()

    def _init_yolo_detector(self, model_size: str = None, confidence: float = None):
        """初始化 YOLO 检测器"""
        try:
            from common.yolo_detector import YOLOFaceDetector

            model_size = model_size or YOLO_MODEL_SIZE
            conf = confidence or YOLO_CONFIDENCE

            self.yolo_detector = YOLOFaceDetector(
                model_size=model_size,
                confidence=conf,
                device="cpu"  # 可根据需要改为 "cuda"
            )
            print(f"YOLO detector initialized (model: yolov8{model_size}, confidence: {conf})")
        except ImportError as e:
            print(f"Warning: YOLO not available, falling back to face_recognition: {e}")
            self.use_yolo = False
        except Exception as e:
            print(f"Warning: YOLO initialization failed, falling back to face_recognition: {e}")
            self.use_yolo = False

    def load_known_faces(self):
        """加载已知人脸特征"""
        self.user_manager.reload_cache()
        print(f"Loaded {len(self.user_manager._cached_encodings)} face encodings")

    def reload_known_faces(self):
        """重新加载已知人脸特征（用户更新后调用）"""
        self.load_known_faces()

    def update_frame(self, frame: np.ndarray):
        """更新待处理的帧"""
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def _process_in_thread(self):
        """在后台线程中处理人脸识别"""
        while self.processing:
            current_time = time.time()
            if current_time - self.last_process_time < self.process_interval:
                time.sleep(0.05)
                continue

            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self.latest_frame.copy()

            # 根据配置选择人脸检测方式
            if self.use_yolo and self.yolo_detector is not None:
                face_locations, face_encodings = self._detect_faces_yolo(frame)
            else:
                face_locations, face_encodings = self._detect_faces_default(frame)

            status = "Standby"
            user_name = None
            confidence = 0.0
            processed_frame = frame.copy()

            if len(face_locations) > 0:
                for face_loc, face_encoding in zip(face_locations, face_encodings):
                    # 处理不同格式的人脸位置
                    if self.use_yolo and self.yolo_detector is not None:
                        # YOLO 返回 (x1, y1, x2, y2) 格式
                        x1, y1, x2, y2 = face_loc
                        left, top, right, bottom = x1, y1, x2, y2
                    else:
                        # face_recognition 返回 (top, right, bottom, left) 格式
                        top, right, bottom, left = face_loc
                        left, top, right, bottom = left, top, right, bottom

                    # 识别用户
                    is_match, user_name, confidence, user_id = \
                        self.user_manager.verify_user(face_encoding, self.tolerance)

                    if is_match:
                        status = "Pass"
                        color = (0, 255, 0)  # 绿色
                        text = f"Pass: {user_name} ({confidence:.2%})"
                    else:
                        status = "Reject"
                        color = (0, 0, 255)  # 红色
                        text = "Reject: Unauthorized"

                    # 绘制边框
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)

                    # 绘制标签背景
                    cv2.rectangle(processed_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)

                    # 绘制文字
                    cv2.putText(processed_frame, text, (left + 6, bottom - 6),
                               cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                    # 记录门禁日志
                    if status == "Pass":
                        self.user_manager.db.log_access(
                            user_id, user_name, "pass", confidence, self.gate_name
                        )
                    else:
                        self.user_manager.db.log_access(
                            -1, "Unauthorized", "reject", 0.0, self.gate_name
                        )

                    # 只处理第一个人脸
                    break

            # 更新结果
            self.result_frame = processed_frame
            self.result_status = status
            self.result_user = user_name
            self.result_confidence = confidence
            self.last_process_time = current_time

    def _detect_faces_yolo(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        使用 YOLO 检测人脸并提取编码
        :param frame: 视频帧
        :return: (人脸位置列表, 人脸编码列表)
        """
        # 缩小帧以提高处理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 使用 YOLO 检测人脸
        detections = self.yolo_detector.detect(rgb_small_frame)

        face_locations = []
        face_encodings = []

        for x1, y1, x2, y2, conf in detections:
            # 缩放回原始尺寸 (YOLO 检测的是缩小后的帧)
            x1 *= 2
            y1 *= 2
            x2 *= 2
            y2 *= 2

            # 转换为 face_recognition 格式 (top, right, bottom, left)
            top, right, bottom, left = y1, x2, y2, x1

            # 提取人脸区域并计算编码
            face_img = rgb_small_frame[y1//2:x2//2, x1//2:y2//2]

            if face_img.size > 0:
                # 使用 face_recognition 计算编码
                face_encoding = face_recognition.face_encodings(
                    rgb_small_frame,
                    [(top//2, right//2, bottom//2, left//2)]
                )

                if face_encoding:
                    face_locations.append((x1, y1, x2, y2))  # 返回 (x1, y1, x2, y2) 格式
                    face_encodings.append(face_encoding[0])

        return face_locations, face_encodings

    def _detect_faces_default(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        使用默认的 face_recognition 检测人脸
        :param frame: 视频帧
        :return: (人脸位置列表, 人脸编码列表)
        """
        # 缩小帧以提高处理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 检测人脸
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # 缩放回原始尺寸
        scaled_locations = []
        for (top, right, bottom, left) in face_locations:
            scaled_locations.append((top * 3, right * 3, bottom * 3, left * 3))

        return scaled_locations, face_encodings

    def set_yolo_enabled(self, enabled: bool):
        """
        动态启用/禁用 YOLO 检测
        :param enabled: 是否启用
        """
        self.use_yolo = enabled
        if enabled and self.yolo_detector is None:
            self._init_yolo_detector()
        print(f"YOLO detection {'enabled' if enabled else 'disabled'}")

    def get_detection_method(self) -> str:
        """
        获取当前使用的检测方法
        :return: 检测方法名称
        """
        return "YOLO" if (self.use_yolo and self.yolo_detector) else "face_recognition"

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
        return len(self.user_manager._cached_encodings)
