"""
门禁控制模块
实现实时人脸识别和门禁控制逻辑
"""

import cv2
import numpy as np
import face_recognition
import threading
import time
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.user_manager import UserManager
from common.config import FACE_RECOGNITION_TOLERANCE


class AccessControl:
    def __init__(self, user_manager: UserManager, tolerance: float = None, gate_name: str = "Main Gate"):
        """
        初始化门禁控制器
        :param user_manager: 用户管理器实例
        :param tolerance: 人脸识别容差阈值
        :param gate_name: 门禁点名称
        """
        self.user_manager = user_manager
        self.tolerance = tolerance or FACE_RECOGNITION_TOLERANCE
        self.gate_name = gate_name

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

            # 缩小帧以提高处理速度
            small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # 检测人脸
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            status = "Standby"
            user_name = None
            confidence = 0.0
            processed_frame = frame.copy()

            if len(face_locations) > 0:
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # 缩放回原始尺寸
                    top *= 3
                    right *= 3
                    bottom *= 3
                    left *= 3

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
