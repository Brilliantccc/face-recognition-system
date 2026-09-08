"""
face_recognition 人脸检测器
使用 dlib 的 HOG/CNN 模型进行人脸检测
"""

import cv2
import numpy as np
from typing import List

from .base import FaceDetector, FaceDetection


class FaceRecognitionDetector(FaceDetector):
    """基于 face_recognition (dlib) 的人脸检测器"""

    def __init__(self, model: str = "hog"):
        """
        初始化 face_recognition 检测器
        :param model: 检测模型 ("hog" 或 "cnn")
        """
        self.model = model
        self._face_recognition = None
        self._load_model()

    def _load_model(self):
        """加载 face_recognition"""
        try:
            import face_recognition
            self._face_recognition = face_recognition
        except ImportError:
            raise ImportError(
                "请安装 face_recognition: pip install face-recognition"
            )

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        检测人脸
        :param image: BGR 格式的图片
        :return: 人脸检测结果列表
        """
        if self._face_recognition is None:
            return []

        # 转换为 RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 缩小以加速
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)

        # 检测人脸
        locations = self._face_recognition.face_locations(small, model=self.model)

        faces = []
        for top, right, bottom, left in locations:
            # 缩放回原尺寸
            x1, y1, x2, y2 = left * 2, top * 2, right * 2, bottom * 2
            faces.append(FaceDetection(x1, y1, x2, y2, 1.0))

        return faces
