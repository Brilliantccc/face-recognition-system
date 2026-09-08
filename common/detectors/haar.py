"""
Haar Cascade 人脸检测器
使用 OpenCV 的 Haar Cascade 分类器
"""

import cv2
import numpy as np
from typing import List

from .base import FaceDetector, FaceDetection


class HaarDetector(FaceDetector):
    """基于 Haar Cascade 的人脸检测器"""

    def __init__(self, cascade_path: str = None):
        """
        初始化 Haar Cascade 检测器
        :param cascade_path: 级联分类器路径
        """
        self.cascade_path = cascade_path
        self.cascade = None
        self._load_model()

    def _load_model(self):
        """加载 Haar Cascade"""
        if self.cascade_path is None:
            self.cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"无法加载 Haar Cascade: {self.cascade_path}")

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        检测人脸
        :param image: BGR 格式的图片
        :return: 人脸检测结果列表
        """
        if self.cascade is None:
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rect = self.cascade.detectMultiScale(gray, 1.3, 5)

        faces = []
        for x, y, w, h in faces_rect:
            faces.append(FaceDetection(x, y, x + w, y + h, 1.0))

        return faces
