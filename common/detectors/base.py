"""
人脸检测器抽象基类
所有检测器必须实现 detect 方法
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class FaceDetection:
    """人脸检测结果"""
    __slots__ = ['x1', 'y1', 'x2', 'y2', 'confidence']

    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence

    def to_tuple(self) -> Tuple[int, int, int, int, float]:
        return (self.x1, self.y1, self.x2, self.y2, self.confidence)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return f"FaceDetection(x1={self.x1}, y1={self.y1}, x2={self.x2}, y2={self.y2}, conf={self.confidence:.3f})"


class FaceDetector(ABC):
    """人脸检测器抽象基类"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        检测图片中的人脸
        :param image: BGR 格式的图片
        :return: 人脸检测结果列表
        """
        pass

    def detect_single(self, image: np.ndarray) -> FaceDetection:
        """检测并返回置信度最高的人脸"""
        detections = self.detect(image)
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)
