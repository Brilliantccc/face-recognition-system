"""
YOLO 人脸检测器
使用 YOLOv11 进行人脸检测
"""

import os
import numpy as np
from typing import List

from .base import FaceDetector, FaceDetection


class YOLODetector(FaceDetector):
    """基于 YOLOv11 的人脸检测器"""

    def __init__(self, model_path: str = None, confidence: float = 0.5, device: str = "cpu"):
        """
        初始化 YOLO 人脸检测器
        :param model_path: 模型文件路径
        :param confidence: 置信度阈值
        :param device: 推理设备 (cpu/cuda)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self):
        """加载 YOLO 人脸检测模型"""
        from ultralytics import YOLO

        # 默认使用 YOLOv11l-face 模型
        if self.model_path is None:
            self.model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "trainer", "models", "yolov11l-face.pt"
            )

        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.model = YOLO(self.model_path)

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        检测人脸
        :param image: BGR 格式的图片
        :return: 人脸检测结果列表
        """
        if self.model is None:
            return []

        results = self.model(
            image,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        faces = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    faces.append(FaceDetection(x1, y1, x2, y2, conf))

        return faces
