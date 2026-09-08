"""
face_recognition 识别器
使用 dlib 的人脸识别模型
"""

import cv2
import numpy as np
from typing import Tuple, Dict

from .base import FaceRecognizer


class FaceRecognitionRecognizer(FaceRecognizer):
    """基于 face_recognition (dlib) 的人脸识别器"""

    def __init__(self, tolerance: float = 0.6):
        """
        初始化 face_recognition 识别器
        :param tolerance: 容差阈值（越小越严格）
        """
        self.tolerance = tolerance
        self._face_recognition = None
        self._embeddings = {}  # {name: [embedding1, embedding2, ...]}
        self._avg_embeddings = {}  # {name: avg_embedding}

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

    def recognize(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        识别人脸
        :param face_image: 人脸图片 (BGR, 已裁剪)
        :return: (姓名, 置信度)
        """
        if self._face_recognition is None or not self._avg_embeddings:
            return "Unknown", 0.0

        embedding = self.get_embedding(face_image)
        if embedding is None:
            return "Unknown", 0.0

        # 与所有注册用户比较
        best_name = "Unknown"
        best_distance = float('inf')

        for name, avg_emb in self._avg_embeddings.items():
            distance = self._face_recognition.face_distance([avg_emb], embedding)[0]
            if distance < best_distance:
                best_distance = distance
                best_name = name

        if best_distance <= self.tolerance:
            confidence = max(0, 1 - best_distance)
            return best_name, confidence

        return "Unknown", 0.0

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        获取人脸 embedding
        :param face_image: 人脸图片 (BGR, 已裁剪)
        :return: embedding 向量 (numpy)
        """
        if self._face_recognition is None:
            return None

        try:
            rgb_img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = self._face_recognition.face_encodings(rgb_img)

            if len(encodings) > 0:
                return encodings[0]
            return None
        except Exception as e:
            print(f"face_recognition embedding error: {e}")
            return None

    def load_database(self, embeddings: Dict[str, np.ndarray]):
        """加载人脸数据库并计算平均 embedding"""
        self._embeddings = {}
        self._avg_embeddings = {}

        for name, emb in embeddings.items():
            if isinstance(emb, list):
                self._embeddings[name] = emb
            else:
                self._embeddings[name] = [emb]

        # 计算每个用户的平均 embedding
        for name, embs in self._embeddings.items():
            if len(embs) > 0:
                self._avg_embeddings[name] = np.mean(embs, axis=0)

    def add_embedding(self, name: str, embedding: np.ndarray):
        """添加单个 embedding"""
        if name not in self._embeddings:
            self._embeddings[name] = []
        self._embeddings[name].append(embedding)
        self._avg_embeddings[name] = np.mean(self._embeddings[name], axis=0)
