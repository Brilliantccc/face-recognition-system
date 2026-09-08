"""
人脸识别器抽象基类
所有识别器必须实现 recognize 和 get_embedding 方法
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import numpy as np


class FaceRecognizer(ABC):
    """人脸识别器抽象基类"""

    @abstractmethod
    def recognize(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        识别人脸
        :param face_image: 人脸图片 (BGR, 已裁剪)
        :return: (姓名, 置信度)，未识别返回 ("Unknown", 0.0)
        """
        pass

    @abstractmethod
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        获取人脸 embedding
        :param face_image: 人脸图片 (BGR, 已裁剪)
        :return: embedding 向量
        """
        pass

    def load_database(self, embeddings: Dict[str, np.ndarray]):
        """
        加载人脸数据库
        :param embeddings: {姓名: embedding} 字典
        """
        self._embeddings = embeddings

    def get_registered_users(self) -> List[str]:
        """获取已注册用户列表"""
        return list(self._embeddings.keys()) if hasattr(self, '_embeddings') else []
