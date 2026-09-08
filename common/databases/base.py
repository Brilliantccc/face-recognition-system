"""
人脸数据库抽象基类
所有数据库必须实现 load 和 save 方法
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import numpy as np


class FaceDatabase(ABC):
    """人脸数据库抽象基类"""

    @abstractmethod
    def load(self) -> Dict[str, np.ndarray]:
        """
        加载所有人脸 embedding
        :return: {姓名: embedding} 字典
        """
        pass

    @abstractmethod
    def save(self, embeddings: Dict[str, np.ndarray]):
        """
        保存人脸 embedding
        :param embeddings: {姓名: embedding} 字典
        """
        pass

    def get_user_embedding(self, name: str) -> np.ndarray:
        """获取单个用户的 embedding"""
        data = self.load()
        return data.get(name)

    def add_user(self, name: str, embedding: np.ndarray):
        """添加单个用户"""
        data = self.load()
        data[name] = embedding
        self.save(data)

    def remove_user(self, name: str):
        """移除单个用户"""
        data = self.load()
        if name in data:
            del data[name]
            self.save(data)

    def list_users(self) -> List[str]:
        """列出所有用户"""
        return list(self.load().keys())
