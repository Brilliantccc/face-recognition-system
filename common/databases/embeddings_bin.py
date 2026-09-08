"""
embeddings.bin 数据库
二进制格式的人脸 embedding 数据库，与 ONNX/C++ 版本兼容
"""

import os
import struct
import numpy as np
from typing import Dict
from collections import defaultdict

from .base import FaceDatabase


class EmbeddingsBinDatabase(FaceDatabase):
    """embeddings.bin 格式的人脸数据库"""

    def __init__(self, db_path: str = None):
        """
        初始化 embeddings.bin 数据库
        :param db_path: 数据库文件路径
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "deploy", "models", "embeddings.bin"
            )
        self.db_path = db_path

    def load(self) -> Dict[str, np.ndarray]:
        """
        加载 embeddings.bin 文件
        :return: {姓名: embedding} 字典（每个用户的多个 embedding 已取平均）
        """
        if not os.path.exists(self.db_path):
            print(f"embeddings.bin not found: {self.db_path}")
            return {}

        try:
            with open(self.db_path, 'rb') as f:
                # 读取头部：版本号, 人数, 维度
                header = struct.unpack('<III', f.read(12))
                version, num_persons, embedding_dim = header

                # 读取所有特征向量
                embeddings_data = f.read(num_persons * embedding_dim * 4)
                embeddings = np.frombuffer(embeddings_data, dtype=np.float32).reshape(num_persons, embedding_dim)

                # 读取姓名并按用户分组
                names = []
                for i in range(num_persons):
                    name_len = struct.unpack('<I', f.read(4))[0]
                    name = f.read(name_len).decode('utf-8')
                    names.append(name)

            # 按用户分组并计算平均 embedding
            user_embeddings = defaultdict(list)
            for name, emb in zip(names, embeddings):
                user_embeddings[name].append(emb)

            # 转换为 {name: avg_embedding}
            result = {}
            for name, embs_list in user_embeddings.items():
                avg_emb = np.mean(embs_list, axis=0)
                # L2 归一化
                norm = np.linalg.norm(avg_emb)
                if norm > 0:
                    avg_emb = avg_emb / norm
                result[name] = avg_emb

            print(f"Loaded {len(result)} users from embeddings.bin")
            return result

        except Exception as e:
            print(f"Failed to load embeddings.bin: {e}")
            return {}

    def save(self, embeddings: Dict[str, np.ndarray]):
        """
        保存 embeddings.bin 文件
        :param embeddings: {姓名: embedding} 字典
        """
        # 准备数据
        names = list(embeddings.keys())
        embeddings_list = [embeddings[name] for name in names]

        if not embeddings_list:
            print("No embeddings to save")
            return

        embedding_dim = embeddings_list[0].shape[0]
        num_persons = len(names)

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        with open(self.db_path, 'wb') as f:
            # 写入头部：版本号=1, 人数, 维度
            f.write(struct.pack('<III', 1, num_persons, embedding_dim))

            # 写入所有特征向量
            for emb in embeddings_list:
                f.write(emb.astype(np.float32).tobytes())

            # 写入姓名
            for name in names:
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)

        print(f"Saved {num_persons} users to embeddings.bin")

    def load_raw(self) -> tuple:
        """
        加载原始数据（保留每个用户的多个 embedding）
        :return: (names_list, embeddings_array, embedding_dim)
        """
        if not os.path.exists(self.db_path):
            return [], np.array([]), 0

        with open(self.db_path, 'rb') as f:
            header = struct.unpack('<III', f.read(12))
            version, num_persons, embedding_dim = header

            embeddings_data = f.read(num_persons * embedding_dim * 4)
            embeddings = np.frombuffer(embeddings_data, dtype=np.float32).reshape(num_persons, embedding_dim)

            names = []
            for i in range(num_persons):
                name_len = struct.unpack('<I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                names.append(name)

        return names, embeddings, embedding_dim
