"""
SQLite 人脸数据库
使用 SQLite 存储人脸 embedding
"""

import os
import sqlite3
import numpy as np
from typing import Dict

from .base import FaceDatabase


class SQLiteDatabase(FaceDatabase):
    """SQLite 格式的人脸数据库"""

    def __init__(self, db_path: str = None):
        """
        初始化 SQLite 数据库
        :param db_path: 数据库文件路径
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "db", "face_database.db"
            )
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建人脸编码表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                encoding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        conn.commit()
        conn.close()

    def load(self) -> Dict[str, np.ndarray]:
        """
        加载所有用户的平均 embedding
        :return: {姓名: embedding} 字典
        """
        if not os.path.exists(self.db_path):
            return {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取所有活跃用户
        cursor.execute("SELECT id, name FROM users WHERE is_active = 1")
        users = cursor.fetchall()

        result = {}
        for user_id, name in users:
            # 获取该用户的所有编码
            cursor.execute(
                "SELECT encoding FROM face_encodings WHERE user_id = ?",
                (user_id,)
            )
            rows = cursor.fetchall()

            if rows:
                # 计算平均 embedding
                encodings = [np.frombuffer(row[0], dtype=np.float64) for row in rows]
                avg_encoding = np.mean(encodings, axis=0)
                result[name] = avg_encoding

        conn.close()
        print(f"Loaded {len(result)} users from SQLite database")
        return result

    def save(self, embeddings: Dict[str, np.ndarray]):
        """
        保存 embeddings 到数据库
        :param embeddings: {姓名: embedding} 字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for name, embedding in embeddings.items():
            # 检查用户是否存在
            cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
            row = cursor.fetchone()

            if row:
                user_id = row[0]
                # 清除旧编码
                cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
            else:
                # 创建新用户
                cursor.execute("INSERT INTO users (name) VALUES (?)", (name,))
                user_id = cursor.lastrowid

            # 保存编码
            encoding_bytes = embedding.astype(np.float64).tobytes()
            cursor.execute(
                "INSERT INTO face_encodings (user_id, encoding) VALUES (?, ?)",
                (user_id, encoding_bytes)
            )

        conn.commit()
        conn.close()
        print(f"Saved {len(embeddings)} users to SQLite database")
