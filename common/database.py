"""
数据库操作模块
使用SQLite存储用户信息和人脸特征
"""

import sqlite3
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class Database:
    def __init__(self, db_path: str = None):
        """
        初始化数据库连接
        :param db_path: 数据库文件路径
        """
        if db_path is None:
            from .config import DATABASE_PATH
            db_path = DATABASE_PATH
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE,
                name TEXT NOT NULL,
                department TEXT,
                phone TEXT,
                role TEXT DEFAULT 'staff',
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 检查并添加缺失的列
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'is_active' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1")
        if 'employee_id' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN employee_id TEXT")
        if 'department' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN department TEXT")
        if 'phone' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        if 'role' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'staff'")
        if 'work_photo' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN work_photo TEXT")

        # 人脸特征表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_encodings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                encoding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        ''')

        # 门禁日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT NOT NULL,
                confidence REAL,
                gate_name TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
            )
        ''')

        # 检查并添加缺失的列
        cursor.execute("PRAGMA table_info(access_log)")
        access_columns = [col[1] for col in cursor.fetchall()]

        if 'gate_name' not in access_columns:
            cursor.execute("ALTER TABLE access_log ADD COLUMN gate_name TEXT")

        conn.commit()
        conn.close()

    def add_user(self, name: str, employee_id: str = None, department: str = None,
                 phone: str = None, role: str = "staff") -> int:
        """
        添加新用户
        :param name: 用户姓名
        :param employee_id: 工号
        :param department: 部门
        :param phone: 联系电话
        :param role: 角色
        :return: 用户ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT INTO users (name, employee_id, department, phone, role) VALUES (?, ?, ?, ?, ?)",
                (name, employee_id, department, phone, role)
            )
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            raise ValueError(f"工号 '{employee_id}' 已存在" if employee_id else "用户信息重复")
        finally:
            conn.close()

    def update_user(self, user_id: int, name: str = None, employee_id: str = None,
                    department: str = None, phone: str = None, is_active: int = None) -> bool:
        """
        更新用户信息
        :param user_id: 用户ID
        :return: 是否更新成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if employee_id is not None:
            updates.append("employee_id = ?")
            params.append(employee_id)
        if department is not None:
            updates.append("department = ?")
            params.append(department)
        if phone is not None:
            updates.append("phone = ?")
            params.append(phone)
        if is_active is not None:
            updates.append("is_active = ?")
            params.append(is_active)

        if not updates:
            return False

        params.append(user_id)
        cursor.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", params)
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def delete_user(self, user_id: int) -> bool:
        """
        软删除用户（标记为不活跃）
        :param user_id: 用户ID
        :return: 是否删除成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 软删除：标记为不活跃
        cursor.execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def hard_delete_user(self, user_id: int) -> bool:
        """
        硬删除用户（从数据库彻底移除，日志中用户名改为"已删除"）
        :param user_id: 用户ID
        :return: 是否删除成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 先删除人脸特征记录
        cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))

        # 更新访问日志，保留记录但标记用户名
        cursor.execute(
            "UPDATE access_log SET user_name = user_name || ' [已删除]' WHERE user_id = ?",
            (user_id,)
        )

        # 彻底删除用户记录
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def delete_face_encodings(self, user_id: int) -> bool:
        """
        删除用户的所有人脸特征记录
        :param user_id: 用户ID
        :return: 是否删除成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def restore_user(self, user_id: int) -> bool:
        """
        恢复已删除的用户
        :param user_id: 用户ID
        :return: 是否恢复成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_active = 1 WHERE id = ?", (user_id,))
        restored = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return restored

    def get_user_by_name(self, name: str) -> Optional[Dict]:
        """
        根据姓名获取用户（包括不活跃的）
        :param name: 用户姓名
        :return: 用户信息
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None

    def get_user(self, user_id: int) -> Optional[Dict]:
        """
        根据ID获取用户
        :param user_id: 用户ID
        :return: 用户信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0], "employee_id": row[1], "name": row[2],
                "department": row[3], "phone": row[4], "role": row[5],
                "is_active": row[6], "created_at": row[7],
                "work_photo": row[8] if len(row) > 8 else None
            }
        return None

    def get_user_by_name(self, name: str) -> Optional[Dict]:
        """
        根据姓名获取用户
        :param name: 用户姓名
        :return: 用户信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0], "employee_id": row[1], "name": row[2],
                "department": row[3], "phone": row[4], "role": row[5],
                "is_active": row[6], "created_at": row[7],
                "work_photo": row[8] if len(row) > 8 else None
            }
        return None

    def get_all_users(self, active_only: bool = False) -> List[Dict]:
        """
        获取所有用户
        :param active_only: 是否只获取启用的用户
        :return: 用户列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if active_only:
            cursor.execute("SELECT * FROM users WHERE is_active = 1 ORDER BY employee_id ASC")
        else:
            cursor.execute("SELECT * FROM users ORDER BY employee_id ASC")

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0], "employee_id": row[1], "name": row[2],
                "department": row[3], "phone": row[4], "role": row[5],
                "is_active": row[6], "created_at": row[7],
                "work_photo": row[8] if len(row) > 8 else None
            }
            for row in rows
        ]

    def search_users(self, keyword: str) -> List[Dict]:
        """
        搜索用户
        :param keyword: 搜索关键词
        :return: 用户列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM users WHERE name LIKE ? OR employee_id LIKE ? OR department LIKE ?",
            (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%")
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0], "employee_id": row[1], "name": row[2],
                "department": row[3], "phone": row[4], "role": row[5],
                "is_active": row[6], "created_at": row[7],
                "work_photo": row[8] if len(row) > 8 else None
            }
            for row in rows
        ]

    def add_face_encoding(self, user_id: int, encoding: np.ndarray, image_path: str = None):
        """
        添加人脸特征
        :param user_id: 用户ID
        :param encoding: 人脸特征向量
        :param image_path: 人脸图片路径
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 统一转为 float32 存储，避免读取时维度错位
        encoding_bytes = encoding.astype(np.float32).tobytes()
        cursor.execute(
            "INSERT INTO face_encodings (user_id, encoding, image_path) VALUES (?, ?, ?)",
            (user_id, encoding_bytes, image_path)
        )
        conn.commit()
        conn.close()

    def get_face_encodings(self, user_id: int = None) -> List[Tuple[int, np.ndarray]]:
        """
        获取人脸特征
        :param user_id: 用户ID，为None时获取所有
        :return: [(user_id, encoding), ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if user_id:
            cursor.execute("SELECT user_id, encoding FROM face_encodings WHERE user_id = ?", (user_id,))
        else:
            cursor.execute("SELECT user_id, encoding FROM face_encodings")

        rows = cursor.fetchall()
        conn.close()

        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    def get_user_faces_with_id(self, user_id: int) -> List[Dict]:
        """
        获取用户的人脸照片列表（包含ID，用于删除）
        :param user_id: 用户ID
        :return: [{'id': face_id, 'image_path': path, 'created_at': time}, ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, image_path, created_at FROM face_encodings WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()

        return [{'id': row[0], 'image_path': row[1], 'created_at': row[2]} for row in rows]

    def get_user_face_count(self, user_id: int) -> int:
        """
        获取用户的人脸照片数量
        :param user_id: 用户ID
        :return: 照片数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM face_encodings WHERE user_id = ?", (user_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def delete_face_encodings(self, user_id: int) -> bool:
        """
        删除用户的所有人脸特征
        :param user_id: 用户ID
        :return: 是否删除成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM face_encodings WHERE user_id = ?", (user_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def delete_face_encoding(self, face_id: int) -> bool:
        """
        删除单个人脸特征
        :param face_id: 人脸特征ID
        :return: 是否删除成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM face_encodings WHERE id = ?", (face_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def set_work_photo(self, user_id: int, path: str) -> bool:
        """
        设置用户工作照片
        :param user_id: 用户ID
        :param path: 照片文件路径
        :return: 是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET work_photo = ? WHERE id = ?", (path, user_id))
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def clear_work_photo(self, user_id: int) -> bool:
        """
        清除用户工作照片
        :param user_id: 用户ID
        :return: 是否成功
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET work_photo = NULL WHERE id = ?", (user_id,))
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    def log_access(self, user_id: int, user_name: str, result: str,
                   confidence: float = None, gate_name: str = None):
        """
        记录门禁日志
        :param user_id: 用户ID
        :param user_name: 用户姓名
        :param result: 结果 (pass/reject/stranger)
        :param confidence: 置信度
        :param gate_name: 门禁点名称
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO access_log (user_id, user_name, result, confidence, gate_name) VALUES (?, ?, ?, ?, ?)",
            (user_id, user_name, result, confidence, gate_name)
        )
        conn.commit()
        conn.close()

    def get_access_log(self, limit: int = 100, user_id: int = None) -> List[Dict]:
        """
        获取门禁日志
        :param limit: 返回条数限制
        :param user_id: 用户ID，为None时获取所有
        :return: 日志列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if user_id:
            cursor.execute(
                "SELECT * FROM access_log WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                (user_id, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM access_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "id": row[0], "user_id": row[1], "user_name": row[2],
                "timestamp": row[3], "result": row[4], "confidence": row[5],
                "gate_name": row[6]
            }
            for row in rows
        ]

    def get_statistics(self) -> Dict:
        """
        获取统计信息
        :return: 统计字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 总用户数
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]

        # 启用用户数
        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        active_users = cursor.fetchone()[0]

        # 总人脸数
        cursor.execute("SELECT COUNT(*) FROM face_encodings")
        total_faces = cursor.fetchone()[0]

        # 今日通行数
        cursor.execute("SELECT COUNT(*) FROM access_log WHERE date(timestamp) = date('now')")
        today_access = cursor.fetchone()[0]

        # 今日通过数
        cursor.execute("SELECT COUNT(*) FROM access_log WHERE date(timestamp) = date('now') AND result = 'pass'")
        today_pass = cursor.fetchone()[0]

        conn.close()

        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_faces": total_faces,
            "today_access": today_access,
            "today_pass": today_pass
        }
