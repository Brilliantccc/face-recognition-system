"""
用户管理模块
处理用户注册、删除和查询
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from .database import Database
from .config import FACES_DIR, FACE_RECOGNITION_TOLERANCE


_face_recognition_module = None
_face_recognition_loaded = False


def _get_face_recognition():
    """延迟加载 face_recognition（加载失败返回 None，不崩溃）"""
    global _face_recognition_module, _face_recognition_loaded
    if _face_recognition_loaded:
        return _face_recognition_module
    _face_recognition_loaded = True
    try:
        import face_recognition
        _face_recognition_module = face_recognition
    except Exception:
        _face_recognition_module = None
    return _face_recognition_module


def imread_safe(filepath):
    """读取图片，支持中文路径（Windows）"""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class UserManager:
    def __init__(self, db: Database, faces_dir: str = None):
        """
        初始化用户管理器
        :param db: 数据库实例
        :param faces_dir: 人脸图片存储目录
        """
        self.db = db
        self.faces_dir = faces_dir or FACES_DIR
        os.makedirs(self.faces_dir, exist_ok=True)

        # 人脸特征缓存
        self._cached_encodings = []
        self._cached_names = []
        self._cached_user_ids = []
        self._cache_valid = False

    def _invalidate_cache(self):
        """使缓存失效"""
        self._cache_valid = False

    def _ensure_cache(self):
        """确保缓存有效"""
        if not self._cache_valid:
            self._cached_encodings, self._cached_names, self._cached_user_ids = \
                self.get_user_encodings()
            self._cache_valid = True

    def register_user(self, name: str, images: List[np.ndarray],
                      employee_id: str = None, department: str = None,
                      phone: str = None) -> int:
        """
        注册新用户（自动去重）
        :param name: 用户姓名
        :param images: 人脸图片列表
        :param employee_id: 工号
        :param department: 部门
        :param phone: 联系电话
        :return: 用户ID
        :raises ValueError: 如果没有检测到任何人脸
        """
        # 检查用户是否已存在（包括不活跃的）
        existing_user = self.db.get_user_by_name(name)
        
        if existing_user:
            user_id = existing_user['id']
            # 如果用户不活跃，重新激活
            if not existing_user.get('is_active', True):
                self.db.restore_user(user_id)
                print(f"  ℹ️ 恢复已存在的用户: {name} (ID: {user_id})")
            else:
                print(f"  ℹ️ 用户已存在: {name} (ID: {user_id})，将添加更多图片")
        else:
            # 添加用户到数据库
            user_id = self.db.add_user(name, employee_id, department, phone)

        # 创建用户目录
        user_dir = self._get_user_dir(user_id, name)
        os.makedirs(user_dir, exist_ok=True)

        # 获取现有照片数量
        existing_count = len([f for f in os.listdir(user_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        saved_count = 0
        fr = _get_face_recognition()
        if fr is None:
            raise RuntimeError("face_recognition/dlib 未安装，无法注册人脸")

        for i, image in enumerate(images):
            # 如果图片太大，先缩小
            h, w = image.shape[:2]
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale)

            # 检测人脸
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = fr.face_locations(rgb_image, model="hog")

            if len(face_locations) == 0:
                face_locations = fr.face_locations(rgb_image, model="cnn")

            if len(face_locations) == 0:
                continue

            # 提取人脸特征
            face_encodings = fr.face_encodings(rgb_image, face_locations)

            if len(face_encodings) > 0:
                # 保存图片（编号从现有数量开始）
                image_path = os.path.join(user_dir, f"{existing_count + saved_count}.jpg")
                # 使用 imencode 保存图片，支持中文路径
                _, img_encoded = cv2.imencode('.jpg', image)
                img_encoded.tofile(image_path)

                # 保存人脸特征到数据库
                self.db.add_face_encoding(user_id, face_encodings[0], image_path)
                saved_count += 1

        # 使缓存失效
        self._invalidate_cache()

        # 如果是新用户且没有保存任何人脸，删除用户并抛出异常
        if saved_count == 0 and not existing_user:
            self.db.delete_user(user_id)
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)
            raise ValueError("所有图片都未检测到人脸，无法注册！")

        return user_id

    def _get_user_dir(self, user_id: int, name: str) -> str:
        """
        获取用户的图片存储目录
        :param user_id: 用户ID
        :param name: 用户姓名
        :return: 目录路径
        """
        # 保留中文、字母、数字、空格、连字符和下划线
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_') or '\u4e00' <= c <= '\u9fff').strip()
        if not safe_name:
            safe_name = "user"
        return os.path.join(self.faces_dir, f"{user_id}_{safe_name}")

    def add_faces_to_user(self, user_id: int, images: List[np.ndarray]) -> int:
        """
        为已有用户添加人脸照片
        :param user_id: 用户ID
        :param images: 人脸图片列表
        :return: 成功添加的数量
        """
        user = self.db.get_user(user_id)
        if not user:
            raise ValueError(f"用户ID {user_id} 不存在")

        # 创建用户目录
        user_dir = self._get_user_dir(user_id, user['name'])
        os.makedirs(user_dir, exist_ok=True)

        # 获取现有照片数量
        existing_count = self.db.get_user_face_count(user_id)
        saved_count = 0

        fr = _get_face_recognition()
        if fr is None:
            raise RuntimeError("face_recognition/dlib 未安装，无法添加人脸")

        for image in images:
            # 如果图片太大，先缩小
            h, w = image.shape[:2]
            if max(h, w) > 1000:
                scale = 1000 / max(h, w)
                image = cv2.resize(image, None, fx=scale, fy=scale)

            # 检测人脸
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = fr.face_locations(rgb_image, model="hog")

            if len(face_locations) == 0:
                face_locations = fr.face_locations(rgb_image, model="cnn")

            if len(face_locations) == 0:
                continue

            # 提取人脸特征
            face_encodings = fr.face_encodings(rgb_image, face_locations)

            if len(face_encodings) > 0:
                # 保存图片
                image_path = os.path.join(user_dir, f"{existing_count + saved_count}.jpg")
                cv2.imwrite(image_path, image)

                # 保存人脸特征到数据库
                self.db.add_face_encoding(user_id, face_encodings[0], image_path)
                saved_count += 1

        # 使缓存失效
        self._invalidate_cache()
        return saved_count

    def delete_user(self, user_id: int, model_dir: str = None) -> bool:
        """
        删除用户（安全删除：删除图片 + 嵌入向量 + 标记不活跃）
        :param user_id: 用户ID
        :param model_dir: 模型目录（用于找到 known_faces.npy）
        :return: 是否删除成功
        """
        user = self.db.get_user(user_id)
        if not user:
            return False
        
        user_name = user['name']
        
        # 1. 删除人脸图片文件
        user_dir = self._get_user_dir(user_id, user_name)
        if os.path.exists(user_dir):
            import shutil
            shutil.rmtree(user_dir)
            print(f"  🗑️ 删除图片目录: {user_dir}")
        
        # 2. 从 known_faces.npy 中移除嵌入向量
        if model_dir:
            known_faces_path = os.path.join(model_dir, "known_faces.npy")
        else:
            known_faces_path = "data/db/known_faces.npy"
        
        if os.path.exists(known_faces_path):
            embeddings = np.load(known_faces_path, allow_pickle=True).item()
            if user_name in embeddings:
                del embeddings[user_name]
                np.save(known_faces_path, embeddings)
                print(f"  🗑️ 从 known_faces.npy 移除: {user_name}")
        
        # 3. 删除数据库中的人脸特征记录
        self.db.delete_face_encodings(user_id)
        
        # 4. 标记用户为不活跃（保留日志记录）
        result = self.db.delete_user(user_id)
        if result:
            self._invalidate_cache()
            print(f"  ✅ 用户已删除: {user_name} (ID: {user_id})")
        
        return result

    def restore_user(self, user_id: int) -> bool:
        """
        恢复已删除的用户
        :param user_id: 用户ID
        :return: 是否恢复成功
        """
        result = self.db.restore_user(user_id)
        if result:
            self._invalidate_cache()
        return result

    def get_all_users(self, active_only: bool = False) -> List[Dict]:
        """
        获取所有用户
        :param active_only: 是否只获取启用的用户
        :return: 用户列表
        """
        return self.db.get_all_users(active_only)

    def get_user(self, user_id: int) -> Optional[Dict]:
        """
        获取用户信息
        :param user_id: 用户ID
        :return: 用户信息
        """
        return self.db.get_user(user_id)

    def search_users(self, keyword: str) -> List[Dict]:
        """
        搜索用户
        :param keyword: 搜索关键词
        :return: 用户列表
        """
        return self.db.search_users(keyword)

    def get_user_encodings(self) -> Tuple[List[np.ndarray], List[str], List[int]]:
        """
        获取所有人脸特征用于识别
        :return: (人脸特征列表, 用户名列表, 用户ID列表)
        """
        encodings_data = self.db.get_face_encodings()

        encodings = []
        names = []
        user_ids = []

        users = {user["id"]: user["name"] for user in self.db.get_all_users(active_only=True)}

        for user_id, encoding in encodings_data:
            if user_id in users:
                encodings.append(encoding)
                names.append(users[user_id])
                user_ids.append(user_id)

        return encodings, names, user_ids

    def load_images_from_files(self, file_paths: List[str]) -> List[np.ndarray]:
        """
        从文件路径加载图片
        :param file_paths: 图片文件路径列表
        :return: 图片列表
        """
        images = []
        for file_path in file_paths:
            if os.path.exists(file_path):
                image = imread_safe(file_path)
                if image is not None:
                    images.append(image)
        return images

    def verify_user(self, face_encoding: np.ndarray, tolerance: float = None) -> Tuple[bool, str, float, int]:
        """
        验证用户身份
        :param face_encoding: 待验证的人脸特征
        :param tolerance: 容差阈值
        :return: (是否通过, 用户名, 置信度, 用户ID)
        """
        if tolerance is None:
            tolerance = FACE_RECOGNITION_TOLERANCE

        self._ensure_cache()

        if len(self._cached_encodings) == 0:
            return False, "Unknown", 0.0, -1

        fr = _get_face_recognition()
        if fr is None:
            return False, "Unknown", 0.0, -1
        face_distances = fr.face_distance(self._cached_encodings, face_encoding)

        best_match_idx = np.argmin(face_distances)
        best_distance = face_distances[best_match_idx]

        if best_distance <= tolerance:
            confidence = max(0, 1 - best_distance)
            return True, self._cached_names[best_match_idx], confidence, self._cached_user_ids[best_match_idx]
        else:
            return False, "Unknown", 0.0, -1

    def reload_cache(self):
        """手动重新加载缓存"""
        self._invalidate_cache()
        self._ensure_cache()
