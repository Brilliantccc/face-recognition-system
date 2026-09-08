"""
人脸数据库模块
支持 embeddings.bin、SQLite 等数据库格式
"""

from .base import FaceDatabase
from .embeddings_bin import EmbeddingsBinDatabase
from .sqlite_db import SQLiteDatabase

# 数据库优先级（auto 模式下按此顺序尝试）
DATABASE_PRIORITY = ["bin", "db"]

# 已注册的数据库
DATABASES = {
    "bin": EmbeddingsBinDatabase,
    "db": SQLiteDatabase,
}


class DatabaseFactory:
    """数据库工厂类"""

    @staticmethod
    def create(backend: str = "auto", **kwargs) -> FaceDatabase:
        """
        创建数据库
        :param backend: 后端名称 ("bin", "db", "auto")
        :param kwargs: 传递给数据库的参数
        :return: 数据库实例
        """
        if backend == "auto":
            return DatabaseFactory._create_auto(**kwargs)

        if backend not in DATABASES:
            raise ValueError(f"未知的数据库后端: {backend}，可选: {list(DATABASES.keys())}")

        return DATABASES[backend](**kwargs)

    @staticmethod
    def _create_auto(**kwargs) -> FaceDatabase:
        """自动选择可用的数据库"""
        errors = []
        for name in DATABASE_PRIORITY:
            try:
                db = DatabaseFactory.create(name, **kwargs)
                # 检查数据库是否有数据
                if db.load():
                    return db
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue

        # 如果没有数据，返回第一个可用的数据库
        for name in DATABASE_PRIORITY:
            try:
                return DatabaseFactory.create(name, **kwargs)
            except Exception:
                continue

        raise RuntimeError(f"没有可用的数据库:\n" + "\n".join(errors))

    @staticmethod
    def list_available() -> dict:
        """列出所有可用的数据库"""
        available = {}
        for name, cls in DATABASES.items():
            try:
                available[name] = {"class": cls, "available": True}
            except Exception as e:
                available[name] = {"class": cls, "available": False, "error": str(e)}
        return available


__all__ = [
    "FaceDatabase",
    "EmbeddingsBinDatabase",
    "SQLiteDatabase",
    "DatabaseFactory",
]
