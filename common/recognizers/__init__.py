"""
人脸识别器模块
支持 MobileFaceNet、face_recognition 等识别器
"""

from .base import FaceRecognizer
from .mobilenet import MobileFaceNetRecognizer
from .face_recognition import FaceRecognitionRecognizer

# 识别器优先级（auto 模式下按此顺序尝试）
RECOGNIZER_PRIORITY = ["mobilenet", "face_recognition"]

# 已注册的识别器
RECOGNIZERS = {
    "mobilenet": MobileFaceNetRecognizer,
    "face_recognition": FaceRecognitionRecognizer,
}


class RecognizerFactory:
    """识别器工厂类"""

    @staticmethod
    def create(backend: str = "auto", **kwargs) -> FaceRecognizer:
        """
        创建识别器
        :param backend: 后端名称 ("mobilenet", "face_recognition", "auto")
        :param kwargs: 传递给识别器的参数
        :return: 识别器实例
        """
        if backend == "auto":
            return RecognizerFactory._create_auto(**kwargs)

        if backend not in RECOGNIZERS:
            raise ValueError(f"未知的识别器后端: {backend}，可选: {list(RECOGNIZERS.keys())}")

        return RECOGNIZERS[backend](**kwargs)

    @staticmethod
    def _create_auto(**kwargs) -> FaceRecognizer:
        """自动选择可用的识别器"""
        errors = []
        for name in RECOGNIZER_PRIORITY:
            try:
                return RecognizerFactory.create(name, **kwargs)
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue

        raise RuntimeError(f"没有可用的识别器:\n" + "\n".join(errors))

    @staticmethod
    def list_available() -> dict:
        """列出所有可用的识别器"""
        available = {}
        for name, cls in RECOGNIZERS.items():
            try:
                available[name] = {"class": cls, "available": True}
            except Exception as e:
                available[name] = {"class": cls, "available": False, "error": str(e)}
        return available


__all__ = [
    "FaceRecognizer",
    "MobileFaceNetRecognizer",
    "FaceRecognitionRecognizer",
    "RecognizerFactory",
]
