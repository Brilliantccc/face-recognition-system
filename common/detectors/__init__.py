"""
人脸检测器模块
支持 YOLO、face_recognition、Haar Cascade 等检测器
"""

from .base import FaceDetector, FaceDetection
from .yolo import YOLODetector
from .face_recognition import FaceRecognitionDetector
from .haar import HaarDetector

# 检测器优先级（auto 模式下按此顺序尝试）
DETECTOR_PRIORITY = ["yolo", "face_recognition", "haar"]

# 已注册的检测器
DETECTORS = {
    "yolo": YOLODetector,
    "face_recognition": FaceRecognitionDetector,
    "haar": HaarDetector,
}


class DetectorFactory:
    """检测器工厂类"""

    @staticmethod
    def create(backend: str = "auto", **kwargs) -> FaceDetector:
        """
        创建检测器
        :param backend: 后端名称 ("yolo", "face_recognition", "haar", "auto")
        :param kwargs: 传递给检测器的参数
        :return: 检测器实例
        """
        if backend == "auto":
            return DetectorFactory._create_auto(**kwargs)

        if backend not in DETECTORS:
            raise ValueError(f"未知的检测器后端: {backend}，可选: {list(DETECTORS.keys())}")

        return DETECTORS[backend](**kwargs)

    @staticmethod
    def _create_auto(**kwargs) -> FaceDetector:
        """自动选择可用的检测器"""
        errors = []
        for name in DETECTOR_PRIORITY:
            try:
                return DetectorFactory.create(name, **kwargs)
            except Exception as e:
                errors.append(f"{name}: {e}")
                continue

        raise RuntimeError(f"没有可用的检测器:\n" + "\n".join(errors))

    @staticmethod
    def list_available() -> dict:
        """列出所有可用的检测器"""
        available = {}
        for name, cls in DETECTORS.items():
            try:
                # 尝试创建实例（不传参数）
                if name == "yolo":
                    # YOLO 需要模型路径，跳过检查
                    available[name] = {"class": cls, "available": True}
                else:
                    available[name] = {"class": cls, "available": True}
            except Exception as e:
                available[name] = {"class": cls, "available": False, "error": str(e)}
        return available


__all__ = [
    "FaceDetector",
    "FaceDetection",
    "YOLODetector",
    "FaceRecognitionDetector",
    "HaarDetector",
    "DetectorFactory",
]
