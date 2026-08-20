"""
YOLO 人脸检测模块
使用 YOLOv8 进行快速人脸检测
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import sys

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class YOLOFaceDetector:
    """基于 YOLOv8 的人脸检测器"""

    def __init__(self, model_size: str = "n", confidence: float = 0.5, device: str = "cpu"):
        """
        初始化 YOLO 人脸检测器
        :param model_size: 模型大小 (n/nano, s/small, m/medium, l/large, x/xlarge)
        :param confidence: 置信度阈值
        :param device: 推理设备 (cpu/cuda)
        """
        self.model_size = model_size
        self.confidence = confidence
        self.device = device
        self.model = None

        self._load_model()

    def _load_model(self):
        """加载 YOLO 人脸检测模型"""
        try:
            from ultralytics import YOLO

            model_name = f"yolov8{self.model_size}.pt"

            # 检查是否有本地的人脸检测模型
            local_model_path = os.path.join(
                os.path.dirname(__file__), "..", "models", f"yolov8{self.model_size}_face.pt"
            )

            if os.path.exists(local_model_path):
                self.model = YOLO(local_model_path)
                print(f"Loaded local face model: {local_model_path}")
            else:
                self.model = YOLO(model_name)
                print(f"Loaded YOLO model: {model_name}")

            print(f"YOLOFaceDetector initialized (device: {self.device})")

        except ImportError:
            raise ImportError(
                "请安装 ultralytics: pip install ultralytics"
            )

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        检测人脸
        :param image: BGR 格式的图片
        :return: 人脸位置列表 [(x1, y1, x2, y2, confidence), ...]
        """
        if self.model is None:
            return []

        results = self.model(
            image,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        faces = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    if self._is_face_class(cls):
                        faces.append((x1, y1, x2, y2, conf))

        return faces

    def _is_face_class(self, class_id: int) -> bool:
        """检查类别是否为人脸"""
        return True

    def detect_with_landmarks(self, image: np.ndarray) -> List[dict]:
        """检测人脸并返回详细信息"""
        if self.model is None:
            return []

        results = self.model(
            image,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        faces = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    if self._is_face_class(cls):
                        face_info = {
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf,
                            "width": x2 - x1,
                            "height": y2 - y1
                        }
                        faces.append(face_info)

        return faces

    def extract_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                     margin: float = 0.2) -> Optional[np.ndarray]:
        """
        从图片中提取人脸区域
        :param image: 原始图片
        :param bbox: 边界框 (x1, y1, x2, y2)
        :param margin: 边距比例
        :return: 提取的人脸图片
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)

        x1_new = max(0, x1 - margin_x)
        y1_new = max(0, y1 - margin_y)
        x2_new = min(w, x2 + margin_x)
        y2_new = min(h, y2 + margin_y)

        face_img = image[y1_new:y2_new, x1_new:x2_new]

        if face_img.size == 0:
            return None

        return face_img


class YOLODetectorFactory:
    """YOLO 检测器工厂类"""

    @staticmethod
    def create_detector(model_size: str = "n", confidence: float = 0.5,
                       device: str = "cpu") -> YOLOFaceDetector:
        return YOLOFaceDetector(
            model_size=model_size,
            confidence=confidence,
            device=device
        )


if __name__ == "__main__":
    detector = YOLOFaceDetector(model_size="n", confidence=0.5)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    print("Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect(frame)

        for x1, y1, x2, y2, conf in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"Face: {conf:.2%}"
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLO Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
