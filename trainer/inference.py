"""
人脸模型推理模块
使用训练好的模型进行人脸识别
"""

import os
import sys
import json
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.models.facenet import MobileFaceNet


class FaceRecognizer:
    """人脸识别器"""
    def __init__(self, model_dir="trainer/models/saved", confidence_threshold=0.6):
        """
        初始化识别器
        :param model_dir: 模型目录
        :param confidence_threshold: 置信度阈值
        """
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 人脸检测器
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default_aligned.xml')
        if self.face_cascade.empty():
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载模型
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.embeddings = {}  # 存储已知人脸的嵌入向量
        self.load_model()

    def load_model(self):
        """加载模型"""
        model_path = os.path.join(self.model_dir, "inference_model.pth")
        info_path = os.path.join(self.model_dir, "model_info.json")

        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            return False

        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = MobileFaceNet(
            embedding_size=checkpoint['embedding_size'],
            num_classes=None  # 推理时不使用分类头
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # 加载类别信息
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                self.class_to_idx = info.get('class_to_idx', {})
                self.idx_to_class = {int(k): v for k, v in info.get('idx_to_class', {}).items()}

        print(f"Model loaded successfully. Classes: {len(self.idx_to_class)}")
        return True

    def detect_faces(self, image):
        """
        检测人脸
        :param image: BGR格式的图片
        :return: 人脸位置列表 [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def preprocess_face(self, image, face_rect):
        """
        预处理人脸
        :param image: 原始图片
        :param face_rect: 人脸位置 (x, y, w, h)
        :return: 预处理后的图片张量
        """
        x, y, w, h = face_rect

        # 扩大人脸区域
        margin = int(0.2 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)

        # 裁剪人脸
        face_img = image[y1:y2, x1:x2]

        # 转换为 PIL Image
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = Image.fromarray(face_img)

        # 应用变换
        face_tensor = self.transform(face_img)

        return face_tensor.unsqueeze(0).to(self.device)

    def get_embedding(self, face_tensor):
        """
        获取人脸嵌入向量
        :param face_tensor: 预处理后的人脸张量
        :return: 嵌入向量
        """
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding.cpu().numpy().flatten()

    def recognize(self, image):
        """
        识别图片中的人脸
        :param image: BGR格式的图片
        :return: 识别结果列表 [(name, confidence, (x, y, w, h)), ...]
        """
        if self.model is None:
            return []

        results = []
        faces = self.detect_faces(image)

        for face_rect in faces:
            # 预处理
            face_tensor = self.preprocess_face(image, face_rect)

            # 获取嵌入
            embedding = self.get_embedding(face_tensor)

            # 计算与已知人脸的距离
            best_name = "Unknown"
            best_confidence = 0.0

            if self.embeddings:
                for name, known_embedding in self.embeddings.items():
                    # 计算余弦相似度
                    similarity = np.dot(embedding, known_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
                    )

                    if similarity > best_confidence and similarity > self.confidence_threshold:
                        best_confidence = similarity
                        best_name = name

            results.append((best_name, best_confidence, tuple(face_rect)))

        return results

    def register_face(self, name, image):
        """
        注册新的人脸
        :param name: 人员姓名
        :param image: BGR格式的图片
        :return: 是否成功
        """
        if self.model is None:
            print("Error: Model not loaded")
            return False

        faces = self.detect_faces(image)
        if len(faces) == 0:
            print("No face detected")
            return False

        # 使用最大的人脸
        face_rect = max(faces, key=lambda f: f[2] * f[3])
        face_tensor = self.preprocess_face(image, face_rect)
        embedding = self.get_embedding(face_tensor)

        # 存储嵌入向量
        if name not in self.embeddings:
            self.embeddings[name] = embedding
        else:
            # 平均新的嵌入
            self.embeddings[name] = (self.embeddings[name] + embedding) / 2

        print(f"Face registered for: {name}")
        return True

    def save_embeddings(self, path="trainer/models/saved/known_faces.npy"):
        """保存已知人脸的嵌入向量"""
        np.save(path, self.embeddings)
        print(f"Embeddings saved to {path}")

    def load_embeddings(self, path="trainer/models/saved/known_faces.npy"):
        """加载已知人脸的嵌入向量"""
        if os.path.exists(path):
            self.embeddings = np.load(path, allow_pickle=True).item()
            print(f"Loaded {len(self.embeddings)} face embeddings")
            return True
        return False


class FaceDetector:
    """简单的人脸检测器（使用 OpenCV）"""
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default_aligned.xml')
        if self.face_cascade.empty():
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, image, min_confidence=0.5):
        """
        检测人脸
        :param image: BGR格式的图片
        :return: 人脸位置列表
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces


def test_recognition():
    """测试识别功能"""
    recognizer = FaceRecognizer()

    # 加载已知人脸
    recognizer.load_embeddings()

    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    print("Press SPACE to register current face, ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 识别
        results = recognizer.recognize(frame)

        # 绘制结果
        for name, confidence, (x, y, w, h) in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            text = f"{name}: {confidence:.2%}" if name != "Unknown" else "Unknown"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Face Recognition Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            name = input("Enter name for registration: ")
            if name:
                recognizer.register_face(name, frame)
                recognizer.save_embeddings()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_recognition()
