"""
人脸模型推理模块
使用训练好的模型进行人脸识别
支持 YOLO 人脸检测 (更快速)
"""

import os
import sys
import json
import cv2
import numpy as np
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用统一的 PyTorch 工具模块
from common import torch_utils
from common.config import USE_YOLO_DETECTION, YOLO_MODEL_SIZE, YOLO_CONFIDENCE, FACE_INPUT_SIZE

# 从 torch_utils 获取 PyTorch 组件
torch = torch_utils.torch
transforms = torch_utils.transforms
Image = torch_utils.Image
TORCH_AVAILABLE = torch_utils.TORCH_AVAILABLE

# 导入模型
MobileFaceNet = None
if TORCH_AVAILABLE:
    try:
        from trainer.models.facenet import MobileFaceNet
    except Exception as e:
        print(f"Warning: Failed to import MobileFaceNet: {e}")


class FaceRecognizer:
    """人脸识别器"""
    def __init__(self, model_dir="trainer/models/saved", confidence_threshold=0.6,
                 use_yolo: bool = None, yolo_model_size: str = None, yolo_confidence: float = None):
        """
        初始化识别器
        :param model_dir: 模型目录
        :param confidence_threshold: 置信度阈值
        :param use_yolo: 是否使用 YOLO 进行人脸检测
        :param yolo_model_size: YOLO 模型大小
        :param yolo_confidence: YOLO 置信度阈值
        """
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        # 使用 torch_utils 统一管理设备选择
        self.device = torch_utils.get_device(prefer_gpu=True)
        if self.device is None:
            raise RuntimeError("PyTorch 未安装")
        print(f"FaceRecognizer using device: {self.device}")

        # YOLO 配置
        self.use_yolo = use_yolo if use_yolo is not None else USE_YOLO_DETECTION
        self.yolo_detector = None

        if self.use_yolo:
            self._init_yolo_detector(yolo_model_size, yolo_confidence)

        # 人脸检测器 (备用)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default_aligned.xml')
        if self.face_cascade.empty():
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 数据变换
        self.transform = transforms.Compose([
            transforms.Resize((FACE_INPUT_SIZE, FACE_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载模型
        self.model = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.embeddings = {}  # 存储已知人脸的嵌入向量
        self.load_model()

    def _init_yolo_detector(self, model_size: str = None, confidence: float = None):
        """初始化 YOLO 检测器"""
        try:
            from common.yolo_detector import YOLOFaceDetector

            model_size = model_size or YOLO_MODEL_SIZE
            conf = confidence or YOLO_CONFIDENCE
            
            # 自动检测 GPU 可用性
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

            self.yolo_detector = YOLOFaceDetector(
                model_size=model_size,
                confidence=conf,
                device=device
            )
            print(f"YOLO detector initialized for FaceRecognizer (device: {device})")
        except ImportError as e:
            print(f"Warning: YOLO not available, using Haar Cascade: {e}")
            self.use_yolo = False
        except Exception as e:
            print(f"Warning: YOLO initialization failed: {e}")
            self.use_yolo = False

    def load_model(self):
        """加载模型（自动查找最新版本）"""
        model_path = os.path.join(self.model_dir, "inference_model.pth")
        info_path = os.path.join(self.model_dir, "model_info.json")

        # 如果直接路径没有模型，自动扫描子目录找最新版本
        if not os.path.exists(model_path) and os.path.isdir(self.model_dir):
            version_dirs = sorted(
                [d for d in os.listdir(self.model_dir)
                 if os.path.isdir(os.path.join(self.model_dir, d))],
                key=lambda d: os.path.getmtime(os.path.join(self.model_dir, d)),
                reverse=True
            )
            for vdir in version_dirs:
                candidate = os.path.join(self.model_dir, vdir, "inference_model.pth")
                if os.path.exists(candidate):
                    self.model_dir = os.path.join(self.model_dir, vdir)
                    model_path = candidate
                    info_path = os.path.join(self.model_dir, "model_info.json")
                    print(f"Auto-found model in: {vdir}")
                    break

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

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测人脸
        :param image: BGR格式的图片
        :return: 人脸位置列表 [(x, y, w, h), ...]
        """
        if self.use_yolo and self.yolo_detector is not None:
            return self._detect_faces_yolo(image)
        else:
            return self._detect_faces_haar(image)

    def _detect_faces_yolo(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        使用 YOLO 检测人脸
        :param image: BGR格式的图片
        :return: 人脸位置列表 [(x, y, w, h), ...]
        """
        detections = self.yolo_detector.detect(image)

        faces = []
        for x1, y1, x2, y2, conf in detections:
            # 转换为 (x, y, w, h) 格式
            w = x2 - x1
            h = y2 - y1
            faces.append((x1, y1, w, h))

        return faces

    def _detect_faces_haar(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        使用 Haar Cascade 检测人脸
        :param image: BGR格式的图片
        :return: 人脸位置列表 [(x, y, w, h), ...]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)
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
        注册新的人脸（保存到数据库）
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

        # 保存到数据库
        try:
            from common.database import Database
            from common.user_manager import UserManager
            from common.config import DATABASE_PATH, FACES_DIR
            
            db = Database(DATABASE_PATH)
            user_manager = UserManager(db, FACES_DIR)
            
            # 检查用户是否已存在
            existing_user = db.get_user_by_name(name)
            if existing_user:
                user_id = existing_user['id']
                if not existing_user.get('is_active', True):
                    db.restore_user(user_id)
                    print(f"恢复已存在的用户: {name}")
            else:
                user_id = db.add_user(name)
            
            # 保存图片
            user_dir = os.path.join(FACES_DIR, str(user_id) + "_" + name)
            os.makedirs(user_dir, exist_ok=True)
            
            existing_count = len([f for f in os.listdir(user_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            image_path = os.path.join(user_dir, f"camera_{existing_count + 1}.jpg")
            cv2.imwrite(image_path, image)
            
            # 保存到数据库
            db.add_face_encoding(user_id, embedding, image_path)
            
            print(f"✅ 注册成功: {name} (ID: {user_id})")
            return True
            
        except Exception as e:
            print(f"❌ 注册失败: {e}")
            return False

    def save_embeddings(self, path="data/db/known_faces.npy"):
        """保存已知人脸的嵌入向量"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.embeddings)
        print(f"Embeddings saved to {path}")

    def load_embeddings(self, path="data/db/known_faces.npy"):
        """加载已知人脸的嵌入向量"""
        if os.path.exists(path):
            self.embeddings = np.load(path, allow_pickle=True).item()
            print(f"Loaded {len(self.embeddings)} face embeddings")
            return True
        return False

    def set_yolo_enabled(self, enabled: bool):
        """
        动态启用/禁用 YOLO 检测
        :param enabled: 是否启用
        """
        self.use_yolo = enabled
        if enabled and self.yolo_detector is None:
            self._init_yolo_detector()
        print(f"YOLO detection {'enabled' if enabled else 'disabled'}")

    def get_detection_method(self) -> str:
        """
        获取当前使用的检测方法
        :return: 检测方法名称
        """
        return "YOLO" if (self.use_yolo and self.yolo_detector) else "Haar Cascade"


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


def test_recognition(model_dir="trainer/models/saved"):
    """测试识别功能"""
    recognizer = FaceRecognizer(model_dir=model_dir)

    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    print("\n" + "=" * 50)
    print("人脸识别测试")
    print("=" * 50)
    print("操作说明:")
    print("  SPACE - 拍照并注册人脸")
    print("  ESC   - 退出")
    print("=" * 50 + "\n")

    current_name = None
    registering = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测人脸
        faces = recognizer.detect_faces(frame)

        # 绘制人脸框
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if not registering else (0, 255, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # 显示状态信息
        if registering:
            cv2.putText(frame, f"Registering: {current_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Press SPACE to register", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            # 暂停摄像头，等待输入姓名
            registering = True
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 50)
            name = input("请输入姓名 (或按 Enter 取消): ").strip()
            if name:
                print(f"正在注册: {name}...")
                # 重新打开摄像头拍照
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                
                ret, frame = cap.read()
                if ret:
                    success = recognizer.register_face(name, frame)
                    if success:
                        print(f"✅ 注册成功: {name}")
                    else:
                        print(f"❌ 注册失败")
            else:
                print("已取消")
            
            registering = False
            # 重新打开摄像头
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition Test")
    parser.add_argument("--model-dir", type=str, default="trainer/models/saved",
                       help="模型目录")
    
    args = parser.parse_args()
    test_recognition(args.model_dir)
