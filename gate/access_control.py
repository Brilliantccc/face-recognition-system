"""
门禁控制模块
实现实时人脸识别和门禁控制逻辑
支持 YOLO 人脸检测 + 训练模型识别
"""

import sys
import os

# 确保项目根目录在最前面
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 尝试导入 torch（在其他模块之前）
torch = None
MobileFaceNet = None
Image = None
ImageDraw = None
ImageFont = None
transforms = None
TORCH_AVAILABLE = False

try:
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from torchvision import transforms

    trainer_dir = os.path.join(project_root, 'trainer')
    if trainer_dir not in sys.path:
        sys.path.insert(0, trainer_dir)
    from trainer.models.facenet import MobileFaceNet
    TORCH_AVAILABLE = True
    print("Torch loaded successfully")
except Exception as e:
    print(f"Warning: torch not available: {e}")

import cv2
import numpy as np
import face_recognition
import threading
import time
from typing import Tuple, Optional, List

from common.user_manager import UserManager
from common.config import FACE_RECOGNITION_TOLERANCE, USE_YOLO_DETECTION, YOLO_MODEL_SIZE, YOLO_CONFIDENCE


class AccessControl:
    def __init__(self, user_manager: UserManager, tolerance: float = None, gate_name: str = "Main Gate",
                 use_yolo: bool = None, yolo_model_size: str = None, yolo_confidence: float = None,
                 use_trained_model: bool = True):
        """
        初始化门禁控制器
        :param user_manager: 用户管理器实例
        :param tolerance: 人脸识别容差阈值
        :param gate_name: 门禁点名称
        :param use_yolo: 是否使用 YOLO 进行人脸检测
        :param yolo_model_size: YOLO 模型大小
        :param yolo_confidence: YOLO 置信度阈值
        :param use_trained_model: 是否使用训练好的模型进行识别
        """
        self.user_manager = user_manager
        self.tolerance = tolerance or FACE_RECOGNITION_TOLERANCE
        self.gate_name = gate_name

        # YOLO 配置 (需要 torch)
        self.use_yolo = (use_yolo if use_yolo is not None else USE_YOLO_DETECTION) and TORCH_AVAILABLE
        self.yolo_detector = None

        if self.use_yolo:
            self._init_yolo_detector(yolo_model_size, yolo_confidence)

        # 训练模型配置 (需要 torch)
        self.use_trained_model = use_trained_model and TORCH_AVAILABLE
        self.trained_model = None
        self.trained_model_info = {}
        self.device = None
        self.transform = None

        # 加载训练模型
        if self.use_trained_model:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self._load_trained_model()

        if not TORCH_AVAILABLE:
            print("Running in face_recognition mode (YOLO and trained model disabled)")

        self.current_status = "Standby"
        self.current_user = None
        self.confidence = 0.0

        # 多线程处理相关
        self.processing = False
        self.latest_frame = None
        self.result_frame = None
        self.result_status = "Standby"
        self.result_user = None
        self.result_confidence = 0.0
        self.frame_lock = threading.Lock()
        self.last_process_time = 0
        self.process_interval = 0.3  # 处理间隔

        # 缓存 PIL 字体（避免每次加载）
        self._pil_font = None
        self._pil_font_small = None

        # 加载已知人脸特征
        self.load_known_faces()

        # 加载训练模型
        if self.use_trained_model:
            self._load_trained_model()

    def _init_yolo_detector(self, model_size: str = None, confidence: float = None):
        """初始化 YOLO 检测器"""
        try:
            from common.yolo_detector import YOLOFaceDetector

            model_size = model_size or YOLO_MODEL_SIZE
            conf = confidence or YOLO_CONFIDENCE

            self.yolo_detector = YOLOFaceDetector(
                model_size=model_size,
                confidence=conf,
                device="cpu"  # 可根据需要改为 "cuda"
            )
            print(f"YOLO detector initialized (model: yolov8{model_size}, confidence: {conf})")
        except ImportError as e:
            print(f"Warning: YOLO not available, falling back to face_recognition: {e}")
            self.use_yolo = False
        except Exception as e:
            print(f"Warning: YOLO initialization failed, falling back to face_recognition: {e}")
            self.use_yolo = False

    def _load_trained_model(self):
        """加载训练好的 MobileFaceNet 模型"""
        if not TORCH_AVAILABLE:
            self.use_trained_model = False
            return

        # 优先使用带分类头的模型
        model_path = "trainer/models/saved/best_model.pth"
        info_path = "trainer/models/saved/model_info.json"

        if not os.path.exists(model_path):
            model_path = "trainer/models/saved/inference_model.pth"

        if not os.path.exists(model_path):
            print(f"Warning: Trained model not found at {model_path}")
            self.use_trained_model = False
            return

        try:
            import json

            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)

            # 检查是否有分类头
            num_classes = checkpoint.get('num_classes', None)
            has_classifier = num_classes is not None and num_classes > 0

            self.trained_model = MobileFaceNet(
                embedding_size=checkpoint['embedding_size'],
                num_classes=num_classes if has_classifier else None
            ).to(self.device)

            self.trained_model.load_state_dict(checkpoint['model_state_dict'])
            self.trained_model.eval()

            # 加载类别信息（优先使用 checkpoint 中的信息）
            if 'idx_to_class' in checkpoint:
                # checkpoint 中的 idx_to_class 使用整数键
                self.trained_model_info = {
                    'idx_to_class': checkpoint['idx_to_class'],
                    'class_to_idx': checkpoint.get('class_to_idx', {}),
                    'num_classes': num_classes,
                    'embedding_size': checkpoint.get('embedding_size', 128)
                }
            elif os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.trained_model_info = json.load(f)

            # 确保数据库中有训练模型中的所有用户
            self._sync_users_with_model()

            print(f"Trained model loaded: {len(self.trained_model_info.get('idx_to_class', {}))} classes")

        except Exception as e:
            print(f"Warning: Failed to load trained model: {e}")
            self.use_trained_model = False

    def _sync_users_with_model(self):
        """同步数据库用户与训练模型类别"""
        idx_to_class = self.trained_model_info.get('idx_to_class', {})

        for _, name in idx_to_class.items():
            # 检查用户是否存在
            existing_user = self.user_manager.db.get_user_by_name(name)
            if not existing_user:
                # 自动添加新用户
                user_id = self.user_manager.db.add_user(name)
                print(f"Auto-added user: {name} (ID: {user_id})")

    def _recognize_with_trained_model(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        使用训练好的模型识别人脸
        :param face_img: 人脸图片 (BGR)
        :return: (姓名, 置信度)
        """
        if self.trained_model is None or self.transform is None:
            return "Unknown", 0.0

        try:
            import torch
            import torch.nn.functional as F

            # 预处理
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                result = self.trained_model(tensor)

            # 获取类别信息
            idx_to_class = self.trained_model_info.get('idx_to_class', {})

            if self.trained_model.classifier is not None:
                # 有分类头，直接输出分类结果
                if isinstance(result, tuple):
                    embeddings, logits = result
                else:
                    logits = result

                probs = F.softmax(logits, dim=1)
                confidence, predicted = probs.max(1)

                class_idx = predicted.item()
                confidence = confidence.item()

                # 获取类别名称（支持整数键和字符串键）
                name = idx_to_class.get(class_idx, idx_to_class.get(str(class_idx), "Unknown"))

                # 训练模型置信度 >0.8 时才直接返回（高置信度才信任）
                # 低置信度回退到 face_recognition，确保正确识别
                if confidence > 0.8:
                    return name, confidence

            # 训练模型置信度不够，使用 face_recognition 辅助判断
            face_encodings = face_recognition.face_encodings(rgb_img)

            if face_encodings:
                is_match, fr_name, fr_confidence, user_id = \
                    self.user_manager.verify_user(face_encodings[0], self.tolerance)

                if is_match:
                    return fr_name, fr_confidence

        except Exception as e:
            print(f"Recognition error: {e}")

        return "Unknown", 0.0

    def load_known_faces(self):
        """加载已知人脸特征"""
        self.user_manager.reload_cache()
        print(f"Loaded {len(self.user_manager._cached_encodings)} face encodings")

    def reload_known_faces(self):
        """重新加载已知人脸特征（用户更新后调用）"""
        self.load_known_faces()

    def _get_pil_font(self, size=24):
        """获取缓存的 PIL 字体"""
        if size == 24 and self._pil_font:
            return self._pil_font
        if size == 16 and self._pil_font_small:
            return self._pil_font_small

        if TORCH_AVAILABLE:
            try:
                font = ImageFont.truetype("msyh.ttc", size)
            except:
                font = ImageFont.load_default()

            if size == 24:
                self._pil_font = font
            else:
                self._pil_font_small = font
            return font
        return None

    def update_frame(self, frame: np.ndarray):
        """更新待处理的帧"""
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def _process_in_thread(self):
        """在后台线程中处理人脸识别"""
        while self.processing:
            current_time = time.time()
            if current_time - self.last_process_time < self.process_interval:
                time.sleep(0.05)
                continue

            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.05)
                    continue
                frame = self.latest_frame.copy()

            # 根据配置选择人脸检测方式
            if self.use_yolo and self.yolo_detector is not None:
                face_locations, face_encodings = self._detect_faces_yolo(frame)
            else:
                face_locations, face_encodings = self._detect_faces_default(frame)

            # Debug: 打印检测结果
            if len(face_locations) > 0:
                print(f"Detected {len(face_locations)} faces")

            status = "Standby"
            user_name = None
            confidence = 0.0
            processed_frame = frame.copy()

            if len(face_locations) > 0:
                for i, face_loc in enumerate(face_locations):
                    # 处理不同格式的人脸位置
                    if self.use_yolo and self.yolo_detector is not None:
                        # YOLO 返回 (x1, y1, x2, y2) 格式
                        x1, y1, x2, y2 = face_loc
                        left, top, right, bottom = x1, y1, x2, y2
                    else:
                        # face_recognition 返回 (top, right, bottom, left) 格式
                        top, right, bottom, left = face_loc
                        left, top, right, bottom = left, top, right, bottom

                    # 识别用户
                    user_id = -1

                    if self.use_trained_model and self.trained_model is not None:
                        # 使用训练模型识别
                        face_img = frame[top:bottom, left:right] if bottom > top and right > left else None
                        if face_img is not None and face_img.size > 0:
                            user_name, confidence = self._recognize_with_trained_model(face_img)
                            if user_name != "Unknown":
                                is_match = True
                                # 从数据库获取用户ID
                                user = self.user_manager.db.get_user_by_name(user_name)
                                if user:
                                    user_id = user['id']
                            else:
                                is_match = False
                        else:
                            is_match = False
                            user_name = "Unknown"
                            confidence = 0.0
                    else:
                        # 使用 face_recognition 识别
                        face_encoding = face_encodings[i] if i < len(face_encodings) else None
                        if face_encoding is not None:
                            is_match, user_name, confidence, user_id = \
                                self.user_manager.verify_user(face_encoding, self.tolerance)
                        else:
                            is_match = False
                            user_name = "Unknown"
                            confidence = 0.0

                    if is_match:
                        status = "Pass"
                        color = (0, 255, 0)  # 绿色
                        text = "Welcome"
                        detail_text = f"{confidence:.2%}"
                    else:
                        status = "Reject"
                        color = (0, 0, 255)  # 红色
                        text = "Unauthorized"
                        detail_text = ""

                    # 绘制边框
                    cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 3)

                    # 绘制标签背景 (主文字)
                    cv2.rectangle(processed_frame, (left, bottom - 50), (right, bottom), color, cv2.FILLED)

                    # 使用 PIL 绘制中文文字
                    if TORCH_AVAILABLE:
                        pil_img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                        pil_draw = ImageDraw.Draw(pil_img)

                        # 使用缓存的字体
                        font = self._get_pil_font(24)

                        pil_draw.text((left + 10, bottom - 45), text, fill=(255, 255, 255), font=font)

                        # 绘制置信度
                        if detail_text:
                            pil_draw.text((left + 10, bottom - 18), detail_text, fill=(255, 255, 255), font=font)

                        processed_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    else:
                        # Fallback: 使用 OpenCV
                        cv2.putText(processed_frame, text, (left + 10, bottom - 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        if detail_text:
                            cv2.putText(processed_frame, detail_text, (left + 10, bottom - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 记录门禁日志
                    if status == "Pass":
                        self.user_manager.db.log_access(
                            user_id, user_name, "pass", confidence, self.gate_name
                        )
                    else:
                        self.user_manager.db.log_access(
                            -1, "Unauthorized", "reject", 0.0, self.gate_name
                        )

                    # 只处理第一个人脸
                    break

            # 更新结果
            self.result_frame = processed_frame
            self.result_status = status
            self.result_user = user_name
            self.result_confidence = confidence
            self.last_process_time = current_time

    def _detect_faces_yolo(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        使用 YOLO 检测人脸并提取编码
        :param frame: 视频帧 (BGR)
        :return: (人脸位置列表, 人脸编码列表)
        """
        # 缩小帧以提高处理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # 使用 YOLO 检测人脸 (传入 BGR 格式)
        detections = self.yolo_detector.detect(small_frame)

        face_locations = []

        for x1, y1, x2, y2, conf in detections:
            # 缩放回原始尺寸 (YOLO 检测的是缩小后的帧)
            x1 *= 2
            y1 *= 2
            x2 *= 2
            y2 *= 2

            face_locations.append((x1, y1, x2, y2))

        # 返回空的 face_encodings（使用训练模型时不需要）
        return face_locations, []

    def _detect_faces_default(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        使用默认的 face_recognition 检测人脸
        :param frame: 视频帧
        :return: (人脸位置列表, 人脸编码列表)
        """
        # 缩小帧以提高处理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.33, fy=0.33)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 检测人脸
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        # 如果使用训练模型，不需要计算 face_encodings
        if self.use_trained_model and self.trained_model is not None:
            # 缩放回原始尺寸
            scaled_locations = [(t*3, r*3, b*3, l*3) for (t, r, b, l) in face_locations]
            return scaled_locations, []

        # 使用 face_recognition 时需要计算编码
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # 缩放回原始尺寸
        scaled_locations = []
        for (top, right, bottom, left) in face_locations:
            scaled_locations.append((top * 3, right * 3, bottom * 3, left * 3))

        return scaled_locations, face_encodings

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
        return "YOLO" if (self.use_yolo and self.yolo_detector) else "face_recognition"

    def get_recognition_method(self) -> str:
        """
        获取当前使用的识别方法
        :return: 识别方法名称
        """
        if self.use_trained_model and self.trained_model is not None:
            classes = self.trained_model_info.get('idx_to_class', {})
            return f"TrainedModel ({len(classes)} classes)"
        return "face_recognition"

    def get_trained_classes(self) -> dict:
        """
        获取训练模型的类别信息
        :return: 类别字典
        """
        return self.trained_model_info.get('idx_to_class', {})

    def start_processing(self):
        """启动后台处理线程"""
        if not self.processing:
            self.processing = True
            self.thread = threading.Thread(target=self._process_in_thread, daemon=True)
            self.thread.start()

    def stop_processing(self):
        """停止后台处理线程"""
        self.processing = False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str, Optional[str], float]:
        """
        处理帧 - 更新最新帧并返回最近的识别结果
        :param frame: 视频帧
        :return: (处理后的帧, 状态, 用户名, 置信度)
        """
        self.update_frame(frame)

        if self.result_frame is not None:
            return self.result_frame, self.result_status, self.result_user, self.result_confidence
        else:
            return frame, "Standby", None, 0.0

    def get_current_status(self) -> Tuple[str, Optional[str], float]:
        """
        获取当前门禁状态
        :return: (状态, 用户名, 置信度)
        """
        return self.current_status, self.current_user, self.confidence

    def get_known_faces_count(self) -> int:
        """
        获取已知人脸数量
        :return: 已知人脸数量
        """
        return len(self.user_manager._cached_encodings)
