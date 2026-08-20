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

import cv2
import numpy as np
import face_recognition
import threading
import time
from typing import Tuple, Optional, List

from common.user_manager import UserManager
from common.config import FACE_RECOGNITION_TOLERANCE, USE_YOLO_DETECTION, YOLO_MODEL_SIZE, YOLO_CONFIDENCE, FACE_INPUT_SIZE, STRICT_REGISTRATION_ONLY
from common import torch_utils

# 门禁控制参数
DOOR_OPEN_DURATION = 5.0       # 开门持续时间（秒）
DOOR_COOLDOWN = 3.0            # 关门后冷却时间（秒），防止重复触发

# 从 torch_utils 获取 PyTorch 组件
torch = torch_utils.torch
TORCH_AVAILABLE = torch_utils.TORCH_AVAILABLE
MobileFaceNet = None

if TORCH_AVAILABLE:
    try:
        from trainer.models.facenet import MobileFaceNet
    except Exception as e:
        print(f"Warning: Failed to import MobileFaceNet: {e}")
        TORCH_AVAILABLE = False


def imread_safe(filepath):
    """读取图片，支持中文路径（Windows）"""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


class AccessControl:
    def __init__(self, user_manager: UserManager, tolerance: float = None, gate_name: str = "Main Gate",
                 use_yolo: bool = None, yolo_model_size: str = None, yolo_confidence: float = None,
                 use_trained_model: bool = False):
        """
        初始化门禁控制器
        :param user_manager: 用户管理器实例
        :param tolerance: 人脸识别容差阈值
        :param gate_name: 门禁点名称
        :param use_yolo: 是否使用 YOLO 进行人脸检测
        :param yolo_model_size: YOLO 模型大小
        :param yolo_confidence: YOLO 置信度阈值
        :param use_trained_model: 是否使用训练好的模型进行识别（默认 False，使用 face_recognition）
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
        self.trained_model_dir = None
        self.device = None
        self.transform = None

        # 设置设备和变换
        if self.use_trained_model and TORCH_AVAILABLE:
            self.device = torch_utils.get_device(prefer_gpu=True)
            self.transform = torch_utils.transforms.Compose([
                torch_utils.transforms.Resize((FACE_INPUT_SIZE, FACE_INPUT_SIZE)),
                torch_utils.transforms.ToTensor(),
                torch_utils.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

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
        self.model_lock = threading.Lock()  # 模型加载/切换锁
        self.last_process_time = 0
        self.process_interval = 0.3  # 处理间隔

        # 缓存 PIL 字体（避免每次加载）
        self._pil_font = None
        self._pil_font_small = None

        # 注册用户 embedding 缓存
        self.registered_embeddings = {}  # {name: embedding_tensor}

        # 门禁开关状态
        self.door_open = False
        self.door_open_time = 0.0        # 开门时刻
        self.door_cooldown_until = 0.0   # 冷却期截止时刻
        self.on_door_open = None         # 回调：开门时触发
        self.on_door_close = None        # 回调：关门时触发

        # 加载已知人脸特征
        self.load_known_faces()

        # 加载训练模型（只调用一次）
        if self.use_trained_model:
            self._load_trained_model()

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
            print(f"YOLO detector initialized (model: yolov8{model_size}, confidence: {conf}, device: {device})")
        except ImportError as e:
            print(f"Warning: YOLO not available, falling back to face_recognition: {e}")
            self.use_yolo = False
        except Exception as e:
            print(f"Warning: YOLO initialization failed, falling back to face_recognition: {e}")
            self.use_yolo = False

    def _load_trained_model(self):
        """加载训练好的 MobileFaceNet 模型（自动查找最新模型）"""
        if not TORCH_AVAILABLE or MobileFaceNet is None:
            self.use_trained_model = False
            return

        models_dir = "trainer/models/saved"
        model_path = None

        # 优先查找带版本号的目录中的 inference_model.pth
        if os.path.exists(models_dir):
            version_dirs = sorted(
                [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))],
                key=lambda d: os.path.getmtime(os.path.join(models_dir, d)),
                reverse=True
            )
            for vdir in version_dirs:
                candidate = os.path.join(models_dir, vdir, "inference_model.pth")
                if os.path.exists(candidate):
                    model_path = os.path.join(models_dir, vdir)
                    break

        if model_path is None:
            print("Warning: No trained model found")
            self.use_trained_model = False
            return

        self.load_model_from_dir(model_path)

    def _sync_users_with_model(self):
        """
        [已禁用] 同步数据库用户与训练模型类别
        为确保门禁安全，不再自动将模型类别用户添加到数据库。
        所有用户必须通过管理员系统手动注册。
        """
        # 安全模式下不自动同步，防止非授权用户被添加
        if STRICT_REGISTRATION_ONLY:
            print("Strict mode: auto-sync disabled. Users must be registered via admin.")
            return

        idx_to_class = self.trained_model_info.get('idx_to_class', {})
        for _, name in idx_to_class.items():
            existing_user = self.user_manager.db.get_user_by_name(name)
            if not existing_user:
                user_id = self.user_manager.db.add_user(name)
                print(f"Auto-added user: {name} (ID: {user_id})")

    def get_available_models(self) -> list:
        """获取所有可用的模型列表"""
        models_dir = "trainer/models/saved"
        available = []

        # 始终添加 face_recognition 选项（兜底方案）
        available.append({
            'name': 'face_recognition',
            'path': 'face_recognition',
            'num_classes': None,
            'is_builtin': True
        })

        if not os.path.exists(models_dir):
            return available
        for name in sorted(os.listdir(models_dir)):
            model_dir = os.path.join(models_dir, name)
            if os.path.isdir(model_dir):
                has_inference = os.path.exists(os.path.join(model_dir, "inference_model.pth"))
                has_best = os.path.exists(os.path.join(model_dir, "best_model.pth"))
                if has_inference or has_best:
                    info_path = os.path.join(model_dir, "model_info.json")
                    num_classes = None
                    if os.path.exists(info_path):
                        import json
                        with open(info_path, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            num_classes = info.get('num_classes')
                    available.append({
                        'name': name,
                        'path': model_dir,
                        'num_classes': num_classes,
                        'is_builtin': False
                    })
        return available

    def load_model_from_dir(self, model_dir: str) -> bool:
        """从指定目录加载模型（线程安全）"""
        import json

        with self.model_lock:
            return self._load_model_from_dir_impl(model_dir)

    def _load_model_from_dir_impl(self, model_dir: str) -> bool:
        """从指定目录加载模型（内部实现）"""
        import json

        inference_path = os.path.join(model_dir, "inference_model.pth")
        best_path = os.path.join(model_dir, "best_model.pth")
        info_path = os.path.join(model_dir, "model_info.json")

        # inference_model.pth 没有分类头，适合 embedding 匹配
        model_file = inference_path if os.path.exists(inference_path) else best_path
        if not os.path.exists(model_file):
            print(f"Warning: No model file in {model_dir}")
            return False

        try:
            checkpoint = torch_utils.load_model_checkpoint(model_file, map_location=self.device)

            # 始终用无分类头的模型，通过 embedding 余弦相似度识别
            self.trained_model = MobileFaceNet(
                embedding_size=checkpoint.get('embedding_size', 128),
                num_classes=None  # 不加载分类头
            ).to(self.device)

            # 只加载模型权重（过滤掉可能存在的 classifier 权重）
            state_dict = checkpoint['model_state_dict']
            filtered = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
            self.trained_model.load_state_dict(filtered, strict=False)
            self.trained_model.eval()

            # 加载类别信息
            if 'idx_to_class' in checkpoint:
                self.trained_model_info = {
                    'idx_to_class': checkpoint['idx_to_class'],
                    'class_to_idx': checkpoint.get('class_to_idx', {}),
                    'num_classes': checkpoint.get('num_classes', len(checkpoint['idx_to_class'])),
                    'embedding_size': checkpoint.get('embedding_size', 128)
                }
            elif os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.trained_model_info = json.load(f)

            # 加载训练时的最佳阈值（如果没有配置，默认使用 0.55）
            self.trained_threshold = self.trained_model_info.get('best_threshold', 0.55)
            self.trained_val_acc = self.trained_model_info.get('best_val_acc', None)

            self.trained_model_dir = model_dir
            self.use_trained_model = True
            self._sync_users_with_model()

            classes = len(self.trained_model_info.get('idx_to_class', {}))
            print(f"Model loaded from {model_dir}: {classes} classes")

            # 构建注册用户 embedding 缓存
            self.build_registered_embeddings()
            return True

        except Exception as e:
            print(f"Warning: Failed to load model from {model_dir}: {e}")
            return False

    def _recognize_with_trained_model(self, face_img: np.ndarray) -> Tuple[str, float]:
        """
        使用训练好的模型识别人脸（只匹配数据库中已注册的激活用户）
        增加二次验证：当 top1 和 top2 相似度差距太小时，拒绝识别（防止相似用户误识别）
        :param face_img: 人脸图片 (BGR)
        :return: (姓名, 置信度 0~100%)
        """
        # 加锁快照，防止模型切换时读到半更新状态
        with self.model_lock:
            model = self.trained_model
            transform = self.transform
            embeddings = dict(self.registered_embeddings)
            threshold = getattr(self, 'trained_threshold', 0.55)
            device = self.device

        if model is None or transform is None:
            return "Unknown", 0.0

        if not embeddings:
            return "Unknown", 0.0

        try:
            import torch.nn.functional as F

            # 预处理
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            pil_img = torch_utils.Image.fromarray(rgb_img)
            tensor = transform(pil_img).unsqueeze(0).to(device)

            # 提取 embedding
            with torch.no_grad():
                embedding = model(tensor)  # [1, 128]
                embedding = F.normalize(embedding, p=2, dim=1)

            # 跟所有注册用户的 embedding 比较余弦相似度
            sims = []
            for name, reg_emb in embeddings.items():
                sim = F.cosine_similarity(embedding, reg_emb.unsqueeze(0)).item()
                sims.append((name, sim))

            # 按相似度降序排序
            sims.sort(key=lambda x: x[1], reverse=True)

            best_name = sims[0][0] if sims else "Unknown"
            best_sim = sims[0][1] if sims else -1.0

            # 相似度低于阈值 → 未识别
            if best_sim < threshold:
                return "Unknown", 0.0

            # 二次验证：top1 和 top2 差距检查
            # 如果两个最相似的用户差距太小，说明模型无法区分，拒绝识别
            MARGIN_THRESHOLD = 0.10  # 最小差距要求
            if len(sims) >= 2:
                second_sim = sims[1][1]
                margin = best_sim - second_sim
                if margin < MARGIN_THRESHOLD:
                    print(f"Ambiguous: '{best_name}'({best_sim:.3f}) vs '{sims[1][0]}'({second_sim:.3f}), "
                          f"margin={margin:.3f} < {MARGIN_THRESHOLD}, rejected")
                    return "Unknown", 0.0

            # 严格模式：验证用户是否在数据库中且已激活
            if STRICT_REGISTRATION_ONLY:
                db_user = self.user_manager.db.get_user_by_name(best_name)
                if not db_user or not db_user.get('is_active', True):
                    print(f"Rejected: '{best_name}' not in active DB users (sim={best_sim:.3f})")
                    return "Unknown", 0.0

            # 置信度 = 相似度直接作为置信度（已超过阈值）
            confidence = best_sim
            return best_name, confidence

        except Exception as e:
            print(f"Recognition error: {e}")

        return "Unknown", 0.0

    def build_registered_embeddings(self):
        """为所有注册用户的照片提取 embedding 缓存"""
        if self.trained_model is None or self.transform is None:
            return

        self.registered_embeddings = {}
        faces_dir = "data/faces"

        if not os.path.exists(faces_dir):
            print("Warning: data/faces/ not found, no registered users to cache")
            return

        import torch.nn.functional as F

        for user_dir_name in os.listdir(faces_dir):
            user_dir = os.path.join(faces_dir, user_dir_name)
            if not os.path.isdir(user_dir):
                continue

            # 从目录名提取用户名（格式：ID_姓名 或 姓名）
            name = user_dir_name
            if "_" in user_dir_name:
                name = user_dir_name.split("_", 1)[1]

            # 收集该用户所有图片的 embedding
            embeddings = []
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            for img_name in os.listdir(user_dir):
                if os.path.splitext(img_name)[1].lower() not in valid_exts:
                    continue
                img_path = os.path.join(user_dir, img_name)
                try:
                    img = imread_safe(img_path)
                    if img is None:
                        continue
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = torch_utils.Image.fromarray(rgb_img)
                    tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        emb = self.trained_model(tensor)
                    embeddings.append(emb.squeeze(0))
                except Exception:
                    continue

            if embeddings:
                # 取所有图片 embedding 的平均值，作为该用户的代表向量
                avg_emb = torch.stack(embeddings).mean(dim=0)
                avg_emb = F.normalize(avg_emb, p=2, dim=0)
                self.registered_embeddings[name] = avg_emb

        print(f"Cached embeddings for {len(self.registered_embeddings)} registered users")

    def load_known_faces(self):
        """加载已知人脸特征"""
        self.user_manager.reload_cache()
        print(f"Loaded {len(self.user_manager._cached_encodings)} face encodings")
        # 重建注册用户 embedding 缓存
        if self.use_trained_model and self.trained_model is not None:
            self.build_registered_embeddings()

    def reload_known_faces(self):
        """重新加载已知人脸特征（用户更新后调用）"""
        self.load_known_faces()

    def _get_pil_font(self, size=24):
        """获取缓存的 PIL 字体"""
        if size == 24 and self._pil_font:
            return self._pil_font
        if size == 16 and self._pil_font_small:
            return self._pil_font_small

        if TORCH_AVAILABLE and torch_utils.Image is not None:
            try:
                from PIL import ImageFont
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

            # 门禁开关状态管理
            self._update_door_state(current_time)

            # 门已打开或在冷却期，跳过识别（节省算力）
            if self.door_open or current_time < self.door_cooldown_until:
                with self.frame_lock:
                    if self.latest_frame is None:
                        time.sleep(0.05)
                        continue
                    # 只更新画面，不做人脸识别
                    self.result_frame = self.latest_frame.copy()
                    self.result_status = "Pass" if self.door_open else "Standby"
                    self.result_user = self.current_user
                    self.result_confidence = 0.0
                self.last_process_time = current_time
                time.sleep(0.1)
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
                        # 触发开门
                        self._open_door(user_name)
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
                    if TORCH_AVAILABLE and torch_utils.Image is not None:
                        from PIL import ImageDraw
                        pil_img = torch_utils.Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
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

    def _open_door(self, user_name: str):
        """开门"""
        now = time.time()
        if self.door_open or now < self.door_cooldown_until:
            return  # 已开门或在冷却期，不重复触发

        self.door_open = True
        self.door_open_time = now
        self.current_user = user_name
        print(f"Door OPEN - {user_name} (auto-close in {DOOR_OPEN_DURATION}s)")

        # 触发开门回调（可连接 GPIO/继电器等）
        if self.on_door_open:
            try:
                self.on_door_open(user_name)
            except Exception as e:
                print(f"Door open callback error: {e}")

    def _update_door_state(self, current_time: float):
        """检查是否需要自动关门"""
        if not self.door_open:
            return

        elapsed = current_time - self.door_open_time
        if elapsed >= DOOR_OPEN_DURATION:
            self.door_open = False
            self.door_cooldown_until = current_time + DOOR_COOLDOWN
            print(f"Door CLOSE (was open for {elapsed:.1f}s, cooldown {DOOR_COOLDOWN}s)")

            # 触发关门回调
            if self.on_door_close:
                try:
                    self.on_door_close()
                except Exception as e:
                    print(f"Door close callback error: {e}")

    def force_open_door(self, duration: float = None):
        """手动强制开门（管理员操作）"""
        dur = duration or DOOR_OPEN_DURATION
        now = time.time()
        self.door_open = True
        self.door_open_time = now
        print(f"Door FORCE OPEN for {dur}s")
        if self.on_door_open:
            try:
                self.on_door_open("Admin")
            except Exception as e:
                print(f"Door open callback error: {e}")

    def _detect_faces_yolo(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        使用 YOLO 检测人脸，再用 face_recognition 提取编码
        :param frame: 视频帧 (BGR)
        :return: (人脸位置列表, 人脸编码列表)
        """
        # 缩小帧以提高处理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # 使用 YOLO 检测人脸
        detections = self.yolo_detector.detect(small_frame)

        face_locations = []
        face_encodings = []

        if len(detections) == 0:
            return face_locations, face_encodings

        # 转换为 RGB 用于 face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]

        # 将 YOLO 检测结果转换为 face_recognition 格式 (top, right, bottom, left)
        fr_locations = []
        for x1, y1, x2, y2, conf in detections:
            # 添加边距
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)

            # 转换为 face_recognition 格式并缩放回原始尺寸
            top = max(0, int((y1 * 2) - margin_y))
            right = min(w, int((x2 * 2) + margin_x))
            bottom = min(h, int((y2 * 2) + margin_y))
            left = max(0, int((x1 * 2) - margin_x))

            fr_locations.append((top, right, bottom, left))
            face_locations.append((x1 * 2, y1 * 2, x2 * 2, y2 * 2))  # 保存原始 YOLO 格式

        # 使用 face_recognition 在完整帧上提取编码（传入人脸位置）
        try:
            face_encodings = face_recognition.face_encodings(rgb_frame, fr_locations)
        except Exception as e:
            print(f"Warning: face_encodings failed: {e}")
            face_encodings = [None] * len(fr_locations)

        return face_locations, face_encodings

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

    def get_active_users_count(self) -> int:
        """获取数据库中已激活的注册用户数量"""
        users = self.user_manager.get_all_users(active_only=True)
        return len(users)

    def is_strict_mode(self) -> bool:
        """是否为严格模式（只允许数据库注册用户通过）"""
        return STRICT_REGISTRATION_ONLY
