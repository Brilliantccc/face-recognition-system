"""
InsightFace ArcFace 识别器
使用 InsightFace buffalo_l 预训练模型（ArcFace-R100, 512维embedding）
与 MobileFaceNet 识别器接口一致，支持对比实验

模型下载：
  从 https://github.com/deepinsight/insightface/tree/master/python-package 下载 buffalo_l 包
  解压后将以下文件放入 deploy/models/buffalo_l/:
    - w600k_r100.onnx  (ArcFace-R100, 推荐, ~250MB)
    或
    - w600k_r50.onnx   (ArcFace-R50, 更快, ~85MB)

用法：
  # 门禁系统自动选择（配置 RECOGNITION_BACKEND="insightface"）
  # 或手动指定：
  from common.recognizers import InsightFaceRecognizer
  recognizer = InsightFaceRecognizer()
"""

import os
import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional

from .base import FaceRecognizer


class InsightFaceRecognizer(FaceRecognizer):
    """基于 InsightFace ArcFace 的人脸识别器"""

    # 预处理参数（与 MobileFaceNet / ONNX 版本一致）
    INPUT_SIZE = 112
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_dir: str = None, device: str = None):
        """
        初始化 InsightFace 识别器
        :param model_dir: buffalo_l 模型目录路径
        :param device: 推理设备 (None=自动选择)
        """
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.session = None
        self._embeddings = {}
        self._threshold = 0.45  # ArcFace-R100 在 512 维空间中的默认阈值
        self.embedding_size = 512

        self._load_model()

    def _find_model_dir(self) -> Optional[str]:
        """自动查找 buffalo_l 模型目录"""
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "models", "buffalo_l"),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "buffalo_l"),
            os.path.join(os.path.dirname(__file__), "..", "..", "deploy", "models", "buffalo"),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return os.path.abspath(path)
        return None

    def _load_model(self):
        """加载 InsightFace ArcFace ONNX 模型"""
        import onnxruntime as ort

        # 自动查找模型目录
        if self.model_dir is None:
            self.model_dir = self._find_model_dir()

        if self.model_dir is None:
            raise FileNotFoundError(
                "未找到 buffalo_l 模型目录，请下载后放入 deploy/models/buffalo_l/\n"
                "下载地址: https://github.com/deepinsight/insightface"
            )

        # 优先 R100（精度高），退而用 R50（速度快）
        model_candidates = [
            ("w600k_r100.onnx", "ArcFace-R100 (512-dim)"),
            ("w600k_r50.onnx",  "ArcFace-R50 (512-dim)"),
        ]

        model_path = None
        model_name = None
        for filename, desc in model_candidates:
            candidate = os.path.join(self.model_dir, filename)
            if os.path.exists(candidate):
                model_path = candidate
                model_name = desc
                break

        if model_path is None:
            available = os.listdir(self.model_dir) if os.path.isdir(self.model_dir) else []
            raise FileNotFoundError(
                f"在 {self.model_dir} 中未找到 w600k_r100.onnx 或 w600k_r50.onnx\n"
                f"目录内容: {available}"
            )

        # 推理设备选择（与 torch_utils 逻辑对齐）
        if self.device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device == "cpu":
            providers = ['CPUExecutionProvider']
        else:
            # 自动选择
            try:
                import torch
                if torch.cuda.is_available():
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                else:
                    providers = ['CPUExecutionProvider']
            except ImportError:
                providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        active_provider = self.session.get_providers()[0]
        print(f"[InsightFace] {model_name} loaded (provider: {active_provider})")

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        预处理人脸图片（与 MobileFaceNet / ONNX 门禁完全一致）
        BGR → RGB → Resize(112×112) → /255 → ImageNet Normalize → CHW → batch
        """
        # BGR → RGB
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        # Resize
        rgb = cv2.resize(rgb, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        # float32, 归一化
        blob = rgb.astype(np.float32) / 255.0
        blob = (blob - self.MEAN) / self.STD
        # HWC → CHW, 添加 batch 维度
        blob = blob.transpose(2, 0, 1)
        blob = np.expand_dims(blob, 0).astype(np.float32)
        return blob

    def recognize(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        识别人脸
        :param face_image: 人脸图片 (BGR, 已裁剪)
        :return: (姓名, 置信度)
        """
        if self.session is None or not self._embeddings:
            return "Unknown", 0.0

        embedding = self.get_embedding(face_image)
        if embedding is None:
            return "Unknown", 0.0

        # 与所有注册用户比较（余弦相似度，embedding 已 L2 归一化）
        best_name = "Unknown"
        best_sim = -1.0

        for name, reg_emb in self._embeddings.items():
            sim = float(np.dot(embedding, reg_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < self._threshold:
            return "Unknown", 0.0

        return best_name, best_sim

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        提取人脸 512 维 embedding
        :param face_image: 人脸图片 (BGR, 已裁剪)
        :return: 512 维 L2 归一化 embedding 向量 (numpy)
        """
        if self.session is None:
            return None

        try:
            blob = self._preprocess(face_image)
            outputs = self.session.run(None, {self.input_name: blob})
            embedding = outputs[0].flatten()

            # InsightFace 模型输出已 L2 归一化，这里做安全检查
            norm = np.linalg.norm(embedding)
            if norm > 0 and abs(norm - 1.0) > 0.01:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            print(f"[InsightFace] embedding error: {e}")
            return None

    def load_database(self, embeddings: Dict[str, np.ndarray]):
        """
        加载人脸数据库（兼容基类接口）
        自动处理 128 维 (MobileFaceNet) 和 512 维 (InsightFace) 的差异
        :param embeddings: {姓名: embedding} 字典
        """
        self._embeddings = {}
        for name, emb in embeddings.items():
            if isinstance(emb, np.ndarray):
                emb_tensor = emb.astype(np.float32)
            else:
                emb_tensor = np.array(emb, dtype=np.float32)

            # 维度检查
            if emb_tensor.shape[0] != self.embedding_size:
                print(f"[InsightFace] 跳过 {name}: embedding 维度 {emb_tensor.shape[0]} != {self.embedding_size}")
                continue

            # 确保 L2 归一化
            norm = np.linalg.norm(emb_tensor)
            if norm > 0:
                emb_tensor = emb_tensor / norm

            self._embeddings[name] = emb_tensor

        print(f"[InsightFace] loaded {len(self._embeddings)} users (dim={self.embedding_size})")
