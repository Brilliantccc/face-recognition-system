"""
MobileFaceNet 识别器
使用训练好的 MobileFaceNet 模型进行人脸识别
"""

import os
import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional

from .base import FaceRecognizer


class MobileFaceNetRecognizer(FaceRecognizer):
    """基于 MobileFaceNet 的人脸识别器"""

    def __init__(self, model_dir: str = None, device: str = None):
        """
        初始化 MobileFaceNet 识别器
        :param model_dir: 模型目录路径
        :param device: 推理设备
        """
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.transform = None
        self._embeddings = {}
        self._threshold = 0.55

        self._load_model()

    def _load_model(self):
        """加载 MobileFaceNet 模型"""
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from common import torch_utils
        self.torch = torch_utils.torch
        self.TORCH_AVAILABLE = torch_utils.TORCH_AVAILABLE

        if not self.TORCH_AVAILABLE:
            raise ImportError("PyTorch 不可用")

        from trainer.models.facenet import MobileFaceNet

        # 自动查找最新模型
        if self.model_dir is None:
            self.model_dir = self._find_latest_model()

        if self.model_dir is None:
            raise FileNotFoundError("没有找到训练好的模型")

        # 设置设备
        if self.device is None:
            self.device = "cuda" if self.torch.cuda.is_available() else "cpu"

        # 设置变换
        self.transform = torch_utils.transforms.Compose([
            torch_utils.transforms.Resize((112, 112)),
            torch_utils.transforms.ToTensor(),
            torch_utils.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 加载模型
        inference_path = os.path.join(self.model_dir, "inference_model.pth")
        best_path = os.path.join(self.model_dir, "best_model.pth")
        model_file = inference_path if os.path.exists(inference_path) else best_path

        checkpoint = torch_utils.load_model_checkpoint(model_file, map_location=self.device)

        self.model = MobileFaceNet(
            embedding_size=checkpoint.get('embedding_size', 128),
            num_classes=None
        ).to(self.device)

        # 只加载模型权重
        state_dict = checkpoint['model_state_dict']
        filtered = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}
        self.model.load_state_dict(filtered, strict=False)
        self.model.eval()

        # 加载阈值
        info_path = os.path.join(self.model_dir, "model_info.json")
        if os.path.exists(info_path):
            import json
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
                self._threshold = info.get('best_threshold', 0.55)

        self._torch_utils = torch_utils

    def _find_latest_model(self) -> Optional[str]:
        """查找最新训练的模型目录"""
        models_root = os.path.join(
            os.path.dirname(__file__), "..", "..", "trainer", "models", "saved"
        )
        if not os.path.exists(models_root):
            return None

        version_dirs = sorted(
            [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))],
            key=lambda d: os.path.getmtime(os.path.join(models_root, d)),
            reverse=True
        )
        for vdir in version_dirs:
            candidate = os.path.join(models_root, vdir, "inference_model.pth")
            if os.path.exists(candidate):
                return os.path.join(models_root, vdir)

        return None

    def recognize(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        识别人脸
        :param face_image: 人脸图片 (BGR, 已裁剪)
        :return: (姓名, 置信度)
        """
        if self.model is None or not self._embeddings:
            return "Unknown", 0.0

        embedding = self.get_embedding(face_image)
        if embedding is None:
            return "Unknown", 0.0

        # 与所有注册用户比较
        import torch.nn.functional as F

        best_name = "Unknown"
        best_sim = -1.0

        for name, reg_emb in self._embeddings.items():
            sim = F.cosine_similarity(
                self.torch.from_numpy(embedding).float().unsqueeze(0),
                reg_emb.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim < self._threshold:
            return "Unknown", 0.0

        return best_name, best_sim

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        获取人脸 embedding
        :param face_image: 人脸图片 (BGR, 已裁剪)
        :return: embedding 向量 (numpy)
        """
        if self.model is None:
            return None

        try:
            # 预处理
            rgb_img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_img = self._torch_utils.Image.fromarray(rgb_img)
            tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

            # 提取 embedding
            with self.torch.no_grad():
                embedding = self.model(tensor)

            return embedding.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"MobileFaceNet embedding error: {e}")
            return None

    def load_database(self, embeddings: Dict[str, np.ndarray]):
        """加载人脸数据库"""
        import torch
        self._embeddings = {}
        for name, emb in embeddings.items():
            if isinstance(emb, np.ndarray):
                self._embeddings[name] = torch.from_numpy(emb).float()
            else:
                self._embeddings[name] = emb.float()
