"""
人脸识别门禁系统 - ONNX Runtime GPU 加速版
性能：延迟 < 15ms/帧，FPS ≥ 60
"""
import cv2
import numpy as np
import onnxruntime as ort
import sqlite3
import time
import struct
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import sys
import io

# 修复 Windows 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class FaceDetector:
    """YOLO 人脸检测器 (ONNX Runtime)"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        self.conf_threshold = conf_threshold

        # 初始化 ONNX Runtime 会话（GPU 加速）
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 640, 640]

        print(f"✅ 检测模型加载成功 (Provider: {self.session.get_providers()[0]})")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        检测人脸
        返回: [(x1, y1, x2, y2, confidence), ...]
        """
        h, w = frame.shape[:2]

        # 预处理
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0

        # 推理
        outputs = self.session.run(None, {self.input_name: img})

        # 后处理
        detections = []
        output = outputs[0]  # [1, num_classes + 4 + 1, num_detections]

        # YOLOv8 输出格式: [x_center, y_center, w, h, class_scores...]
        if len(output.shape) == 3:
            output = output[0]  # [num_classes + 4 + 1, num_detections]

        # 转置为 [num_detections, num_classes + 4 + 1]
        output = output.T

        for det in output:
            # 解析检测结果
            x_center, y_center, box_w, box_h = det[:4]
            class_scores = det[4:-1]  # 去掉最后的 objectness score
            obj_conf = det[-1]

            # 找到最高置信度的类别
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]

            # 只保留人脸类别 (class_id == 0 for face)
            if class_id == 0 and obj_conf * class_conf > self.conf_threshold:
                # 转换为角点坐标并缩放回原图尺寸
                x1 = int((x_center - box_w / 2) * w / 640)
                y1 = int((y_center - box_h / 2) * h / 640)
                x2 = int((x_center + box_w / 2) * w / 640)
                y2 = int((y_center + box_h / 2) * h / 640)

                # 边界检查
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                detections.append((x1, y1, x2, y2, float(obj_conf * class_conf)))

        # NMS
        return self._nms(detections, 0.4)

    def _nms(self, detections: List, threshold: float) -> List:
        """非极大值抑制"""
        if not detections:
            return []

        # 按置信度排序
        detections.sort(key=lambda x: x[4], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            # 移除重叠框
            detections = [
                det for det in detections
                if self._iou(best, det) < threshold
            ]

        return keep

    def _iou(self, box1, box2) -> float:
        """计算 IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0


class FaceRecognizer:
    """MobileFaceNet 人脸识别器 (ONNX Runtime GPU)"""

    # 预处理参数（必须与训练一致）
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    INPUT_SIZE = 112

    def __init__(self, model_path: str):
        # 初始化 ONNX Runtime 会话（GPU 加速）
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        print(f"✅ 识别模型加载成功 (Provider: {self.session.get_providers()[0]})")

    def preprocess(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """预处理（必须与训练一致）"""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # 1. 扩大人脸区域 20%
        mx = int((x2 - x1) * 0.2)
        my = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        # 2. 裁剪并 resize
        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (self.INPUT_SIZE, self.INPUT_SIZE))

        # 3. 归一化
        face = face.astype(np.float32) / 255.0
        face = (face - self.MEAN) / self.STD

        # 4. HWC -> CHW, 添加 batch 维度
        face = face.transpose(2, 0, 1)
        face = np.expand_dims(face, 0).astype(np.float32)

        return face

    def extract_embedding(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """提取人脸特征向量"""
        tensor = self.preprocess(frame, bbox)
        outputs = self.session.run(None, {self.input_name: tensor})
        embedding = outputs[0].flatten()

        # L2 归一化
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class FaceDatabase:
    """人脸数据库"""

    def __init__(self, db_path: str):
        self.embeddings = []
        self.names = []
        self._load_database(db_path)

    def _load_database(self, path: str):
        """加载二进制格式的人脸数据库"""
        if not Path(path).exists():
            print(f"⚠️ 数据库文件不存在: {path}")
            return

        with open(path, 'rb') as f:
            # 读取头部：版本号, 人数, 维度
            header = struct.unpack('<III', f.read(12))
            version, num_persons, embedding_dim = header

            print(f"📊 数据库: {num_persons} 人, 维度 {embedding_dim}")

            # 读取所有特征向量
            embeddings_data = f.read(num_persons * embedding_dim * 4)
            embeddings = np.frombuffer(embeddings_data, dtype=np.float32).reshape(num_persons, embedding_dim)

            # 读取姓名
            for i in range(num_persons):
                name_len = struct.unpack('<I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                self.names.append(name)
                self.embeddings.append(embeddings[i])

        print(f"✅ 加载了 {len(self.names)} 个人脸数据")

    def search(self, embedding: np.ndarray, threshold: float = 0.6) -> Tuple[Optional[str], float]:
        """搜索最匹配的人脸"""
        if not self.embeddings:
            return None, 0.0

        # 计算余弦相似度
        similarities = []
        for db_emb in self.embeddings:
            sim = np.dot(embedding, db_emb)
            similarities.append(sim)

        similarities = np.array(similarities)
        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        if best_sim >= threshold:
            return self.names[best_idx], float(best_sim)
        return None, float(best_sim)


class GateSystem:
    """门禁系统主类"""

    def __init__(self, config_path: str):
        # 加载配置
        config_path = Path(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 转换为绝对路径（相对于 deploy 目录）
        deploy_dir = config_path.parent.parent  # config -> deploy
        for key in ['detector_model', 'recognizer_model', 'database_path']:
            if key in self.config:
                self.config[key] = str(deploy_dir / self.config[key])

        print("=" * 50)
        print("    人脸识别门禁系统 v2.0 (ONNX GPU)")
        print("=" * 50)

        # 初始化组件
        print("\n[1/3] Loading models...")
        self.detector = FaceDetector(
            self.config['detector_model'],
            conf_threshold=0.5
        )
        self.recognizer = FaceRecognizer(self.config['recognizer_model'])

        print("\n[2/3] Loading database...")
        self.database = FaceDatabase(self.config['database_path'])

        self.threshold = self.config.get('recognition_threshold', 0.6)

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        """处理单帧图像"""
        results = []

        # 检测人脸
        detections = self.detector.detect(frame)

        for x1, y1, x2, y2, conf in detections:
            bbox = (x1, y1, x2, y2)

            # 提取特征并识别
            embedding = self.recognizer.extract_embedding(frame, bbox)
            name, similarity = self.database.search(embedding, self.threshold)

            results.append({
                'bbox': (x1, y1, x2, y2),
                'is_known': name is not None,
                'name': name or 'Stranger',
                'similarity': similarity,
                'detection_conf': conf
            })

        return results

    def run(self):
        """运行门禁系统"""
        print("\n[3/3] Starting recognition loop...")
        print("Press ESC to exit")
        print("=" * 50)

        # 打开摄像头
        cap = cv2.VideoCapture(self.config.get('camera_id', 0))
        if not cap.isOpened():
            print("❌ Failed to open camera!")
            return

        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.get('input_width', 640))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.get('input_height', 480))

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            t0 = time.time()
            results = self.process_frame(frame)
            inference_ms = (time.time() - t0) * 1000

            # 绘制结果
            for r in results:
                x1, y1, x2, y2 = r['bbox']

                # 边界框颜色
                color = (0, 255, 0) if r['is_known'] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 标签
                if r['is_known']:
                    label = f"{r['name']} ({int(r['similarity']*100)}%)"
                    # 门禁控制：识别成功
                    print(f"🔓 ACCESS GRANTED: {r['name']} (sim: {r['similarity']:.2f})")
                else:
                    label = "Stranger"

                # 绘制标签背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+5, y1), color, -1)
                cv2.putText(frame, label, (x1+2, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # 计算 FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # 显示 FPS 和推理时间
            info = f"FPS: {int(fps)} | Inference: {inference_ms:.1f}ms"
            cv2.putText(frame, info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Recognition Gate System", frame)

            # ESC 退出
            if cv2.waitKey(1) == 27:
                break

        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*50}")
        print(f"Session ended. Processed {frame_count} frames.")
        print(f"{'='*50}")


if __name__ == '__main__':
    config_path = Path(__file__).parent / 'config' / 'config.json'
    system = GateSystem(str(config_path))
    system.run()
