#!/usr/bin/env python3
"""
使用 MobileFaceNet 重新生成人脸数据库
"""
import sys
import os
import sqlite3
import numpy as np
import cv2
import struct
from pathlib import Path

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_mobilefacenet(model_path: str):
    """加载 MobileFaceNet ONNX 模型"""
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    return session


def load_detector(model_path: str):
    """加载 YOLOv11-face ONNX 检测器"""
    import onnxruntime as ort
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    return session


def detect_with_yolo(session, frame: np.ndarray, conf_threshold: float = 0.5) -> list:
    """使用 YOLOv11-face 检测人脸"""
    import onnxruntime as ort

    h, w = frame.shape[:2]

    # 预处理
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0

    # 推理
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})

    # 后处理
    output = outputs[0]
    if len(output.shape) == 3:
        output = output[0]

    output = output.T

    detections = []
    for det in output:
        x_center, y_center, box_w, box_h, conf = det[:5]

        if conf > conf_threshold:
            x1 = int((x_center - box_w / 2) * w / 640)
            y1 = int((y_center - box_h / 2) * h / 640)
            x2 = int((x_center + box_w / 2) * w / 640)
            y2 = int((y_center + box_h / 2) * h / 640)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            detections.append((x1, y1, x2, y2))

    return detections


def preprocess_face(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """预处理人脸（与 gate_onnx.py 一致）"""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    # 扩大人脸区域 20%
    mx = int((x2 - x1) * 0.2)
    my = int((y2 - y1) * 0.2)
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)

    # 裁剪并 resize
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (112, 112))

    # 归一化
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    face = face.astype(np.float32) / 255.0
    face = (face - MEAN) / STD

    # HWC -> CHW, 添加 batch 维度
    face = face.transpose(2, 0, 1)
    face = np.expand_dims(face, 0).astype(np.float32)

    return face


def extract_embedding(session, frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """使用 MobileFaceNet 提取人脸特征"""
    input_name = session.get_inputs()[0].name

    # 预处理
    tensor = preprocess_face(frame, bbox)

    # 推理
    outputs = session.run(None, {input_name: tensor})
    embedding = outputs[0].flatten()

    # L2 归一化
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def regenerate_database(db_path: str, output_path: str, model_path: str):
    """使用 MobileFaceNet 重新生成数据库"""
    print(f"=" * 60)
    print(f"Regenerating database with MobileFaceNet")
    print(f"=" * 60)

    # 加载 MobileFaceNet 模型
    print(f"\n[1/3] Loading models...")
    session = load_mobilefacenet(model_path)
    print(f"  OK MobileFaceNet loaded: {model_path}")

    # 加载 YOLOv11-face 检测器（与实时识别一致）
    detector_path = os.path.join(project_root, 'deploy', 'models', 'yolov11l-face.onnx')
    detector = load_detector(detector_path)
    print(f"  OK Detector loaded: {detector_path}")

    # 连接数据库
    print(f"\n[2/3] Loading user data...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询所有活跃用户
    cursor.execute("""
        SELECT u.id, u.name
        FROM users u
        WHERE u.is_active = 1
    """)
    users = cursor.fetchall()
    print(f"  OK Found {len(users)} users")

    # 清除旧编码
    cursor.execute("DELETE FROM face_encodings")
    conn.commit()

    # 为每个用户生成编码
    print(f"\n[3/3] Generating encodings...")
    all_names = []
    all_embeddings = []

    for user_id, user_name in users:
        print(f"\n  Processing: {user_name}")

        # 尝试默认路径
        images_dir = os.path.join(project_root, "data", "faces", f"{user_id}_{user_name}")

        if not os.path.exists(images_dir):
            print(f"    SKIP: Directory not found: {images_dir}")
            continue

        # 获取所有图片
        image_files = [f for f in os.listdir(images_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"    SKIP: No images found")
            continue

        print(f"    Found {len(image_files)} images")

        # 生成编码
        encodings_count = 0
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)

            # 读取图片（支持中文路径）
            data = np.fromfile(img_path, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # 检测人脸（使用 YOLOv11，与实时识别一致）
            faces = detect_with_yolo(detector, frame, conf_threshold=0.5)
            if not faces:
                continue

            # 使用最大的人脸
            face = max(faces, key=lambda f: (f[2]-f[0]) * (f[3]-f[1]))
            x1, y1, x2, y2 = face

            # 提取嵌入
            embedding = extract_embedding(session, frame, (x1, y1, x2, y2))

            # 保存到数据库
            embedding_blob = embedding.astype(np.float32).tobytes()
            cursor.execute(
                "INSERT INTO face_encodings (user_id, encoding, image_path) VALUES (?, ?, ?)",
                (user_id, embedding_blob, img_path)
            )

            all_names.append(user_name)
            all_embeddings.append(embedding)
            encodings_count += 1

        print(f"    Generated {encodings_count} encodings")

    conn.commit()
    conn.close()

    # 导出二进制文件
    print(f"\nExporting to {output_path}...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    with open(output_path, 'wb') as f:
        # 文件头: 版本号, 人数, 维度
        f.write(struct.pack('<III', 1, len(all_names), 128))

        # 写入所有 embedding
        f.write(embeddings_array.tobytes())

        # 写入所有人名
        for name in all_names:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

    print(f"\nDone!")
    print(f"  - Total encodings: {len(all_names)}")
    print(f"  - Output file: {output_path}")
    print(f"  - File size: {os.path.getsize(output_path)} bytes")

    # 打印统计
    from collections import Counter
    name_counts = Counter(all_names)
    print(f"\nEncodings per user:")
    for name, count in name_counts.items():
        print(f"  - {name}: {count}")


if __name__ == '__main__':
    db_path = os.path.join(project_root, 'data', 'db', 'face_access.db')
    output_path = os.path.join(project_root, 'deploy', 'models', 'embeddings_mbn.bin')
    model_path = os.path.join(project_root, 'deploy', 'models', 'mobilefacenet.onnx')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    regenerate_database(db_path, output_path, model_path)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    regenerate_database(db_path, output_path, model_path)
