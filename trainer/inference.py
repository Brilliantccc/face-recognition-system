"""
人脸识别测试工具
使用模块化架构（检测器 + 识别器）进行实时识别测试

用法:
  python trainer/inference.py
  python trainer/inference.py --model-dir trainer/models/saved/optimal_v2
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_recognition(model_dir="trainer/models/saved"):
    """使用模块化架构测试识别功能"""
    from common.recognizers import MobileFaceNetRecognizer
    from common.detectors import DetectorFactory
    from common.user_manager import UserManager
    from common.database import Database
    from common.config import DATABASE_PATH, FACES_DIR

    # 初始化检测器
    detector = DetectorFactory.create("auto")
    print(f"Detector: {type(detector).__name__}")

    # 初始化识别器
    recognizer = MobileFaceNetRecognizer(model_dir=model_dir)
    print(f"Recognizer: MobileFaceNet (device: {recognizer.device})")

    # 加载数据库
    db = Database(DATABASE_PATH)
    user_manager = UserManager(db, FACES_DIR)

    # 尝试加载已有编码
    try:
        from common.databases import EmbeddingsBinDatabase
        bin_db = EmbeddingsBinDatabase(model="mobilenet")
        embeddings = bin_db.load()
        if embeddings:
            recognizer.load_database(embeddings)
            print(f"Loaded {len(embeddings)} users from embeddings.bin (mobilenet)")
    except Exception:
        # fallback: 从 SQLite 加载
        encodings = db.get_face_encodings(model="mobilenet")
        if encodings:
            from collections import defaultdict
            user_encodings = defaultdict(list)
            for user_id, enc in encodings:
                user_encodings[user_id].append(enc)
            users = db.get_all_users()
            user_map = {u['id']: u['name'] for u in users}
            emb_dict = {}
            for user_id, encs in user_encodings.items():
                name = user_map.get(user_id, f"user_{user_id}")
                avg = np.mean(encs, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                emb_dict[name] = avg
            recognizer.load_database(emb_dict)
            print(f"Loaded {len(emb_dict)} users from SQLite")

    # 打开摄像头
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera!")
        return

    print("\n" + "=" * 50)
    print("人脸识别测试 (模块化架构)")
    print("检测器: {} | 识别器: MobileFaceNet".format(type(detector).__name__))
    print("ESC - 退出")
    print("=" * 50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 检测人脸
        detections = detector.detect(frame)

        for det in detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2

            # 裁剪 + margin
            h, w = frame.shape[:2]
            face_w, face_h = x2 - x1, y2 - y1
            mx, my = int(face_w * 0.2), int(face_h * 0.2)
            fx1, fy1 = max(0, x1 - mx), max(0, y1 - my)
            fx2, fy2 = min(w, x2 + mx), min(h, y2 + my)
            face_img = frame[fy1:fy2, fx1:fx2]

            if face_img.size == 0:
                continue

            # 识别
            name, conf = recognizer.recognize(face_img)

            # 绘制
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{name} ({conf:.2%})" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition Test", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition Test")
    parser.add_argument("--model-dir", type=str, default="trainer/models/saved",
                        help="模型目录")
    args = parser.parse_args()

    test_recognition(args.model_dir)
