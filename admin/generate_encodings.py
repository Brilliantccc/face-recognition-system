"""
为已有用户生成人脸编码
使用训练好的 MobileFaceNet 模型，不依赖 dlib/face_recognition
可命令行运行：python admin/generate_encodings.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from common.database import Database
from common.config import DATABASE_PATH, FACES_DIR


def generate_encodings():
    """为所有没有编码的用户生成人脸编码"""
    print("Loading model...")
    try:
        from trainer.inference import FaceRecognizer
        # 使用 optimal_v2 模型
        model_dir = os.path.join(os.path.dirname(__file__), "..", "trainer", "models", "saved", "optimal_v2")
        recognizer = FaceRecognizer(model_dir=model_dir)
        if not recognizer.load_model():
            print("[FAIL] Model load failed")
            return
    except Exception as e:
        print(f"[FAIL] Load inference module failed: {e}")
        return

    print(f"[OK] Model loaded (device: {recognizer.device})")

    db = Database(DATABASE_PATH)
    users = db.get_all_users(active_only=True)

    total_new = 0
    for user in users:
        user_id = user['id']
        name = user['name']

        # 检查是否已有编码
        existing_encodings = db.get_face_encodings(user_id)
        if len(existing_encodings) > 0:
            print(f"  [SKIP] {name} (ID={user_id}) - already has {len(existing_encodings)} encodings")
            continue

        # 查找用户目录
        user_dir = None
        for d in os.listdir(FACES_DIR):
            if d.startswith(f"{user_id}_"):
                user_dir = os.path.join(FACES_DIR, d)
                break

        if user_dir is None or not os.path.exists(user_dir):
            print(f"  [WARN] {name} (ID={user_id}) - no photo dir, skip")
            continue

        # 读取所有照片
        images = []
        for f in sorted(os.listdir(user_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(user_dir, f)
                try:
                    img_array = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        images.append((f, img_path, img))
                except Exception as e:
                    print(f"  [WARN] {name}: {f} read failed - {e}")

        if not images:
            print(f"  [WARN] {name} (ID={user_id}) - no valid images, skip")
            continue

        # 为每张照片生成编码
        saved = 0
        for img_name, img_path, img in images:
            faces = recognizer.detect_faces(img)
            if len(faces) == 0:
                print(f"  [WARN] {name}: {img_name} - no face detected")
                continue

            # 使用最大的人脸
            face_rect = max(faces, key=lambda f: f[2] * f[3])
            face_tensor = recognizer.preprocess_face(img, face_rect)
            embedding = recognizer.get_embedding(face_tensor)

            if embedding is not None:
                db.add_face_encoding(user_id, embedding, img_path)
                saved += 1

        print(f"  [OK] {name} (ID={user_id}) - generated {saved}/{len(images)} encodings")
        total_new += saved

    print(f"\nDone! Generated {total_new} new encodings")


if __name__ == "__main__":
    generate_encodings()
