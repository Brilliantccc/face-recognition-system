"""
为已有用户生成人脸编码
支持两种模型：
1. face_recognition (基于 dlib，128维，推荐用于验证任务)
2. trained_model (训练的 MobileFaceNet，128维，用于分类任务)

重要：--model trained 时会自动查找最新训练的模型（与门禁系统加载逻辑一致）。

用法:
  python admin/generate_encodings.py --model face_recognition
  python admin/generate_encodings.py --model trained --force
"""
import os
import sys
import json
import cv2
import numpy as np
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.database import Database
from common.config import DATABASE_PATH, FACES_DIR, DATA_DIR

# 模型选择配置文件（保存在 data/ 目录，已被 .gitignore 排除）
MODEL_CONFIG_PATH = os.path.join(DATA_DIR, ".encoding_model.json")


def imread_safe(filepath):
    """读取图片，支持中文路径"""
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def get_last_model():
    """获取上次使用的编码模型"""
    if os.path.exists(MODEL_CONFIG_PATH):
        try:
            with open(MODEL_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                return config.get("model", "face_recognition")
        except Exception:
            pass
    return "face_recognition"


def save_last_model(model_name):
    """保存上次使用的编码模型"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump({"model": model_name}, f)


def clear_encodings():
    """清除所有编码"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM face_encodings")
    conn.commit()
    conn.close()


def generate_with_face_recognition(progress_queue=None):
    """使用 face_recognition 生成编码"""
    def report(msg_type, *args):
        if progress_queue:
            progress_queue.put((msg_type, *args))
        else:
            if msg_type == 'progress':
                print(f"  [{args[0]}/{args[1]}] {args[2]}")
            elif msg_type == 'done':
                print(f"\nDone! Generated {args[0]} new encodings")
            elif msg_type == 'error':
                print(f"\nError: {args[0]}")

    report('progress', 0, 1, "Loading face_recognition...")

    try:
        import face_recognition
        report('progress', 0, 1, "face_recognition loaded")
    except ImportError:
        report('error', "face_recognition 未安装，请运行: pip install face_recognition")
        return 0

    db = Database(DATABASE_PATH)
    users = db.get_all_users(active_only=True)

    total_new = 0
    total_users = len(users)

    for user_idx, user in enumerate(users):
        user_id = user['id']
        name = user['name']

        user_dir = None
        for d in os.listdir(FACES_DIR):
            if d.startswith(f"{user_id}_"):
                user_dir = os.path.join(FACES_DIR, d)
                break

        if user_dir is None or not os.path.exists(user_dir):
            report('progress', user_idx + 1, total_users, f"{name} - no photo dir, skip")
            continue

        images = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for f in sorted(os.listdir(user_dir)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                img_path = os.path.join(user_dir, f)
                img = imread_safe(img_path)
                if img is not None:
                    images.append((f, img_path, img))

        if not images:
            report('progress', user_idx + 1, total_users, f"{name} - no valid images, skip")
            continue

        saved = 0
        for img_name, img_path, img in images:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img, model="hog")

            if len(face_locations) == 0:
                continue

            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            if len(face_encodings) > 0:
                db.add_face_encoding(user_id, face_encodings[0], img_path)
                saved += 1

        report('progress', user_idx + 1, total_users, f"{name} - {saved}/{len(images)} encodings")
        total_new += saved

    report('done', total_new)
    return total_new


def _find_latest_model_dir():
    """
    自动查找最新训练的模型目录（与 gate/access_control.py 的加载逻辑一致）。
    优先查找带 inference_model.pth 的版本目录，按修改时间倒序。
    """
    models_root = os.path.join(os.path.dirname(__file__), "..", "trainer", "models", "saved")
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


def generate_with_trained_model(progress_queue=None):
    """使用训练模型生成编码"""
    def report(msg_type, *args):
        if progress_queue:
            progress_queue.put((msg_type, *args))
        else:
            if msg_type == 'progress':
                print(f"  [{args[0]}/{args[1]}] {args[2]}")
            elif msg_type == 'done':
                print(f"\nDone! Generated {args[0]} new encodings")
            elif msg_type == 'error':
                print(f"\nError: {args[0]}")

    report('progress', 0, 1, "Loading trained model...")

    try:
        import importlib.util
        inference_path = os.path.join(os.path.dirname(__file__), "..", "trainer", "inference.py")
        spec = importlib.util.spec_from_file_location("trainer.inference", inference_path)
        inference_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_mod)
        FaceRecognizer = inference_mod.FaceRecognizer

        # 自动查找最新模型（与门禁系统 _load_trained_model 逻辑一致）
        model_dir = _find_latest_model_dir()
        if model_dir is None:
            report('error', "No trained model found in trainer/models/saved/")
            return 0

        report('progress', 0, 1, f"Using model: {os.path.basename(model_dir)}")
        recognizer = FaceRecognizer(model_dir=model_dir)
        if not recognizer.load_model():
            report('error', "Model load failed")
            return 0
    except Exception as e:
        report('error', f"Load inference module failed: {e}")
        return 0

    report('progress', 0, 1, f"Model loaded (device: {recognizer.device})")

    db = Database(DATABASE_PATH)
    users = db.get_all_users(active_only=True)

    total_new = 0
    total_users = len(users)

    for user_idx, user in enumerate(users):
        user_id = user['id']
        name = user['name']

        user_dir = None
        for d in os.listdir(FACES_DIR):
            if d.startswith(f"{user_id}_"):
                user_dir = os.path.join(FACES_DIR, d)
                break

        if user_dir is None or not os.path.exists(user_dir):
            report('progress', user_idx + 1, total_users, f"{name} - no photo dir, skip")
            continue

        images = []
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for f in sorted(os.listdir(user_dir)):
            if os.path.splitext(f)[1].lower() in valid_exts:
                img_path = os.path.join(user_dir, f)
                img = imread_safe(img_path)
                if img is not None:
                    images.append((f, img_path, img))

        if not images:
            report('progress', user_idx + 1, total_users, f"{name} - no valid images, skip")
            continue

        saved = 0
        for img_name, img_path, img in images:
            faces = recognizer.detect_faces(img)
            if len(faces) == 0:
                continue

            face_rect = max(faces, key=lambda f: f[2] * f[3])
            face_tensor = recognizer.preprocess_face(img, face_rect)
            embedding = recognizer.get_embedding(face_tensor)

            if embedding is not None:
                db.add_face_encoding(user_id, embedding, img_path)
                saved += 1

        report('progress', user_idx + 1, total_users, f"{name} - {saved}/{len(images)} encodings")
        total_new += saved

    report('done', total_new)
    return total_new


def generate_encodings(progress_queue=None, model="face_recognition", force=False):
    """
    生成人脸编码
    :param progress_queue: 进度队列
    :param model: 模型名称 ("face_recognition" 或 "trained")
    :param force: 强制重新生成
    """
    last_model = get_last_model()

    # 检查是否需要重新生成
    if not force and model == last_model:
        # 检查是否已有编码
        db = Database(DATABASE_PATH)
        users = db.get_all_users(active_only=True)
        has_encodings = False
        for user in users:
            if len(db.get_face_encodings(user['id'])) > 0:
                has_encodings = True
                break

        if has_encodings:
            if progress_queue:
                progress_queue.put(('done', 0))
            else:
                print("编码已存在且模型未变更，跳过（使用 --force 强制重新生成）")
            return 0

    # 清除旧编码（切换模型时）
    if force or model != last_model:
        clear_encodings()
        save_last_model(model)

    # 根据模型选择生成方式
    if model == "face_recognition":
        return generate_with_face_recognition(progress_queue)
    elif model == "trained":
        return generate_with_trained_model(progress_queue)
    else:
        if progress_queue:
            progress_queue.put(('error', f"未知模型: {model}"))
        return 0


# ============================================================
# 异步版本
# ============================================================

def _subprocess_reader(pipe, queue):
    """后台线程：读取子进程 stdout"""
    import re
    for line in iter(pipe.readline, ''):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'\[(\d+)/(\d+)\]\s*(.*)', line)
        if m:
            current, total, text = int(m.group(1)), int(m.group(2)), m.group(3)
            queue.put(('progress', current, total, text))
        elif line.startswith('Done!'):
            nums = re.findall(r'\d+', line)
            count = int(nums[0]) if nums else 0
            queue.put(('done', count))
        elif line.startswith('Error:'):
            queue.put(('error', line[6:].strip()))
        elif 'loaded' in line.lower() or 'loading' in line.lower():
            queue.put(('progress', 0, 1, line))
    pipe.close()


class _SubprocessHandle:
    """模拟 multiprocessing.Process 接口"""
    def __init__(self, proc):
        self._proc = proc

    @property
    def pid(self):
        return self._proc.pid

    def is_alive(self):
        return self._proc.poll() is None

    @property
    def exitcode(self):
        return self._proc.returncode


def generate_encodings_async(model="face_recognition", force=False):
    """
    异步生成编码
    :param model: 模型名称
    :param force: 强制重新生成
    :return: (process_handle, queue)
    """
    import subprocess
    import threading
    import multiprocessing

    queue = multiprocessing.Queue()

    script = os.path.join(os.path.dirname(__file__), 'generate_encodings.py')
    cmd = [sys.executable, '-u', script, '--model', model]
    if force:
        cmd.append('--force')

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )

    reader = threading.Thread(target=_subprocess_reader, args=(proc.stdout, queue), daemon=True)
    reader.start()

    handle = _SubprocessHandle(proc)
    return handle, queue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成人脸编码")
    parser.add_argument("--model", choices=["face_recognition", "trained"], default="face_recognition",
                       help="编码模型: face_recognition (推荐) 或 trained")
    parser.add_argument("--force", action="store_true", help="强制重新生成")
    args = parser.parse_args()
    generate_encodings(model=args.model, force=args.force)
