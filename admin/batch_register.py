"""
批量注册模块
从 data/uploads 目录读取人员照片，批量注册到数据库
支持命令行调用和 GUI 复用

注意：此模块不依赖 face_recognition/dlib，只做照片存储和数据库写入。
人脸编码由门禁系统启动时统一生成。
"""
import os
import sys

import cv2
import numpy as np

from common.config import DATABASE_PATH, FACES_DIR

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def scan_uploads(uploads_dir):
    """扫描 uploads 目录，返回人员列表"""
    if not os.path.exists(uploads_dir):
        return []
    persons = []
    for name in sorted(os.listdir(uploads_dir)):
        person_dir = os.path.join(uploads_dir, name)
        if not os.path.isdir(person_dir):
            continue
        count = len([f for f in os.listdir(person_dir)
                     if os.path.splitext(f)[1].lower() in VALID_EXTS])
        if count > 0:
            persons.append({'name': name, 'count': count})
    return persons


def batch_register(faces_dir="data/uploads", db_path=None, user_manager=None):
    """
    批量注册人脸图片（已存在的用户会跳过或恢复，不会重复添加）
    直接操作数据库，不触发 face_recognition/dlib 加载。
    人脸编码由门禁系统启动时统一生成。
    Returns: dict: {'success': int, 'skip': int, 'reactivate': int, 'fail': int, 'errors': list, 'persons': list}
    """
    result = {'success': 0, 'skip': 0, 'reactivate': 0, 'fail': 0, 'errors': [], 'persons': []}
    if not os.path.exists(faces_dir):
        return result

    from common.database import Database
    db = Database(db_path or DATABASE_PATH)

    persons = scan_uploads(faces_dir)
    result['persons'] = persons

    for person in persons:
        person_dir = os.path.join(faces_dir, person['name'])

        # 检查用户是否已存在（包括不活跃的）
        existing = db.get_user_by_name(person['name'])
        if existing:
            if existing.get('is_active', True):
                # 已存在且活跃 → 跳过
                result['skip'] += 1
                continue
            else:
                # 已存在但不活跃 → 恢复
                user_id = existing['id']
                db.restore_user(user_id)
                result['reactivate'] += 1
                # 获取现有照片数量
                user_dir = None
                for d in os.listdir(FACES_DIR):
                    if d.startswith(f"{user_id}_"):
                        user_dir = os.path.join(FACES_DIR, d)
                        break
                if user_dir is None:
                    safe_name = "".join(c for c in person['name']
                                       if c.isalnum() or c in (' ', '-', '_') or '一' <= c <= '鿿').strip()
                    if not safe_name:
                        safe_name = "user"
                    user_dir = os.path.join(FACES_DIR, f"{user_id}_{safe_name}")
                existing_count = len([f for f in os.listdir(user_dir)
                                     if os.path.splitext(f)[1].lower() in VALID_EXTS]) if os.path.exists(user_dir) else 0
        else:
            # 新用户 → 创建
            user_id = db.add_user(person['name'])
            safe_name = "".join(c for c in person['name']
                               if c.isalnum() or c in (' ', '-', '_') or '一' <= c <= '鿿').strip()
            if not safe_name:
                safe_name = "user"
            user_dir = os.path.join(FACES_DIR, f"{user_id}_{safe_name}")
            existing_count = 0

        # 创建用户目录
        os.makedirs(user_dir, exist_ok=True)

        # 读取图片
        valid_images = []
        for img_name in os.listdir(person_dir):
            if os.path.splitext(img_name)[1].lower() in VALID_EXTS:
                img_path = os.path.join(person_dir, img_name)
                try:
                    img_array = np.fromfile(img_path, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        valid_images.append((img_name, img_path))
                except Exception as e:
                    result['errors'].append(f"{person['name']}: {img_name} - {e}")

        if not valid_images:
            result['errors'].append(f"{person['name']}: 无有效图片")
            result['fail'] += 1
            continue

        # 保存照片
        saved = 0
        for idx, (img_name, src_path) in enumerate(valid_images):
            dst_path = os.path.join(user_dir, f"{existing_count + idx}.jpg")
            try:
                img_array = np.fromfile(src_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is not None:
                    _, encoded = cv2.imencode('.jpg', img)
                    encoded.tofile(dst_path)
                    saved += 1
            except Exception as e:
                result['errors'].append(f"{person['name']}: {img_name} 保存失败 - {e}")

        if saved == 0 and existing is None:
            db.delete_user(user_id)
            result['errors'].append(f"{person['name']}: 所有照片保存失败")
            result['fail'] += 1
        else:
            result['success'] += 1

    return result


# ============================================================
# 多进程版本 —— 在子进程中执行，隔离任何潜在崩溃
# ============================================================

def _worker(uploads_dir, result_queue):
    """子进程入口：执行 batch_register 并把结果放回队列"""
    try:
        result = batch_register(uploads_dir)
        result_queue.put(('ok', result))
    except Exception as e:
        result_queue.put(('error', str(e)))


def batch_register_async(uploads_dir, callback):
    """
    异步批量注册（多进程）
    :param uploads_dir: uploads 目录路径
    :param callback: 完成回调 callback(result_dict)
    :return: (process, queue, callback)
    """
    import multiprocessing
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_worker, args=(uploads_dir, queue), daemon=True)
    proc.start()
    return proc, queue, callback
