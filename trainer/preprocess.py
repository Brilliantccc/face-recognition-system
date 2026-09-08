"""
CASIA-WebFace 数据集预处理脚本

功能:
  1. 使用 YOLO / face_recognition 检测原始图片中的人脸
  2. 裁剪并对齐人脸到 112×112 标准化尺寸
  3. 按 train/val 比例划分数据集
  4. 输出到 trainer/data/aligned/ 目录

使用方法:
  # 1. 把 CASIA-WebFace 解压到 trainer/data/raw/CASIA-WebFace/
  # 2. 运行本脚本:
  python trainer/preprocess.py

  # 或指定参数:
  python trainer/preprocess.py --raw-dir trainer/data/raw/CASIA-WebFace \
                               --output-dir trainer/data/aligned \
                               --size 112 --val-ratio 0.1 --method yolo

目录结构要求:
  trainer/data/raw/CASIA-WebFace/
  ├── 0000001/
  │   ├── 001.jpg
  │   ├── 002.jpg
  │   └── ...
  ├── 0000002/
  │   └── ...
  └── ...

输出结构:
  trainer/data/aligned/
  ├── train/
  │   ├── 0000001/
  │   │   ├── 001.jpg  (112×112 对齐后)
  │   │   └── ...
  │   └── ...
  └── val/
      ├── 0000001/
      └── ...
"""

import os
import sys
import cv2
import shutil
import random
import argparse
import numpy as np
from datetime import datetime

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class FacePreprocessor:
    """人脸数据预处理器（使用模块化架构）"""

    def __init__(self, output_size=112, method="yolo"):
        """
        初始化预处理器
        :param output_size: 输出图片尺寸 (112×112)
        :param method: 人脸检测方法 ("yolo", "face_recognition", "haar", "auto")
        """
        self.output_size = output_size
        self.method = method
        self.detector = None

        self._init_detector()

    def _init_detector(self):
        """初始化人脸检测器"""
        from common.detectors import DetectorFactory

        try:
            if self.method == "auto":
                self.detector = DetectorFactory.create("auto")
            else:
                self.detector = DetectorFactory.create(self.method)
            print(f"[Preprocess] 检测器已加载: {type(self.detector).__name__}")
        except Exception as e:
            print(f"[Preprocess] 检测器加载失败: {e}")
            raise

    def detect_face(self, image):
        """
        检测图片中的人脸，返回最大的人脸边界框
        :param image: BGR 格式图片
        :return: (x1, y1, x2, y2) 或 None
        """
        if self.detector is None:
            return None

        try:
            detections = self.detector.detect(image)
            if detections:
                # 取置信度最高的人脸
                best = max(detections, key=lambda d: d.confidence if hasattr(d, 'confidence') else 1.0)
                if hasattr(best, 'x1'):
                    return (best.x1, best.y1, best.x2, best.y2)
                else:
                    return (best[0], best[1], best[2], best[3])
        except Exception as e:
            print(f"[Preprocess] 检测失败: {e}")

        return None

    def crop_and_resize(self, image, bbox, margin=0.2):
        """
        裁剪人脸区域并缩放到标准尺寸
        :param image: 原始图片
        :param bbox: 边界框 (x1, y1, x2, y2)
        :param margin: 边距比例
        :return: 112×112 的人脸图片
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # 计算扩展边距
        face_w = x2 - x1
        face_h = y2 - y1
        margin_x = int(face_w * margin)
        margin_y = int(face_h * margin)

        # 扩展边界
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)

        # 裁剪
        face_img = image[y1:y2, x1:x2]

        if face_img.size == 0:
            return None

        # 缩放到标准尺寸
        face_img = cv2.resize(face_img, (self.output_size, self.output_size),
                              interpolation=cv2.INTER_LINEAR)

        return face_img

    def process_image(self, image_path):
        """
        处理单张图片：检测人脸 → 裁剪对齐 → 返回标准化图片
        :param image_path: 图片路径
        :return: 处理后的图片 (numpy array) 或 None
        """
        image = cv2.imread(image_path)
        if image is None:
            return None

        bbox = self.detect_face(image)
        if bbox is None:
            return None

        face_img = self.crop_and_resize(image, bbox)
        return face_img


def preprocess_dataset(raw_dir, output_dir, output_size=112, val_ratio=0.1,
                       method="yolo", max_per_person=None, skip_existing=True):
    """
    预处理整个数据集

    Args:
        raw_dir:          原始数据集目录
        output_dir:       输出目录
        output_size:      输出图片尺寸
        val_ratio:        验证集比例
        method:           人脸检测方法
        max_per_person:   每人最大图片数 (None 表示不限制)
        skip_existing:    是否跳过已处理的图片
    """
    print("=" * 60)
    print("CASIA-WebFace 数据集预处理")
    print("=" * 60)
    print(f"原始目录: {raw_dir}")
    print(f"输出目录: {output_dir}")
    print(f"输出尺寸: {output_size}×{output_size}")
    print(f"验证集比例: {val_ratio}")
    print(f"检测方法: {method}")
    print("=" * 60)

    # 检查原始目录
    if not os.path.exists(raw_dir):
        print(f"❌ 原始目录不存在: {raw_dir}")
        print(f"\n请先下载 CASIA-WebFace 数据集并解压到:")
        print(f"  {raw_dir}")
        return

    # 创建输出目录
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 初始化预处理器
    preprocessor = FacePreprocessor(output_size=output_size, method=method)

    # 获取所有人员目录
    person_dirs = sorted([
        d for d in os.listdir(raw_dir)
        if os.path.isdir(os.path.join(raw_dir, d))
    ])

    print(f"\n找到 {len(person_dirs)} 个人员目录")

    # 统计
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    persons_processed = 0

    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for i, person_name in enumerate(person_dirs):
        person_raw_dir = os.path.join(raw_dir, person_name)

        # 获取该人的所有图片
        images = sorted([
            f for f in os.listdir(person_raw_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])

        if not images:
            continue

        # 限制每人图片数
        if max_per_person and len(images) > max_per_person:
            images = random.sample(images, max_per_person)

        # 决定 train/val 划分
        random.shuffle(images)
        
        # 确保训练集至少有 1 张图片
        if len(images) <= 2:
            # 图片太少，全部放入训练集，验证集为空
            val_images = []
            train_images = images
        else:
            val_count = max(1, int(len(images) * val_ratio))
            # 确保训练集至少有 1 张
            if val_count >= len(images):
                val_count = len(images) - 1
            val_images = images[:val_count]
            train_images = images[val_count:]

        # 处理图片
        person_processed = 0
        for split, split_images in [("train", train_images), ("val", val_images)]:
            split_dir = train_dir if split == "train" else val_dir
            person_out_dir = os.path.join(split_dir, person_name)
            os.makedirs(person_out_dir, exist_ok=True)

            for img_name in split_images:
                img_path = os.path.join(person_raw_dir, img_name)
                out_path = os.path.join(person_out_dir, img_name)

                # 跳过已处理的
                if skip_existing and os.path.exists(out_path):
                    total_skipped += 1
                    person_processed += 1
                    continue

                # 处理图片
                face_img = preprocessor.process_image(img_path)

                if face_img is not None:
                    cv2.imwrite(out_path, face_img)
                    total_processed += 1
                    person_processed += 1
                else:
                    total_failed += 1

        persons_processed += 1

        # 进度显示
        if (i + 1) % 100 == 0 or (i + 1) == len(person_dirs):
            print(f"  [{i+1}/{len(person_dirs)}] "
                  f"已处理: {total_processed} | "
                  f"跳过: {total_skipped} | "
                  f"失败: {total_failed}")

    print("\n" + "=" * 60)
    print("预处理完成!")
    print(f"  处理人员: {persons_processed}")
    print(f"  成功处理: {total_processed}")
    print(f"  跳过(已存在): {total_skipped}")
    print(f"  失败(未检测到人脸): {total_failed}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)


def check_raw_dataset(raw_dir):
    """检查原始数据集状态"""
    print(f"\n检查原始数据集: {raw_dir}")

    if not os.path.exists(raw_dir):
        print(f"  ❌ 目录不存在!")
        return False

    person_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    total_images = 0
    for person in person_dirs:
        person_dir = os.path.join(raw_dir, person)
        imgs = [f for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        total_images += len(imgs)

    print(f"  ✅ 找到 {len(person_dirs)} 人, {total_images} 张图片")
    return True


def main():
    parser = argparse.ArgumentParser(description="CASIA-WebFace 数据集预处理")
    parser.add_argument("--raw-dir", type=str,
                        default="trainer/data/raw/CASIA-WebFace",
                        help="原始数据集目录")
    parser.add_argument("--output-dir", type=str,
                        default="trainer/data/aligned",
                        help="输出目录")
    parser.add_argument("--size", type=int, default=112,
                        help="输出图片尺寸 (默认 112)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例 (默认 0.1)")
    parser.add_argument("--method", type=str, default="yolo",
                        choices=["yolo", "face_recognition", "haar"],
                        help="人脸检测方法")
    parser.add_argument("--max-per-person", type=int, default=None,
                        help="每人最大图片数 (默认不限制)")
    parser.add_argument("--check", action="store_true",
                        help="只检查数据集，不处理")
    parser.add_argument("--no-skip", action="store_true",
                        help="不跳过已处理的图片 (重新处理)")

    args = parser.parse_args()

    if args.check:
        check_raw_dataset(args.raw_dir)
        return

    preprocess_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        output_size=args.size,
        val_ratio=args.val_ratio,
        method=args.method,
        max_per_person=args.max_per_person,
        skip_existing=not args.no_skip
    )


if __name__ == "__main__":
    main()
