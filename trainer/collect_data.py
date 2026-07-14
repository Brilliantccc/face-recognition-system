"""
人脸数据收集工具
从摄像头或图片文件夹收集人脸数据并进行标注
"""

import cv2
import os
import sys
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FaceDataCollector:
    def __init__(self, output_dir="trainer/data"):
        """
        初始化数据收集器
        :param output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default_aligned.xml')
        if self.face_cascade.empty():
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # 创建目录
        os.makedirs(output_dir, exist_ok=True)

    def collect_from_camera(self, person_name: str, num_samples: int = 50):
        """
        从摄像头收集人脸数据
        :param person_name: 人员姓名
        :param num_samples: 需要的样本数量
        """
        person_dir = os.path.join(self.output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open camera!")
            return

        print(f"Collecting face data for: {person_name}")
        print(f"Target: {num_samples} samples")
        print("Press SPACE to capture, ESC to finish")

        count = 0
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测人脸
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # 绘制人脸框
            display_frame = frame.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 显示信息
            cv2.putText(display_frame, f"Count: {count}/{num_samples}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE: Capture | ESC: Finish", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Face Data Collection", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                if len(faces) > 0:
                    # 取最大的人脸
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                    # 扩大人脸区域
                    margin = int(0.2 * max(w, h))
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)

                    face_img = frame[y1:y2, x1:x2]

                    # 保存图片
                    img_path = os.path.join(person_dir, f"{count:04d}.jpg")
                    cv2.imwrite(img_path, face_img)
                    count += 1
                    print(f"Captured {count}/{num_samples}")
                else:
                    print("No face detected! Please face the camera.")

        cap.release()
        cv2.destroyAllWindows()
        print(f"Data collection complete: {count} samples saved to {person_dir}")

    def collect_from_folder(self, source_dir: str, person_name: str):
        """
        从文件夹收集人脸数据
        :param source_dir: 源图片文件夹
        :param person_name: 人员姓名
        """
        person_dir = os.path.join(self.output_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        # 支持的图片格式
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        count = 0
        for file_name in os.listdir(source_dir):
            if file_name.lower().endswith(valid_exts):
                img_path = os.path.join(source_dir, file_name)
                frame = cv2.imread(img_path)

                if frame is None:
                    continue

                # 检测人脸
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    # 取最大的人脸
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                    # 扩大人脸区域
                    margin = int(0.2 * max(w, h))
                    x1 = max(0, x - margin)
                    y1 = max(0, y - margin)
                    x2 = min(frame.shape[1], x + w + margin)
                    y2 = min(frame.shape[0], y + h + margin)

                    face_img = frame[y1:y2, x1:x2]

                    # 保存图片
                    save_path = os.path.join(person_dir, f"{count:04d}.jpg")
                    cv2.imwrite(save_path, face_img)
                    count += 1
                    print(f"Processed: {file_name} -> {person_name}/{count:04d}.jpg")

        print(f"Folder collection complete: {count} samples saved to {person_dir}")

    def create_dataset_info(self):
        """
        创建数据集信息文件
        """
        dataset_info = {
            "name": "face_dataset",
            "created_at": datetime.now().isoformat(),
            "persons": []
        }

        for person_name in os.listdir(self.output_dir):
            person_dir = os.path.join(self.output_dir, person_name)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                dataset_info["persons"].append({
                    "name": person_name,
                    "num_images": len(images)
                })

        info_path = os.path.join(self.output_dir, "dataset_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        print(f"Dataset info saved to {info_path}")
        return dataset_info

    def split_dataset(self, train_ratio: float = 0.8):
        """
        将数据集分为训练集和测试集
        :param train_ratio: 训练集比例
        """
        train_dir = os.path.join(self.output_dir, "train")
        test_dir = os.path.join(self.output_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for person_name in os.listdir(self.output_dir):
            person_dir = os.path.join(self.output_dir, person_name)
            if not os.path.isdir(person_dir) or person_name in ('train', 'test'):
                continue

            images = sorted([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            split_idx = int(len(images) * train_ratio)

            # 创建人员目录
            train_person_dir = os.path.join(train_dir, person_name)
            test_person_dir = os.path.join(test_dir, person_name)
            os.makedirs(train_person_dir, exist_ok=True)
            os.makedirs(test_person_dir, exist_ok=True)

            # 复制文件
            import shutil
            for img in images[:split_idx]:
                src = os.path.join(person_dir, img)
                dst = os.path.join(train_person_dir, img)
                shutil.copy2(src, dst)

            for img in images[split_idx:]:
                src = os.path.join(person_dir, img)
                dst = os.path.join(test_person_dir, img)
                shutil.copy2(src, dst)

            print(f"{person_name}: {split_idx} train, {len(images) - split_idx} test")

        print("Dataset split complete!")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Face Data Collection Tool")
    parser.add_argument("--mode", choices=["camera", "folder", "split", "info"], default="camera",
                       help="Collection mode")
    parser.add_argument("--name", type=str, help="Person name")
    parser.add_argument("--source", type=str, help="Source folder path (for folder mode)")
    parser.add_argument("--num", type=int, default=50, help="Number of samples (for camera mode)")
    parser.add_argument("--output", type=str, default="trainer/data", help="Output directory")

    args = parser.parse_args()

    collector = FaceDataCollector(args.output)

    if args.mode == "camera":
        if not args.name:
            print("Error: Please specify person name with --name")
            return
        collector.collect_from_camera(args.name, args.num)

    elif args.mode == "folder":
        if not args.name or not args.source:
            print("Error: Please specify --name and --source")
            return
        collector.collect_from_folder(args.source, args.name)

    elif args.mode == "split":
        collector.split_dataset()

    elif args.mode == "info":
        info = collector.create_dataset_info()
        print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
