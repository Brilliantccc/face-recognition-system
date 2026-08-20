"""
人脸模型微调脚本 — 用门禁实际数据微调预训练模型
目标：让模型知道"这个人是否注册过"，而不是"这是哪个人"

用法:
  python trainer/finetune_local.py              # 微调模型
  python trainer/finetune_local.py --calibrate  # 仅校准阈值
"""

import os
import sys
import json
import numpy as np
import cv2
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import torch_utils
from common.config import FACES_DIR, DATABASE_PATH

if not torch_utils.TORCH_AVAILABLE:
    print("❌ PyTorch 未安装，无法微调")
    sys.exit(1)

torch = torch_utils.torch
nn = torch_utils.nn
transforms = torch_utils.transforms
Image = torch_utils.Image
from torch.utils.data import DataLoader, Dataset
from trainer.models.facenet import MobileFaceNet
from torch.nn import functional as F
from torch.amp import autocast, GradScaler


# ============================================================
# 数据集：加载本地门禁用户照片
# ============================================================

class LocalFaceDataset(Dataset):
    """加载本地门禁用户照片"""

    def __init__(self, faces_dir, db_path, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM users WHERE is_active = 1")
        users = cursor.fetchall()
        conn.close()

        if not users:
            print("❌ 没有活跃用户")
            return

        for idx, (user_id, name) in enumerate(users):
            self.class_to_idx[user_id] = idx
            self.idx_to_class[idx] = (user_id, name)

        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        for user_id, name in users:
            user_dir = None
            for d in os.listdir(faces_dir):
                if d.startswith(f"{user_id}_"):
                    user_dir = os.path.join(faces_dir, d)
                    break

            if user_dir is None or not os.path.exists(user_dir):
                print(f"  ⚠️ {name}: 未找到照片目录")
                continue

            img_count = 0
            for img_name in os.listdir(user_dir):
                if os.path.splitext(img_name)[1].lower() in valid_exts:
                    img_path = os.path.join(user_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[user_id]))
                    img_count += 1

            print(f"  👤 {name}: {img_count} 张照片")

        print(f"\n📊 总计: {len(self.samples)} 张照片, {len(self.class_to_idx)} 个用户")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_array = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 微调器（验证任务导向）
# ============================================================

class LocalFinetuner:
    """
    用本地数据微调模型
    目标：让模型的 embedding 更适合"验证"任务
    - 同一个人的不同照片 → embedding 应该接近
    - 不同人的照片 → embedding 应该远离
    """

    def __init__(self, base_model_dir="trainer/models/saved/optimal_v2",
                 output_dir="trainer/models/saved/finetuned_local"):
        self.base_model_dir = base_model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.device = torch_utils.get_device(prefer_gpu=True)
        print(f"🖥️ 设备: {self.device}")

        # 数据变换（增强版：模拟真实门禁场景）
        self.train_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = self._load_base_model()

    def _load_base_model(self):
        """加载预训练模型"""
        model_path = os.path.join(self.base_model_dir, "inference_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(self.base_model_dir, "best_model.pth")

        print(f"📂 加载基础模型: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 推断分类数
        num_classes = 10575
        for key in state_dict:
            if 'classifier.weight' in key:
                num_classes = state_dict[key].shape[0]
                break

        print(f"  原始分类数: {num_classes}")

        model = MobileFaceNet(embedding_size=128, num_classes=num_classes)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)

        return model

    def finetune(self, epochs=20, lr=0.001, batch_size=8):
        """
        微调模型
        策略：冻结特征层，只训练分类头
        目的：让 embedding 空间更适合验证任务
        """
        print("\n" + "=" * 50)
        print("🔧 开始微调（验证任务导向）")
        print("=" * 50)

        # 加载数据
        train_dataset = LocalFaceDataset(FACES_DIR, DATABASE_PATH, self.train_transform)
        if len(train_dataset) == 0:
            print("❌ 没有可用的训练数据")
            return

        num_users = len(train_dataset.class_to_idx)
        print(f"  👥 微调用户数: {num_users}")

        # 数据增强：如果数据太少，重复采样
        if len(train_dataset.samples) < 30:
            print("  ⚠️ 数据较少，启用增强采样")
            from torch.utils.data import ConcatDataset
            repeat_times = max(5, 30 // len(train_dataset.samples))
            datasets = [LocalFaceDataset(FACES_DIR, DATABASE_PATH, self.train_transform)
                       for _ in range(repeat_times)]
            train_dataset = ConcatDataset(datasets)
            print(f"  📊 增强后: {len(train_dataset)} 张照片")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)

        # 替换分类头
        if hasattr(self.model, 'classifier'):
            old_out = self.model.classifier.out_features
            if old_out != num_users:
                print(f"  🔄 替换分类头: {old_out} → {num_users}")
                self.model.classifier = nn.Linear(
                    self.model.classifier.in_features, num_users
                ).to(self.device)

        # 冻结特征提取层，只训练分类头
        print("  🔒 冻结特征提取层，只训练分类头")
        for name, param in self.model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False

        # 优化器
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()

        # 训练
        best_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

            scheduler.step()

            train_acc = 100.0 * correct / total if total > 0 else 0
            avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0

            print(f"  Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Acc: {train_acc:.1f}%")

            if train_acc > best_acc:
                best_acc = train_acc
                self._save_model(num_users, "best_finetuned.pth")

        # 保存最终模型
        self._save_model(num_users, "final_finetuned.pth")
        print(f"\n✅ 微调完成！最佳准确率: {best_acc:.1f}%")
        print(f"📁 模型保存在: {self.output_dir}")

        # 自动校准阈值
        print("\n" + "=" * 50)
        print("🎯 自动校准阈值")
        print("=" * 50)
        calibrate_threshold(self.output_dir)

    def _save_model(self, num_classes, filename):
        """保存模型"""
        model_state = self.model.state_dict()
        inference_state = {k: v for k, v in model_state.items()
                          if 'classifier' not in k and 'fc' not in k}

        save_path = os.path.join(self.output_dir, filename)
        torch.save(inference_state, save_path)

        config = {
            "model_name": "FaceNet-Finetuned-Local",
            "architecture": "MobileFaceNet + ArcFace",
            "num_classes": num_classes,
            "embedding_size": 128,
            "input_size": [3, 112, 112],
            "finetuned": True,
            "base_model": self.base_model_dir,
            "description": {
                "zh": "基于本地门禁数据微调的人脸识别模型，优化验证任务"
            }
        }
        config_path = os.path.join(self.output_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


# ============================================================
# 阈值校准（验证任务专用）
# ============================================================

def calibrate_threshold(model_dir=None):
    """
    校准识别阈值
    测试不同阈值，找到最佳值
    目标：在误识率 < 1% 的前提下，尽量提高识别率
    """
    from common.database import Database
    from common.user_manager import UserManager

    db = Database(DATABASE_PATH)
    user_manager = UserManager(db)

    encodings, names, user_ids = user_manager.get_user_encodings()
    if len(encodings) == 0:
        print("❌ 没有人脸编码数据")
        return

    print(f"📊 共 {len(encodings)} 个编码, {len(set(names))} 个用户")

    # 转换为 numpy
    encodings_np = np.array([e.cpu().numpy() if hasattr(e, 'cpu') else np.array(e) for e in encodings])

    # 计算所有编码之间的距离矩阵
    n = len(encodings_np)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(encodings_np[i] - encodings_np[j])

    # 同一个人的距离（正样本对）
    same_person_dists = []
    diff_person_dists = []

    for i in range(n):
        for j in range(i + 1, n):
            if user_ids[i] == user_ids[j]:
                same_person_dists.append(dist_matrix[i][j])
            else:
                diff_person_dists.append(dist_matrix[i][j])

    if not same_person_dists:
        print("⚠️ 每个用户只有1张照片，无法计算同人距离")
        print("  建议：为每个用户补充更多照片（至少5张）")
        return

    print(f"\n📊 距离统计:")
    print(f"  同人距离: 平均 {np.mean(same_person_dists):.4f}, 最大 {np.max(same_person_dists):.4f}")
    print(f"  异人距离: 平均 {np.mean(diff_person_dists):.4f}, 最小 {np.min(diff_person_dists):.4f}")

    # 测试不同阈值
    thresholds = np.arange(0.30, 0.70, 0.02)

    print(f"\n{'阈值':>6} | {'识别率':>8} | {'误识率':>8} | {'评价':>10}")
    print("-" * 50)

    best_threshold = 0.45
    best_score = 0

    for threshold in thresholds:
        # 识别率：同人距离 < 阈值的比例
        true_accept = sum(1 for d in same_person_dists if d <= threshold)
        recall = true_accept / len(same_person_dists) * 100 if same_person_dists else 0

        # 误识率：异人距离 < 阈值的比例
        false_accept = sum(1 for d in diff_person_dists if d <= threshold)
        far = false_accept / len(diff_person_dists) * 100 if diff_person_dists else 0

        # 评分：识别率高 + 误识率低
        score = recall - far * 10  # 误识惩罚更重

        rating = ""
        if far < 1 and recall > 80:
            rating = "⭐ 最佳"
            if score > best_score:
                best_score = score
                best_threshold = threshold
        elif far < 5:
            rating = "✅ 可用"
        else:
            rating = "❌ 误识高"

        print(f"{threshold:>6.2f} | {recall:>7.1f}% | {far:>7.1f}% | {rating}")

    print(f"\n🎯 建议阈值: {best_threshold:.2f}")
    print(f"\n💡 修改方法:")
    print(f"   打开 common/config.py，修改:")
    print(f"   FACE_RECOGNITION_TOLERANCE = {best_threshold:.2f}")

    # 自动更新 config.py
    config_path = os.path.join(os.path.dirname(__file__), "..", "common", "config.py")
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 替换阈值
        import re
        new_content = re.sub(
            r'FACE_RECOGNITION_TOLERANCE\s*=\s*[\d.]+',
            f'FACE_RECOGNITION_TOLERANCE = {best_threshold:.2f}',
            content
        )

        if new_content != content:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"\n✅ 已自动更新 config.py 中的阈值为 {best_threshold:.2f}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="人脸模型微调（验证任务导向）")
    parser.add_argument("--calibrate", action="store_true", help="仅校准阈值")
    parser.add_argument("--epochs", type=int, default=20, help="微调轮数")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    args = parser.parse_args()

    if args.calibrate:
        calibrate_threshold()
    else:
        finetuner = LocalFinetuner()
        finetuner.finetune(epochs=args.epochs, lr=args.lr)
