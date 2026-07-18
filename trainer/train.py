"""
人脸模型训练脚本
支持 MobileFaceNet + ArcFace Loss
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.models.facenet import MobileFaceNet, ArcFaceLoss


class FaceDataset(Dataset):
    """人脸数据集"""
    def __init__(self, data_dir, transform=None):
        """
        初始化数据集
        :param data_dir: 数据目录（包含 train 或 train/test 子目录）
        :param transform: 数据变换
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载数据列表"""
        # 查找训练数据目录
        train_dir = os.path.join(self.data_dir, "train")
        if not os.path.exists(train_dir):
            train_dir = self.data_dir

        # 获取所有人员目录
        persons = sorted([d for d in os.listdir(train_dir)
                         if os.path.isdir(os.path.join(train_dir, d))])

        # 建立类别映射
        for idx, person in enumerate(persons):
            self.class_to_idx[person] = idx
            self.idx_to_class[idx] = person

            person_dir = os.path.join(train_dir, person)
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, img_name)
                    self.samples.append((img_path, idx))

        print(f"Loaded {len(self.samples)} images from {len(persons)} persons")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # 加载图片
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class FaceTrainer:
    """人脸模型训练器"""
    def __init__(self, data_dir="trainer/data", model_dir="trainer/models/saved"):
        """
        初始化训练器
        :param data_dir: 数据目录
        :param model_dir: 模型保存目录
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 数据变换
        self.train_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def prepare_data(self, batch_size=32):
        """准备数据加载器"""
        # 加载训练集
        full_dataset = FaceDataset(self.data_dir, transform=self.train_transform)
        self.class_to_idx = full_dataset.class_to_idx
        self.idx_to_class = full_dataset.idx_to_class
        num_classes = len(self.class_to_idx)

        # 加载验证集
        val_dir = os.path.join(self.data_dir, "test")
        if os.path.exists(val_dir):
            val_dataset = FaceDataset(self.data_dir, transform=self.val_transform)
        else:
            # 如果没有验证集，从训练集中划分
            val_size = int(0.2 * len(full_dataset))
            train_size = len(full_dataset) - val_size
            full_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

        self.train_dataset = full_dataset

        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, num_classes

    def train(self, epochs=50, batch_size=32, learning_rate=0.001):
        """
        训练模型
        :param epochs: 训练轮数
        :param batch_size: 批次大小
        :param learning_rate: 学习率
        """
        print("=" * 50)
        print("Starting Face Recognition Model Training")
        print("=" * 50)

        # 准备数据
        train_loader, val_loader, num_classes = self.prepare_data(batch_size)
        print(f"Number of classes: {num_classes}")

        # 创建模型
        model = MobileFaceNet(embedding_size=128, num_classes=num_classes).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # 使用交叉熵损失函数（带类别权重，处理数据不平衡）
        # 计算类别权重（样本少的类别权重高）
        class_counts = [0] * num_classes
        for _, labels in train_loader:
            for label in labels:
                class_counts[label.item()] += 1

        total_samples = sum(class_counts)
        class_weights = [total_samples / (num_classes * count) if count > 0 else 1.0 for count in class_counts]
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        print(f"Class counts: {class_counts}")
        print(f"Class weights: {[f'{w:.2f}' for w in class_weights]}")

        criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device)

        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        best_val_acc = 0.0

        for epoch in range(epochs):
            # 训练阶段
            model.train()
            criterion.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                output = model(images)
                if isinstance(output, tuple):
                    embeddings, logits = output
                else:
                    logits = output

                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    output = model(images)
                    if isinstance(output, tuple):
                        embeddings, logits = output
                    else:
                        logits = output

                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total

            # 更新学习率
            scheduler.step(val_loss)

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            # 打印信息
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                  f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(model, num_classes, self.class_to_idx)
                print(f"  -> Best model saved (Val Acc: {val_acc:.2f}%)")

        # 保存训练历史
        self.save_history(history)

        print("=" * 50)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        print("=" * 50)

        return history

    def save_model(self, model, num_classes, class_to_idx):
        """保存模型"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'embedding_size': 128,
            'class_to_idx': class_to_idx,
            'idx_to_class': {v: k for k, v in class_to_idx.items()}
        }

        model_path = os.path.join(self.model_dir, "best_model.pth")
        torch.save(checkpoint, model_path)

        # 同时保存为推理格式
        self.export_for_inference(model, num_classes, class_to_idx)

    def export_for_inference(self, model, num_classes, class_to_idx):
        """导出用于推理的模型"""
        # 创建推理模型（无分类头）
        inference_model = MobileFaceNet(embedding_size=128, num_classes=None)
        inference_model.load_state_dict(model.state_dict(), strict=False)

        # 保存
        inference_checkpoint = {
            'model_state_dict': inference_model.state_dict(),
            'num_classes': num_classes,
            'embedding_size': 128,
            'class_to_idx': class_to_idx,
            'idx_to_class': {v: k for k, v in class_to_idx.items()}
        }

        inference_path = os.path.join(self.model_dir, "inference_model.pth")
        torch.save(inference_checkpoint, inference_path)

        # 保存类别信息
        info_path = os.path.join(self.model_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_classes': num_classes,
                'embedding_size': 128,
                'class_to_idx': class_to_idx,
                'idx_to_class': {str(v): k for k, v in class_to_idx.items()}
            }, f, indent=2, ensure_ascii=False)

    def save_history(self, history):
        """保存训练历史"""
        history_path = os.path.join(self.model_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")

    def load_model(self, model_path=None):
        """加载模型"""
        if model_path is None:
            model_path = os.path.join(self.model_dir, "best_model.pth")

        checkpoint = torch.load(model_path, map_location=self.device)

        model = MobileFaceNet(
            embedding_size=checkpoint['embedding_size'],
            num_classes=checkpoint['num_classes']
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])

        return model, checkpoint.get('class_to_idx', {})


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition Model Training")
    parser.add_argument("--data", type=str, default="trainer/data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="trainer/models/saved", help="Model save directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    trainer = FaceTrainer(args.data, args.model_dir)
    trainer.train(args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
