"""
人脸模型训练脚本 - RTX 3060 Laptop 最优版
针对 6GB 显存优化，使用 AMP 加速
"""

import os
import sys
import json
import numpy as np
import warnings

# 忽略 AMP 相关的学习率调度器警告（PyTorch 已知问题）
warnings.filterwarnings('ignore', category=UserWarning, message='.*lr_scheduler.step.*')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import torch_utils

if torch_utils.TORCH_AVAILABLE:
    torch = torch_utils.torch
    nn = torch_utils.nn
    transforms = torch_utils.transforms
    Image = torch_utils.Image
    from torch.utils.data import DataLoader, Dataset
    from trainer.models.facenet import MobileFaceNet, ArcFaceLoss
    from torch.nn import functional as F
    from torch.amp import autocast, GradScaler


class FaceDataset(Dataset):
    """人脸数据集"""
    def __init__(self, data_dir, transform=None, class_to_idx=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = class_to_idx if class_to_idx is not None else {}
        self.idx_to_class = {}
        self._load_data()

    def _load_data(self):
        persons = sorted([d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d))])
        if not self.class_to_idx:
            for idx, person in enumerate(persons):
                self.class_to_idx[person] = idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for person in persons:
            if person not in self.class_to_idx:
                continue
            idx = self.class_to_idx[person]
            person_dir = os.path.join(self.data_dir, person)
            for img_name in os.listdir(person_dir):
                if os.path.splitext(img_name)[1].lower() in valid_exts:
                    img_path = os.path.join(person_dir, img_name)
                    self.samples.append((img_path, idx))
        print(f"Loaded {len(self.samples)} images from {len(persons)} persons")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class OptimalTrainer:
    """RTX 3060 最优训练器"""
    
    def __init__(self, data_dir="trainer/data/aligned", 
                 model_dir="trainer/models/saved", use_gpu=True):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.device = torch_utils.get_device(prefer_gpu=use_gpu)
        print(f"Using device: {self.device}")
        
        # 启用 AMP (自动混合精度)
        self.use_amp = torch.cuda.is_available()
        print(f"AMP (混合精度): {'启用' if self.use_amp else '禁用'}")
        
        # 启用 cuDNN benchmark
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("cuDNN benchmark: 启用")

        # 数据增强
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def prepare_data(self, batch_size=128):
        """准备数据"""
        train_dir = os.path.join(self.data_dir, "train")
        if not os.path.exists(train_dir):
            train_dir = self.data_dir
        
        train_dataset = FaceDataset(train_dir, transform=self.train_transform)
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = train_dataset.idx_to_class
        num_classes = len(self.class_to_idx)

        val_dir = os.path.join(self.data_dir, "val")
        if os.path.exists(val_dir):
            val_dataset = FaceDataset(val_dir, transform=self.val_transform, 
                                     class_to_idx=self.class_to_idx)
            print(f"Loaded validation set from {val_dir}")
        else:
            val_size = int(0.2 * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )

        # i9-12900H + RTX 3060 优化
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=6,            # 6 个 worker
            pin_memory=True,          # 锁页内存加速
            persistent_workers=True,  # 保持进程存活
            prefetch_factor=4,        # 预加载 4 个 batch
            drop_last=True            # 丢弃最后不完整的batch
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, 
            num_workers=6,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4
        )

        return train_loader, val_loader, num_classes

    def save_checkpoint(self, model, criterion, optimizer, scheduler, epoch, 
                        best_val_acc, history, num_classes, class_to_idx, 
                        gradient_clip_max_norm, patience_counter, scaler=None):
        """保存完整训练断点（用于续训）"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'arcface_weight': criterion.weight.data,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'history': history,
            'num_classes': num_classes,
            'embedding_size': 128,
            'class_to_idx': class_to_idx,
            'idx_to_class': {v: k for k, v in class_to_idx.items()},
            'gradient_clip_max_norm': gradient_clip_max_norm,
            'patience_counter': patience_counter,
        }
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        checkpoint_path = os.path.join(self.model_dir, "checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """加载训练断点（用于续训）"""
        if not os.path.exists(checkpoint_path):
            return None
        print(f"📂 找到训练断点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        return checkpoint

    def train(self, epochs=100, batch_size=128, learning_rate=0.1, resume=False):
        """训练模型，支持断点续训"""
        import torch.optim as optim

        print("=" * 60)
        print("Face Recognition Training (RTX 3060 Optimal)")
        print("=" * 60)

        # 检查是否有断点可恢复
        checkpoint_path = os.path.join(self.model_dir, "checkpoint.pth")
        start_epoch = 0
        resume_checkpoint = None
        
        if resume and os.path.exists(checkpoint_path):
            resume_checkpoint = self.load_checkpoint(checkpoint_path)
            if resume_checkpoint:
                start_epoch = resume_checkpoint['epoch'] + 1
                self.class_to_idx = resume_checkpoint.get('class_to_idx', {})
                self.idx_to_class = resume_checkpoint.get('idx_to_class', {})
                print(f"🔄 将从 Epoch {start_epoch + 1} 继续训练（上次到 Epoch {resume_checkpoint['epoch'] + 1}）")
                print(f"   上次最佳验证准确率: {resume_checkpoint.get('best_val_acc', 0):.2f}%")

        train_loader, val_loader, num_classes = self.prepare_data(batch_size)
        print(f"Number of classes: {num_classes}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Num workers: {train_loader.num_workers}")

        # 模型只返回 embeddings（不带分类头）
        model = MobileFaceNet(embedding_size=128, num_classes=None).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # ArcFace Loss
        criterion = ArcFaceLoss(
            embedding_size=128,
            num_classes=num_classes,
            s=32.0,    # 稳定缩放因子
            m=0.20     # 稳定间隔
        ).to(self.device)
        print(f"Using ArcFace Loss (s=32.0, m=0.20)")

        # 参数分组：卷积层和分类层使用不同的 weight_decay
        # ArcFace 的 W 已被 L2 归一化，不应施加正则化
        model_params = list(model.parameters())
        arcface_params = list(criterion.parameters())
        
        optimizer = optim.SGD([
            {'params': model_params, 'weight_decay': 4e-5},      # 卷积层：轻量正则化
            {'params': arcface_params, 'weight_decay': 0}        # 分类层：不施加正则
        ], lr=learning_rate, momentum=0.9, nesterov=True)

        # OneCycleLR 学习率调度
        total_steps = epochs * len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,           # 使用命令行传入的学习率
            total_steps=total_steps,
            pct_start=0.10,             # 预热 10%
            div_factor=25,              # 初始 lr = max_lr / 25
            final_div_factor=1000       # 最终 lr = max_lr / 1000
        )

        # AMP
        scaler = GradScaler('cuda') if self.use_amp else None

        history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 15
        
        # 梯度监控和恢复机制
        gradient_clip_max_norm = 5.0      # 初始梯度裁剪阈值
        loss_explosion_threshold = 50.0    # Loss 爆炸阈值（随机模型初始loss约22，50为真正异常）
        loss_nan_count = 0                 # Loss NaN 计数
        max_loss_nan_count = 3             # 最大容忍 NaN 次数
        last_good_checkpoint = None        # 上一个好的 checkpoint
        recovery_count = 0                 # 恢复次数
        max_recovery_count = 3             # 最大恢复次数

        # 从断点恢复状态
        if resume_checkpoint:
            model.load_state_dict(resume_checkpoint['model_state_dict'])
            criterion.weight.data = resume_checkpoint['arcface_weight']
            optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            best_val_acc = resume_checkpoint.get('best_val_acc', 0.0)
            history = resume_checkpoint.get('history', history)
            gradient_clip_max_norm = resume_checkpoint.get('gradient_clip_max_norm', 5.0)
            patience_counter = resume_checkpoint.get('patience_counter', 0)
            if scaler is not None and 'scaler_state_dict' in resume_checkpoint:
                scaler.load_state_dict(resume_checkpoint['scaler_state_dict'])
            print(f"✅ 断点恢复成功！继续训练...")

        for epoch in range(start_epoch, epochs):
            model.train()
            criterion.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            total_batches = len(train_loader)
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if self.use_amp:
                    with autocast('cuda'):
                        embeddings = model(images)
                        loss, logits = criterion(embeddings, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(criterion.parameters()),
                        max_norm=gradient_clip_max_norm  # 使用动态阈值
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    embeddings = model(images)
                    loss, logits = criterion(embeddings, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(criterion.parameters()),
                        max_norm=gradient_clip_max_norm  # 使用动态阈值
                    )
                    optimizer.step()

                # 学习率调度（必须在 optimizer.step() 之后）
                scheduler.step()

                # 梯度监控（在累加 loss 之前检查，防止 NaN 污染统计）
                current_loss = loss.item()
                if np.isnan(current_loss) or np.isinf(current_loss):
                    loss_nan_count += 1
                    print(f"  ⚠️ 检测到 NaN/Inf Loss (第 {loss_nan_count} 次)")
                    
                    if loss_nan_count >= max_loss_nan_count:
                        # 尝试恢复
                        if recovery_count < max_recovery_count and last_good_checkpoint:
                            recovery_count += 1
                            print(f"  🔄 尝试恢复到上一个好的 checkpoint (第 {recovery_count} 次)")
                            
                            # 加载上一个好的 checkpoint
                            checkpoint = torch.load(last_good_checkpoint, map_location=self.device)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            criterion.weight.data = checkpoint['arcface_weight']
                            
                            # 降低学习率和梯度裁剪阈值
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            gradient_clip_max_norm *= 0.5
                            
                            # 重置计数器
                            loss_nan_count = 0
                            print(f"  ✅ 恢复成功，学习率降低 50%，梯度裁剪阈值: {gradient_clip_max_norm:.2f}")
                        else:
                            print(f"  ❌ 无法恢复，停止训练")
                            break
                    continue  # 跳过此 batch，不累加 NaN loss
                
                train_loss += current_loss

                # Loss 爆炸检测
                if current_loss > loss_explosion_threshold:
                    print(f"  ⚠️ Loss 爆炸: {current_loss:.2f} > {loss_explosion_threshold}")
                    
                    if recovery_count < max_recovery_count and last_good_checkpoint:
                        recovery_count += 1
                        print(f"  🔄 尝试恢复到上一个好的 checkpoint")
                        
                        checkpoint = torch.load(last_good_checkpoint, map_location=self.device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        criterion.weight.data = checkpoint['arcface_weight']
                        
                        # 降低学习率
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5
                        gradient_clip_max_norm *= 0.5
                        
                        loss_nan_count = 0
                        print(f"  ✅ 恢复成功，学习率降低 50%")
                        continue
                
                with torch.no_grad():
                    # 用原始余弦相似度计算准确率（不含margin，与验证一致）
                    W = F.normalize(criterion.weight, p=2, dim=1)
                    cosine_acc = F.linear(embeddings, W)
                    _, predicted = cosine_acc.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % 500 == 0:
                    current_acc = 100. * train_correct / train_total
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"  Batch [{batch_idx+1}/{total_batches}] "
                          f"Loss: {loss.item():.4f} "
                          f"Acc: {current_acc:.2f}% "
                          f"LR: {current_lr:.6f}")

            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total

            model.eval()
            criterion.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    if self.use_amp:
                        with autocast('cuda'):
                            embeddings = model(images)
                    else:
                        embeddings = model(images)
                    
                    W = F.normalize(criterion.weight, p=2, dim=1)
                    cosine = F.linear(embeddings, W)
                    _, predicted = cosine.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = 100. * val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
                  f"Val Acc: {val_acc:.2f}% LR: {current_lr:.6f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                checkpoint_path = self.save_model(model, num_classes, self.class_to_idx, criterion, epoch)
                last_good_checkpoint = checkpoint_path  # 记录好的 checkpoint
                print(f"  -> Best model saved (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            
            # 每个 epoch 结束后保存训练断点（用于续训）
            self.save_checkpoint(model, criterion, optimizer, scheduler, epoch, 
                               best_val_acc, history, num_classes, self.class_to_idx,
                               gradient_clip_max_norm, patience_counter, scaler)

        # 训练完成后删除断点文件
        final_checkpoint = os.path.join(self.model_dir, "checkpoint.pth")
        if os.path.exists(final_checkpoint):
            os.remove(final_checkpoint)
            print("训练完成，已清理断点文件")

        # 计算最佳阈值
        best_threshold = self._find_best_threshold(model, val_loader, criterion)
        print(f"最佳识别阈值: {best_threshold:.4f}")

        # 保存阈值到 model_info.json
        info_path = os.path.join(self.model_dir, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            info['best_threshold'] = best_threshold
            info['best_val_acc'] = best_val_acc
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)

        history_path = os.path.join(self.model_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        print("=" * 60)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        print("=" * 60)

        return history

    def _find_best_threshold(self, model, val_loader, criterion):
        """在验证集上搜索最佳识别阈值（最大化准确率）"""
        import torch.nn.functional as F

        model.eval()
        criterion.eval()

        all_cosines = []  # 所有余弦相似度
        all_labels = []   # 标签（是否同一人）

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                embeddings = model(images)
                W = F.normalize(criterion.weight, p=2, dim=1)
                cosine_scores = F.linear(embeddings, W)  # [B, num_classes]

                # 对每个样本，取其真实类别的余弦分数（正样本）
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    pos_score = cosine_scores[i, true_label].item()
                    # 负样本：除真实类别外的最大分数
                    neg_scores = cosine_scores[i].clone()
                    neg_scores[true_label] = -1.0
                    neg_score = neg_scores.max().item()

                    all_cosines.append(pos_score)
                    all_labels.append(1)  # 正样本

                    all_cosines.append(neg_score)
                    all_labels.append(0)  # 负样本

        all_cosines = np.array(all_cosines)
        all_labels = np.array(all_labels)

        # 搜索最佳阈值
        best_acc = 0.0
        best_threshold = 0.40  # 默认值

        for threshold in np.arange(0.1, 0.95, 0.01):
            pred_pos = all_cosines >= threshold
            acc = (pred_pos == (all_labels == 1)).mean()
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        print(f"  阈值搜索结果: threshold={best_threshold:.4f}, acc={best_acc*100:.2f}%")
        return float(best_threshold)

    def save_model(self, model, num_classes, class_to_idx, criterion, epoch):
        """保存模型"""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'arcface_weight': criterion.weight.data,
            'num_classes': num_classes,
            'embedding_size': 128,
            'class_to_idx': class_to_idx,
            'idx_to_class': {v: k for k, v in class_to_idx.items()},
            'epoch': epoch
        }
        model_path = os.path.join(self.model_dir, "best_model.pth")
        torch.save(checkpoint, model_path)

        inference_checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_classes': num_classes,
            'embedding_size': 128,
            'class_to_idx': class_to_idx,
            'idx_to_class': {v: k for k, v in class_to_idx.items()}
        }
        torch.save(inference_checkpoint, os.path.join(self.model_dir, "inference_model.pth"))

        arcface_path = os.path.join(self.model_dir, "arcface_weight.pth")
        torch.save(criterion.weight.data, arcface_path)

        info_path = os.path.join(self.model_dir, "model_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_classes': num_classes,
                'embedding_size': 128,
                'class_to_idx': class_to_idx,
                'idx_to_class': {str(v): k for k, v in class_to_idx.items()}
            }, f, indent=2, ensure_ascii=False)
        
        return model_path  # 返回保存路径

    def save_history(self, history):
        """保存训练历史"""
        history_path = os.path.join(self.model_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        return history_path


def get_next_version(model_dir, prefix="optimal"):
    """获取下一个版本号"""
    import glob
    os.makedirs(model_dir, exist_ok=True)
    existing_versions = glob.glob(os.path.join(model_dir, f"{prefix}_v*"))
    if not existing_versions:
        return f"{prefix}_v1"
    
    version_nums = []
    for v in existing_versions:
        try:
            basename = os.path.basename(v)
            num = int(basename.split("_v")[1])
            version_nums.append(num)
        except (ValueError, IndexError):
            continue
    
    if not version_nums:
        return f"{prefix}_v1"
    
    return f"{prefix}_v{max(version_nums) + 1}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimal Face Training for RTX 3060")
    parser.add_argument("--data", type=str, default="trainer/data/aligned")
    parser.add_argument("--model-dir", type=str, default="trainer/models/saved")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数（默认100轮）")
    parser.add_argument("--batch-size", type=int, default=128, help="批次大小（默认128）")
    parser.add_argument("--lr", type=float, default=0.01, help="峰值学习率（默认0.01）")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true", help="从上次断点继续训练")

    args = parser.parse_args()

    # 续训模式下，查找可恢复的版本目录
    if args.resume and os.path.exists(args.model_dir):
        import glob as glob_module
        version_dirs = sorted(glob_module.glob(os.path.join(args.model_dir, "optimal_v*")))
        if version_dirs:
            for vdir in reversed(version_dirs):
                ckpt = os.path.join(vdir, "checkpoint.pth")
                if os.path.exists(ckpt):
                    args.model_dir = vdir
                    print(f"🔄 找到可恢复的断点: {vdir}")
                    break
            else:
                args.model_dir = version_dirs[-1]
                print(f"📂 使用最新版本目录: {version_dirs[-1]}")

    # 续训模式下，如果 model_dir 已存在且包含断点，直接使用
    if args.resume and os.path.exists(os.path.join(args.model_dir, "checkpoint.pth")):
        print(f"📂 续训模式，模型目录: {args.model_dir}")
    else:
        # 验证数据目录
        train_dir = os.path.join(args.data, "train")
        val_dir = os.path.join(args.data, "val")
        if not os.path.exists(train_dir):
            print(f"❌ 训练集目录不存在: {train_dir}")
            print(f"请先运行预处理: python trainer/preprocess.py")
            return
        if not os.path.exists(val_dir):
            print(f"❌ 验证集目录不存在: {val_dir}")
            print(f"请先运行预处理: python trainer/preprocess.py")
            return

        # 自动创建版本文件夹
        version = get_next_version(args.model_dir, prefix="optimal")
        version_dir = os.path.join(args.model_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        args.model_dir = version_dir
        print(f"📁 模型将保存到: {version_dir}")

    trainer = OptimalTrainer(args.data, args.model_dir, use_gpu=not args.cpu)
    trainer.train(args.epochs, args.batch_size, args.lr, resume=args.resume)
    
    print(f"\n✅ 训练完成！模型保存在: {args.model_dir}")


if __name__ == "__main__":
    main()
