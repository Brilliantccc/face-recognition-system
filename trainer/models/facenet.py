"""
FaceNet 模型定义
轻量级的人脸识别网络，适合嵌入式设备
"""

import sys
import os

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 使用统一的 PyTorch 工具模块
from common import torch_utils

TORCH_AVAILABLE = torch_utils.TORCH_AVAILABLE

if TORCH_AVAILABLE:
    torch = torch_utils.torch
    nn = torch_utils.nn
    F = torch.nn.functional


    class BasicConv2d(nn.Module):
        """基本卷积块"""
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))


    class InvertedResidual(nn.Module):
        """MobileNetV2 的倒残差块（修正版）"""
        def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
            super().__init__()
            self.stride = stride
            self.use_residual = (stride == 1 and in_channels == out_channels)

            hidden_dim = int(in_channels * expand_ratio)
            layers = []

            # 1. 扩展层（1x1 Conv）- 仅当 expand_ratio != 1
            if expand_ratio != 1:
                layers.extend([
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True)
                ])

            # 2. 深度可分离卷积（3x3 DW Conv）
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

            # 3. 投影层（1x1 Conv）- 必须存在，且无激活函数
            layers.extend([
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            ])

            self.conv = nn.Sequential(*layers)

        def forward(self, x):
            if self.use_residual:
                return x + self.conv(x)
            return self.conv(x)


    class MobileFaceNet(nn.Module):
        """
        MobileFaceNet - 轻量级人脸识别网络（修正版）
        适合移动端和嵌入式设备部署
        """
        def __init__(self, embedding_size=128, num_classes=None, input_size=112):
            """
            初始化 MobileFaceNet
            :param embedding_size: 嵌入向量维度
            :param num_classes: 分类类别数（训练时使用）
            :param input_size: 输入图像尺寸（默认112）
            """
            super().__init__()
            self.embedding_size = embedding_size

            # 输入层
            self.conv1 = BasicConv2d(3, 32, 3, 2, 1)

            # MobileNetV2 倒残差块
            self.inverted_blocks = nn.Sequential(
                InvertedResidual(32, 16, 1, 1),
                InvertedResidual(16, 24, 2, 6),
                InvertedResidual(24, 24, 1, 6),
                InvertedResidual(24, 32, 2, 6),
                InvertedResidual(32, 32, 1, 6),
                InvertedResidual(32, 32, 1, 6),
                InvertedResidual(32, 64, 2, 6),
                InvertedResidual(64, 64, 1, 6),
                InvertedResidual(64, 64, 1, 6),
                InvertedResidual(64, 128, 1, 6),
                InvertedResidual(128, 128, 1, 6),
                InvertedResidual(128, 128, 1, 6),
            )

            # 输出层
            self.conv2 = BasicConv2d(128, 128, 3, 2, 1)
            self.conv3 = BasicConv2d(128, 128, 3, 2, 1)

            # 使用全局平均池化替代固定尺寸全连接
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, embedding_size)

            # 分类头（训练时使用）
            self.classifier = nn.Linear(embedding_size, num_classes) if num_classes else None

        def forward(self, x):
            x = self.conv1(x)                     # 112x112 -> 56x56
            x = self.inverted_blocks(x)           # 56x56 -> ... -> 7x7
            x = self.conv2(x)                     # 7x7 -> 3x3
            x = self.conv3(x)                     # 3x3 -> 1x1 (or 2x2)

            x = self.global_avg_pool(x)           # -> 1x1
            x = x.view(x.size(0), -1)             # flatten
            x = self.fc(x)                        # -> embedding_size

            # L2 归一化
            x = F.normalize(x, p=2, dim=1)

            if self.classifier is not None:
                return x, self.classifier(x)

            return x


    class ArcFaceLoss(nn.Module):
        """
        ArcFace Loss - 角度间隔损失函数（修正版）
        提高人脸识别的区分度
        """
        def __init__(self, embedding_size, num_classes, s=32.0, m=0.20):
            """
            初始化 ArcFace Loss
            :param embedding_size: 嵌入向量维度
            :param num_classes: 类别数
            :param s: 缩放因子（默认32.0，稳定训练）
            :param m: 角度间隔（默认0.20，稳定训练）
            """
            super().__init__()
            self.s = s
            self.m = m
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
            nn.init.xavier_uniform_(self.weight)
            self.ce = nn.CrossEntropyLoss()

        def forward(self, embeddings, labels):
            # 归一化权重和嵌入
            W = F.normalize(self.weight, p=2, dim=1)
            cosine = F.linear(embeddings, W)

            # 计算角度
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

            # 添加间隔
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)

            # 更稳定的计算方式
            target_logits = torch.cos(theta + self.m * one_hot)

            # 对于非目标类，保留原始 cosine
            logits = (target_logits - cosine) * one_hot + cosine
            logits = logits * self.s

            loss = self.ce(logits, labels)
            return loss, logits  # 同时返回损失和logits，用于计算准确率


    class FocalLoss(nn.Module):
        """
        Focal Loss - 解决样本不平衡问题（修正版）
        """
        def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            return focal_loss

else:
    # PyTorch 不可用时，定义占位类
    print("[facenet] Warning: PyTorch not available, model classes are placeholders")

    class BasicConv2d:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for BasicConv2d")

    class InvertedResidual:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for InvertedResidual")

    class MobileFaceNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for MobileFaceNet")

    class ArcFaceLoss:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for ArcFaceLoss")

    class FocalLoss:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for FocalLoss")


if __name__ == "__main__" and TORCH_AVAILABLE:
    # 测试模型
    model = MobileFaceNet(embedding_size=128, num_classes=100)
    x = torch.randn(2, 3, 112, 112)

    # 训练模式
    embedding, logits = model(x)
    print(f"Embedding shape: {embedding.shape}")  # [2, 128]
    print(f"Logits shape: {logits.shape}")        # [2, 100]

    # 测试推理模式
    model.eval()
    with torch.no_grad():
        embedding_only = model(x[:1])
        print(f"Inference embedding shape: {embedding_only.shape}")  # [1, 128]
