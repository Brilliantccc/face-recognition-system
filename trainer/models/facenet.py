"""
FaceNet 模型定义
轻量级的人脸识别网络，适合嵌入式设备
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    """基本卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNetV2 的倒残差块"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            # 扩展层
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])

        # 深度可分离卷积
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # 投影层
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileFaceNet(nn.Module):
    """
    MobileFaceNet - 轻量级人脸识别网络
    适合移动端和嵌入式设备部署
    """
    def __init__(self, embedding_size=128, num_classes=None):
        """
        初始化 MobileFaceNet
        :param embedding_size: 嵌入向量维度
        :param num_classes: 分类类别数（训练时使用）
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

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 7 * 7, embedding_size)

        # 分类头（训练时使用）
        self.classifier = nn.Linear(embedding_size, num_classes) if num_classes else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.inverted_blocks(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)

        # L2 归一化
        x = F.normalize(x, p=2, dim=1)

        if self.classifier is not None and self.training:
            return x, self.classifier(x)

        return x


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss - 角度间隔损失函数
    提高人脸识别的区分度
    """
    def __init__(self, embedding_size, num_classes, s=30.0, m=0.50):
        """
        初始化 ArcFace Loss
        :param embedding_size: 嵌入向量维度
        :param num_classes: 类别数
        :param s: 缩放因子
        :param m: 角度间隔
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
        target_logits = torch.cos(theta + self.m * one_hot)

        # 缩放
        logits = target_logits * self.s

        return self.ce(logits, labels)


class FocalLoss(nn.Module):
    """
    Focal Loss - 解决样本不平衡问题
    """
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


if __name__ == "__main__":
    # 测试模型
    model = MobileFaceNet(embedding_size=128, num_classes=100)
    x = torch.randn(2, 3, 112, 112)
    embedding, logits = model(x)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Logits shape: {logits.shape}")
