import torch
import torch.nn as nn


# ============================================================
# 1. ResNet 基本残差块
# ============================================================

class BasicBlock(nn.Module):
    """用于 CIFAR-10 ResNet-34 的 BasicBlock（不是 Bottleneck）。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # 主分支：Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 尺寸或通道数变化时使用 projection shortcut，否则直接恒等映射。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


# ============================================================
# 2. CIFAR-10 风格 ResNet-34
# ============================================================

class SimpleCNN(nn.Module):
    """保持 SimpleCNN 接口不变的 CIFAR-10 风格 ResNet-34。"""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 本实验仅将模型深度从 ResNet-18 提升到 ResNet-34。
        #
        # ResNet-18 的四个 stage 为：
        # [2, 2, 2, 2]
        #
        # ResNet-34 的四个 stage 为：
        # [3, 4, 6, 3]
        #
        # 两者都使用 BasicBlock，
        # ResNet-34 通过增加 Residual Block 数量提高网络深度和特征表达能力。
        #
        # 本实验用于观察在已经使用 Mixup + RandomErasing 的情况下，
        # 增加 ResNet 深度是否还能进一步提升 CIFAR-10 分类准确率。

        # CIFAR-10 输入只有 32x32，因此使用 3x3、stride=1 的开头卷积，
        # 不使用 ImageNet ResNet 的 7x7 卷积，也不加入开头 MaxPool。
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Stage 1：3 个 BasicBlock，保持 64x32x32。
        self.stage1 = self._make_stage(64, 64, blocks=3, first_stride=1)

        # Stage 2：首个 block stride=2，32x32 -> 16x16。
        self.stage2 = self._make_stage(64, 128, blocks=4, first_stride=2)

        # Stage 3：首个 block stride=2，16x16 -> 8x8。
        self.stage3 = self._make_stage(128, 256, blocks=6, first_stride=2)

        # Stage 4：首个 block stride=2，8x8 -> 4x4。
        self.stage4 = self._make_stage(256, 512, blocks=3, first_stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    @staticmethod
    def _make_stage(in_channels, out_channels, blocks, first_stride):
        """构建一个 stage：仅第一个 BasicBlock 负责必要的下采样。"""
        layers = [BasicBlock(in_channels, out_channels, stride=first_stride)]
        layers.extend(
            BasicBlock(out_channels, out_channels, stride=1)
            for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
