from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import SimpleCNN


# =========================
# 1. 路径和设备
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)


# =========================
# 2. 数据预处理
# =========================

# RandomErasing 先在单张训练图片内随机擦除局部区域，
# DataLoader 组成 batch 后，再在训练循环中执行 CutMix。
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    ),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.2),
        ratio=(0.3, 3.3),
        value=0
    )
])

# Validation 和 Test 不使用随机增强，也不使用 CutMix。
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# =========================
# 3. Dataset
# =========================

train_full_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=train_transform
)

val_full_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=eval_transform
)

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=eval_transform
)


# =========================
# 4. 划分 Train / Validation
# =========================

total_size = len(train_full_dataset)
train_size = 45000
val_size = 5000

# 保持 baseline 的数据划分随机种子不变。
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(total_size, generator=generator).tolist()

train_indices = indices[:train_size]
val_indices = indices[train_size:]

train_dataset = Subset(train_full_dataset, train_indices)
val_dataset = Subset(val_full_dataset, val_indices)


# =========================
# 5. DataLoader
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# =========================
# 6. 模型、损失函数、优化器和调度器
# =========================

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[50, 75],
    gamma=0.1
)


# =========================
# 7. 训练参数
# =========================

num_epochs = 100
cutmix_alpha = 1.0
best_val_accuracy = 0.0


# 假设图片 A 是猫，
# 图片 B 是狗。
#
# CutMix 会把图片 B 的一个矩形区域剪下来，
# 替换图片 A 的对应区域。
#
# 如果最终图片中：
# 70% 面积来自猫，
# 30% 面积来自狗，
#
# 那么损失就是：
# loss = 0.7 * 猫对应的 loss + 0.3 * 狗对应的 loss
#
# 与 Mixup 不同：
# Mixup 是整张图片线性叠加，
# CutMix 是直接替换局部矩形区域。


def rand_bbox(size, lam):
    """根据 lambda 在图片范围内随机生成 CutMix 矩形区域。"""

    W = size[3]
    H = size[2]

    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(
        cx - cut_w // 2,
        0,
        W
    )
    bby1 = np.clip(
        cy - cut_h // 2,
        0,
        H
    )
    bbx2 = np.clip(
        cx + cut_w // 2,
        0,
        W
    )
    bby2 = np.clip(
        cy + cut_h // 2,
        0,
        H
    )

    return bbx1, bby1, bbx2, bby2


# =========================
# 8. 开始训练
# =========================

for epoch in range(num_epochs):

    # 训练阶段：只在这里使用 CutMix。
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 从 Beta(alpha, alpha) 分布采样当前 batch 的初始面积比例。
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)

        # 在当前 batch 内随机配对图片，并保留两组原始标签。
        index = torch.randperm(
            images.size(0),
            device=device
        )
        labels_a = labels
        labels_b = labels[index]

        bbx1, bby1, bbx2, bby2 = rand_bbox(
            images.size(),
            lam
        )

        # 使用 clone()，避免原地破坏当前 batch 的原始 images。
        mixed_images = images.clone()
        mixed_images[
            :,
            :,
            bby1:bby2,
            bbx1:bbx2
        ] = images[
            index,
            :,
            bby1:bby2,
            bbx1:bbx2
        ]

        # 边界裁剪可能改变实际替换面积，因此按真实面积重新计算 lambda。
        lam = 1 - (
            (bbx2 - bbx1)
            * (bby2 - bby1)
            / (
                images.size(2)
                * images.size(3)
            )
        )

        optimizer.zero_grad()
        outputs = model(mixed_images)

        # 不创建 one-hot 标签；两个 CrossEntropyLoss 按实际面积比例加权。
        loss = (
            lam * criterion(outputs, labels_a)
            + (1 - lam) * criterion(outputs, labels_b)
        )

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation 阶段完全使用原始图片和真实标签，不使用 CutMix。
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = correct / total
    average_loss = running_loss / len(train_loader)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Loss: {average_loss:.4f}, "
        f"Validation Accuracy: {val_accuracy:.4f}, "
        f"LR: {current_lr:.6f}"
    )

    # 保持原有的最佳 Validation 模型保存逻辑。
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL_PATH)
        print("保存新的最佳模型！")

    # 每个 epoch 结束后调用一次。
    scheduler.step()


# =========================
# 9. 训练结束并加载最佳模型
# =========================

print()
print("训练完成！")
print("最佳 Validation Accuracy：", best_val_accuracy)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)
model.eval()


# =========================
# 10. 最终 Test：不使用 CutMix
# =========================

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total

print()
print("=========================")
print("最终测试结果")
print("=========================")
print(f"Test Accuracy: {test_accuracy:.4f}")
