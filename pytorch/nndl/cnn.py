"""卷积神经网络通用模块：LeNet-5、Plain/Residual block、Net 工厂。

这些模块是第 5 章定义的"经典 CNN 积木"，跨章节复用（chap7 重新拿 LeNet-5 做 BS /
weight init / BN ablation）。所有模块都基于 `nn.Module`，用 PyTorch autograd。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet-5 经典手写数字识别架构（输入 28×28 灰度图，10 类）。

    结构：
        Conv(1→6, 5×5, pad=2) → Pool → Conv(6→16, 5×5) → Pool
        → FC(400→120) → FC(120→84) → FC(84→n_class)
    """

    def __init__(self, n_class: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # 28x28 -> 28x28
        self.pool1 = nn.MaxPool2d(2, 2)                          # -> 14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # -> 10x10
        self.pool2 = nn.MaxPool2d(2, 2)                          # -> 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PlainBlock(nn.Module):
    """Plain（无残差）卷积块：Conv-BN-ReLU-Conv-BN，最后再 ReLU。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out)


class ResBlock(nn.Module):
    """Residual 块：在 PlainBlock 基础上加 shortcut `out = ReLU(F(x) + shortcut(x))`。

    若维度不匹配（stride > 1 或 channel 改变），用 1×1 卷积调整 shortcut。
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class Net(nn.Module):
    """用一种 block 工厂搭一个浅 ResNet（默认 6 个 block，三组下采样）。

    `block` 是 `PlainBlock` 或 `ResBlock`——同一个骨架装不同 block 类型，用来对照
    残差连接对训练的影响。
    """

    def __init__(self, block, n_class: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(block(16, 16, stride=1), block(16, 16, stride=1))
        self.stage2 = nn.Sequential(block(16, 32, stride=2), block(32, 32, stride=1))
        self.stage3 = nn.Sequential(block(32, 64, stride=2), block(64, 64, stride=1))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, n_class)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
