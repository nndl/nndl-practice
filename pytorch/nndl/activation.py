"""激活函数：Logistic / Softmax。

PyTorch 内置 `torch.sigmoid` / `torch.softmax` 已经做了数值稳定化，本模块保留手写
版本，用于第 3 章实现 Logistic / Softmax 回归时的教学讲解。
"""
from __future__ import annotations

import torch


def logistic(x):
    """Logistic 函数：σ(x) = 1 / (1 + exp(-x))。"""
    return 1 / (1 + torch.exp(-x))


def softmax(X):
    """数值稳定的 Softmax，输入形状 `[N, D]`，沿最后一维归一化。

    实现要点：先减去每行最大值再 `exp`，避免上溢出；分母至少含一个 `exp(0)=1`，
    避免下溢出。
    """
    x_max = X.max(dim=1, keepdim=True).values
    x_exp = torch.exp(X - x_max)
    partition = x_exp.sum(dim=1, keepdim=True)
    return x_exp / partition
