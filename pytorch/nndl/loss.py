"""损失函数：MSE、二分类交叉熵、多分类交叉熵。

`BinaryCrossEntropyLoss` / `MultiCrossEntropyLoss` 设计上跟 `Model_LR` / `Model_SR`
配套——只算 `forward`，反向梯度由 `Model_*.backward(labels)` 手动写出（合成的
Sigmoid + BCE / Softmax + CE 梯度极简）。
"""
from __future__ import annotations

import torch

from .op import Op


def mean_squared_error(y_true, y_pred):
    """均方误差。`y_true` 和 `y_pred` 形状一致。"""
    assert y_true.shape[0] == y_pred.shape[0]
    return ((y_true - y_pred) ** 2).mean()


# 别名，跟 chap1/2 inline 习惯保持一致
mse = mean_squared_error


class BinaryCrossEntropyLoss(Op):
    """二分类交叉熵：`predicts ∈ (0, 1)^{N×1}`、`labels ∈ {0, 1}^{N×1}`。

    `R(w, b) = -1/N · (yᵀ log ŷ + (1-y)ᵀ log(1-ŷ))`

    实现里对 `predicts` 做了 `clamp(eps, 1-eps)` 避免 `log(0)`。
    """

    def __init__(self):
        super().__init__()
        self.predicts = None
        self.labels = None

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        N = predicts.shape[0]
        eps = 1e-9
        p = predicts.clamp(eps, 1 - eps)
        loss = -1 / N * (labels.t() @ torch.log(p) + (1 - labels).t() @ torch.log(1 - p))
        return loss.squeeze()


class MultiCrossEntropyLoss(Op):
    """多分类交叉熵：`predicts ∈ (0, 1)^{N×C}`、`labels ∈ {0,…,C-1}^N`（长整型）。

    由于 `y` 是 one-hot，损失等价于「正确类别预测概率的负对数均值」：

        R(W, b) = -1/N · Σ log [ŷ⁽ⁿ⁾]_{y⁽ⁿ⁾}
    """

    def __init__(self):
        super().__init__()
        self.predicts = None
        self.labels = None

    def forward(self, predicts, labels):
        self.predicts = predicts
        self.labels = labels
        N = predicts.shape[0]
        eps = 1e-12
        p_correct = predicts[torch.arange(N), labels].clamp(eps, 1.0)
        return -torch.log(p_correct).mean()
