"""线性分类模型：Logistic 回归 / Softmax 回归（自定义 Op 风格）。

参数存放在 `params` 字典里、梯度存放在 `grads` 字典里，配合 `SimpleBatchGD`
更新；与基于 PyTorch autograd 的 `nn.Linear + nn.BCEWithLogitsLoss` 等价（但更便
于讲清楚反向传播）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .op import Op
from .activation import logistic, softmax


class Model_LR(Op):
    """Logistic 回归：`ŷ = σ(X w + b)`，输出 `[N, 1]` 的预测概率。

    `backward(labels)` 直接给出 BCE + sigmoid 的合成梯度：
    `∂R/∂w = -1/N · Xᵀ (y - ŷ)`，`∂R/∂b = -1/N · 1ᵀ (y - ŷ)`。
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.params = {
            "w": torch.zeros(input_size, 1),
            "b": torch.zeros(1),
        }
        self.grads = {}
        self.X = None
        self.outputs = None

    def forward(self, inputs):
        self.X = inputs
        score = inputs @ self.params["w"] + self.params["b"]
        self.outputs = logistic(score)
        return self.outputs

    def backward(self, labels):
        N = labels.shape[0]
        self.grads["w"] = -1 / N * (self.X.t() @ (labels - self.outputs))
        self.grads["b"] = -1 / N * (labels - self.outputs).sum()


class Model_SR(Op):
    """Softmax 回归：`ŷ = softmax(X W + b)`，输出 `[N, C]` 的预测概率分布。

    `backward(labels)` 直接给出 CrossEntropy + softmax 的合成梯度：
    `∂R/∂W = -1/N · Xᵀ (y_onehot - ŷ)`，`∂R/∂b = -1/N · 1ᵀ (y_onehot - ŷ)`。
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.params = {
            "W": torch.zeros(input_size, output_size),
            "b": torch.zeros(output_size),
        }
        self.grads = {}
        self.X = None
        self.outputs = None
        self.output_size = output_size

    def forward(self, inputs):
        self.X = inputs
        score = inputs @ self.params["W"] + self.params["b"]
        self.outputs = softmax(score)
        return self.outputs

    def backward(self, labels):
        N = labels.shape[0]
        y_onehot = F.one_hot(labels, num_classes=self.output_size).float()
        self.grads["W"] = -1 / N * (self.X.t() @ (y_onehot - self.outputs))
        self.grads["b"] = -1 / N * (y_onehot - self.outputs).sum(dim=0)
