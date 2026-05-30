"""优化器：基类 + 简单批量梯度下降。

跟 `nndl.classify.Model_LR` / `Model_SR` 这种"参数存在 `params` 字典里、梯度存在
`grads` 字典里"的自定义算子配套使用。需要 PyTorch autograd 的话直接用
`torch.optim.*` 即可。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """优化器基类。子类实现 `step()`，从 `model.grads` 读梯度更新 `model.params`。"""

    def __init__(self, init_lr, model):
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        raise NotImplementedError


class SimpleBatchGD(Optimizer):
    """全批量梯度下降：`θ ← θ - α · ∂R/∂θ`。"""

    def step(self):
        if isinstance(self.model.params, dict):
            for key in self.model.params.keys():
                self.model.params[key] = (
                    self.model.params[key] - self.init_lr * self.model.grads[key]
                )


class BatchGD(Optimizer):
    """支持多层模型的全批量梯度下降：遍历 `model.layers`，对每层 `params` 字典里的
    参数按 `θ ← θ - α·∂R/∂θ` 更新。适用于参数分散在各层的模型（如 `nndl.mlp.MLP_3L`）。"""

    def step(self):
        for layer in self.model.layers:
            if hasattr(layer, "params") and isinstance(layer.params, dict):
                for key in layer.params.keys():
                    layer.params[key] = (
                        layer.params[key] - self.init_lr * layer.grads[key]
                    )
