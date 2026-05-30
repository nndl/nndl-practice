"""Op 风格的三层 MLP 及其组件（对应 chap7 正则化小节）。

延续 chap1 的 `Op` 接口：手写前向/反向，参数与梯度分别存在各层的 `params` / `grads`
字典里，搭配 `nndl.optim.BatchGD`（遍历 `model.layers` 更新各层参数）使用。各章
notebook 在首次引入处内联展示完整实现，本模块提供等价的工程化版本供复用。
"""
from __future__ import annotations

import torch

from .op import Op


class ReLU(Op):
    """ReLU 算子：前向逐元素取正；反向按“前向是否大于 0”把上游梯度透传或置零。"""

    def __init__(self):
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs.clamp(min=0)

    def backward(self, outputs_grads):
        return outputs_grads * (self.inputs > 0).to(outputs_grads.dtype)


class LinearLayer(Op):
    """线性层算子：z = X·W + b，手写前向/反向，参数梯度写入 `self.grads`。"""

    def __init__(self, input_size, output_size, name="fc"):
        super().__init__()
        self.name = name
        self.params = {
            "W": torch.randn(input_size, output_size),
            "b": torch.zeros(output_size),
        }
        self.grads = {}
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return torch.matmul(inputs, self.params["W"]) + self.params["b"]

    def backward(self, grads):
        self.grads["W"] = torch.matmul(self.inputs.t(), grads)
        self.grads["b"] = torch.sum(grads, dim=0)
        return torch.matmul(grads, self.params["W"].t())


class MLP_3L(Op):
    """三层 MLP：两组 Linear+ReLU，末层 Linear 直接输出分数（logits）。

    参数分散在各 `LinearLayer` 中（见 `self.layers`），需配合
    `nndl.optim.BatchGD` 这类遍历 `layers` 的优化器更新。

    `layers_size`：长度为 4 的列表 `[输入维, 隐藏1, 隐藏2, 输出维]`。
    """

    def __init__(self, layers_size):
        self.fc1 = LinearLayer(layers_size[0], layers_size[1], name="fc1")
        self.act_fn1 = ReLU()
        self.fc2 = LinearLayer(layers_size[1], layers_size[2], name="fc2")
        self.act_fn2 = ReLU()
        self.fc3 = LinearLayer(layers_size[2], layers_size[3], name="fc3")
        self.layers = [self.fc1, self.act_fn1, self.fc2, self.act_fn2, self.fc3]

    def forward(self, X):
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        a2 = self.act_fn2(z2)
        z3 = self.fc3(a2)
        return z3

    def backward(self, loss_grad_z3):
        g = self.fc3.backward(loss_grad_z3)
        g = self.act_fn2.backward(g)
        g = self.fc2.backward(g)
        g = self.act_fn1.backward(g)
        self.fc1.backward(g)
