"""算子（Op）基类。

`Op` 是本书自定义算子库的统一接口：子类实现 `forward` / `backward`，由 `__call__`
分发。这里的"算子"不依赖 PyTorch autograd——`backward` 由用户手动推导并实现，便于
理解链式法则与反向传播的细节。第 1 章 1.3 节是它的设计动机。
"""
from __future__ import annotations


class Op:
    """算子基类：子类按需实现 forward / backward。

    - 实例可以像函数那样调用：`op(...)` 等价于 `op.forward(...)`，与 PyTorch
      `nn.Module` 约定一致。
    - `forward(*args, **kwargs)`：前向计算；参数数量与含义由子类约定。算子常用单
      输入 (`forward(x)`)，损失常用两输入 (`forward(logits, labels)`)。
    - `backward(*args, **kwargs)`：反向计算；通常接收上游对本算子输出的梯度，返回
      对输入或参数的梯度。模型/损失算子有时直接接收 labels，具体签名以各子类为准。

    模型类（如 `Model_LR`、`Model_SR`）一般在子类里再持有 `params` / `grads` 两个
    字典，由 backward 写入梯度、由优化器读取并更新参数。
    """

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError
