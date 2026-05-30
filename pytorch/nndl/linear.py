"""线性算子与最小二乘法求解器。

行向量样本约定：`X ∈ R^{N×D}`，模型 `y = X w + b`。
"""
from __future__ import annotations

import torch

from .op import Op


class Linear(Op):
    """线性算子 `y = X w + b`（参数存放在 `params` 字典里）。

    参数：
    - `input_size`：特征维度 `D`。

    属性：
    - `params['w']`：权重矩阵 `[D, 1]`（一元/单输出回归用）；如需多输出可在外部
      替换为 `[D, C]`。
    - `params['b']`：偏置 `[1]`。
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size
        self.params = {
            "w": torch.zeros(input_size, 1),
            "b": torch.zeros(1),
        }

    def forward(self, X):
        assert X.shape[1] == self.input_size
        return X @ self.params["w"] + self.params["b"]


def optimizer_lsm(model: Linear, X, y, reg_lambda: float = 0.0):
    """最小二乘法（可带 L2 正则化）求解 Linear 算子的参数。

    解法：对均方误差关于 `b` 求偏导得到 `b* = ȳ - x̄ᵀ w*`；代回再对 `w` 求偏导得到

        w* = (XᵀX + λI)⁻¹ Xᵀ (y - ȳ)

    其中 `X` 已经减去均值，**偏置不参与正则化**。

    结果原地写入 `model.params['w']` 与 `model.params['b']`。
    """
    D = X.shape[1]
    x_bar = X.mean(dim=0, keepdim=True)
    y_bar = y.mean()
    x_sub = X - x_bar
    A = x_sub.T @ x_sub + reg_lambda * torch.eye(D, dtype=X.dtype, device=X.device)
    rhs = x_sub.T @ (y - y_bar)
    w = torch.linalg.solve(A, rhs)
    b = y_bar - x_bar @ w
    model.params["w"] = w
    model.params["b"] = b.squeeze(0)
    return model
