"""chap2 机器学习概述 sanity tests.

覆盖：
- 多项式回归闭式解：M=3 在 sin 上能拟合到位
- L2 正则在 M=8 上能改善测试误差
- Runner.fit() 能让训练损失下降
"""
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# --------------------------- 多项式回归核心 --------------------------- #
def _poly_basis(x, degree):
    if degree == 0:
        return torch.empty(x.shape[0], 0)
    return torch.cat([x ** k for k in range(1, degree + 1)], dim=1)


def _fit_predict(X_tr, y_tr, X_eval, degree, lam=0.0):
    Phi_t = _poly_basis(X_tr, degree)
    Phi_e = _poly_basis(X_eval, degree)
    Phi_t1 = torch.cat([Phi_t, torch.ones_like(X_tr[:, :1])], dim=1)
    Phi_e1 = torch.cat([Phi_e, torch.ones_like(X_eval[:, :1])], dim=1)
    D = Phi_t1.shape[1]
    I = torch.eye(D)
    I[-1, -1] = 0          # bias 不正则化
    A = Phi_t1.T @ Phi_t1 + lam * I
    w = torch.linalg.solve(A, Phi_t1.T @ y_tr)
    return Phi_e1 @ w


def _sin_data(n, noise=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 1, generator=g)
    y = torch.sin(2 * math.pi * x) + noise * torch.randn(n, 1, generator=g)
    return x, y


def test_poly_M3_fits_sin_better_than_M1():
    X_tr, y_tr = _sin_data(50, noise=0.1)
    X_te, y_te = _sin_data(200, noise=0.0)        # 无噪声评估真分布
    mse_m1 = ((_fit_predict(X_tr, y_tr, X_te, 1) - y_te) ** 2).mean().item()
    mse_m3 = ((_fit_predict(X_tr, y_tr, X_te, 3) - y_te) ** 2).mean().item()
    assert mse_m3 < mse_m1 * 0.5, f"M=3 should crush M=1 on sin: m1={mse_m1}  m3={mse_m3}"


def test_l2_improves_overfit_at_M8():
    # 评估在 sin 真分布上密集采样的"干净"surface 上：单纯比较拟合曲线 vs 真函数。
    X_tr, y_tr = _sin_data(15, noise=0.5)
    X_eval = torch.linspace(0, 1, 500).unsqueeze(1)
    y_eval = torch.sin(2 * math.pi * X_eval)
    mse_no  = ((_fit_predict(X_tr, y_tr, X_eval, 8, lam=0.0) - y_eval) ** 2).mean().item()
    mse_reg = ((_fit_predict(X_tr, y_tr, X_eval, 8, lam=0.1) - y_eval) ** 2).mean().item()
    assert mse_reg < mse_no, f"L2 reg should help on M=8: no={mse_no}  reg={mse_reg}"


# ----------------------------- Runner -------------------------------- #
class Runner:
    """与 notebook 中 Runner 类的最小镜像，便于独立测试。"""
    def __init__(self, model, optimizer, loss_fn, metric_fn=None):
        self.model, self.optimizer, self.loss_fn = model, optimizer, loss_fn
        self.metric_fn = metric_fn

    def fit(self, train_loader, num_epochs=20):
        history = []
        for _ in range(num_epochs):
            self.model.train()
            running, n = 0.0, 0
            for x, y in train_loader:
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(x), y)
                loss.backward()
                self.optimizer.step()
                running += loss.item() * x.size(0)
                n += x.size(0)
            history.append(running / n)
        return history

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        fn = self.metric_fn or self.loss_fn
        total, n = 0.0, 0
        for x, y in loader:
            total += fn(self.model(x), y).item() * x.size(0)
            n += x.size(0)
        return total / n


def test_runner_reduces_loss():
    torch.manual_seed(0)
    N, D = 200, 4
    X = torch.randn(N, D)
    true_w = torch.randn(D, 1)
    y = X @ true_w + 0.1 * torch.randn(N, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    model = nn.Linear(D, 1)
    runner = Runner(model, optim.Adam(model.parameters(), lr=0.05), nn.MSELoss())
    history = runner.fit(loader, num_epochs=50)

    assert history[-1] < history[0] * 0.1, f"loss didn't drop: {history[0]} → {history[-1]}"
    assert history[-1] < 0.1, f"final loss too high: {history[-1]}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
