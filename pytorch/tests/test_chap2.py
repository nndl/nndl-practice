"""chap2 机器学习概述 sanity tests.

覆盖：
- 多项式回归闭式解：M=3 在 sin 上能拟合到位
- L2 正则在 M=8 上能改善测试误差
- RunnerV1 把闭式解 + 评估 + save/load 串成一条流水线
"""
import math
import tempfile
from pathlib import Path

import torch

from nndl.runner import RunnerV1


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


# ----------------------------- RunnerV1 -------------------------------- #
class _Linear:
    """与 notebook 中 Linear 算子结构等价的最小镜像。"""
    def __init__(self, input_size):
        self.params = {
            'w': torch.zeros(input_size, 1),
            'b': torch.zeros(1),
        }

    def __call__(self, X):
        return X @ self.params['w'] + self.params['b']


def _optimizer_lsm(model, X, y, reg_lambda=0.0):
    D = X.shape[1]
    x_bar = X.mean(dim=0, keepdim=True)
    y_bar = y.mean()
    x_sub = X - x_bar
    A = x_sub.T @ x_sub + reg_lambda * torch.eye(D)
    rhs = x_sub.T @ (y - y_bar)
    w = torch.linalg.solve(A, rhs)
    b = y_bar - x_bar @ w
    model.params['w'] = w
    model.params['b'] = b.squeeze(0)


def _mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def test_runnerv1_closed_form_recovers_params():
    """RunnerV1 求闭式解后能拿到与真值非常接近的 (w, b)。"""
    torch.manual_seed(0)
    N = 200
    X = torch.linspace(-3, 3, N).unsqueeze(1)
    true_w, true_b = 2.0, 1.0
    y = true_w * X + true_b + 0.1 * torch.randn(N, 1)

    runner = RunnerV1(_Linear(input_size=1), _optimizer_lsm)
    runner.fit(X, y)
    w = runner.model.params['w'].item()
    b = runner.model.params['b'].item()
    assert abs(w - true_w) < 0.05, f"w drift: {w} vs {true_w}"
    assert abs(b - true_b) < 0.05, f"b drift: {b} vs {true_b}"


def test_runnerv1_save_load_roundtrip():
    """save → load 后评估结果一致。"""
    torch.manual_seed(0)
    N, D = 200, 4
    X = torch.randn(N, D)
    y = X @ torch.randn(D, 1) + 0.1 * torch.randn(N, 1)

    runner = RunnerV1(_Linear(input_size=D), _optimizer_lsm)
    runner.fit(X, y)
    mse_before = runner.evaluate(X, y, _mse)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "linear.pt"
        runner.save(p)

        runner2 = RunnerV1(_Linear(input_size=D), _optimizer_lsm)
        runner2.load(p)
        mse_after = runner2.evaluate(X, y, _mse)

    assert abs(mse_before - mse_after) < 1e-9, f"save/load drift: {mse_before} vs {mse_after}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
