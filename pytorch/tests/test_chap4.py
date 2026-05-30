"""chap4 前馈神经网络 sanity tests.

覆盖：
- moons 上 2-层 MLP 测试准确率 > 0.95
- iris 上 RunnerV3 best-checkpoint 能恢复到比训练末尾更好（或相当）的 dev metric
- 手算反向梯度与 autograd 在 1e-6 量级一致
"""
import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from nndl.runner import RunnerV3


# ---- make_moons：与 notebook 同一来源（sklearn 自带）----
from sklearn.datasets import make_moons as _sk_make_moons

def make_moons(n=400, noise=0.15, seed=0):
    X_np, y_np = _sk_make_moons(n_samples=n, noise=noise, random_state=seed)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return X, y


def test_moons_mlp_high_accuracy():
    torch.manual_seed(0)
    X_all, y_all = make_moons(n=400, noise=0.15)
    X_tr, y_tr = X_all[:300], y_all[:300]
    X_te, y_te = X_all[300:], y_all[300:]

    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    opt = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(500):
        loss = loss_fn(model(X_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        acc = ((torch.sigmoid(model(X_te)) >= 0.5).float() == y_te).float().mean().item()
    assert acc > 0.95, f"moons MLP acc too low: {acc}"


# ---- manual backward vs autograd ----
def test_manual_backward_matches_autograd():
    torch.manual_seed(0)
    x = torch.randn(3)
    y = torch.tensor(1.0)
    W1 = torch.randn(4, 3); b1 = torch.zeros(4)
    W2 = torch.randn(4);    b2 = torch.zeros(())

    # forward
    z1 = W1 @ x + b1
    a1 = torch.sigmoid(z1)
    z2 = W2 @ a1 + b2
    a2 = torch.sigmoid(z2)

    # manual
    dz2 = a2 - y
    dW2 = dz2 * a1
    db2 = dz2
    da1 = dz2 * W2
    dz1 = da1 * a1 * (1 - a1)
    dW1 = dz1.unsqueeze(1) * x.unsqueeze(0)
    db1 = dz1

    # autograd
    W1a = W1.clone().requires_grad_()
    b1a = b1.clone().requires_grad_()
    W2a = W2.clone().requires_grad_()
    b2a = b2.clone().requires_grad_()
    a2a = torch.sigmoid(W2a @ torch.sigmoid(W1a @ x + b1a) + b2a)
    F.binary_cross_entropy(a2a, y).backward()

    assert (dW1 - W1a.grad).abs().max().item() < 1e-6
    assert (db1 - b1a.grad).abs().max().item() < 1e-6
    assert (dW2 - W2a.grad).abs().max().item() < 1e-6
    assert (db2 - b2a.grad).abs().max().item() < 1e-6


# ---- RunnerV3 best-model logic ----
def test_runner_best_checkpoint_recoverable():
    """训练后加载 best.pt，其 dev metric 应当 >= 训练任一 epoch 末的 dev metric。"""
    from sklearn.datasets import load_iris
    torch.manual_seed(0)
    X, y = load_iris(return_X_y=True)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    perm = torch.randperm(len(X))
    Xtr, ytr = X[perm[:90]], y[perm[:90]]
    Xdv, ydv = X[perm[90:120]], y[perm[90:120]]
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr = (Xtr - mu) / sd; Xdv = (Xdv - mu) / sd

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=16, shuffle=True)
    dev_loader   = DataLoader(TensorDataset(Xdv, ydv), batch_size=32)

    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
    runner = RunnerV3(
        model, optim.Adam(model.parameters(), lr=0.01),
        nn.CrossEntropyLoss(),
        metric_fn=lambda l, y: (l.argmax(1) == y).float().mean().item(),
        higher_is_better=True,
    )
    with tempfile.TemporaryDirectory() as td:
        best_path = Path(td) / "iris_best.pt"
        runner.fit(train_loader, dev_loader, num_epochs=80, log_every=200,
                   best_path=str(best_path))
        best_observed = max(runner.history["dev_metric"])
        # 训练结束的模型（可能已经退化）
        _, end_acc = runner._eval(dev_loader)
        # 加载 best.pt
        model.load_state_dict(torch.load(str(best_path), weights_only=True))
        _, loaded_acc = runner._eval(dev_loader)

    assert loaded_acc >= end_acc - 1e-9, f"loaded best {loaded_acc} < end {end_acc}"
    assert loaded_acc >= best_observed - 1e-9
    assert loaded_acc > 0.85, f"iris dev acc too low: {loaded_acc}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
