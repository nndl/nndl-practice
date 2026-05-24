"""chap3 线性模型 sanity tests.

跑通三种线性回归求解方式并检查它们彼此一致；分类部分检查 BCE/CE loss 能稳定收敛到合理精度。
"""
import torch
import torch.nn as nn
import torch.optim as optim


def _linear_data(n=200, w=2.0, b=1.0, noise=0.3, seed=0):
    torch.manual_seed(seed)
    x = torch.linspace(-3, 3, n).unsqueeze(1)
    y = w * x + b + noise * torch.randn_like(x)
    return x, y


def test_closed_form_recovers_params():
    x, y = _linear_data()
    X = torch.cat([x, torch.ones_like(x)], dim=1)
    w = torch.linalg.pinv(X) @ y
    assert abs(w[0].item() - 2.0) < 0.1
    assert abs(w[1].item() - 1.0) < 0.1


def test_gd_matches_closed_form():
    x, y = _linear_data()
    X = torch.cat([x, torch.ones_like(x)], dim=1)
    w_closed = (torch.linalg.pinv(X) @ y).squeeze()

    w = torch.zeros(2, 1)
    for _ in range(1000):
        grad = (X.T @ (X @ w - y)) * 2.0 / x.size(0)
        w = w - 0.05 * grad
    diff = (w.squeeze() - w_closed).abs().max().item()
    assert diff < 1e-2, f"GD diverged from closed-form: {diff}"


def test_nn_linear_trains():
    x, y = _linear_data()
    model = nn.Linear(1, 1)
    opt = optim.SGD(model.parameters(), lr=0.05)
    for _ in range(500):
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.2


def _two_clusters(n=200, seed=0):
    torch.manual_seed(seed)
    pos = torch.randn(n, 2) + torch.tensor([2.0, 2.0])
    neg = torch.randn(n, 2) + torch.tensor([-2.0, -2.0])
    X = torch.cat([pos, neg], dim=0)
    y = torch.cat([torch.ones(n), torch.zeros(n)]).unsqueeze(1)
    return X, y


def test_logistic_regression_high_accuracy():
    X, y = _two_clusters()
    model = nn.Linear(2, 1)
    opt = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(300):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (torch.sigmoid(model(X)) >= 0.5).float()
        acc = (pred == y).float().mean().item()
    assert acc > 0.95


def _three_clusters(n=150, seed=0):
    torch.manual_seed(seed)
    centers = [(0.0, 3.0), (-3.0, -2.0), (3.0, -2.0)]
    Xs, ys = [], []
    for k, (cx, cy) in enumerate(centers):
        Xs.append(torch.randn(n, 2) + torch.tensor([cx, cy]))
        ys.append(torch.full((n,), k, dtype=torch.long))
    return torch.cat(Xs), torch.cat(ys)


def test_softmax_regression_high_accuracy():
    X, y = _three_clusters()
    model = nn.Linear(2, 3)
    opt = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(500):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (model(X).argmax(dim=1) == y).float().mean().item()
    assert acc > 0.9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
