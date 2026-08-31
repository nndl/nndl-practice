"""chap3 线性模型 sanity tests.

覆盖两个 notebook 的核心训练链路：
- 线性模型-上：手写 `Model_LR` + `BinaryCrossEntropyLoss` + `SimpleBatchGD` 在 Moon 风格二分类数据上能收敛到 > 0.8 准确率
- 线性模型-下：手写 `Model_SR` + `MultiCrossEntropyLoss` + `SimpleBatchGD` 在 Multi1000 测试集上达到与书稿一致的准确率
"""
import math

import torch

from nndl import (
    Model_LR,
    Model_SR,
    BinaryCrossEntropyLoss,
    MultiCrossEntropyLoss,
    SimpleBatchGD,
    accuracy,
)
from nndl.data import make_multiclass_classification


def _moons(n=400, noise=0.2, seed=0):
    """跟 chap3-上 notebook 的 make_moons 同构的弯月形二分类数据。"""
    torch.manual_seed(seed)
    n_out = n // 2
    n_in = n - n_out
    outer_x = torch.cos(torch.linspace(0, math.pi, n_out))
    outer_y = torch.sin(torch.linspace(0, math.pi, n_out))
    inner_x = 1 - torch.cos(torch.linspace(0, math.pi, n_in))
    inner_y = 0.5 - torch.sin(torch.linspace(0, math.pi, n_in))
    X = torch.stack(
        [torch.cat([outer_x, inner_x]), torch.cat([outer_y, inner_y])], dim=1
    )
    y = torch.cat([torch.zeros(n_out, 1), torch.ones(n_in, 1)])
    X = X + noise * torch.randn_like(X)
    return X, y


def test_logistic_regression_separates_moons():
    X, y = _moons()
    model = Model_LR(input_size=2)
    optimizer = SimpleBatchGD(init_lr=0.1, model=model)
    loss_fn = BinaryCrossEntropyLoss()

    for _ in range(500):
        preds = model(X)
        _ = loss_fn(preds, y)
        model.backward(y)
        optimizer.step()

    acc = accuracy(model(X), y)
    assert acc > 0.8, f"Logistic 回归在弯月数据上准确率应 > 0.8，实际 {acc:.3f}"


def _multi1000(seed=287):
    """复现 chap3-下 Notebook 与书稿使用的 Multi1000 参数。"""
    return make_multiclass_classification(
        n_samples=1000, n_features=2, n_classes=3,
        noise=0.15, label_noise=0.0, seed=seed)


def test_softmax_regression_separates_three_clusters():
    X, y = _multi1000()
    X_train, y_train = X[:640], y[:640]
    X_test, y_test = X[800:], y[800:]
    model = Model_SR(input_size=2, output_size=3)
    optimizer = SimpleBatchGD(init_lr=0.1, model=model)
    loss_fn = MultiCrossEntropyLoss()

    for _ in range(500):
        preds = model(X_train)
        _ = loss_fn(preds, y_train)
        model.backward(y_train)
        optimizer.step()

    acc = accuracy(model(X_test), y_test)
    assert acc > 0.70, f"Softmax 回归在 Multi1000 测试集准确率应 > 0.70，实际 {acc:.3f}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
