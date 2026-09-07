"""chap5 卷积神经网络 sanity tests.

覆盖：
- 朴素 conv2d 与 nn.Conv2d 数值一致
- LeNet5 前向输出形状正确，参数量 ~62k
- ResBlock / PlainBlock 同输入下形状一致，参数量相近
- LeNet5 在合成 32x32 数据上能拟合（loss 下降）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _conv2d_naive(X, W, stride=1, padding=0):
    if padding:
        X = F.pad(X, (padding,)*4)
    H, Wd = X.shape; kH, kW = W.shape
    H_out = (H - kH) // stride + 1
    W_out = (Wd - kW) // stride + 1
    out = torch.zeros(H_out, W_out)
    for i in range(H_out):
        for j in range(W_out):
            h, w = i * stride, j * stride
            out[i, j] = (X[h:h+kH, w:w+kW] * W).sum()
    return out


def test_naive_conv_matches_nn_conv2d():
    torch.manual_seed(0)
    x = torch.arange(25, dtype=torch.float32).reshape(5, 5)
    k = torch.tensor([[1., 0, -1], [1, 0, -1], [1, 0, -1]])
    manual = _conv2d_naive(x, k)

    conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)
    with torch.no_grad():
        conv.weight.copy_(k.reshape(1, 1, 3, 3))
    out = conv(x.reshape(1, 1, 5, 5)).squeeze()
    assert (manual - out).abs().max().item() < 1e-6


def _notebook_classes(part):
    """Load class definitions only, so tests exercise the teaching code without training cells."""
    import ast
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / f"chap5卷积神经网络/卷积神经网络-{part}.ipynb"
    namespace = {"torch": torch, "nn": nn, "F": F}
    for cell in json.loads(path.read_text(encoding="utf8"))["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in {"Conv2d", "Pool2D", "Model_LeNet", "ResBlock"}]
        if classes:
            exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def LeNet5(n_class=10):
    return _notebook_classes("上")["Model_LeNet"](in_channels=1, num_classes=n_class)


def test_lenet_shapes_and_params():
    m = LeNet5()
    out = m(torch.zeros(2, 1, 32, 32))
    assert tuple(out.shape) == (2, 10)
    n_params = sum(p.numel() for p in m.parameters())
    assert 60_000 < n_params < 65_000, f"unexpected param count: {n_params}"


def test_lenet_trains_on_synthetic():
    """Two clusters of 32x32 noise — one class around 0, one around 0.5; LeNet should reach >90% on train."""
    torch.manual_seed(0)
    n = 100
    X = torch.cat([torch.randn(n, 1, 32, 32) * 0.1,
                   torch.randn(n, 1, 32, 32) * 0.1 + 0.5])
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    # Two-class output enough
    model = LeNet5(n_class=2)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for _ in range(30):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
        history.append(loss.item())
    assert history[-1] < history[0] * 0.3, f"loss didn't drop: {history[0]} → {history[-1]}"
    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean().item()
    assert acc > 0.95, f"train acc too low: {acc}"


def _PlainBlock(in_ch, out_ch, stride=1):
    return _notebook_classes("下")["ResBlock"](in_ch, out_ch, stride, use_residual=False)


def _ResBlock(in_ch, out_ch, stride=1):
    return _notebook_classes("下")["ResBlock"](in_ch, out_ch, stride, use_residual=True)


def test_rectangular_pooling_matches_values_and_gradients():
    pool_class = _notebook_classes("上")["Pool2D"]
    torch.manual_seed(91)
    for mode in ("max", "avg"):
        x = torch.randn(2, 3, 5, 8, dtype=torch.double, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        actual = pool_class((2, 3), mode, 2)(x)
        expected = getattr(F, mode + "_pool2d")(y, (2, 3), stride=2)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward(); expected.sum().backward()
        torch.testing.assert_close(x.grad, y.grad)


def test_residual_toggle_preserves_initial_parameters():
    block = _notebook_classes("下")["ResBlock"]
    torch.manual_seed(3)
    plain = block(3, 5, stride=2, use_residual=False)
    torch.manual_seed(3)
    residual = block(3, 5, stride=2, use_residual=True)
    assert plain.state_dict().keys() == residual.state_dict().keys()
    for name, value in plain.state_dict().items():
        torch.testing.assert_close(value, residual.state_dict()[name])


def test_blocks_shapes():
    x = torch.randn(2, 16, 8, 8)
    for Block in (_PlainBlock, _ResBlock):
        b1 = Block(16, 16, stride=1)
        assert tuple(b1(x).shape) == (2, 16, 8, 8)
        b2 = Block(16, 32, stride=2)
        assert tuple(b2(x).shape) == (2, 32, 4, 4)


def test_resblock_skip_identity_when_zero_residual():
    """If both conv weights are zero, ResBlock with matching shapes reduces to ReLU(shortcut(x)) = ReLU(x)."""
    block = _ResBlock(16, 16, stride=1)
    for p in [block.conv1.weight, block.conv2.weight]:
        nn.init.zeros_(p)
    block.eval()                       # BN in eval mode uses running stats (initialized to identity-like)
    x = torch.randn(1, 16, 4, 4)
    out = block(x)
    # In eval, BN(0)=0 (running_mean=0, running_var=1, conv outputs are 0 -> bn outputs are 0)
    # so out = ReLU(0 + x) = ReLU(x)
    assert torch.allclose(out, F.relu(x))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
