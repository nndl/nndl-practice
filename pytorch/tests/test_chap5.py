"""chap5 卷积神经网络 sanity tests.

覆盖：
- 朴素 conv2d 与 nn.Conv2d 数值一致
- LeNet5 前向输出形状正确，参数量 ~62k
- ResBlock / PlainBlock 同输入下形状一致，参数量相近
- LeNet5 在合成 28x28 数据上能拟合（loss 下降）
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


class LeNet5(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def test_lenet_shapes_and_params():
    m = LeNet5()
    out = m(torch.zeros(2, 1, 28, 28))
    assert tuple(out.shape) == (2, 10)
    n_params = sum(p.numel() for p in m.parameters())
    assert 60_000 < n_params < 65_000, f"unexpected param count: {n_params}"


def test_lenet_trains_on_synthetic():
    """Two clusters of 28x28 noise — one class around 0, one around 0.5; LeNet should reach >90% on train."""
    torch.manual_seed(0)
    n = 100
    X = torch.cat([torch.randn(n, 1, 28, 28) * 0.1,
                   torch.randn(n, 1, 28, 28) * 0.1 + 0.5])
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


class _PlainBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(x)))


class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = (nn.Identity() if (in_ch == out_ch and stride == 1)
                         else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_ch)))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


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
