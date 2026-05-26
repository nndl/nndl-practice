"""chap1 实践基础 sanity tests.

覆盖 chap1 notebook 涵盖的工具点：Tensor 基本操作、广播、形状操作、autograd、
nn.Module 训练循环、Dataset/DataLoader、state_dict 保存/加载。
"""
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ---- Tensor 基础 ----
def test_tensor_creation_shape_dtype():
    a = torch.tensor([1.0, 2.0, 3.0])
    assert a.shape == (3,)
    assert a.dtype == torch.float32

    d = torch.randn(2, 3)
    assert d.shape == (2, 3)


def test_dtype_cast():
    x = torch.randn(3, 4)
    assert x.to(torch.long).dtype == torch.long
    assert x.float().dtype == torch.float32


# ---- 广播 ----
def test_broadcasting_addition():
    x = torch.tensor([[1.0, 2.0, 3.0]])    # (1, 3)
    y = torch.tensor([[10.0], [20.0]])     # (2, 1)
    z = x + y
    assert z.shape == (2, 3)
    expected = torch.tensor([[11.0, 12.0, 13.0],
                             [21.0, 22.0, 23.0]])
    assert torch.allclose(z, expected)


def test_matmul_and_reduction():
    A = torch.randn(3, 4)
    B = torch.randn(4, 5)
    assert (A @ B).shape == (3, 5)

    M = torch.arange(12).reshape(3, 4).float()
    assert torch.allclose(M.sum(), torch.tensor(66.0))
    assert M.sum(dim=0).shape == (4,)
    assert M.mean(dim=1, keepdim=True).shape == (3, 1)


# ---- 形状操作 ----
def test_view_permute_squeeze():
    x = torch.arange(24)
    y = x.view(2, 3, 4)
    assert y.shape == (2, 3, 4)
    z = y.permute(2, 0, 1)
    assert z.shape == (4, 2, 3)

    a = torch.zeros(1, 3, 1)
    assert a.squeeze().shape == (3,)
    assert a.unsqueeze(0).shape == (1, 1, 3, 1)


def test_cat_stack():
    u = torch.zeros(2, 3)
    v = torch.ones(2, 3)
    assert torch.cat([u, v], dim=0).shape == (4, 3)
    assert torch.stack([u, v], dim=0).shape == (2, 2, 3)


# ---- autograd ----
def test_simple_backward():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    assert torch.allclose(x.grad, torch.tensor([2.0, 4.0, 6.0]))


def test_no_grad_context():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    with torch.no_grad():
        y = x * 2
    assert not y.requires_grad


def test_detach_disconnects_graph():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    z = (x * 3).detach()
    assert not z.requires_grad


# ---- nn.Module + 训练循环 ----
class TinyMLP(nn.Module):
    def __init__(self, in_dim=2, hid=16, out=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, out),
        )

    def forward(self, x):
        return self.net(x)


def test_tiny_mlp_trains():
    torch.manual_seed(0)
    X = torch.randn(256, 2)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(256, 1)

    model = TinyMLP()
    opt = optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    initial = loss_fn(model(X), y).item()
    for _ in range(200):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    final = loss.item()
    assert final < initial * 0.1, f"loss did not drop enough: {initial} → {final}"


# ---- Dataset / DataLoader ----
class XOR(Dataset):
    def __init__(self):
        self.X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        self.y = torch.tensor([0, 1, 1, 0])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def test_dataloader_iterates_all_samples():
    ds = XOR()
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    seen = 0
    for x, y in loader:
        seen += x.size(0)
    assert seen == len(ds)


# ---- state_dict 保存与加载 ----
def test_state_dict_roundtrip():
    torch.manual_seed(0)
    model = TinyMLP()
    X = torch.randn(8, 2)
    with torch.no_grad():
        ref = model(X)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tinymlp.pt"
        torch.save(model.state_dict(), p)

        model2 = TinyMLP()
        # 加载前先验证 model2 和 model 输出不同（参数未对齐）
        with torch.no_grad():
            assert not torch.allclose(model2(X), ref, atol=1e-4)

        model2.load_state_dict(torch.load(p, weights_only=True))
        with torch.no_grad():
            assert torch.allclose(model2(X), ref)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
