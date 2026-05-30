"""nndl 工具包的 sanity 测试：算子 / 模型 / loss / metric / 数据集等公共 API。

只验证形状和数学正确性，不跑训练。完整端到端训练在 test_chap2 / test_chap4 里覆盖。
"""
import math

import torch
import torch.nn as nn

import nndl


# ---- Op / Linear / optimizer_lsm ---------------------------------------- #
def test_linear_forward_shape():
    m = nndl.Linear(input_size=4)
    out = m(torch.randn(3, 4))
    assert out.shape == (3, 1)


def test_optimizer_lsm_recovers_params():
    torch.manual_seed(0)
    N = 200
    X = torch.linspace(-3, 3, N).unsqueeze(1)
    y = 2 * X + 1 + 0.1 * torch.randn(N, 1)
    m = nndl.Linear(input_size=1)
    nndl.optimizer_lsm(m, X, y)
    assert abs(m.params["w"].item() - 2.0) < 0.05
    assert abs(m.params["b"].item() - 1.0) < 0.05


def test_optimizer_lsm_ridge_shrinks_w():
    """L2 正则越大，求出的 |w| 应该越小。"""
    torch.manual_seed(0)
    N, D = 100, 5
    X = torch.randn(N, D)
    y = X @ torch.randn(D, 1) + 0.1 * torch.randn(N, 1)
    m0 = nndl.Linear(input_size=D); nndl.optimizer_lsm(m0, X, y, reg_lambda=0.0)
    m1 = nndl.Linear(input_size=D); nndl.optimizer_lsm(m1, X, y, reg_lambda=10.0)
    assert m1.params["w"].norm() < m0.params["w"].norm()


# ---- Activation ---------------------------------------------------------- #
def test_logistic_softmax():
    x = torch.tensor([0.0, 100.0, -100.0])
    out = nndl.logistic(x)
    assert torch.allclose(out, torch.tensor([0.5, 1.0, 0.0]), atol=1e-3)

    sm = nndl.softmax(torch.tensor([[1000.0, 1000.0, 1000.0]]))   # 数值稳定性
    assert torch.allclose(sm, torch.full((1, 3), 1.0 / 3), atol=1e-4)


# ---- Model_LR / Model_SR ------------------------------------------------ #
def test_model_lr_forward_backward():
    torch.manual_seed(0)
    model = nndl.Model_LR(input_size=4)
    X = torch.randn(3, 4)
    y = torch.tensor([[1.0], [0.0], [1.0]])
    out = model(X)
    assert out.shape == (3, 1)
    assert torch.all(out >= 0) and torch.all(out <= 1)
    # 全 0 初始化时 forward 全是 0.5
    assert torch.allclose(out, torch.full((3, 1), 0.5))
    model.backward(y)
    assert "w" in model.grads and "b" in model.grads
    assert model.grads["w"].shape == (4, 1)


def test_model_sr_forward_backward():
    torch.manual_seed(0)
    model = nndl.Model_SR(input_size=4, output_size=3)
    X = torch.randn(2, 4)
    labels = torch.tensor([0, 2])
    out = model(X)
    assert out.shape == (2, 3)
    # 全 0 初始化时每类概率 1/3
    assert torch.allclose(out, torch.full((2, 3), 1.0 / 3), atol=1e-6)
    model.backward(labels)
    assert model.grads["W"].shape == (4, 3)
    assert model.grads["b"].shape == (3,)


# ---- Loss ---------------------------------------------------------------- #
def test_bce_loss():
    bce = nndl.BinaryCrossEntropyLoss()
    p = torch.tensor([[0.5], [0.5]])
    y = torch.tensor([[1.0], [0.0]])
    loss = bce(p, y)
    # -log(0.5) ≈ 0.6931
    assert abs(loss.item() - math.log(2)) < 1e-4


def test_mce_loss():
    mce = nndl.MultiCrossEntropyLoss()
    p = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
    y = torch.tensor([0, 1])
    loss = mce(p, y)
    expected = -(math.log(0.5) + math.log(0.6)) / 2
    assert abs(loss.item() - expected) < 1e-5


def test_mse():
    y_true = torch.tensor([1.0, 2.0, 3.0])
    y_pred = torch.tensor([1.0, 2.0, 4.0])
    assert abs(nndl.mean_squared_error(y_true, y_pred).item() - 1.0 / 3) < 1e-6


# ---- Metric -------------------------------------------------------------- #
def test_accuracy_binary_and_multi():
    bin_preds = torch.tensor([[0.9], [0.2], [0.7], [0.1]])
    bin_labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert nndl.accuracy(bin_preds, bin_labels) == 0.75

    multi_logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    multi_labels = torch.tensor([1, 1, 0])
    assert abs(nndl.accuracy(multi_logits, multi_labels) - 1.0 / 3) < 1e-6


# ---- Optimizer / SimpleBatchGD ------------------------------------------ #
def test_simple_batch_gd_updates_params():
    torch.manual_seed(0)
    model = nndl.Model_LR(input_size=3)
    opt = nndl.SimpleBatchGD(init_lr=0.1, model=model)
    X = torch.randn(8, 3)
    y = torch.tensor([[1.0], [0.0]] * 4)
    out = model(X)
    model.backward(y)
    w_before = model.params["w"].clone()
    opt.step()
    w_after = model.params["w"]
    assert not torch.allclose(w_before, w_after)


# ---- CNN modules --------------------------------------------------------- #
def test_lenet5_shape():
    m = nndl.LeNet5(n_class=10)
    out = m(torch.zeros(2, 1, 28, 28))
    assert out.shape == (2, 10)


def test_plain_res_block_shape():
    pb = nndl.PlainBlock(16, 16)
    rb = nndl.ResBlock(16, 32, stride=2)
    x = torch.randn(2, 16, 8, 8)
    assert pb(x).shape == (2, 16, 8, 8)
    assert rb(x).shape == (2, 32, 4, 4)   # stride=2 下采样


def test_net_shape():
    plain_net = nndl.Net(nndl.PlainBlock)
    res_net = nndl.Net(nndl.ResBlock)
    x = torch.randn(2, 3, 32, 32)
    assert plain_net(x).shape == (2, 10)
    assert res_net(x).shape == (2, 10)


# ---- RNN modules --------------------------------------------------------- #
def test_my_srn_shape():
    m = nndl.MySRN(vocab=10, embed=16, hidden=32, n_class=19)
    out = m(torch.zeros(2, 5, dtype=torch.long))
    assert out.shape == (2, 19)


def test_my_lstm_shape():
    m = nndl.MyLSTMModel(vocab=10, embed=16, hidden=32, n_class=19)
    out = m(torch.zeros(2, 5, dtype=torch.long))
    assert out.shape == (2, 19)


# ---- Attention modules --------------------------------------------------- #
def test_additive_attention_masking():
    torch.manual_seed(0)
    att = nndl.AdditiveAttention(hidden_size=8, attn_size=16)
    H = torch.randn(2, 5, 8)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])
    ctx, w = att(H, mask)
    assert ctx.shape == (2, 8)
    assert w.shape == (2, 5)
    # row 0 全有效 → softmax 之和为 1
    assert abs(w[0].sum().item() - 1.0) < 1e-5
    # row 1 padding 位置（idx 3, 4）权重应为 0
    assert w[1, 3].item() < 1e-5
    assert w[1, 4].item() < 1e-5


def test_scaled_dot_attention_self_identity():
    """Q=K=V=I_4 时，注意力矩阵对角线应主导。"""
    X = torch.eye(4) + 0.01 * torch.randn(4, 4)
    _, A = nndl.scaled_dot_attention(X, X, X)
    diag = A.diag()
    off_diag = A - torch.diag(diag)
    assert (diag.mean() > off_diag.mean()).item()


def test_multihead_attention_shape():
    mha = nndl.MultiHeadAttention(embed_dim=32, n_heads=4)
    x = torch.randn(2, 7, 32)
    out, attn = mha(x)
    assert out.shape == (2, 7, 32)
    assert attn.shape == (2, 4, 7, 7)


def test_sinusoidal_pe_first_columns():
    pe = nndl.SinusoidalPE(d_model=4, max_len=10)
    pe_table = pe.pe
    # 第 0 维（i=0）：sin(pos / 1) = sin(pos)
    assert torch.allclose(pe_table[:5, 0], torch.sin(torch.arange(5).float()), atol=1e-6)
    # 第 1 维（i=0 的 cos 分量）：cos(pos)
    assert torch.allclose(pe_table[:5, 1], torch.cos(torch.arange(5).float()), atol=1e-6)


def test_transformer_block_shape():
    blk = nndl.TransformerBlock(d_model=32, n_heads=4, ff_dim=64, dropout=0.0)
    x = torch.randn(2, 7, 32)
    out = blk(x)
    assert out.shape == (2, 7, 32)


# ---- nndl.data ----------------------------------------------------------- #
def test_data_make_moons():
    X, y = nndl.data.make_moons(50, noise=0.1)
    assert X.shape == (50, 2)
    assert y.shape == (50,)
    assert set(y.tolist()) == {0.0, 1.0}


def test_data_make_digit_sum():
    X, y = nndl.data.make_digit_sum(n_samples=20, L=8, seed=0)
    assert X.shape == (20, 8)
    assert torch.equal(y, X.sum(dim=1))


def test_data_sentiment_dataset_collate():
    ds = nndl.data.SentimentDS(n=10, seed=0)
    assert len(ds) == 10
    x, y = ds[0]
    assert x.dim() == 1 and isinstance(y, int)
    # collate 把变长序列 pad 到同长
    batch = [ds[i] for i in range(3)]
    padded, lens, labels = nndl.data.sentiment_collate(batch)
    assert padded.dim() == 2 and padded.shape[0] == 3
    assert padded.shape[1] == lens.max().item()
    assert labels.shape == (3,)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
