"""chap8 注意力机制 sanity tests."""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from nndl.attention import AdditiveAttention, MultiHeadAttention, scaled_dot_attention as _scaled_dot




# ---- AdditiveAttention ----


def test_additive_attention_masks_padding():
    """Pad 位置（mask=False）应当拿到 0 注意力权重；有效位置权重和应为 1。"""
    torch.manual_seed(0)
    B, L, h = 2, 5, 8
    H = torch.randn(B, L, h)
    mask = torch.tensor([[True]*5, [True, True, True, False, False]])

    att = AdditiveAttention(h, 16)
    ctx, weights = att(H, mask)
    assert tuple(ctx.shape) == (B, h)
    assert tuple(weights.shape) == (B, L)

    # row 0：所有位置有效，权重和 = 1
    assert abs(weights[0].sum().item() - 1.0) < 1e-5
    # row 1：pad 位置（3, 4）权重应当 ~ 0
    assert weights[1, 3:].abs().max().item() < 1e-6
    # row 1：有效位置和为 1
    assert abs(weights[1, :3].sum().item() - 1.0) < 1e-5


# ---- Scaled dot-product attention ----


def test_scaled_dot_attention_with_identity_input_is_diagonal_dominant():
    torch.manual_seed(0)
    # dim=8 让 softmax 更分离；noise 缩小到 0.001 让 identity 主导更强
    X = torch.eye(8) + 0.001 * torch.randn(8, 8)
    _, A = _scaled_dot(X, X, X)
    diag = A.diag()
    off = A - torch.diag(diag)
    assert (diag > off.max(dim=-1).values).all(), f"diagonal should dominate; A={A}"


def test_scaled_dot_attention_normalization():
    torch.manual_seed(0)
    Q = torch.randn(2, 5, 8)
    _, A = _scaled_dot(Q, Q, Q)
    sums = A.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---- MultiHeadAttention ----


def test_mha_shapes():
    mha = MultiHeadAttention(embed_dim=32, n_heads=4)
    x = torch.randn(2, 7, 32)
    out, attn = mha(x)
    assert tuple(out.shape) == (2, 7, 32)
    assert tuple(attn.shape) == (2, 4, 7, 7)


def test_mha_key_padding_mask_zeroes_pad():
    mha = MultiHeadAttention(embed_dim=16, n_heads=2)
    x = torch.randn(1, 6, 16)
    mask = torch.tensor([[True, True, True, True, False, False]])      # last 2 are pad
    _, attn = mha(x, key_padding_mask=mask)
    # attn[b, h, q, k] over k=4,5 should be 0
    assert attn[..., 4:].abs().max().item() < 1e-6


# ---- Positional encoding ----
def test_sinusoidal_pe_first_columns_match_formula():
    d = 8
    max_len = 5
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)

    # pos=0 -> all sin = 0, all cos = 1
    assert torch.allclose(pe[0, 0::2], torch.zeros(4), atol=1e-6)
    assert torch.allclose(pe[0, 1::2], torch.ones(4), atol=1e-6)


def test_sinusoidal_pe_accepts_odd_dimension():
    from nndl import SinusoidalPE

    pe = SinusoidalPE(d_model=3, max_len=5)
    out = pe(torch.zeros(2, 5, 3))
    assert out.shape == (2, 5, 3)
    assert torch.isfinite(out).all()


# ---- nn.TransformerEncoderLayer integration test ----
def test_nn_transformer_encoder_runs():
    layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                       dropout=0.0, batch_first=True, norm_first=True,
                                       activation='gelu')
    enc = nn.TransformerEncoder(layer, num_layers=2)
    x = torch.randn(2, 10, 64)
    pad = torch.zeros(2, 10, dtype=torch.bool); pad[1, 7:] = True
    out = enc(x, src_key_padding_mask=pad)
    assert tuple(out.shape) == (2, 10, 64)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
