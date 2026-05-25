"""chap8 注意力机制 sanity tests."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- AdditiveAttention ----
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size, attn_size=64):
        super().__init__()
        self.W = nn.Linear(hidden_size, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)

    def forward(self, H, mask):
        scores = self.v(torch.tanh(self.W(H))).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        return torch.bmm(attn.unsqueeze(1), H).squeeze(1), attn


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
def _scaled_dot(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn


def test_scaled_dot_attention_with_identity_input_is_diagonal_dominant():
    torch.manual_seed(0)
    X = torch.eye(4) + 0.01 * torch.randn(4, 4)
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
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def _split(self, x):
        B, L, _ = x.shape
        return x.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, key_padding_mask=None):
        Q = self._split(self.W_q(x)); K = self._split(self.W_k(x)); V = self._split(self.W_v(x))
        mask = key_padding_mask[:, None, None, :] if key_padding_mask is not None else None
        out, attn = _scaled_dot(Q, K, V, mask=mask)
        B, h, L, d = out.shape
        return self.W_o(out.transpose(1, 2).reshape(B, L, h * d)), attn


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
