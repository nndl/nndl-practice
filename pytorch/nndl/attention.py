"""注意力机制模块：第 8 章定义的可复用注意力组件。

- 加性（Additive）注意力：跟 BiLSTM 配合用，输入 `(H, mask)`。
- 点积（Scaled Dot-Product）注意力：函数式，便于嵌入到任意模型里。
- 多头注意力（MultiHead）：Transformer 的核心模块。
- 正弦位置编码（Sinusoidal PE）：register_buffer，不参与训练。
- TransformerBlock：pre-LN 风格的标准 Transformer 编码块。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Additive attention（chap8 上）                                                #
# --------------------------------------------------------------------------- #
class AdditiveAttention(nn.Module):
    """加性（Bahdanau）注意力：`scores = vᵀ tanh(W H)`。

    输入：
    - `H`：`[B, L, hidden]` 编码器输出
    - `mask`：`[B, L]` 布尔掩码，`True` 表示有效位置

    返回 `(context, attn)`：`context` 形状 `[B, hidden]`、`attn` 形状 `[B, L]`。
    """

    def __init__(self, hidden_size: int, attn_size: int = 64):
        super().__init__()
        self.W = nn.Linear(hidden_size, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)

    def forward(self, H, mask):
        scores = self.v(torch.tanh(self.W(H))).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), H).squeeze(1)
        return context, attn


# --------------------------------------------------------------------------- #
# Scaled dot-product attention（chap8 下）                                      #
# --------------------------------------------------------------------------- #
def scaled_dot_attention(Q, K, V, mask=None):
    """缩放点积注意力（函数式）：`Attn(Q, K, V) = softmax(QKᵀ/√d_k) V`。

    输入：
    - `Q, K, V`：`[..., L, d_k]` / `[..., L, d_k]` / `[..., L, d_v]`
    - `mask`：可选，`[..., L_q, L_k]` 布尔掩码（`True` 表示有效）。

    返回 `(out, attn)`：`out` 形状 `[..., L_q, d_v]`、`attn` 形状 `[..., L_q, L_k]`。
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores, dim=-1)
    return attn @ V, attn


# --------------------------------------------------------------------------- #
# MultiHead attention（chap8 下）                                               #
# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    """多头自注意力：`Q/K/V` 各自线性映射后切成 `n_heads` 个头并行做缩放点积。

    输入 `x`：`[B, L, embed_dim]`；可选 `key_padding_mask`：`[B, L]`（True=有效）。
    输出 `(out, attn)`：`out` 形状 `[B, L, embed_dim]`、`attn` 形状 `[B, n_heads, L, L]`。
    """

    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x):
        B, L, _ = x.shape
        return x.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, key_padding_mask=None):
        Q = self._split_heads(self.W_q(x))
        K = self._split_heads(self.W_k(x))
        V = self._split_heads(self.W_v(x))

        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]

        out, attn = scaled_dot_attention(Q, K, V, mask=mask)
        B, h, L, d = out.shape
        out = out.transpose(1, 2).reshape(B, L, h * d)
        return self.W_o(out), attn


# --------------------------------------------------------------------------- #
# Sinusoidal positional encoding（chap8 下）                                    #
# --------------------------------------------------------------------------- #
class SinusoidalPE(nn.Module):
    """正弦余弦位置编码：`pe[pos, 2i] = sin(pos / 10000^{2i/d})`、`pe[pos, 2i+1] = cos(...)`。

    用 `register_buffer` 保存，不参与训练；forward 直接把 `pe[:L]` 加到输入上。
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(1)
        assert L <= self.pe.size(0), \
            f"序列长度 {L} 超过 max_len={self.pe.size(0)}，请增大 SinusoidalPE(max_len=...)"
        return x + self.pe[:L]


# --------------------------------------------------------------------------- #
# Transformer encoder block（chap8 下）                                         #
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Pre-LN Transformer 编码块：Attn → 残差 → FFN → 残差。

    `ff_dim` 通常是 `d_model * 4`；`dropout` 用在残差路径上。
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(self.norm1(x), key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
