"""循环神经网络通用模块：手写 SRN、手写 LSTM 包装。

`MySRN` 是从零手写的 Simple Recurrent Network（朴素 RNN），第 6 章用来跟 `nn.RNN`
逐元素对照；`MyLSTMModel` 是基于 `nn.LSTM` 的封装，方便在 chap6 上的「序列累加」
任务里跟 SRN 做对比。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class MySRN(nn.Module):
    """手写 SRN：`h_t = tanh(e_t W_x + h_{t-1} W_h + b)`。

    用 identity 初始化 `W_h`（接近 RNN 训练初期的稳定参数化）。第 6 章用来跟
    `nn.RNN` 的输出做逐元素对照。

    输入：`x ∈ {0..9}^{N×L}` 整数索引；输出 `[N, n_class]` logits。
    """

    def __init__(self, vocab: int = 10, embed: int = 32, hidden: int = 64,
                 n_class: int = 19):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.Wx = nn.Parameter(torch.randn(embed, hidden) / math.sqrt(embed))
        self.Wh = nn.Parameter(torch.eye(hidden))
        self.b = nn.Parameter(torch.zeros(hidden))
        self.fc = nn.Linear(hidden, n_class)

    def forward(self, x):
        e = self.emb(x)
        h = torch.zeros(e.size(0), self.Wh.size(0), device=e.device)
        for t in range(e.size(1)):
            h = torch.tanh(e[:, t] @ self.Wx + h @ self.Wh + self.b)
        return self.fc(h)


class MyLSTMModel(nn.Module):
    """`nn.LSTM` 包装：Embedding → LSTM → 末步 hidden → FC。

    输入约定同 `MySRN`（整数索引序列）。
    """

    def __init__(self, vocab: int = 10, embed: int = 32, hidden: int = 64,
                 n_class: int = 19):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_class)

    def forward(self, x):
        e = self.emb(x)
        _, (h, _) = self.lstm(e)
        # h: [num_layers, B, hidden]，取最后一层末步隐状态（1 层时等价于 h.squeeze(0)）
        return self.fc(h[-1])
