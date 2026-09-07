"""循环神经网络通用模块：手写SRN与nn.LSTM包装。

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

    使用单位矩阵初始化 `W_h`，用于观察初始化对递推的影响。
    非线性激活仍可能使长程梯度衰减；也可与 `nn.RNN` 做数值对照。

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
        h = e.new_zeros(e.size(0), self.Wh.size(0))
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


class NativeRecurrent(nn.Module):
    """把书中单层SRN/LSTM的参数复制到框架算子，返回末隐状态。

    LSTM门顺序为输入门、遗忘门、候选状态、输出门（i,f,g,o）。
    手写算子只有一份偏置，因此将框架的第二份偏置固定为零。
    只支持本章单向、单层、无投影、无暂退的循环层。
    """

    def __init__(self, manual, kind="rnn"):
        super().__init__()
        if kind not in ("rnn", "lstm"):
            raise ValueError("kind must be rnn or lstm")
        self.kind = kind
        factory = nn.RNN if kind == "rnn" else nn.LSTM
        self.cell = factory(manual.input_size, manual.hidden_size, batch_first=True)
        self.cell.to(next(manual.parameters()))
        with torch.no_grad():
            if kind == "rnn":
                weight_ih, weight_hh = manual.W.T, manual.U.T
                bias = manual.b.flatten()
            else:
                gates = ("i", "f", "c", "o")
                weight_ih = torch.cat([getattr(manual, "W_" + g).T for g in gates])
                weight_hh = torch.cat([getattr(manual, "U_" + g).T for g in gates])
                bias = torch.cat([getattr(manual, "b_" + g).flatten() for g in gates])
            self.cell.weight_ih_l0.copy_(weight_ih)
            self.cell.weight_hh_l0.copy_(weight_hh)
            self.cell.bias_ih_l0.copy_(bias)
            self.cell.bias_hh_l0.zero_()
        self.cell.bias_hh_l0.requires_grad_(False)

    def forward(self, inputs):
        _, state = self.cell(inputs)
        if self.kind == "lstm":
            state = state[0]
        return state[-1]
