"""跨章节复用的合成数据集生成函数。

| 函数 / 类 | 首次出现 | 用途 |
|---|---|---|
| `make_moons` | chap3 上 | 二分类弯月数据集（Moon1000） |
| `make_multiclass_classification` | chap3 下 | 三分类簇数据集（Multi1000） |
| `make_digit_sum` | chap6 上 | 数字求和序列任务（DigitSum） |
| `make_sentiment_sample` / `SentimentDS` / `sentiment_collate` | chap8 上 | 情感分类变长序列（chap8 上下共用） |
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Moon1000（chap3 上）                                                          #
# --------------------------------------------------------------------------- #
def make_moons(n_samples: int = 1000, shuffle: bool = True, noise=None, seed=None):
    """两个弯月形分布的二分类数据集。

    返回 `(X, y)`：`X` 形状 `[n_samples, 2]`、`y` 形状 `[n_samples]`（float，0/1）。
    `seed` 非 None 时用局部 Generator，使 shuffle / noise 可复现（默认 None 走全局 RNG，行为同前）。
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    n_out = n_samples // 2
    n_in = n_samples - n_out

    outer_x = torch.cos(torch.linspace(0, math.pi, n_out))
    outer_y = torch.sin(torch.linspace(0, math.pi, n_out))
    inner_x = 1 - torch.cos(torch.linspace(0, math.pi, n_in))
    inner_y = 0.5 - torch.sin(torch.linspace(0, math.pi, n_in))

    X = torch.stack([
        torch.cat([outer_x, inner_x]),
        torch.cat([outer_y, inner_y]),
    ], dim=1)
    y = torch.cat([torch.zeros(n_out), torch.ones(n_in)])

    if shuffle:
        idx = torch.randperm(X.shape[0], generator=g)
        X = X[idx]; y = y[idx]
    if noise is not None:
        X = X + torch.normal(mean=0.0, std=noise, size=X.shape, generator=g)
    return X, y


# --------------------------------------------------------------------------- #
# Multi1000（chap3 下）                                                         #
# --------------------------------------------------------------------------- #
def make_multiclass_classification(n_samples: int = 100, n_features: int = 2,
                                   n_classes: int = 3, shuffle: bool = True,
                                   noise: float = 0.1, seed=None):
    """从 `n_classes` 个簇采样的多分类数据集。

    返回 `(X, y)`：`X` 形状 `[n_samples, n_features]`、`y` 形状 `[n_samples]`（long）。
    `seed` 非 None 时用局部 Generator，使采样 / 噪声 / shuffle 可复现（默认 None 走全局 RNG，行为同前）。
    """
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    n_per_class = [n_samples // n_classes] * n_classes
    for i in range(n_samples - sum(n_per_class)):
        n_per_class[i % n_classes] += 1

    X = torch.zeros(n_samples, n_features)
    y = torch.zeros(n_samples, dtype=torch.int64)

    centroids = torch.randperm(2 ** n_features, generator=g)[:n_classes]
    centroids_bin = np.unpackbits(centroids.numpy().astype("uint8")).reshape(-1, 8)[:, -n_features:]
    centroids = torch.tensor(centroids_bin, dtype=torch.float32)
    centroids = 1.5 * centroids - 1

    X[:, :n_features] = torch.randn(n_samples, n_features, generator=g)

    stop = 0
    for k, centroid in enumerate(centroids):
        start, stop = stop, stop + n_per_class[k]
        y[start:stop] = k % n_classes
        X_k = X[start:stop, :n_features]
        A = 2 * torch.rand(n_features, n_features, generator=g) - 1
        X_k = X_k @ A + centroid
        X[start:stop, :n_features] = X_k

    if noise > 0:
        noise_mask = torch.rand(n_samples, generator=g) < noise
        n_noisy = int(noise_mask.sum())
        y[noise_mask] = torch.randint(0, n_classes, (n_noisy,), generator=g)
    if shuffle:
        idx = torch.randperm(n_samples, generator=g)
        X = X[idx]; y = y[idx]
    return X, y


# --------------------------------------------------------------------------- #
# DigitSum（chap6 上）                                                          #
# --------------------------------------------------------------------------- #
def make_digit_sum(n_samples: int, L: int, seed: int = 0):
    """生成长度为 `L` 的数字序列以及它们的求和。

    每个样本是 `L` 个 0-9 的整数 token，标签是它们的和（取值 0..9L）。返回
    `(X, y)`：`X` 形状 `[n_samples, L]` long、`y` 形状 `[n_samples]` long。
    """
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 10, (n_samples, L), generator=g)
    y = X.sum(dim=1)
    return X, y


# --------------------------------------------------------------------------- #
# Sentiment（chap8 上下共用）                                                   #
# --------------------------------------------------------------------------- #
# 词表设计：前 PAD=0、POS=[1,2,3]、NEG=[4,5,6]、FILL=[7..VOCAB-1] 是无情感填充词
VOCAB = 25
PAD = 0
POS = [1, 2, 3]
NEG = [4, 5, 6]
FILL = list(range(7, VOCAB))


def make_sentiment_sample(g, L_range=(35, 46), n_sentiment: int = 2):
    """生成一条情感序列：长度在 `L_range` 内，随机选 POS / NEG 池放 `n_sentiment` 个情感词。

    返回 `(tokens, label)`：`tokens` 是 `[L]` 的 long tensor，`label` 是 int（0=neg / 1=pos）。
    """
    L = torch.randint(L_range[0], L_range[1], (1,), generator=g).item()
    label = torch.randint(0, 2, (1,), generator=g).item()
    tokens = [FILL[torch.randint(0, len(FILL), (1,), generator=g).item()] for _ in range(L)]
    sent_pool = POS if label == 1 else NEG
    positions = torch.randperm(L, generator=g)[:n_sentiment].tolist()
    for p in positions:
        tokens[p] = sent_pool[torch.randint(0, 3, (1,), generator=g).item()]
    return torch.tensor(tokens, dtype=torch.long), label


class SentimentDS(Dataset):
    """情感分类合成数据集，初始化时用给定种子生成 `n` 条 `(tokens, label)` 对。"""

    def __init__(self, n: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.data = [make_sentiment_sample(g) for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def sentiment_collate(batch):
    """`DataLoader.collate_fn`：把变长 token 序列 pad 到同长，返回 `(padded, lengths, labels)`。

    与 `RunnerV3` 的 `*inputs, y = batch` 解包约定兼容。
    """
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    return (
        pad_sequence(seqs, batch_first=True, padding_value=PAD),
        lengths,
        torch.tensor(labels),
    )
