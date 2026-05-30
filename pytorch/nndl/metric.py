"""评价指标。"""
from __future__ import annotations

import torch


def accuracy(preds, labels):
    """通用 accuracy：自动判断二分类（概率）/ 多分类（logits 或概率）。

    - 当 `preds` 形状是 `[N, 1]`：按概率阈值 0.5 取二分类预测；
    - 当 `preds` 形状是 `[N, C]`（C > 1）：取 argmax 作为多分类预测；
    - 当 `preds` 是一维：直接当作类别索引。

    `labels` 会自动 reshape 到与 `preds` 预测一致。返回 Python float，方便直接
    作为 `RunnerV3` 的 `metric_fn` 使用。
    """
    if preds.dim() > 1 and preds.shape[1] == 1:
        preds = (preds >= 0.5).float()
    elif preds.dim() > 1:
        preds = preds.argmax(dim=1)
    labels = labels.reshape(preds.shape)
    return (preds == labels).float().mean().item()
