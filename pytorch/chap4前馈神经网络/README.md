# chap4 前馈神经网络（PyTorch）

| Notebook | 内容 |
|---|---|
| [`前馈神经网络-上.ipynb`](前馈神经网络-上.ipynb) | 激活函数族（sigmoid/tanh/relu/leaky_relu/elu/softplus）+ 手算反向 vs `autograd` 对比 + moons 二分类 + hidden/激活函数 ablation |
| [`前馈神经网络-下.ipynb`](前馈神经网络-下.ipynb) | iris 三分类 + `RunnerV2`（best-model 追踪）+ `DataLoader` |

## 实现要点

- **logits-only 输出**：模型最后一层不要叠 sigmoid / softmax；二分类用 `BCEWithLogitsLoss`，多分类用 `CrossEntropyLoss`，PyTorch 内部融合激活 + 交叉熵，数值更稳。
- **手算反向 vs autograd**：chap4 上把一个 2 层 MLP 的反向手工推一遍（关键化简 $\partial \mathcal{L}/\partial z_2 = \hat y - y$），再用 `autograd` 跑同一个网络，验证逐元素一致到 $10^{-7}$ 量级。
- **`RunnerV2`** 在 chap2 `Runner` 基础上加了：
  - `metric_fn(logits, y) -> float` 和 `higher_is_better=True/False`
  - 训练循环里实时比较 dev metric，超过历史最佳就把 `state_dict` 写到 `best_path`
  - `history` 多记一份 `dev_loss`
- **iris 数据**用 `sklearn.datasets.load_iris`；标准化的 mean/std 只在训练集上拟合（避免信息泄露）。
- moons 数据：notebook 内 `make_moons` 用 `torch.Generator` 控制随机性，零外部依赖。

## 测试

```bash
python -m pytest pytorch/tests/test_chap4.py -v
```
