# chap4 前馈神经网络（PyTorch）

| 材料 | 内容 |
|---|---|
| [`前馈神经网络-上.ipynb`](前馈神经网络-上.ipynb) | 激活函数族 + 手算反向 vs `autograd` + 书中两层 MLP/Runner 的手写版与 PyTorch 版 + moons 二分类 + 对照实验 |
| [`前馈神经网络-下.ipynb`](前馈神经网络-下.ipynb) | iris 三分类 + `RunnerV3`（DataLoader + best-model 追踪）|

## 实现要点

- **主线与书稿一致**：chap4 上依次实现 `Linear`、`Logistic`、`Model_MLP_L2`、`RunnerV2_1`，再改写为基于 `nn.Module` 和 autograd 的 `Model_MLP_L2_V2`、`RunnerV2_2`；`Model_MLP_L5` 放在后面的深层网络实验中。
- **损失函数约定**：为对应书中的推导，手写模型和同名 PyTorch 模型输出概率并使用 BCE；后续对照实验直接输出 logits，配合 `BCEWithLogitsLoss`，避免重复 sigmoid 且具有更好的数值稳定性。
- **手算反向 vs autograd**：chap4 上把一个 2 层 MLP 的反向手工推一遍（关键化简 $\partial \mathcal{L}/\partial z_2 = \hat y - y$），再用 `autograd` 跑同一个网络，验证逐元素一致到 $10^{-7}$ 量级。
- **`RunnerV3`** 在 chap3 `RunnerV2` 基础上做了两件事：
  - 用 `DataLoader` 喂入训练数据，每个 epoch 内部按小批量遍历；
  - 用 `metric_fn(logits, y) -> float` + `higher_is_better=True/False` 解耦 metric 与 loss，dev metric 超过历史最佳就把 **`state_dict`** 写到 `best_path`；
  - `history` 同时记录 `train_loss`、`dev_loss`、`dev_metric`。
- **iris 数据**用 `sklearn.datasets.load_iris`；标准化的 mean/std 只在训练集上拟合（避免信息泄露）。
- moons 数据由 `sklearn.datasets.make_moons` 生成，并通过 `random_state` 固定随机性。

## 测试

```bash
python -m pytest pytorch/tests/test_chap4.py -v
```
