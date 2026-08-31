# chap4 前馈神经网络（PyTorch）

| 材料 | 内容 |
|---|---|
| [`前馈神经网络-上.ipynb`](前馈神经网络-上.ipynb) | 激活函数族（sigmoid/tanh/relu/leaky_relu/elu/softplus）+ 手算反向 vs `autograd` 对比 + moons 二分类 + hidden/激活函数 ablation |
| [`前馈神经网络-下.ipynb`](前馈神经网络-下.ipynb) | iris 三分类 + `RunnerV3`（DataLoader + best-model 追踪）|
| [`book_code.py`](book_code.py) | 书稿同名代码：`Model_MLP_L2`、`RunnerV2_1`、`Model_MLP_L2_V2`、`RunnerV2_2`、`Model_MLP_L5` 等 |

## 实现要点

- **logits-only 输出**：模型最后一层不要叠 sigmoid / softmax；二分类用 `BCEWithLogitsLoss`，多分类用 `CrossEntropyLoss`，PyTorch 内部融合激活 + 交叉熵，数值更稳。
- **手算反向 vs autograd**：chap4 上把一个 2 层 MLP 的反向手工推一遍（关键化简 $\partial \mathcal{L}/\partial z_2 = \hat y - y$），再用 `autograd` 跑同一个网络，验证逐元素一致到 $10^{-7}$ 量级。
- **`RunnerV3`** 在 chap3 `RunnerV2` 基础上做了两件事：
  - 用 `DataLoader` 喂入训练数据，每个 epoch 内部按小批量遍历；
  - 用 `metric_fn(logits, y) -> float` + `higher_is_better=True/False` 解耦 metric 与 loss，dev metric 超过历史最佳就把 **`state_dict`** 写到 `best_path`；
  - `history` 同时记录 `train_loss`、`dev_loss`、`dev_metric`。
- **iris 数据**用 `sklearn.datasets.load_iris`；标准化的 mean/std 只在训练集上拟合（避免信息泄露）。
- moons 数据：notebook 内 `make_moons` 用 `torch.Generator` 控制随机性，零外部依赖。

## 书稿代码对照

书稿按“手写算子 → 手写反向传播 → PyTorch 预定义算子”的顺序展开，因此保留了多个过渡版本；Notebook 为便于完整运行，采用了更紧凑的现代 PyTorch 路线。书中出现的同名类统一收录在 [`book_code.py`](book_code.py)：

| 书稿代码 | 对应内容 |
|---|---|
| `Linear`、`Logistic`、`Model_MLP_L2` | 手写两层 MLP 的前向与反向传播 |
| `BinaryCrossEntropyLoss`、`BatchGD`、`RunnerV2_1` | 手写损失、参数更新和训练循环 |
| `Model_MLP_L2_V2`、`RunnerV2_2` | `nn.Module`、autograd 与 `state_dict` 版本 |
| `Model_MLP_L5` | 梯度消失和死亡 ReLU 对照实验 |

Notebook 中的 `nn.Sequential` 与 `RunnerV3` 是上述实现继续工程化后的写法，两条路线可以对照学习。

## 测试

```bash
python -m pytest pytorch/tests/test_chap4.py -v
```
