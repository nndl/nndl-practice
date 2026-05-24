# chap3 线性模型（PyTorch）

| Notebook | 内容 |
|---|---|
| [`线性模型-上.ipynb`](线性模型-上.ipynb) | 一元 / 多元线性回归：合成数据 + 闭式解 + 手写梯度下降 + `nn.Linear` 训练 |
| [`线性模型-下.ipynb`](线性模型-下.ipynb) | Logistic 回归（二分类）与 Softmax 回归（多分类）+ 决策边界可视化 |

## 实现要点

- **数值稳定的 loss**：二分类用 `BCEWithLogitsLoss`、多分类用 `CrossEntropyLoss`；它们在内部分别做 sigmoid+BCE、log_softmax+NLL，模型最后一层**只输出 logits**，不要再叠 sigmoid / softmax。
- **闭式解 vs 梯度下降**：在合成数据上两种方法的最终参数应接近一致；`tests/test_chap3.py` 里有相应断言。
- **可视化决策边界**：用 `torch.meshgrid(..., indexing='xy')` 生成网格点，`contourf` 画概率/区域。

## 测试

```bash
python -m pytest pytorch/tests/test_chap3.py -v
```

或直接 `python pytorch/tests/test_chap3.py` 顺序跑。
