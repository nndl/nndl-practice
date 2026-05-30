# chap3 线性模型（PyTorch）

| Notebook | 内容 |
|---|---|
| [`线性模型-上.ipynb`](线性模型-上.ipynb) | Logistic 回归二分类：Moon1000 合成数据 + 手写 `Model_LR` + `BinaryCrossEntropyLoss` + `SimpleBatchGD` + 决策边界可视化 |
| [`线性模型-下.ipynb`](线性模型-下.ipynb) | Softmax 回归多分类：先用 Multi1000 三簇合成数据演示，再做鸢尾花 Iris 分类实战 |

## 实现要点

- **自定义 Op 算子风格**：延续第 1、2 章的 `Op` 接口手写 `Model_LR` / `Model_SR`，sigmoid / softmax 放在模型 forward 里；`backward(labels)` 直接给出 BCE+sigmoid / CE+softmax 的合成梯度——绕过自动求导，把反向传播讲透。
- **数值稳定**：Softmax 减最大值避免 `exp` 上溢/下溢；`BinaryCrossEntropyLoss` / `MultiCrossEntropyLoss` 把概率 `clamp` 到 `(eps, 1-eps)` 避免 `log(0)`。
- **`RunnerV2` 引入早停法**：训练时同时跑验证集，保留 dev 上最优的 `model.params` 字典；本章全程用 `SimpleBatchGD` 全批量梯度下降，不再是闭式解。
- **可视化决策边界**：用 `torch.meshgrid(..., indexing='xy')` 生成网格点，`contourf` 画概率/区域。

## 测试

```bash
python -m pytest pytorch/tests/test_chap3.py -v
```

或直接 `python pytorch/tests/test_chap3.py` 顺序跑。
