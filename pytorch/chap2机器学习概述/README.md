# chap2 机器学习概述（PyTorch）

| Notebook | 内容 |
|---|---|
| [`机器学习概述-上.ipynb`](机器学习概述-上.ipynb) | 机器学习五要素 + 线性回归最小 pipeline + 多项式回归（欠/过拟合 + L2 正则） |
| [`机器学习概述-下.ipynb`](机器学习概述-下.ipynb) | `RunnerV1`（闭式解 + 评估 + save/load）+ 加州房价回归案例 |

## 实现要点

- **五要素 → PyTorch 对应物**：`Dataset`/`DataLoader`、`nn.Module`、`nn.MSELoss`、`torch.optim.*`、自定义 metric。本章侧重数学/算法，`RunnerV1` 的闭式解一步到位、不需要梯度下降，因此 chap2 下没有用到 `DataLoader`。
- **闭式最小二乘**：用 `torch.linalg.lstsq`（数值稳定，内部 SVD/QR）；带 L2 正则时用 `torch.linalg.solve` 解 $(X^\top X + \lambda I)\, w = X^\top y$，**偏置项不参与正则化**。
- **多项式回归** = 多项式基函数变换 + 线性回归。模型容量随阶数增长，数据少时易过拟合。
- **加州房价**：用 `sklearn.datasets.fetch_california_housing`（20640 样本、8 特征，目标 `MedHouseVal`）。波士顿房价数据集 sklearn 1.2 起已整体移除，本仓库不再使用。
- **标准化的统计量**只能在训练集上拟合，再用同一组 `mean/std` 变换测试集，避免信息泄露。
- **保存 / 加载**：`RunnerV1` 把 `model.params` 字典写到磁盘（自定义 Op 算子，没有 `state_dict`）；第 4 章 `RunnerV3` 才切到 PyTorch 的 `state_dict`。

## `Runner` 接口约定

本章引入的 `RunnerV1` 只覆盖闭式解场景；第 3、4 章会循序渐进扩展到 `RunnerV2` / `RunnerV3`。三版本的完整实现都打包在 `pytorch/nndl/runner.py` 里，后续章节可直接 `from nndl.runner import RunnerV1/V2/V3` 复用。

```python
runner = RunnerV1(model, optimizer)              # optimizer 是闭式求解器函数
runner.fit(X, y)                                 # 调一次求解器
mse_val = runner.evaluate(X, y, metric_fn=mse)
y_hat   = runner.predict(X)
runner.save(path);  runner.load(path)            # 存取 model.params 字典
```

## 测试

```bash
python -m pytest pytorch/tests/test_chap2.py -v
```
