# chap2 机器学习概述（PyTorch）

| Notebook | 内容 |
|---|---|
| [`机器学习概述-上.ipynb`](机器学习概述-上.ipynb) | 机器学习五要素 + 线性回归最小 pipeline + 多项式回归（欠/过拟合 + L2 正则） |
| [`机器学习概述-下.ipynb`](机器学习概述-下.ipynb) | `Runner` 类（训练 / 评估 / 保存 / 加载） + 波士顿房价回归案例 |

## 实现要点

- **五要素 → PyTorch 对应物**：`Dataset`/`DataLoader`、`nn.Module`、`nn.MSELoss`、`torch.optim.*`、自定义 metric。本章只在 chap2 下用到 `DataLoader`；chap2 上侧重数学/算法。
- **闭式最小二乘**：用 `torch.linalg.lstsq`（数值稳定，内部 SVD/QR）；带 L2 正则时用 `torch.linalg.solve` 解 $(X^\top X + \lambda I)\, w = X^\top y$，**偏置项不参与正则化**。
- **多项式回归** = 多项式基函数变换 + 线性回归。模型容量随阶数增长，数据少时易过拟合。
- **波士顿房价**：去掉了原数据集的 `b` 列（按种族的统计变量，存在伦理问题；sklearn 1.0 起已停止内置），保留 12 个特征。
- **归一化的统计量**只能在训练集上拟合，再用同一组 `min/max` 变换测试集，避免信息泄露。
- **保存 / 加载**用 `torch.save(model.state_dict(), ...)` 和 `model.load_state_dict(torch.load(...))`，不要直接 pickle 整个 model。

## `Runner` 接口约定

后续章节会扩展这个 `Runner`（验证集 / 早停 / metric 记录 / lr 调度等），保持以下基础接口稳定：

```python
runner = Runner(model, optimizer, loss_fn, metric_fn=None)
runner.fit(train_loader, dev_loader=None, num_epochs=100, log_every=10)
metric = runner.evaluate(loader)
y_hat  = runner.predict(x)
runner.save(path);  runner.load(path)
```

## 测试

```bash
python -m pytest pytorch/tests/test_chap2.py -v
```
