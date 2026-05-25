# chap7 网络优化与正则化（PyTorch）

| Notebook | 内容 |
|---|---|
| [`网络优化与正则化.ipynb`](网络优化与正则化.ipynb) | 优化器对比（Beale 二维 loss 面）、batch size 影响、Xavier/Kaiming 初始化、BatchNorm、Dropout、`lr_scheduler` 三种调度器 |

## 实现要点

- **优化器**：`torch.optim.{SGD, RMSprop, Adam, AdamW, …}`；`SGD(params, lr=..., momentum=0.9, weight_decay=5e-4)` 是 CV 经典配方。注意 `weight_decay` 直接挂在 optimizer 上，等价于 L2 正则。
- **mini-batch size**：决定每个 epoch 走多少步（`len(data) / batch_size`）。太小方差大，太大每 epoch 步数少；32-256 是常见甜区。
- **初始化**：
  - PyTorch `nn.Linear / Conv2d` 默认就是 Kaiming uniform，大多数情况下不用手动初始化。
  - 手动选择时：`nn.init.xavier_normal_/uniform_` 配 tanh/sigmoid，`nn.init.kaiming_normal_/uniform_(..., nonlinearity='relu')` 配 ReLU 类。
  - 经验法则：让信号通过深网络后每层方差大致保持稳定。
- **BatchNorm**：`nn.BatchNorm1d/2d`。`model.train()` 用当前 batch 的统计量；`model.eval()` 用训练期间累计的 running mean/var。让大学习率成为可能。
- **Dropout**：`nn.Dropout(p)`。`model.eval()` 自动关掉，不需要手动处理。常用 0.2-0.5。
- **`lr_scheduler`**：
  - 调用顺序 **`optimizer.step()` 然后 `scheduler.step()`**（PyTorch 1.1+ 的硬性要求）
  - `CosineAnnealingLR(opt, T_max=epochs)` 是现代默认
  - `ReduceLROnPlateau` 适合根据 dev 指标动态降 lr：`scheduler.step(dev_metric)`

## 关于 Adam 在 Beale 函数上"发散"

notebook 里的二维 Beale 测试中，Adam 在这个特定起点跑飞了——这是合成 loss 面的常见现象（参见 [On the Convergence of Adam (Reddi et al., ICLR'18)](https://openreview.net/forum?id=ryQu7f-RZ)）。在真实深度网络上，Adam 依然是无脑默认且通常足够好的选择。

## 测试

```bash
python -m pytest pytorch/tests/test_chap7.py -v
```
