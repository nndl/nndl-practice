# chap5 卷积神经网络（PyTorch）

| Notebook | 内容 |
|---|---|
| [`卷积神经网络-上.ipynb`](卷积神经网络-上.ipynb) | 朴素 conv2d vs `nn.Conv2d`、形状公式、Laplacian 边缘检测、LeNet-5 + MNIST 子集训练 + 第一层 kernel 可视化 |
| [`卷积神经网络-下.ipynb`](卷积神经网络-下.ipynb) | `PlainBlock` vs `ResBlock`，同深度的 Plain-Net vs ResNet 在 CIFAR-10 子集上对比；`torchvision.models.resnet18` API 演示 |

## 实现要点

- 深度学习里的 "conv" 实际是 cross-correlation（kernel 不翻转）。`nn.Conv2d` 即是该操作。
- **形状公式**：$H_\text{out} = \lfloor (H + 2p - k) / s \rfloor + 1$。`padding = k // 2, stride = 1` 时尺寸保持不变。
- **`Conv → BN → ReLU`** 是现代 CNN 的标配三件套；BN 让深网络的训练曲线更平滑。
- **残差连接**：`out + shortcut(x)`；当 `in_ch != out_ch` 或 `stride > 1` 时，shortcut 用 `1x1 Conv` 投影对齐。
- **MNIST/CIFAR 用 torchvision**：`datasets.MNIST` / `datasets.CIFAR10` 自动下载到 `~/.cache/torch_data`（首次运行时下载，约 50MB + 150MB）。
- notebook 出于执行时间考虑用了小 subset（MNIST 5000/1000，CIFAR-10 5000/1000）。完整数据 + 更多 epoch 在 GPU 上跑 ~5min 内能到 ~99% / ~70%。

## 测试

```bash
python -m pytest pytorch/tests/test_chap5.py -v
```
