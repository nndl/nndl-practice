# 第7章：网络优化与正则化（PyTorch）

[Notebook](网络优化与正则化.ipynb)与第7章按同一顺序展开：批大小与学习率缩放、手写优化器的Sphere轨迹和线性拟合、鞍点曲面、Xavier初始化、BN/LN、L2惩罚、匹配系数的权重衰减及暂退法。最后补充学习率调度器。

在本目录打开并顺序运行。MNIST复用`~/.cache/torch_data`，首次运行需下载。图像实验使用5000条训练、1000条验证，训练5轮。正则化实验固定400条Moon样本，按200/100/100划分；四种配置各更新50000次，用验证集选择后才评价测试集。

需要核对的约定：

- 线性拟合损失是通常MSE的一半，每轮按样本数重新统计。
- 手写AdaGrad、RMSprop与Adam的epsilon位于平方根外；Adam一次step内所有参数共用迭代编号。
- BN的momentum在本手写实现中表示旧统计量权重，与PyTorch接口相反。运行统计量要随模型保存。
- 无动量SGD下，L2与权重衰减仅在系数及作用参数范围匹配时等价；AdamW使用解耦衰减。
- 初始化应检查有效方差和gain。Linear/Conv2d内部默认初始化不能只凭“Kaiming”这个函数名理解成ReLU的He方差。
- 框架Dropout通过train()/eval()切换；本章手写Op则显式传递mode。
- 调度器按回合还是按更新调用，需要与其参数单位一致。ReduceLROnPlateau还要传入验证指标。

从`pytorch`目录运行相关测试：

```bash
python -m pytest tests/test_chap7.py -q
```
