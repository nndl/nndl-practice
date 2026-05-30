"""nndl：《神经网络与深度学习：案例与实践》随书工具包。

按书的结构组织：算子 → 模型 → 损失 → 优化 → 训练框架。各章 notebook 在首次引入
处会内联展示完整实现以便讲解；本包提供等价的工程化版本，供后续章节和测试复用。

公共 API 列表（按章节出现顺序）：

- `Op`：算子基类（chap1）
- `Linear`、`optimizer_lsm`：线性算子与最小二乘求解（chap2）
- `logistic`、`softmax`：激活函数（chap3）
- `Model_LR`、`Model_SR`：Logistic / Softmax 回归（chap3）
- `BinaryCrossEntropyLoss`、`MultiCrossEntropyLoss`、`mean_squared_error`（aka `mse`）（chap2/3）
- `accuracy`：通用准确率（chap3+）
- `Optimizer`、`SimpleBatchGD`：自定义算子配套的梯度下降（chap3）
- `RunnerV1`、`RunnerV2`、`RunnerV3`：循序渐进的训练框架（chap1/3/4）
- `LeNet5`、`PlainBlock`、`ResBlock`、`Net`：CNN 通用模块（chap5）
- `MySRN`、`MyLSTMModel`：手写 RNN 单元（chap6）
- `AdditiveAttention`、`scaled_dot_attention`、`MultiHeadAttention`、`SinusoidalPE`、`TransformerBlock`：注意力模块（chap8）
- `nndl.data`：合成数据集（Moon1000 / Multi1000 / DigitSum / Sentiment）
"""

from .op import Op
from .linear import Linear, optimizer_lsm
from .activation import logistic, softmax
from .loss import (
    mean_squared_error,
    mse,
    BinaryCrossEntropyLoss,
    MultiCrossEntropyLoss,
)
from .metric import accuracy
from .optim import Optimizer, SimpleBatchGD, BatchGD
from .mlp import ReLU, LinearLayer, MLP_3L
from .classify import Model_LR, Model_SR
from .cnn import LeNet5, PlainBlock, ResBlock, Net
from .rnn import MySRN, MyLSTMModel
from .attention import (
    AdditiveAttention,
    scaled_dot_attention,
    MultiHeadAttention,
    SinusoidalPE,
    TransformerBlock,
)
from .runner import RunnerV1, RunnerV2, RunnerV3
from . import data  # 子模块，按需 `from nndl.data import make_moons` 等

__all__ = [
    # ops / models
    "Op", "Linear", "optimizer_lsm",
    "logistic", "softmax",
    "Model_LR", "Model_SR",
    # loss / metric
    "mean_squared_error", "mse",
    "BinaryCrossEntropyLoss", "MultiCrossEntropyLoss",
    "accuracy",
    # optim
    "Optimizer", "SimpleBatchGD", "BatchGD",
    # op-based MLP (chap7 正则化)
    "ReLU", "LinearLayer", "MLP_3L",
    # runners
    "RunnerV1", "RunnerV2", "RunnerV3",
    # cnn / rnn / attention
    "LeNet5", "PlainBlock", "ResBlock", "Net",
    "MySRN", "MyLSTMModel",
    "AdditiveAttention", "scaled_dot_attention",
    "MultiHeadAttention", "SinusoidalPE", "TransformerBlock",
    # data
    "data",
]
