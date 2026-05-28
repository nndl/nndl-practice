# PyTorch 实现

《神经网络与深度学习：案例与实践》10 个章节的 PyTorch 实现。第一版印刷书的 PaddlePaddle 实现保留在 [practice-in-paddle](https://github.com/nndl/practice-in-paddle)。

**各章进度表和章节入口见 [仓库根 README](../README.md#pytorch-章节进度)**（单一来源，避免双份维护）。

---

## 环境要求

| 组件 | 最低版本 | 说明 |
|---|---|---|
| Python | **3.11+**（64-bit） | 推荐 3.11/3.12；3.10 应当也跑得通但未持续验证，3.13 须留意 `torch` wheel 是否已发布 |
| pip    | 25+              | 旧版 pip 解析 PyTorch 索引可能拿不到最新 wheel |
| PyTorch | **2.7+** | 部分 chap 用到 `nn.functional.scaled_dot_product_attention` 的新行为、`torch.compile` 等 2.x 后期特性 |
| torchvision | **0.22+** | chap5 用 `datasets.MNIST` / `CIFAR10` 自动下载 |
| CUDA（可选） | 12.6+ | GPU 训练；CPU 也能跑完所有 notebook（chap5 / chap10 会慢一些） |

完整依赖见 [`requirements.txt`](requirements.txt)：还包括 `numpy`、`pandas`、`matplotlib`、`scikit-learn`（chap4 下 iris）、`jupyter`、`pytest`。

## 安装

```bash
# 1) 装一个干净的 venv（推荐）
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 2) 装 PyTorch
# CPU（约 200MB）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU（需要 CUDA 12.6+）
# 参见 https://pytorch.org/get-started/ 选对应的 wheel

# 3) 装其余依赖
pip install -r pytorch/requirements.txt
```

## 跑通一个章节

```bash
# 在仓库根目录
jupyter notebook pytorch/chap5卷积神经网络/卷积神经网络-上.ipynb
```

或者用 nbconvert 在命令行端到端执行（CI / 复跑用）：

```bash
jupyter nbconvert --to notebook --execute \
    pytorch/chap5卷积神经网络/卷积神经网络-上.ipynb \
    --output /tmp/out.ipynb --ExecutePreprocessor.timeout=600
```

> **数据集**：chap5（MNIST / CIFAR-10）首次运行会通过 `torchvision.datasets` 自动下载到 `~/.cache/torch_data`（约 200 MB），后续 run 直接读缓存。chap4 下的 iris 由 `sklearn.datasets.load_iris` 内置，不联网。波士顿房价的 CSV 已提交到 [`dataset/boston_house_prices.csv`](dataset/boston_house_prices.csv)。

## 跑测试

每章配一份 `pytest` sanity 检查（关键算子 / 关键结论），放在 [`tests/`](tests/) 下：

```bash
# 在仓库根目录运行
python -m pytest pytorch/tests/ -v
```

CPU 上跑完所有测试约 1 分钟。

## 共享工具包 `nndl/`

[`pytorch/nndl/`](nndl/) 是随书工程化版本的算子 / 模型 / 训练框架，被多个章节复用。各章 notebook 在**首次引入**处会内联展示完整实现帮助讲解；后续章节 / 测试通过 `from nndl import ...` 直接调用工程化版本。

按章节出现顺序：

| 模块 | 主要内容 | 首次出现 |
|---|---|---|
| `op` | 算子基类 `Op` | chap1 |
| `linear`, `optim` | `Linear`、`optimizer_lsm`、`SimpleBatchGD` | chap2 |
| `activation`, `classify`, `loss`, `metric` | `logistic`/`softmax`、`Model_LR`/`Model_SR`、CE / MSE、`accuracy` | chap3 |
| `runner` | `RunnerV1` → `RunnerV2` → `RunnerV3`（训练框架渐进升级） | chap1 / chap3 / chap4 |
| `cnn` | `LeNet5`、`PlainBlock` / `ResBlock` / `Net` | chap5 |
| `rnn` | `MySRN`、`MyLSTMModel` | chap6 |
| `attention` | `AdditiveAttention`、`scaled_dot_attention`、`MultiHeadAttention`、`SinusoidalPE`、`TransformerBlock` | chap8 |
| `data` | 合成数据集（Moon1000 / Multi1000 / DigitSum / Sentiment） | 多章共用 |

## 与 Paddle 版的对应

PyTorch 版与 PaddlePaddle 版章节一一对应，编号一致；两个版本可独立使用。Paddle 版只覆盖前 8 章；图神经网络（chap9）与大语言模型与智能体（chap10）是 PyTorch 版独有。
