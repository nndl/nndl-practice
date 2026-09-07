# PyTorch 实现

《神经网络与深度学习：案例与实践》10 个章节的 PyTorch 实现。第一版印刷书的 PaddlePaddle 实现保留在 [practice-in-paddle](https://github.com/nndl/practice-in-paddle)。

预编译 PyTorch 包自带所需的 CUDA 运行时组件；普通使用通常不要求另装同版本的完整 CUDA Toolkit。自定义 CUDA 扩展等编译任务另有要求。

**各章目录和代码入口见 [仓库根 README](../README.md#章节目录)**（单一来源，避免双份维护）。

---

## 环境要求

| 组件 | 最低版本 | 说明 |
|---|---|---|
| Python | **3.11+**（64-bit） | 推荐 3.11/3.12；其他 Python 版本应检查所选 PyTorch 与依赖包的兼容范围 |
| pip | 与当前 Python 兼容 | 安装前更新，使用对应解释器的 `python -m pip` |
| PyTorch | **2.7+** | 部分 chap 用到 `nn.functional.scaled_dot_product_attention` 的新行为、`torch.compile` 等 2.x 后期特性 |
| torchvision | **0.22+** | chap5 用 `datasets.MNIST` / `CIFAR10` 自动下载 |
| NVIDIA GPU（可选） | 与所选 PyTorch 构建兼容的驱动 | 按官方安装页选择 CUDA 构建；本书常规示例可用 CPU，较长训练宜先缩小规模 |

完整依赖见 [`requirements.txt`](requirements.txt)：还包括 `numpy`、`pandas`、`matplotlib`、`scikit-learn`（chap4 下 iris）、`networkx` 与 `torch-geometric`（chap9）、`jupyter`、`pytest`。

## 安装

```bash
# 以下命令从仓库根目录 nndl-practice 执行
# 1) 装一个干净的 venv（推荐）
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# 2) 装 PyTorch
# CPU（Linux / Windows；macOS 使用官方对应命令）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU（按显卡驱动与所选构建的兼容条件安装）
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

> **数据集**：chap2 下的加州房价由 `sklearn.datasets.fetch_california_housing` 获取并缓存在本机 scikit-learn 数据目录；chap4 下的 iris 由 `sklearn.datasets.load_iris` 内置，不联网。chap5（MNIST / CIFAR-10）首次运行会通过 `torchvision.datasets` 自动下载到 `~/.cache/torch_data`（约 200 MB），后续 run 直接读缓存。

chap10 上支持快速冒烟模式：把环境变量 `NNDL_QUICK_RUN` 设为 `1` 后，只训练20步、批大小改为8；不设置时仍使用书中的完整超参数。

```powershell
$env:NNDL_QUICK_RUN = '1'
jupyter notebook pytorch/chap10大语言模型与智能体/大语言模型与智能体-上.ipynb
```

## 跑测试

每章配一份 `pytest` sanity 检查（关键算子 / 关键结论），放在 [`tests/`](tests/) 下：

```bash
# 在仓库根目录运行
python -m pytest pytorch/tests/ -v
```

测试耗时取决于设备和已安装的可选依赖；完整训练实验另需运行对应Notebook。

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
| `text` | 训练词表、IMDB预处理与批次补齐 | chap6 / chap8 |
| `llm` | NanoGPT、LoRA、采样与完整模型KV缓存 | chap10 |
| `data` | 合成数据集（Moon1000 / Multi1000 / DigitSum / Sentiment） | 多章共用 |

## 与 Paddle 版的对应

两版的前8章主题对应，可独立使用；本轮实验配置、实现与指标以PyTorch版为准。Paddle 版只覆盖前 8 章；图神经网络（chap9）与大语言模型与智能体（chap10）是 PyTorch 版独有。
