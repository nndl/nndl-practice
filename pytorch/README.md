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

完整依赖见 [`requirements.txt`](requirements.txt)：还包括 `numpy`、`pandas`、`matplotlib`、`scikit-learn`、`networkx` 与 `torch-geometric`（chap9）、`jupyter`、`pytest`。

## 安装

```bash
# 以下命令从仓库根目录 nndl-practice 执行
# 1) 装一个干净的 venv（推荐）
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2) 装 PyTorch
# CPU（Linux / Windows；macOS 使用官方对应命令）
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# GPU（按显卡驱动与所选构建的兼容条件安装）
# 参见 https://pytorch.org/get-started/ 选对应的 wheel

# 3) 装其余依赖
python -m pip install -r pytorch/requirements.txt
```

如果 Notebook 提示找不到已安装的模块，先检查当前内核。可用以下命令注册环境，再在 Jupyter 的内核菜单中选择 `Python (nndl-practice)`：

```bash
python -m ipykernel install --user --name nndl-practice --display-name "Python (nndl-practice)"
```

## 跑通一个章节

```bash
# 在仓库根目录
python -m jupyter notebook "pytorch/chap1实践基础/实践基础.ipynb"
```

或者用 nbconvert 在命令行端到端执行（CI / 复跑用）：

```bash
python -m jupyter nbconvert --to notebook --execute "pytorch/chap1实践基础/实践基础.ipynb" --output=chap1-executed --output-dir=.outputs --ExecutePreprocessor.timeout=600
```

执行结果写入仓库根目录下的 `.outputs/`，保留原 Notebook。命令在 PowerShell 和 Bash 中均可单行运行。后续章节按各章 README 从头顺序执行；较长训练需要相应增大单元格超时。

## 数据准备

| 章节 | 数据来源与准备方式 |
|---|---|
| 第 1、3、4、7 章 | 合成数据或 scikit-learn 内置 Iris，无需单独下载 |
| 第 2 章下 | Notebook 调用 `fetch_california_housing`，首次联网，随后复用 scikit-learn 缓存 |
| 第 5 章 | Notebook 通过 torchvision 下载 MNIST / CIFAR-10 到 `~/.cache/torch_data` |
| 第 6 章下、第 8 章上 | 运行下方 IMDB 准备命令，写入 `pytorch/datasets/imdb/` |
| 第 8 章下 | 运行下方 LCQMC / BERT 字表准备命令，写入 `pytorch/datasets/` |
| 第 9 章 | PyG 首次下载 Cora / PROTEINS，缓存于本章 `data/` |
| 第 10 章上 | 首次下载 TinyShakespeare；下篇读取上篇保存的预训练检查点 |

```bash
# 在仓库根目录运行；只准备要学习的章节所需数据
python pytorch/datasets/download.py --only=imdb
python pytorch/datasets/download.py --only=lcqmc,bert_vocab
```

下载脚本会检查全部必需文件，损坏或缺失时重新准备；失败时以非零状态退出。具体来源、划分和缓存说明见 [`datasets/README.md`](datasets/README.md)。

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

测试包括逐章算子核验，以及数据准备中断恢复、学习率调度、评价状态等回归用例。数据准备测试使用本地小压缩包，不联网；完整训练实验另需运行对应 Notebook。

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

`RunnerV3.evaluate()` 和 `predict()` 在关闭梯度的评价模式中执行，结束后恢复调用前的模式。训练损失和指标按样本数加权汇总；空数据会给出明确错误。学习率历史记录该回合实际使用的学习率。`ReduceLROnPlateau` 接收验证指标，其 `mode` 要与 `higher_is_better` 一致；普通调度器每回合调用一次。需要逐次更新学习率时，应使用章节中的自定义训练循环。接口约定参见 [PyTorch 调度器文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)。

## 与 Paddle 版的对应

两版的前8章主题对应，可独立使用；本轮实验配置、实现与指标以PyTorch版为准。Paddle 版只覆盖前 8 章；图神经网络（chap9）与大语言模型与智能体（chap10）是 PyTorch 版独有。
