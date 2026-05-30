# datasets —— PyTorch 笔记本的本地数据

各章 notebook 用相对路径 `../datasets/<name>` 读取（见各章“数据加载”cell 里的候选路径列表）。
数据较大且为公开数据集，**已 gitignore 不入库**；用下面的 `download.py` 一键准备。

| 目录 / 文件 | 用于 |
|---|---|
| `imdb/` | chap6 循环 / chap8 注意力（IMDB 情感，gzip）|
| `lcqmc/` | chap8 注意力（LCQMC 中文句对，gzip）|
| `cifar-10-batches-py/` | chap5 卷积（CIFAR-10 原始 pickle）|
| `bert-base-chinese/vocab.txt` | chap8 LCQMC Transformer 精确复刻字表（21128）|
| `boston_house_prices.csv` | chap2 机器学习概述（线性回归）|

## 准备数据

本目录自带 `download.py`，从国内可达镜像（ModelScope / 百度 BOS 等）一键下载，可反复运行（已存在则跳过）：

```bash
# 在仓库根目录 nndl-practice 下运行
python pytorch/datasets/download.py
# 或只下指定数据集
python pytorch/datasets/download.py --only=lcqmc,bert_vocab
```

- MNIST 由 `torchvision.datasets.MNIST(download=True)` 自动下载；Iris 由 `sklearn.datasets.load_iris()` 内置，均无需在此准备。
- 数据文件本身不入库（见仓库 `.gitignore`）。各 notebook 的候选路径里还保留了 `../../../practice-in-paddle/dataset/...` 作为回退，所以即便这里为空、只要隔壁 practice-in-paddle 备好数据也能跑。
