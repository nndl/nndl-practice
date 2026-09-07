# Notebook 数据准备

从仓库根目录运行以下命令，数据会写入本目录。数据文件不入 Git；首次准备需要联网，完成后可复用本地文件。

| 数据 | 用途 | 准备命令 |
|---|---|---|
| IMDB | 第6章下、第8章上：影评情感分类 | `python pytorch/datasets/download.py --only=imdb` |
| LCQMC 与 BERT 中文字表 | 第8章下：中文句对匹配 | `python pytorch/datasets/download.py --only=lcqmc,bert_vocab` |
| CIFAR-10 原始批次 | 本地原始数据副本 | `python pytorch/datasets/download.py --only=cifar10` |

不加 `--only` 时依次准备以上四项。第5章 Notebook 自行通过 torchvision 下载 MNIST / CIFAR-10 到 `~/.cache/torch_data`，与本目录的 CIFAR-10 副本是两个位置，通常无需重复下载。

第2章当前使用加州房价，由 Notebook 调用 `sklearn.datasets.fetch_california_housing` 获取；Iris 由 scikit-learn 内置。旧 `--only=boston` 入口仅作兼容保留，不在默认下载列表中。

## 文件与实验划分

- `imdb/{train,dev,test,vocab}.txt.gz`：每条影评为 `标签<TAB>文本`。沿用原流程按文件名排序，再用种子42从官方训练部分分出5,000条验证数据；训练/验证/测试为20,000/5,000/25,000条。统一小写、HTML换行和空白处理。词表只统计训练文本，预留 `[PAD]=0`、`[UNK]=1`，最多再保留50,000个词。
- `lcqmc/{train,dev,test}.txt.gz`：每行为 `句子A<TAB>句子B<TAB>标签`，标签为0或1；去除表头，保留原始数据划分。
- `bert-base-chinese/vocab.txt`：21,128项中文BERT字表。本章按字符查表，不加载BERT预训练权重，也不执行WordPiece分词。
- `cifar-10-batches-py/`：5个训练批次、测试批次与类别元数据。

各章短程实验还会从以上数据中抽取子集，具体规模以 Notebook 为准。

## 中断与重试

脚本只读取压缩包中的所需普通文件，不展开整个目录。IMDB按压缩包顺序读取，再恢复原有排序和随机划分，避免在磁盘上生成数万个小文件。

已有数据只有在全部必需文件齐全时才跳过；gzip文件还会检查是否可完整解压。缺失、空文件或损坏的gzip会触发重新准备。新数据先写入临时目录，全部转换成功后再逐文件替换目标；转换失败时保留原数据与下载包，便于重试。

下载中断会清理未完成的 `.part` 文件。参数拼写错误、缺失数据划分、格式错误和无效BERT字表都会使命令失败，按报错处理后重跑原命令即可。CIFAR-10缓存检查覆盖文件齐全和非空，不替代torchvision的内容校验。

## 下载来源

IMDB和CIFAR-10分别使用[Stanford原始数据](https://ai.stanford.edu/~amaas/data/sentiment/)与[Toronto原始数据](https://www.cs.toronto.edu/~kriz/cifar.html)，LCQMC使用百度BOS上的PaddleHub数据包，BERT字表使用ModelScope镜像。准确下载地址统一维护在[download.py](download.py)的 `SOURCES` 中；各来源的网络可达性可能不同。
