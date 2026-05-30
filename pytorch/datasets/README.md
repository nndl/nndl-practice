# datasets —— PyTorch 笔记本的本地数据

各章 notebook 用相对路径 `../datasets/<name>` 读取（见各章“数据加载”cell 里的候选路径列表）。
数据较大且为公开数据集，**已 gitignore 不入库**；换机器时按下面重新准备即可。

| 目录 | 用于 |
|---|---|
| `imdb/` | chap6 循环 / chap8 注意力（IMDB 情感，gzip）|
| `lcqmc/` | chap8 注意力（LCQMC 中文句对，gzip）|
| `cifar-10-batches-py/` | chap5 卷积（CIFAR-10 原始 pickle）|
| `bert-base-chinese/vocab.txt` | chap8 LCQMC Transformer 精确复刻字表（21128）|

## 重新准备

数据从同工作区 `practice-in-paddle/dataset` 复制而来（那边的 `download.py` 可从国内镜像下载，含 `bert_vocab`）：

```powershell
# 1) 先在 practice-in-paddle 下载齐（含 bert 词表）
python ..\..\..\practice-in-paddle\dataset\download.py
# 2) 复制到这里
$src = '..\..\..\practice-in-paddle\dataset'
foreach ($d in 'imdb','lcqmc','cifar-10-batches-py','bert-base-chinese') { Copy-Item -Recurse -Force "$src\$d" . }
```

各 notebook 的候选路径里仍保留 `../../../practice-in-paddle/dataset/...` 作为回退，
所以即使这里为空、只要 practice-in-paddle 备好数据也能跑。
