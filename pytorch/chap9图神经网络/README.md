# chap9 图神经网络（PyTorch）

| Notebook | 内容 |
|---|---|
| [`图神经网络.ipynb`](图神经网络.ipynb) | 从零纯 PyTorch 实现 GCN / GraphSAGE / GAT / GIN；Zachary's Karate Club 上 4 种 GNN 半监督节点分类对比；图级任务玩具数据集（分子图回归）；PyTorch Geometric 等价写法 |

## 实现要点

- **消息传递抽象**：所有 GNN 共享 `message → aggregate → update` 三步。`MessagePassing` 抽象基类把这三步独立出来，子类只重写需要的部分（mean aggregator 演示在基类上的最小实现）。
- **GCN 归一化**：$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 一次性算好缓存——$\tilde{A} = A + I$ 加自环避免每层都丢失自身信息。手写时小心：每层都要用归一化邻接矩阵相乘，不是只乘一次。
- **GraphSAGE**：写成针对单节点 $i$ 的聚合形式（`CONCAT(h_i, AGG(neighbors))` 后过 `W`），支持邻居采样。`AGG` 可换 mean / sum / max / LSTM。
- **GAT 注意力 + 邻居 mask**：先算所有 pair 的 $\alpha_{ij}$，再用邻接矩阵的 0 位置填 `-inf` 让 softmax 自然忽略非邻居。多头是 $K$ 套独立 $(W, \boldsymbol{a})$ 并行算，最后 concat 或平均。
- **GIN 用 sum 而非 mean**：mean 聚合会损失邻居数量信息（区分不了 $\{a\}$ 和 $\{a, a\}$），GIN 改用 sum 保证单射；再过 MLP（多层 ReLU）拟合任意单射函数。$\epsilon$ 可学习时调整中心节点自身贡献。
- **节点级 vs 图级任务**：节点分类直接拿每个节点的最终隐藏向量过 softmax；图级任务（分子图回归）需要 **readout**——把所有节点的表示 sum / mean / max 聚合成单个图级向量。GIN 论文推荐拼接每层的 readout，让不同深度的子结构信息都参与。
- **半监督设定**：Karate Club 只有 2 个标签节点（教练 Mr. Hi 和管理员 Officer），其余 32 个全部 unlabeled。loss 只在标注节点上计算，但 forward 全图同时传消息——这正是 GNN 半监督的关键。
- **PyG 等价写法**：实际项目用 [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)，`edge_index` 稀疏表示比稠密邻接矩阵省显存。本章手写版本作为"学具"，理解原理后切到 `GCNConv` / `SAGEConv` / `GATConv` / `GINConv` 一行替换即可。

## 测试

```bash
python -m pytest pytorch/tests/test_chap9.py -v
```
