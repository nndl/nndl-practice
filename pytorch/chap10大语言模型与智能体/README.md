# chap10 大语言模型与智能体（PyTorch）

| Notebook | 内容 |
|---|---|
| [`大语言模型与智能体-上.ipynb`](大语言模型与智能体-上.ipynb) | 字符级 nanoGPT 从零实现：因果多头自注意力、pre-LN Transformer block、TinyShakespeare 上预训练；解码策略（greedy / temperature / top-k / top-p）；KV cache 推理加速 |
| [`大语言模型与智能体-下.ipynb`](大语言模型与智能体-下.ipynb) | LoRA 低秩适配器；SFT（大小写转换玩具任务）；DPO 直接偏好优化；ReAct 智能体雏形（Calculator + Search 工具）；RAG 检索增强生成 |

数据集：`tinyshakespeare.txt` 由 notebook 自动从 karpathy/char-rnn 下载（约 1.1 MB），已加入 `.gitignore`。

## 实现要点

### 上：nanoGPT 预训练与采样

- **decoder-only 现代设计**：token + 可学习位置嵌入 → $N$ 个 pre-LN block（LN → MHA → 残差 → LN → FFN → 残差）→ 末尾 LN → 线性层映射到词表。pre-LN 比 post-LN 训练稳定。
- **因果自注意力**：用 `torch.triu(diagonal=1).bool()` 上三角 mask 把 `att.masked_fill(mask, -inf)` 屏蔽未来位置；`qkv` 投影合并到一个 `Linear(n_embd, 3 * n_embd)` 节省一次 matmul。
- **训练目标**：next-token 预测的交叉熵，等价于把 `[x_0, ..., x_{T-1}]` 当输入、`[x_1, ..., x_T]` 当目标（错开一位），所有位置并行算 loss。
- **AdamW + warmup + cosine decay**：`betas=(0.9, 0.95)`、`weight_decay=0.1` 是 GPT/Llama 现代实践。warmup 防早期梯度爆炸，cosine 让末期学习率平滑下降。
- **采样策略**：`temperature` 调整分布尖锐度；`top-k` 只在前 $k$ 个 token 上重归一化采样；`top-p`（nucleus）取累积概率到 $p$ 的最小集合。三者可以叠加。
- **KV cache**：朴素 `generate` 每步重算整个前缀（$O(T^2)$）。缓存每层 attention 的 K/V，新一步只算新位置 q 再与缓存 K/V 做注意力，复杂度降到 $O(T)$。生产推理（vLLM、TGI 等）必备优化。

### 下：微调、对齐与智能体

- **LoRA**：冻结基础 `Linear` 的 $\boldsymbol{W}_0$，只学习低秩增量 $\Delta\boldsymbol{W} = \boldsymbol{B}\boldsymbol{A}$（$\boldsymbol{A}\in\mathbb{R}^{r\times d}$、$\boldsymbol{B}\in\mathbb{R}^{d\times r}$）。$\boldsymbol{B}$ 初始化为 0 保证训练开始时 LoRA 输出等价原模型。`scaling = alpha / r` 解耦秩和学习率。
- **SFT 训练**：在 instruction/response 数据对上继续 next-token loss，**只在 response 部分算 loss**——把 prompt 部分的 target 填 `-100`，`F.cross_entropy` 默认忽略 `-100`。本章用"输入小写文本，输出大写"的玩具任务演示。
- **DPO**：把 RLHF 三步（偏好数据 → 奖励模型 → PPO）合并成单一对比损失，不需要单独训奖励模型、也不需要 RL：

  $$\mathcal{L}_\mathrm{DPO} = -\log\sigma\Big(\beta\big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_\mathrm{ref}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_\mathrm{ref}(y_l|x)}\big]\Big).$$

  $\pi_\mathrm{ref}$ 是冻结的参考模型（通常 SFT 后的副本），$\beta$ 控制偏离参考的程度。
- **ReAct**：让 LM 交替输出 `Thought → Action(tool, args) → Observation(tool result) → ...` 直到 `Answer:`。本章用一个 mock LLM 演示完整控制流（算术 → `Calculator`、事实查询 → `Search`）；真实场景下 LLM 调用替换 `mock_llm` 即可。
- **RAG**：知识库 chunk → embedding → 向量数据库；检索 top-$k$ chunk → 拼接到 prompt → LLM 回答。本章用简单的向量内积检索 + mock embedding 演示流程，生产用 FAISS / Milvus / Chroma 替换。

## 测试

```bash
python -m pytest pytorch/tests/test_chap10.py -v
```
