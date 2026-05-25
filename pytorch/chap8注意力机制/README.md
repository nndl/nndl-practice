# chap8 注意力机制（PyTorch）

| Notebook | 内容 |
|---|---|
| [`注意力机制-上.ipynb`](注意力机制-上.ipynb) | 加性注意力 + masking + BiLSTM 上加注意力做情感分类 + 注意力权重可视化 |
| [`注意力机制-下.ipynb`](注意力机制-下.ipynb) | scaled dot-product attention、多头注意力、sinusoidal 位置编码、Transformer Encoder（Pre-LN）、与 `nn.TransformerEncoder` 等价 |

## 实现要点

- **加性注意力**：$s_t = v^\top \tanh(W h_t)$。一份 query（这里就是 LSTM 隐状态序列）。
- **Scaled dot-product attention**：$\text{softmax}(Q K^\top / \sqrt{d_k}) V$。$\sqrt{d_k}$ 缩放避免高维下点积过大、softmax 集中。
- **mask padding**：把 score 设成 $-\infty$ 让 softmax 自然为 0。不要先 softmax 再乘 0/1 mask（会破坏归一化）。
- **多头**：把 $Q/K/V$ 切到 $h$ 个低维子空间并行；后接一个 `W_o` 投影回原维度。
- **位置编码**：自注意力本身位置无感；sinusoidal PE 或 learned PE 都常用。
- **Transformer block** = MHA + 残差 + LayerNorm + FFN + 残差 + LayerNorm。**Pre-LN**（norm 在 sublayer 前）训练更稳，是现代默认。
- **`nn.TransformerEncoderLayer`** 的 `src_key_padding_mask` 约定**与教科书相反**：`True` 表示要 **ignore** 的位置（padding）。手写转用内置时要 `~mask` 取反。

## 测试

```bash
python -m pytest pytorch/tests/test_chap8.py -v
```
