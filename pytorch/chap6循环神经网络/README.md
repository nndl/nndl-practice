# chap6 循环神经网络（PyTorch）

| Notebook | 内容 |
|---|---|
| [`循环神经网络-上.ipynb`](循环神经网络-上.ipynb) | SRN 从头实现 + `nn.RNN` 等价验证；SRN vs LSTM 在数字记忆任务上不同序列长度的表现；`clip_grad_norm_` 的效果 |
| [`循环神经网络-下.ipynb`](循环神经网络-下.ipynb) | 双向 LSTM、变长序列 `pad_sequence` + `pack_padded_sequence`；信号集中在开头的长序列上 bi-LSTM 100% vs uni-LSTM 50% |

## 实现要点

- **`nn.RNN` 的权重布局**：`weight_ih_l0` shape `[hidden, input]`，`weight_hh_l0` shape `[hidden, hidden]`——和手写的 `Wx (input, hidden)` / `Wh (hidden, hidden)` **互为转置**。手写 SRN 与 `nn.RNN` 对齐时记得 `.T`。
- **`nn.LSTM`** 返回 `(out, (h, c))`；单向时 `h.shape = [num_layers, B, hidden]`，双向时 `h.shape = [num_layers * 2, B, hidden]`，最后一层的 forward / backward 隐状态分别是 `h[-2]` / `h[-1]`（或 `h[0] / h[1]` 当 `num_layers=1`）。
- **梯度截断**：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` 写在 `loss.backward()` 之后、`opt.step()` 之前。RNN/Transformer 训练都默认打开。
- **变长序列**：
  - `pad_sequence(seqs, batch_first=True, padding_value=PAD)` → 把同 batch 序列 pad 到等长
  - `pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)` → 跳过 PAD 位置；**`lengths` 必须在 CPU**
  - `nn.Embedding(..., padding_idx=PAD)` → PAD token 的 embedding 梯度归零
- **双向 vs 单向**：当任务有效信息分布在序列两端时（NER / POS / 阅读理解），bi-LSTM 显著优于 uni-LSTM——chap6 下用的合成任务把信号放在开头 + 长尾，单向 LSTM 卡在 50% 随机水平。

## 测试

```bash
python -m pytest pytorch/tests/test_chap6.py -v
```
