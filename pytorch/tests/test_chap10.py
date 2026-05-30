"""chap10 大语言模型与智能体 sanity tests.

覆盖 chap10 上 / 下 notebook 涵盖的工具点：因果掩码、CausalSelfAttention、
NanoGPT 前向 + 损失、采样策略（top-k / top-p）、LoRALinear 初始等价、
SFT 的 ignore_index、DPO 损失公式形状。
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- 因果掩码 ----
def test_causal_mask_is_strict_upper_triangular():
    """因果掩码：True 表示要屏蔽的未来位置（严格上三角，对角线 False）。"""
    T = 5
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    # 对角线及以下为 False（保留），上三角为 True（屏蔽）
    assert (mask.diag() == False).all()
    assert (mask[0, 1:] == True).all()                        # 行 0 的右侧全是未来
    assert (mask[T - 1, :] == False).all()                    # 最后一行没有未来


def test_softmax_with_causal_mask_is_lower_triangular():
    """masked_fill(mask, -inf) + softmax 让注意力权重在上三角处为 0。"""
    T = 6
    scores = torch.randn(T, T)
    mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    # 上三角处必须为 0
    assert attn[mask].abs().max().item() < 1e-6
    # 每行和为 1
    assert torch.allclose(attn.sum(dim=-1), torch.ones(T), atol=1e-5)


# ---- CausalSelfAttention ----
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


def test_causal_self_attention_shape():
    attn = CausalSelfAttention(n_embd=32, n_head=4, block_size=16)
    x = torch.randn(2, 8, 32)
    out = attn(x)
    assert tuple(out.shape) == (2, 8, 32)


def test_causal_self_attention_position_t_uses_only_positions_le_t():
    """位置 t 的输出不应受位置 > t 的输入影响（因果性）。"""
    torch.manual_seed(0)
    attn = CausalSelfAttention(n_embd=16, n_head=2, block_size=10)
    attn.eval()
    x = torch.randn(1, 6, 16)
    out_full = attn(x)

    # 修改位置 4 之后的输入，前 4 个位置的输出应当不变
    x_perturbed = x.clone()
    x_perturbed[:, 4:, :] += torch.randn_like(x_perturbed[:, 4:, :])
    out_perturbed = attn(x_perturbed)
    assert torch.allclose(out_full[:, :4, :], out_perturbed[:, :4, :], atol=1e-5)


# ---- NanoGPT ----
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer=2, n_head=2, n_embd=32, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def test_nanogpt_forward_shape():
    model = NanoGPT(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32)
    idx = torch.randint(0, 65, (2, 10))
    logits, loss = model(idx)
    assert tuple(logits.shape) == (2, 10, 65)
    assert loss is None


def test_nanogpt_loss_computed_when_targets_given():
    model = NanoGPT(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32)
    idx = torch.randint(0, 65, (2, 10))
    targets = torch.randint(0, 65, (2, 10))
    _, loss = model(idx, targets)
    assert loss is not None
    assert loss.shape == ()                                   # scalar
    # 未训练模型应当接近均匀分布预测，loss ≈ log(vocab_size) = log(65) ≈ 4.17
    assert 3.5 < loss.item() < 5.0


def test_nanogpt_trains_on_simple_task():
    """1 步训练后 loss 应当下降（最小可信训练信号）。"""
    torch.manual_seed(0)
    model = NanoGPT(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    idx = torch.randint(0, 10, (4, 8))
    targets = torch.randint(0, 10, (4, 8))
    initial_loss = model(idx, targets)[1].item()
    for _ in range(50):
        _, loss = model(idx, targets)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < initial_loss * 0.5


# ---- 采样策略 ----
def test_top_k_filtering_keeps_only_k_logits_finite():
    """top-k 之后只有 top-k 个 logit 是有限值。"""
    torch.manual_seed(0)
    logits = torch.randn(2, 20)
    k = 5
    v, _ = torch.topk(logits, k=k)
    logits_filtered = logits.clone()
    logits_filtered[logits_filtered < v[:, [-1]]] = float('-inf')
    finite_per_row = torch.isfinite(logits_filtered).sum(dim=-1)
    assert (finite_per_row == k).all()


def test_top_p_filtering_drops_low_prob_tail():
    """top-p (nucleus)：cumulative prob 超过 p 的位置之后全部 -inf。"""
    torch.manual_seed(0)
    logits = torch.randn(1, 20)
    p_threshold = 0.9
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    mask = cum_probs > p_threshold
    # 标准实现：右移一位，保证第一个超过 p 的位置被保留
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = float('-inf')
    filtered = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
    # 重新归一化后，保留位置的累计概率应当 >= p_threshold
    probs = F.softmax(filtered, dim=-1)
    kept_probs = F.softmax(logits, dim=-1)[probs > 0]
    assert kept_probs.sum().item() >= p_threshold


# ---- LoRA ----
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=4, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(torch.zeros(r, in_f))
        self.B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling


def test_lora_at_init_equals_base():
    """初始化时 B=0，LoRA 增量 BA=0，输出应当等于 base Linear。"""
    torch.manual_seed(0)
    base = nn.Linear(64, 32, bias=False)
    lora = LoRALinear(base, r=4, alpha=16)
    x = torch.randn(3, 64)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_lora_only_a_and_b_trainable():
    """base 的权重不可训练，只有 A、B 是可训练参数。"""
    base = nn.Linear(64, 32, bias=False)
    lora = LoRALinear(base, r=4, alpha=16)
    trainable = [(name, p) for name, p in lora.named_parameters() if p.requires_grad]
    trainable_names = {name for name, _ in trainable}
    assert trainable_names == {'A', 'B'}, f"unexpected trainable: {trainable_names}"


def test_lora_param_count_reduction():
    """LoRA 显著降低可训练参数量。"""
    d = 128
    base = nn.Linear(d, d, bias=False)
    lora = LoRALinear(base, r=4, alpha=16)
    full = sum(p.numel() for p in base.parameters())            # d^2
    train = sum(p.numel() for p in lora.parameters() if p.requires_grad)  # 2*d*r
    assert train == 2 * d * 4
    assert full / train >= 10                                   # 至少 10x 压缩


def test_lora_gradient_only_flows_through_ab():
    """反向传播后，base 的 grad 应当是 None，A/B 的 grad 应当非零。"""
    torch.manual_seed(0)
    base = nn.Linear(8, 8, bias=False)
    lora = LoRALinear(base, r=2, alpha=4)
    # 让 B 非零，否则 LoRA 路径输出为 0、上游也没梯度
    with torch.no_grad():
        lora.B.add_(0.1 * torch.randn_like(lora.B))
    x = torch.randn(4, 8)
    out = lora(x).sum()
    out.backward()
    assert lora.base.weight.grad is None
    assert lora.A.grad is not None and lora.A.grad.abs().max() > 0
    assert lora.B.grad is not None and lora.B.grad.abs().max() > 0


# ---- SFT 的 ignore_index ----
def test_cross_entropy_ignore_index_skips_prompt():
    """target 设成 -100 的位置不参与 loss 计算（SFT 关键技巧）。"""
    V = 10
    logits = torch.randn(5, V, requires_grad=True)
    # 全部 -100 → loss 应该是 0（或 nan，取决于实现，这里测对应数值）
    targets_all_ignored = torch.full((5,), -100, dtype=torch.long)
    loss_all = F.cross_entropy(logits, targets_all_ignored)
    assert torch.isnan(loss_all) or loss_all.item() == 0.0    # 均值除以 0 的实现差异

    # 一半 ignored，一半有效
    targets_half = torch.tensor([-100, -100, 3, 5, 7])
    loss_half = F.cross_entropy(logits, targets_half)
    # 只用后 3 个位置计算
    loss_manual = F.cross_entropy(logits[2:], targets_half[2:])
    assert torch.allclose(loss_half, loss_manual, atol=1e-5)


# ---- DPO 损失 ----
def test_dpo_loss_formula_shapes_and_sign():
    """DPO 损失：当 win 优势 > lose 时损失应当较小（正向训练信号）。"""
    beta = 0.1
    # lp_w - lr_w 比 lp_l - lr_l 大很多 → policy 已经偏好 win → loss 小
    lp_w, lr_w = 1.0, 0.0
    lp_l, lr_l = 0.0, 1.0
    logits_win = beta * ((lp_w - lr_w) - (lp_l - lr_l))
    loss_win = -F.logsigmoid(torch.tensor(logits_win)).item()

    # 反过来：policy 偏好 lose → loss 大
    lp_w_bad, lp_l_bad = 0.0, 1.0
    logits_bad = beta * ((lp_w_bad - lr_w) - (lp_l_bad - lr_l))
    loss_bad = -F.logsigmoid(torch.tensor(logits_bad)).item()

    assert loss_win < loss_bad, f"policy 偏好 win 时 loss 应当更小：{loss_win} vs {loss_bad}"


def test_dpo_loss_is_zero_when_logits_large():
    """当 (Δ_w - Δ_l) 极大时，sigmoid → 1，-log(1) → 0。"""
    beta = 1.0
    logits = beta * 20.0                                       # 非常大
    loss = -F.logsigmoid(torch.tensor(logits)).item()
    assert loss < 1e-6


# ---- ReAct 解析 ----
def test_react_action_pattern_extraction():
    """ReAct 主循环用正则从模型输出里抽 Action(tool, args)。"""
    import re
    out = 'Thought: I need to compute 2+3.\nAction: Calculator("2+3")'
    m = re.search(r'Action: (\w+)\("(.+?)"\)', out)
    assert m is not None
    assert m.group(1) == 'Calculator'
    assert m.group(2) == '2+3'


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
