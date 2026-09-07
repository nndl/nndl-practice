"""Exact model definitions used in the chapter 10 text and Notebooks."""
import math
import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def forward_cached(model, idx, cache=None):
    """Incremental evaluation within the absolute-position context window.

    Cache entries are (key, value) for each block. This function does not
    slide or renumber cached positions. Call with model.eval().
    """
    if model.training:
        raise ValueError('缓存前向仅用于评价模式')
    if idx.ndim != 2 or idx.shape[1] < 1:
        raise ValueError('输入须包含至少一个词元')
    if cache is None:
        cache = [None] * len(model.blocks)
    if len(cache) != len(model.blocks):
        raise ValueError('缓存层数与模型不符')
    start = 0 if cache[0] is None else cache[0][0].shape[2]
    batch, length = idx.shape
    end = start + length
    if end > model.block_size:
        raise ValueError('超出绝对位置窗口；不能直接裁剪缓存并重编号')
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(start, end, device=idx.device))
    x = model.drop(x)
    new_cache = []
    for block, previous in zip(model.blocks, cache):
        attention = block.attn
        h = block.ln1(x)
        q, k, v = attention.qkv(h).chunk(3, dim=-1)
        q, k, v = [z.reshape(batch, length, attention.n_head, attention.head_dim).transpose(1, 2)
                   for z in (q, k, v)]
        if previous is not None:
            expected = (batch, attention.n_head, start, attention.head_dim)
            if previous[0].shape != expected or previous[1].shape != expected:
                raise ValueError('缓存形状不一致')
            k = torch.cat([previous[0], k], dim=2)
            v = torch.cat([previous[1], v], dim=2)
        elif start != 0:
            raise ValueError('部分层缺少缓存')
        # Rectangular query/key matrices need the absolute query offset.
        allowed = torch.arange(end, device=idx.device)[None, :] <= torch.arange(start, end, device=idx.device)[:, None]
        score = (q @ k.transpose(-2, -1)) / math.sqrt(attention.head_dim)
        weights = score.masked_fill(~allowed, -float('inf')).softmax(-1)
        h = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, -1)
        x = x + attention.proj(h)
        x = x + block.ffn(block.ln2(x))
        new_cache.append((k, v))
    return model.head(model.ln_f(x)), new_cache


@torch.no_grad()
def generate_cached(model, prompt_ids, max_new_tokens):
    """Greedy generation without sliding the learned absolute-position window."""
    if max_new_tokens < 0 or prompt_ids.shape[1] + max_new_tokens > model.block_size:
        raise ValueError('生成总长度不得超出模型位置窗口')
    was_training = model.training
    model.eval()
    idx = prompt_ids.clone().to(next(model.parameters()).device)
    current, cache = idx, None
    try:
        for _ in range(max_new_tokens):
            logits, cache = forward_cached(model, current, cache)
            current = logits[:, -1].argmax(-1, keepdim=True)
            idx = torch.cat([idx, current], dim=1)
        return idx
    finally:
        model.train(was_training)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        if n_head <= 0 or n_embd % n_head != 0:
            raise ValueError('头数须为正并整除隐藏维数')
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # q, k, v 投影合在一个矩阵里
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        # 因果掩码
        mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        if not 1 <= T <= self.mask.shape[0]:
            raise ValueError('序列长度超出注意力窗口')
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:T, :T], float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),                                                # GPT-2 风格
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
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=128, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if not 1 <= T <= self.block_size:
            raise ValueError('输入长度须在 1 与 block_size 之间')
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError('temperature 须为有限非负数')
    if top_k is not None and (not isinstance(top_k, int) or not 1 <= top_k <= logits.shape[-1]):
        raise ValueError('top_k 须在 1 与词表大小之间')
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError('top_p 须在 (0, 1] 内')
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        values, indices = logits.topk(top_k, dim=-1)
        logits = torch.full_like(logits, -float('inf')).scatter(-1, indices, values)
    if top_p is not None:
        values, indices = logits.sort(descending=True, dim=-1)
        cumulative = F.softmax(values, dim=-1).cumsum(dim=-1)
        remove = cumulative >= top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        values = values.masked_fill(remove, -float('inf'))
        logits = torch.full_like(logits, -float('inf')).scatter(-1, indices, values)
    return torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)

@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0,
             top_k=None, top_p=None, stop_id=None, generator=None):
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0 or max_new_tokens < 0:
        raise ValueError('提示须为非空二维词元张量，生成长度不得为负')
    was_training = model.training; model.eval()
    idx = prompt_ids.clone().to(next(model.parameters()).device)
    finished = torch.zeros(len(idx), dtype=torch.bool, device=idx.device)
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -model.block_size:])
            next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
            if stop_id is not None:
                next_id = torch.where(finished[:, None], stop_id, next_id)
                finished |= next_id[:, 0] == stop_id
            idx = torch.cat([idx, next_id], dim=1)
            if stop_id is not None and finished.all():
                break
        return idx
    finally:
        model.train(was_training)

class LoRALinear(nn.Module):
    """包装一个 nn.Linear，加上 LoRA 增量 (rank-r) 适配器。"""

    def __init__(self, base: nn.Linear, r=4, alpha=16):
        super().__init__()
        if not isinstance(r, int) or r <= 0:
            raise ValueError('LoRA秩须为正整数')
        self.base = base
        self.base.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        in_f, out_f = base.in_features, base.out_features
        self.A = nn.Parameter(base.weight.new_zeros(r, in_f))
        self.B = nn.Parameter(base.weight.new_zeros(out_f, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B 保持全 0：训练开始时 BA = 0，等价于原模型

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scaling

def apply_lora_to_model(model, r=8, alpha=16, target_names=('qkv', 'proj')):
    """精确匹配叶模块名；避免再次注入导致嵌套适配器。"""
    if any(isinstance(m, LoRALinear) for m in model.modules()):
        raise ValueError('模型已经含有LoRA；请从未注入的副本开始')
    matches = [(name, m) for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and name.split('.')[-1] in target_names]
    if not matches:
        raise ValueError('没有匹配的线性层，请核对模块名称')
    for name, module in matches:
        parent = model
        *path, leaf = name.split('.')
        for part in path:
            parent = getattr(parent, part)
        setattr(parent, leaf, LoRALinear(module, r=r, alpha=alpha))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'可训练 {n_train:,} / 总 {n_total:,} ({n_train/n_total*100:.2f}%)')
    return model
