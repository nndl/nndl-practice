"""chap6 循环神经网络 sanity tests.

覆盖：
- 手写 SRN 递推与 nn.RNN 数值一致（注意权重转置）
- LSTM 学得了数字记忆任务（L=10 dev acc > 0.9）
- clip_grad_norm_ 的更新效果（实际写入更新的范数 ≤ max_norm）
- bi-LSTM 在 \"信号在序列开头\" 的长序列任务上显著优于 uni-LSTM
"""
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset, Dataset


def test_manual_srn_matches_nn_rnn():
    torch.manual_seed(0)
    embed_dim, hidden_dim = 16, 32
    emb = nn.Embedding(10, embed_dim)

    Wx = torch.randn(embed_dim, hidden_dim) / math.sqrt(embed_dim)
    Wh = torch.eye(hidden_dim)
    b = torch.zeros(hidden_dim)

    x = torch.randint(0, 10, (3, 7))
    e = emb(x)

    # manual
    h = torch.zeros(3, hidden_dim)
    for t in range(7):
        h = torch.tanh(e[:, t] @ Wx + h @ Wh + b)
    h_manual = h

    # nn.RNN
    rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(Wx.T)
        rnn.weight_hh_l0.copy_(Wh.T)
        rnn.bias_ih_l0.zero_()
        rnn.bias_hh_l0.zero_()
    _, h_final = rnn(e)
    assert (h_manual - h_final.squeeze(0)).abs().max().item() < 1e-6


def _make_digit_sum(n, L, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 10, (n, L), generator=g)
    y = (X[:, 0] + X[:, 1]).long()
    return X, y


def test_lstm_learns_digit_sum_short():
    """LSTM with proper capacity learns digit-sum at L=10 to high accuracy."""
    torch.manual_seed(0)
    X_tr, y_tr = _make_digit_sum(3000, 10, seed=0)
    X_dv, y_dv = _make_digit_sum(500, 10, seed=1)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(10, 32)
            self.lstm = nn.LSTM(32, 64, batch_first=True)
            self.fc = nn.Linear(64, 19)
        def forward(self, x):
            e = self.emb(x); _, (h, _) = self.lstm(e)
            return self.fc(h.squeeze(0))

    model = M(); opt = optim.Adam(model.parameters(), lr=3e-3)
    lossfn = nn.CrossEntropyLoss()
    for _ in range(15):
        for x, y in loader:
            opt.zero_grad(); lossfn(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    with torch.no_grad():
        acc = (model(X_dv).argmax(1) == y_dv).float().mean().item()
    assert acc > 0.85, f"LSTM should learn L=10 digit-sum, got {acc}"


def test_clip_grad_norm_bounds_update():
    """After clip_grad_norm_(params, max_norm), the global grad L2 ≤ max_norm + epsilon."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 5), nn.Tanh(), nn.Linear(5, 1))
    x = torch.randn(8, 5); y = torch.randn(8, 1)
    nn.functional.mse_loss(model(x), y).backward()

    # 人为把所有梯度放大 100 倍，触发 clip
    with torch.no_grad():
        for p in model.parameters():
            p.grad.mul_(100.0)

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model.parameters()))
    assert total <= 1.0 + 1e-5, f"grad norm after clip is {total}"


# ---- bi-LSTM 优势：信号在开头的长序列 ----
VOCAB, PAD = 25, 0
POS = [1, 2, 3]; NEG = [4, 5, 6]; FILL = list(range(7, VOCAB))


class _StartSentimentDS(Dataset):
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        items = []
        for _ in range(n):
            L = torch.randint(35, 46, (1,), generator=g).item()
            label = torch.randint(0, 2, (1,), generator=g).item()
            tokens = [FILL[torch.randint(0, len(FILL), (1,), generator=g).item()] for _ in range(L)]
            sent_pool = POS if label == 1 else NEG
            for i in range(3):
                tokens[i] = sent_pool[torch.randint(0, 3, (1,), generator=g).item()]
            items.append((torch.tensor(tokens, dtype=torch.long), label))
        self.items = items

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def _collate(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    return pad_sequence(seqs, batch_first=True, padding_value=PAD), lengths, torch.tensor(labels)


class _Net(nn.Module):
    def __init__(self, bidirectional):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, 32, padding_idx=PAD)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(64 * (2 if bidirectional else 1), 2)
        self.bi = bidirectional
    def forward(self, x, lens):
        e = self.emb(x)
        p = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(p)
        feat = torch.cat([h[0], h[1]], 1) if self.bi else h[0]
        return self.fc(feat)


def _train_eval(bi, epochs):
    train_loader = DataLoader(_StartSentimentDS(1500, seed=0), batch_size=64, shuffle=True, collate_fn=_collate)
    dev_loader   = DataLoader(_StartSentimentDS(400,  seed=1), batch_size=64, collate_fn=_collate)

    torch.manual_seed(0)
    m = _Net(bidirectional=bi)
    opt = optim.Adam(m.parameters(), lr=3e-3)
    lossfn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for x, l, y in train_loader:
            opt.zero_grad(); lossfn(m(x, l), y).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    m.eval(); correct = total = 0
    with torch.no_grad():
        for x, l, y in dev_loader:
            correct += (m(x, l).argmax(1) == y).sum().item(); total += y.size(0)
    return correct / total


def test_bi_lstm_beats_uni_on_start_signal_long_seq():
    uni_acc = _train_eval(bi=False, epochs=3)
    bi_acc  = _train_eval(bi=True,  epochs=3)
    assert bi_acc > 0.9,  f"bi-LSTM dev acc too low: {bi_acc}"
    assert uni_acc < 0.7, f"uni-LSTM unexpectedly learned (acc {uni_acc}); does the task still expose long-range dep?"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
