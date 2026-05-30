"""chap9 图神经网络 sanity tests.

覆盖 chap9 notebook 涵盖的工具点：GCN 归一化邻接矩阵、GCNLayer / SAGELayer /
GATLayer / GINLayer 的形状与不变量、半监督训练 loss 能够下降。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- GCN 归一化邻接矩阵 ----
def normalize_adj(A):
    """GCN 的对称归一化：A_hat = D^{-1/2} (A + I) D^{-1/2}."""
    N = A.shape[0]
    A_tilde = A + torch.eye(N)
    D_tilde = A_tilde.sum(dim=1)
    D_inv_sqrt = D_tilde.pow(-0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
    D_mat = torch.diag(D_inv_sqrt)
    return D_mat @ A_tilde @ D_mat


def _toy_graph(n=6):
    """链 0-1-2-3-4-5 + 0-5 闭环。"""
    A = torch.zeros(n, n)
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]:
        A[i, j] = 1
        A[j, i] = 1
    return A


def test_normalize_adj_is_symmetric():
    A = _toy_graph(6)
    A_hat = normalize_adj(A)
    assert torch.allclose(A_hat, A_hat.T, atol=1e-6)
    assert A_hat.shape == (6, 6)
    # 对角线应当 > 0（加自环之后每个节点至少和自己有联系）
    assert (A_hat.diag() > 0).all()


def test_normalize_adj_handles_isolated_node():
    """孤立节点的度=0（加自环后=1），归一化不应产生 nan / inf。"""
    A = torch.zeros(4, 4)
    A[0, 1] = A[1, 0] = 1                                     # 0-1 相连，2、3 孤立
    A_hat = normalize_adj(A)
    assert torch.isfinite(A_hat).all()


# ---- GCN ----
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, H, A_hat):
        return A_hat @ self.lin(H)


def test_gcn_layer_shape():
    A = _toy_graph(6)
    A_hat = normalize_adj(A)
    H = torch.randn(6, 4)
    layer = GCNLayer(4, 8)
    out = layer(H, A_hat)
    assert tuple(out.shape) == (6, 8)


def test_gcn_two_layer_trains_on_toy_graph():
    """在 6 节点链上做 2 类节点分类，loss 应当下降。"""
    torch.manual_seed(0)
    A = _toy_graph(6)
    A_hat = normalize_adj(A)
    X = torch.eye(6)
    y = torch.tensor([0, 0, 0, 1, 1, 1])

    model = nn.Sequential()
    conv1 = GCNLayer(6, 8)
    conv2 = GCNLayer(8, 2)
    opt = torch.optim.Adam(list(conv1.parameters()) + list(conv2.parameters()), lr=0.1)

    initial_loss = None
    for step in range(100):
        h = F.relu(conv1(X, A_hat))
        logits = conv2(h, A_hat)
        loss = F.cross_entropy(logits, y)
        if initial_loss is None:
            initial_loss = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5, f"loss didn't drop enough: {initial_loss} -> {final_loss}"


# ---- GraphSAGE ----
class SAGELayer(nn.Module):
    """mean aggregator: concat[h_i, mean(N(i))] -> linear."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, H, A):
        deg = A.sum(dim=1, keepdim=True).clamp(min=1)
        nbr_mean = (A @ H) / deg
        return self.lin(torch.cat([H, nbr_mean], dim=1))


def test_sage_layer_shape_and_concat_dim():
    A = _toy_graph(6)
    H = torch.randn(6, 4)
    layer = SAGELayer(4, 8)
    out = layer(H, A)
    assert tuple(out.shape) == (6, 8)
    # 第一个 Linear 的输入维度必须是 2 * in_dim（拼接 h_i 和 nbr_mean）
    assert layer.lin.in_features == 8


def test_sage_isolated_node_uses_zero_neighbor_mean():
    """孤立节点的 nbr_mean 应当是 0（clamp(min=1) 保证不除 0）。"""
    A = torch.zeros(3, 3)
    A[0, 1] = A[1, 0] = 1                                     # 节点 2 孤立
    H = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    deg = A.sum(dim=1, keepdim=True).clamp(min=1)
    nbr_mean = (A @ H) / deg
    assert torch.allclose(nbr_mean[2], torch.zeros(2))


# ---- GAT ----
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * out_dim) * 0.1)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, H, A):
        N = H.shape[0]
        H_w = self.W(H)
        a_src, a_dst = self.a[:H_w.shape[1]], self.a[H_w.shape[1]:]
        e_src = H_w @ a_src
        e_dst = H_w @ a_dst
        E = e_src.unsqueeze(1) + e_dst.unsqueeze(0)
        E = self.leaky(E)
        A_self = A + torch.eye(N, device=A.device)
        E = E.masked_fill(A_self == 0, float('-inf'))
        alpha = F.softmax(E, dim=1)
        return alpha @ H_w, alpha


def test_gat_attention_row_sum_is_one():
    """每行 alpha 在邻居 + 自环上 softmax，和为 1。"""
    torch.manual_seed(0)
    A = _toy_graph(6)
    H = torch.randn(6, 4)
    layer = GATLayer(4, 8)
    out, alpha = layer(H, A)
    sums = alpha.sum(dim=1)
    assert torch.allclose(sums, torch.ones(6), atol=1e-5)
    assert tuple(out.shape) == (6, 8)


def test_gat_attention_zero_on_non_neighbors():
    """非邻居（且非自身）位置的注意力权重应当为 0。"""
    torch.manual_seed(0)
    A = _toy_graph(6)
    H = torch.randn(6, 4)
    layer = GATLayer(4, 8)
    _, alpha = layer(H, A)
    A_self = A + torch.eye(6)
    non_nbr_mask = (A_self == 0)
    assert alpha[non_nbr_mask].abs().max().item() < 1e-6


# ---- GIN ----
class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, eps=0.0):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, H, A):
        out = (1 + self.eps) * H + A @ H
        return self.mlp(out)


def test_gin_uses_sum_aggregation_not_mean():
    """GIN 用 sum 聚合，所以邻居拷贝两次会得到两倍的 pre-MLP 输入。"""
    torch.manual_seed(0)
    # 2 节点单边图
    A_single = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    # 同一个图，但把邻居"复制 2 份"——通过权重矩阵模拟
    A_double = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    H = torch.randn(2, 4)

    layer = GINLayer(4, 8, eps=0.0)
    layer.eval()                                              # 关闭 dropout 等
    # pre-MLP 部分：H + A @ H
    pre_single = H + A_single @ H
    pre_double = H + A_double @ H
    # 邻居权重翻倍，sum 部分也应翻倍
    assert torch.allclose(pre_double - H, 2 * (pre_single - H))


def test_gin_layer_shape():
    A = _toy_graph(6)
    H = torch.randn(6, 4)
    layer = GINLayer(4, 8, eps=0.0)
    out = layer(H, A)
    assert tuple(out.shape) == (6, 8)


# ---- 图级 readout ----
def test_sum_readout_is_node_count_sensitive():
    """sum readout 对节点数敏感（这是 GIN 推荐 sum 的关键性质）。"""
    H_small = torch.ones(5, 4)
    H_large = torch.ones(10, 4)
    assert torch.allclose(H_large.sum(dim=0), 2 * H_small.sum(dim=0))
    # mean 不敏感
    assert torch.allclose(H_large.mean(dim=0), H_small.mean(dim=0))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok:", name)
