"""chap7 网络优化与正则化 sanity tests.

覆盖：
- Kaiming 初始化在深 ReLU 网络上能保持激活方差稳定
- BatchNorm 在 train / eval 模式下的统计量来源不同
- Dropout 在 eval 模式下被自动关掉
- lr_scheduler.CosineAnnealingLR 的衰减形状正确
- weight_decay 的作用（参数 L2 范数随训练减小）
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def test_kaiming_init_preserves_activation_variance():
    """10 层 ReLU 网络，Kaiming 初始化后每层输出 std 在 0.7-2.0 之间（同量级，不衰减）。"""
    torch.manual_seed(0)
    layers = []
    for _ in range(10):
        layers.append(nn.Linear(512, 512)); layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    x = torch.randn(64, 512)
    stds = []
    with torch.no_grad():
        for m in model:
            x = m(x)
            if isinstance(m, nn.Linear):
                stds.append(x.std().item())
    # 10 层后激活的 std 应当与第一层在同一量级；naive 初始化会衰减到 << 1
    assert 0.5 < stds[-1] < 3.0, f"kaiming should preserve std; final std={stds[-1]}, full={stds}"


def test_naive_init_decays_variance():
    torch.manual_seed(0)
    layers = []
    for _ in range(10):
        layers.append(nn.Linear(512, 512)); layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.05)
            nn.init.zeros_(m.bias)
    x = torch.randn(64, 512)
    with torch.no_grad():
        for m in model:
            x = m(x)
    assert x.std().item() < 0.3, f"naive std=0.05 should decay through 10 layers"


def test_batchnorm_train_vs_eval():
    """train() 用 batch 统计；eval() 用 running 统计——两种模式下相同输入会得到不同结果（在 running stats 未稳定时）。"""
    torch.manual_seed(0)
    bn = nn.BatchNorm1d(8, momentum=0.1)
    x = torch.randn(16, 8) * 3 + 5      # 显著偏离 0 均值 1 方差

    bn.train()
    out_train = bn(x)
    # train 模式下，每个特征通道在 batch 内被标准化，输出 std 接近 1、均值接近 0
    assert out_train.std(dim=0).mean().item() < 1.3
    assert out_train.mean(dim=0).abs().mean().item() < 1e-5

    # 首次 eval 时 running_mean 接近 0 (初值)、running_var 接近 1，所以输出基本是 x 自身
    bn.eval()
    out_eval = bn(x)
    # eval 输出未经 batch 归一化，应当远离 train 模式的输出
    assert (out_train - out_eval).abs().mean().item() > 0.5


def test_dropout_eval_is_identity():
    torch.manual_seed(0)
    drop = nn.Dropout(p=0.9)
    x = torch.randn(100, 10)

    drop.train()
    out_train = drop(x)
    # 大约 90% 的元素被置零
    zero_frac = (out_train == 0).float().mean().item()
    assert zero_frac > 0.7, f"dropout 0.9 should zero most elements; zeroed={zero_frac}"

    drop.eval()
    out_eval = drop(x)
    assert torch.allclose(out_eval, x)


def test_cosine_lr_scheduler_shape():
    """CosineAnnealingLR 的 lr 应当从 initial 平滑衰减到 eta_min。"""
    model = nn.Linear(2, 2)
    opt = optim.SGD(model.parameters(), lr=0.1)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=0.0)
    lrs = [opt.param_groups[0]['lr']]
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]['lr'])
    # 单调递减，且大致到 0
    assert lrs[0] > lrs[50] > lrs[-1]
    assert lrs[-1] < 1e-3
    # 半程时大约 lr=0.05（cosine 在 T/2 处）
    assert abs(lrs[50] - 0.05) < 0.01


def test_weight_decay_shrinks_params_more_than_no_decay():
    """同样的训练步骤下，weight_decay > 0 会让参数 L2 范数比 weight_decay=0 小。"""
    def train(wd, seed=0):
        torch.manual_seed(seed)
        model = nn.Linear(10, 1)
        nn.init.normal_(model.weight, std=1.0)
        opt = optim.SGD(model.parameters(), lr=0.1, weight_decay=wd)
        x = torch.zeros(8, 10); y = torch.zeros(8, 1)        # 全 0 数据 → loss 只想让 w·0+b ≈ 0
        for _ in range(50):
            opt.zero_grad()
            F.mse_loss(model(x), y).backward()
            opt.step()
        return model.weight.norm().item()

    no_wd = train(wd=0.0)
    with_wd = train(wd=0.5)
    assert with_wd < no_wd, f"weight_decay should shrink params: no_wd={no_wd}  with_wd={with_wd}"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok:", name)
