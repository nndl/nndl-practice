"""三个版本的 Runner，对应书里循序渐进的训练框架。

| 版本       | 关键能力                                            | 引入章节 |
|------------|-----------------------------------------------------|----------|
| RunnerV1   | 直接调最小二乘法解析解；只做保存                    | 第 1 章末尾 |
| RunnerV2   | 加入梯度下降法训练；记录 train/dev 历史；保存最优   | 第 3 章 |
| RunnerV3   | 用 DataLoader 做小批量训练；state_dict 保存；支持   | 第 4 章 |
|            | device / grad_clip / 多输入解包 / scheduler / 早停  |          |

RunnerV3 已经够用于绝大多数任务，后续章节都基于它做轻量扩展。
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# RunnerV1：闭式解 + 保存                                                      #
# --------------------------------------------------------------------------- #
class RunnerV1:
    """最小版 Runner：把"求解析解 + 保存参数"封装成一个对象。

    - model：自定义 Op 算子（持有 params 字典）
    - optimizer：求解器函数 `optimizer(model, X, y, **kwargs)`，
      在 model.params 上原地写入闭式解。
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def fit(self, X, y, **kwargs):
        self.optimizer(self.model, X, y, **kwargs)

    def predict(self, X):
        return self.model(X)

    def evaluate(self, X, y, metric_fn):
        return metric_fn(y, self.model(X)).item()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.params, path)

    def load(self, path):
        self.model.params = torch.load(path, weights_only=True)


# --------------------------------------------------------------------------- #
# RunnerV2：全批量梯度下降 + 早停                                              #
# --------------------------------------------------------------------------- #
class RunnerV2:
    """全批量梯度下降；记录 train/dev 历史到 history 字典；保存 dev 最优模型。

    - model：自定义 Op 算子（持有 params/grads 字典并实现 backward(labels)）
    - optimizer：实现 step() 的对象（如 SimpleBatchGD）
    - metric / loss_fn：评价指标与损失函数（返回 tensor 或 float 均可）
    """

    def __init__(self, model, optimizer, metric, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        # 训练过程指标统一收纳到 history 字典中
        self.history = {
            "train_loss": [], "dev_loss": [],
            "train_score": [], "dev_score": [],
        }

    def train(self, train_set, dev_set, num_epochs=100, log_epochs=100,
              save_path="model_best.pt"):
        # 用 -inf 初始化保证首轮一定触发对比
        best_score = -float("inf")
        X, y = train_set
        for epoch in range(num_epochs):
            # 训练一步
            logits = self.model(X)
            trn_loss = self.loss_fn(logits, y)
            if hasattr(trn_loss, "item"):
                trn_loss = trn_loss.item()
            trn_score = self.metric(logits, y)
            if hasattr(trn_score, "item"):
                trn_score = trn_score.item()
            self.model.backward(y)
            self.optimizer.step()
            self.history["train_loss"].append(trn_loss)
            self.history["train_score"].append(trn_score)
            # 验证集评估 + 写入历史 + best-checkpoint
            dev_score, dev_loss = self.evaluate(dev_set)
            self.history["dev_loss"].append(dev_loss)
            self.history["dev_score"].append(dev_score)
            if dev_score > best_score:
                self.save_model(save_path)
                print(f"best score updated: {best_score:.5f} -> {dev_score:.5f}")
                best_score = dev_score
            if (epoch + 1) % log_epochs == 0:
                print(f"[Train] epoch {epoch+1}  loss {trn_loss:.4f}  score {trn_score:.4f}")
                print(f"[Dev]   epoch {epoch+1}  loss {dev_loss:.4f}  score {dev_score:.4f}")

    def evaluate(self, data_set):
        """纯查询：不写入历史。
        训练循环里由 train 自己 append，最终测试集评估不会污染 dev 历史。
        """
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        if hasattr(loss, "item"):
            loss = loss.item()
        score = self.metric(logits, y)
        if hasattr(score, "item"):
            score = score.item()
        return score, loss

    def predict(self, X):
        return self.model(X)

    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(self.model.params, save_path)

    def load_model(self, save_path):
        self.model.params = torch.load(save_path, weights_only=True)


# --------------------------------------------------------------------------- #
# RunnerV3：DataLoader + state_dict                                            #
# --------------------------------------------------------------------------- #
class RunnerV3:
    """小批量训练 + 任意 metric + best-checkpoint（state_dict）。

    - model：nn.Module 子类
    - optimizer：torch.optim.Optimizer
    - loss_fn：例如 nn.CrossEntropyLoss()
    - metric_fn(out, y) -> float：标量评价指标；缺省时以 dev_loss 作 metric
    - higher_is_better：metric 越大越好（accuracy）还是越小越好（loss/MAE）
    - device：若给定（'cuda' / 'cpu' / torch.device），构造时把 model 搬到该 device，
      fit/_eval/predict 时自动把每个 batch 的张量也搬过去。默认 None 表示不接管 device。
    """

    def __init__(self, model, optimizer, loss_fn, metric_fn=None,
                 higher_is_better=True, device=None):
        self.device = torch.device(device) if device is not None else None
        if self.device is not None:
            model = model.to(self.device)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.higher_is_better = higher_is_better
        self.history = {"train_loss": [], "dev_loss": [], "dev_metric": [], "lr": []}

    # ---- 内部工具：把 batch 里的张量搬到 self.device ------------------------ #
    def _to_device(self, batch):
        if self.device is None:
            return batch
        return tuple(t.to(self.device) if torch.is_tensor(t) else t for t in batch)

    def fit(self, train_loader, dev_loader=None, num_epochs=100, log_every=10,
            best_path=None, grad_clip_norm=None, lr_scheduler=None,
            patience=None, seed=None):
        """- `dev_loader=None`：只跟踪 train_loss，跳过 dev 评估和 best-checkpoint。
        - `log_every=None`：完全不打印 epoch 日志（用于内部 ablation 调用等）。
        - `grad_clip_norm`：若非 None，在 `backward()` 与 `step()` 之间调
          `nn.utils.clip_grad_norm_(params, grad_clip_norm)`（RNN/Attention 常用）。
        - `lr_scheduler`：若给定，每个 epoch 末调一次 `lr_scheduler.step()`。
        - `patience`：连续 N 轮 dev_metric 无改善则提前停止（仅当 dev_loader 提供时生效）。
        - `seed`：固定 torch 随机数种子，便于复现实验。

        train_loader 每个 batch 解包为 `*inputs, y`，模型按 `model(*inputs)` 调用——
        因此既支持普通 `(x, y)`，也支持变长序列的 `(padded, lengths, y)` 等多输入约定。
        """
        if seed is not None:
            torch.manual_seed(seed)
        best = -float("inf") if self.higher_is_better else float("inf")
        no_improve = 0
        for epoch in range(num_epochs):
            self.model.train()
            running, n = 0.0, 0
            for batch in train_loader:
                *inputs, y = self._to_device(batch)
                self.optimizer.zero_grad()
                loss = self.loss_fn(self.model(*inputs), y)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()
                bs = inputs[0].size(0)
                running += loss.item() * bs
                n += bs
            train_loss = running / n
            self.history["train_loss"].append(train_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            if lr_scheduler is not None:
                lr_scheduler.step()

            should_log = log_every is not None and (epoch + 1) % log_every == 0
            if dev_loader is not None:
                dev_loss, dev_metric = self._eval(dev_loader)
                self.history["dev_loss"].append(dev_loss)
                self.history["dev_metric"].append(dev_metric)
                improved = (dev_metric > best) if self.higher_is_better else (dev_metric < best)
                if improved:
                    best = dev_metric
                    no_improve = 0
                    if best_path is not None:
                        Path(best_path).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), best_path)
                else:
                    no_improve += 1
                if should_log:
                    tag = " *" if improved else ""
                    print(f"epoch {epoch+1:4d}  train_loss={train_loss:.4f}  "
                          f"dev_loss={dev_loss:.4f}  dev_metric={dev_metric:.4f}{tag}")
                if patience is not None and no_improve >= patience:
                    if log_every is not None:
                        print(f"early stop at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break
            elif should_log:
                print(f"epoch {epoch+1:4d}  train_loss={train_loss:.4f}")

    @torch.no_grad()
    def _eval(self, loader):
        self.model.eval()
        total_loss, total_m, n = 0.0, 0.0, 0
        for batch in loader:
            *inputs, y = self._to_device(batch)
            out = self.model(*inputs)
            bs = inputs[0].size(0)
            total_loss += self.loss_fn(out, y).item() * bs
            if self.metric_fn is not None:
                total_m += self.metric_fn(out, y) * bs
            n += bs
        dev_loss = total_loss / n
        dev_metric = (total_m / n) if self.metric_fn is not None else dev_loss
        return dev_loss, dev_metric

    def evaluate(self, loader):
        return self._eval(loader)

    @torch.no_grad()
    def predict(self, *inputs):
        """跟 fit 的多输入解包对齐；普通用法 `runner.predict(x)` 仍然成立。"""
        self.model.eval()
        if self.device is not None:
            inputs = tuple(t.to(self.device) if torch.is_tensor(t) else t for t in inputs)
        return self.model(*inputs)

    def save(self, path):
        """手动保存当前 model 的 state_dict（与 V1/V2 接口一致）。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
