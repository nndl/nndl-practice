"""第 4 章书稿同名代码。

两个 Notebook 保留了更紧凑的 PyTorch 教学路线；本文件集中提供书稿中按“手写算子
→ 手写反向传播 → PyTorch 预定义算子”顺序出现的类，方便读者直接查找、导入和运行。
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PYTORCH_ROOT = Path(__file__).resolve().parents[1]
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))

from nndl.op import Op
from nndl.optim import Optimizer


def _as_float(value):
    return value.item() if hasattr(value, "item") else float(value)


class Linear(Op):
    """书稿中的多输出线性层算子：Y = XW + b。"""

    def __init__(self, input_size, output_size, name,
                 weight_init=torch.randn, bias_init=torch.zeros):
        self.params = {
            "W": weight_init(size=[input_size, output_size]),
            "b": bias_init(size=[1, output_size]),
        }
        self.grads = {}
        self.inputs = None
        self.name = name

    def forward(self, inputs):
        self.inputs = inputs
        return torch.matmul(inputs, self.params["W"]) + self.params["b"]

    def backward(self, grads):
        self.grads["W"] = torch.matmul(self.inputs.T, grads)
        self.grads["b"] = torch.sum(grads, dim=0, keepdim=True)
        return torch.matmul(grads, self.params["W"].T)


class Logistic(Op):
    """Logistic 激活算子及其手写反向传播。"""

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.params = None

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = torch.sigmoid(inputs)
        return self.outputs

    def backward(self, outputs_grads):
        return outputs_grads * self.outputs * (1.0 - self.outputs)


class Model_MLP_L2(Op):
    """书稿中的两层 MLP：两组 Linear + Logistic，反向过程完全手写。"""

    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size, name="fc1")
        self.act_fn1 = Logistic()
        self.fc2 = Linear(hidden_size, output_size, name="fc2")
        self.act_fn2 = Logistic()
        self.layers = [self.fc1, self.act_fn1, self.fc2, self.act_fn2]

    def forward(self, X):
        z1 = self.fc1(X)
        a1 = self.act_fn1(z1)
        z2 = self.fc2(a1)
        return self.act_fn2(z2)

    def backward(self, loss_grad_a2):
        loss_grad_z2 = self.act_fn2.backward(loss_grad_a2)
        loss_grad_a1 = self.fc2.backward(loss_grad_z2)
        loss_grad_z1 = self.act_fn1.backward(loss_grad_a1)
        return self.fc1.backward(loss_grad_z1)


class BinaryCrossEntropyLoss(Op):
    """与 ``Model_MLP_L2`` 配套的二分类交叉熵和反向入口。"""

    def __init__(self, model):
        self.predicts = None
        self.labels = None
        self.num = None
        self.model = model

    def forward(self, predicts, labels):
        self.num = predicts.shape[0]
        self.predicts = predicts.clamp(1e-7, 1.0 - 1e-7)
        self.labels = labels
        loss = -(
            labels.T @ torch.log(self.predicts)
            + (1.0 - labels).T @ torch.log(1.0 - self.predicts)
        ) / self.num
        return loss.squeeze()

    def backward(self):
        inputs_grads = -(
            self.labels / self.predicts
            - (1.0 - self.labels) / (1.0 - self.predicts)
        ) / self.num
        return self.model.backward(inputs_grads)


class BatchGD(Optimizer):
    """遍历多层手写算子的全批量梯度下降。"""

    def step(self):
        for layer in self.model.layers:
            if isinstance(getattr(layer, "params", None), dict):
                for key in layer.params:
                    layer.params[key] = (
                        layer.params[key] - self.init_lr * layer.grads[key]
                    )


class RunnerV2_1:
    """手写算子版 Runner：逐层保存参数，并从损失算子启动反向传播。"""

    def __init__(self, model, optimizer, metric, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.history = {
            "train_loss": [], "dev_loss": [],
            "train_score": [], "dev_score": [],
        }

    def train(self, train_set, dev_set, num_epochs=100, log_epochs=100,
              save_path=None):
        best_score = -float("inf")
        X, y = train_set
        for epoch in range(num_epochs):
            predicts = self.model(X)
            trn_loss = self.loss_fn(predicts, y)
            trn_score = _as_float(self.metric(predicts, y))
            self.loss_fn.backward()
            self.optimizer.step()
            self.history["train_loss"].append(_as_float(trn_loss))
            self.history["train_score"].append(trn_score)

            dev_score, dev_loss = self.evaluate(dev_set)
            self.history["dev_loss"].append(dev_loss)
            self.history["dev_score"].append(dev_score)
            if dev_score > best_score:
                best_score = dev_score
                if save_path:
                    self.save_model(save_path)
            if log_epochs and (epoch + 1) % log_epochs == 0:
                print(
                    f"[Train] epoch {epoch + 1}/{num_epochs}  "
                    f"loss {_as_float(trn_loss):.4f}"
                )

    def evaluate(self, data_set):
        X, y = data_set
        predicts = self.model(X)
        loss = _as_float(self.loss_fn(predicts, y))
        score = _as_float(self.metric(predicts, y))
        return score, loss

    def predict(self, X):
        return self.model(X)

    def save_model(self, save_path):
        state = {
            layer.name: layer.params
            for layer in self.model.layers
            if isinstance(getattr(layer, "params", None), dict)
        }
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    def load_model(self, save_path):
        state = torch.load(save_path, weights_only=True)
        for layer in self.model.layers:
            if getattr(layer, "name", None) in state:
                layer.params = state[layer.name]


def Normal(mean=0.0, std=1.0):
    return partial(nn.init.normal_, mean=mean, std=std)


def Constant(value=0.0):
    return partial(nn.init.constant_, val=value)


def Uniform(low=-1.0, high=1.0):
    return partial(nn.init.uniform_, a=low, b=high)


def Kaiming():
    return partial(nn.init.kaiming_normal_, nonlinearity="relu")


class Model_MLP_L2_V2(nn.Module):
    """使用 ``nn.Linear`` 与 autograd 重写的两层 MLP。"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        nn.init.normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, inputs):
        hidden = torch.sigmoid(self.fc1(inputs))
        return torch.sigmoid(self.fc2(hidden))


class RunnerV2_2:
    """PyTorch autograd 版 Runner：使用 ``state_dict`` 保存模型。"""

    def __init__(self, model, optimizer, metric, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.history = {
            "train_loss": [], "dev_loss": [],
            "train_score": [], "dev_score": [],
        }

    def train(self, train_set, dev_set, num_epochs=100, log_epochs=100,
              save_path="model_best.pt", custom_print_log=None):
        best_score = -float("inf")
        X, y = train_set
        for epoch in range(num_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            predicts = self.model(X)
            trn_loss = self.loss_fn(predicts, y)
            trn_score = _as_float(self.metric(predicts, y))
            trn_loss.backward()
            if custom_print_log is not None:
                custom_print_log(self.model)
            self.optimizer.step()
            self.history["train_loss"].append(_as_float(trn_loss))
            self.history["train_score"].append(trn_score)

            dev_score, dev_loss = self.evaluate(dev_set)
            self.history["dev_loss"].append(dev_loss)
            self.history["dev_score"].append(dev_score)
            if dev_score > best_score:
                best_score = dev_score
                if save_path:
                    self.save_model(save_path)
            if log_epochs and (epoch + 1) % log_epochs == 0:
                print(
                    f"[Train] epoch {epoch + 1}/{num_epochs}  "
                    f"loss {_as_float(trn_loss):.4f}"
                )

    @torch.no_grad()
    def evaluate(self, data_set):
        self.model.eval()
        X, y = data_set
        predicts = self.model(X)
        loss = _as_float(self.loss_fn(predicts, y))
        score = _as_float(self.metric(predicts, y))
        return score, loss

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        return self.model(X)

    def save_model(self, save_path):
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, save_path):
        self.model.load_state_dict(torch.load(save_path, weights_only=True))


class Model_MLP_L5(nn.Module):
    """书稿中用于观察梯度消失与死亡 ReLU 的五层 MLP。"""

    def __init__(self, input_size, output_size, act="sigmoid",
                 w_init=None, b_init=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, 3)
        self.fc5 = nn.Linear(3, output_size)
        if act == "sigmoid":
            self.act = torch.sigmoid
        elif act == "relu":
            self.act = F.relu
        elif act == "lrelu":
            self.act = F.leaky_relu
        else:
            raise ValueError("act 只能是 'sigmoid'、'relu' 或 'lrelu'")
        self.init_weights(w_init or Kaiming(), b_init or Constant(value=1.0))

    def init_weights(self, w_init, b_init):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                w_init(module.weight)
                b_init(module.bias)

    def forward(self, inputs):
        outputs = self.act(self.fc1(inputs))
        outputs = self.act(self.fc2(outputs))
        outputs = self.act(self.fc3(outputs))
        outputs = self.act(self.fc4(outputs))
        return torch.sigmoid(self.fc5(outputs))


__all__ = [
    "Linear", "Logistic", "Model_MLP_L2", "BinaryCrossEntropyLoss",
    "BatchGD", "RunnerV2_1", "Normal", "Constant", "Uniform", "Kaiming",
    "Model_MLP_L2_V2", "RunnerV2_2", "Model_MLP_L5",
]
