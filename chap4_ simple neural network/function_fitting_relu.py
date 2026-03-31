import math
from pathlib import Path

import numpy as np


def target_function(x: np.ndarray) -> np.ndarray:
    return np.sin(1.5 * x) + 0.3 * x**2


class ReLUNetwork:
    def __init__(self, hidden_dim: int = 64, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, math.sqrt(2.0), size=(1, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0.0, math.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h1 = np.maximum(self.z1, 0.0)
        self.y_pred = self.h1 @ self.W2 + self.b2
        return self.y_pred

    def backward(self, y_true: np.ndarray) -> float:
        n = y_true.shape[0]
        diff = self.y_pred - y_true
        loss = float(np.mean(diff**2))

        grad_y = 2.0 * diff / n
        grad_W2 = self.h1.T @ grad_y
        grad_b2 = np.sum(grad_y, axis=0, keepdims=True)

        grad_h1 = grad_y @ self.W2.T
        grad_z1 = grad_h1 * (self.z1 > 0.0)
        grad_W1 = self.x.T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        self.grads = {
            "W1": grad_W1,
            "b1": grad_b1,
            "W2": grad_W2,
            "b2": grad_b2,
        }
        return loss

    def step(self, lr: float) -> None:
        self.W1 -= lr * self.grads["W1"]
        self.b1 -= lr * self.grads["b1"]
        self.W2 -= lr * self.grads["W2"]
        self.b2 -= lr * self.grads["b2"]


def build_dataset(seed: int = 7):
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-3.0, 3.0, size=(512, 1))
    y_train = target_function(x_train)

    x_test = np.linspace(-3.0, 3.0, 256, dtype=np.float64).reshape(-1, 1)
    y_test = target_function(x_test)
    return x_train, y_train, x_test, y_test


def save_results(x_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    output = Path("function_fitting_predictions.csv")
    data = np.concatenate([x_test, y_test, y_pred], axis=1)
    np.savetxt(
        output,
        data,
        delimiter=",",
        header="x,true_y,pred_y",
        comments="",
    )


def main() -> None:
    x_train, y_train, x_test, y_test = build_dataset()
    model = ReLUNetwork(hidden_dim=64, seed=42)

    epochs = 5000
    learning_rate = 1e-2

    for epoch in range(1, epochs + 1):
        model.forward(x_train)
        train_loss = model.backward(y_train)
        model.step(learning_rate)

        if epoch % 500 == 0 or epoch == 1:
            test_pred = model.forward(x_test)
            test_loss = float(np.mean((test_pred - y_test) ** 2))
            print(
                f"epoch={epoch:4d} "
                f"train_loss={train_loss:.6f} "
                f"test_loss={test_loss:.6f}"
            )

    final_pred = model.forward(x_test)
    final_test_loss = float(np.mean((final_pred - y_test) ** 2))
    print(f"final_test_loss={final_test_loss:.6f}")
    save_results(x_test, y_test, final_pred)
    print("saved=function_fitting_predictions.csv")


if __name__ == "__main__":
    main()
