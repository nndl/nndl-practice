"""训练工具的可观察行为：调度、评价状态、批次汇总和空输入。"""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nndl.runner import RunnerV3


def make_runner(metric_fn=None):
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return RunnerV3(model, optimizer, nn.MSELoss(), metric_fn=metric_fn)


def make_loader(size=5, batch_size=2):
    return DataLoader(TensorDataset(torch.zeros(size, 1), torch.arange(size).float()[:, None]),
                      batch_size=batch_size)


def test_evaluate_weights_partial_batch_and_returns_floats():
    runner = make_runner(metric_fn=lambda out, y: (out - y).abs().mean())
    loss, metric = runner.evaluate(make_loader())
    assert loss == pytest.approx(6.0)
    assert metric == pytest.approx(2.0)
    assert isinstance(loss, float) and isinstance(metric, float)


@pytest.mark.parametrize("training", [True, False])
def test_queries_restore_mode_and_disable_gradients(training):
    runner = make_runner()
    runner.model.train(training)
    modes = []
    handle = runner.model.register_forward_pre_hook(
        lambda model, args: modes.append((model.training, torch.is_grad_enabled())))
    try:
        runner.evaluate(make_loader())
        assert runner.model.training == training
        out = runner.predict(torch.ones(2, 1))
        assert runner.model.training == training
        assert not out.requires_grad
        assert all(mode == (False, False) for mode in modes)
    finally:
        handle.remove()


def test_query_failure_restores_training_mode():
    runner = make_runner()
    with pytest.raises(RuntimeError):
        runner.predict(torch.zeros(2, 3))
    assert runner.model.training


@pytest.mark.parametrize("method", ["fit", "evaluate"])
def test_empty_loader_has_actionable_error(method):
    runner = make_runner()
    with pytest.raises(ValueError, match="empty|no samples"):
        getattr(runner, method)(make_loader(size=0))
    assert runner.history["train_loss"] == []
    assert runner.model.training


def test_plateau_uses_dev_metric_after_evaluation():
    runner = make_runner(metric_fn=lambda out, y: 0.75)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        runner.optimizer, mode="max", factor=0.5, patience=0)
    runner.fit(make_loader(), make_loader(), num_epochs=3, log_every=None,
               lr_scheduler=scheduler)
    assert scheduler.best == pytest.approx(0.75)
    assert runner.history["lr"] == pytest.approx([0.1, 0.1, 0.05])
    assert runner.optimizer.param_groups[0]["lr"] == pytest.approx(0.025)


def test_plateau_requires_validation_before_training():
    runner = make_runner()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(runner.optimizer)
    with pytest.raises(ValueError, match="dev_loader"):
        runner.fit(make_loader(), lr_scheduler=scheduler)
    assert runner.history["train_step_loss"] == []


def test_plateau_rejects_metric_direction_mismatch():
    runner = make_runner(metric_fn=lambda out, y: 0.75)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(runner.optimizer, mode="min")
    with pytest.raises(ValueError, match="mode"):
        runner.fit(make_loader(), make_loader(), lr_scheduler=scheduler)
    assert runner.history["train_step_loss"] == []


def test_epoch_scheduler_and_early_stopping_keep_completed_epochs():
    runner = make_runner()
    scheduler = torch.optim.lr_scheduler.StepLR(runner.optimizer, step_size=1, gamma=0.5)
    runner.fit(make_loader(), make_loader(), num_epochs=5, log_every=None,
               lr_scheduler=scheduler, patience=1)
    assert runner.history["lr"] == pytest.approx([0.1, 0.05])
    assert scheduler.last_epoch == 2
    assert len(runner.history["dev_loss"]) == 2
