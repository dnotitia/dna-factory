"""Completion tables follow log_completions_steps, not every logging call."""

from types import SimpleNamespace

import pytest
from trl import GRPOTrainer

from dna_factory.dnotitia_grpo_trainer import DnotitiaGRPOTrainer


def _trainer(step, interval=100, enabled=True):
    trainer = DnotitiaGRPOTrainer.__new__(DnotitiaGRPOTrainer)
    trainer.log_completions = enabled
    trainer.log_completions_steps = interval
    trainer._last_completion_log_step = -1
    trainer.state = SimpleNamespace(global_step=step)
    return trainer


def test_first_step_then_every_interval(monkeypatch):
    """Step 1 always uploads; later tables land on 100, 200, ... counted from 0."""
    seen = []

    def fake_log(self, logs, start_time=None):
        seen.append((self.state.global_step, self.log_completions))

    monkeypatch.setattr(GRPOTrainer, "log", fake_log)
    trainer = _trainer(step=0)

    for step in (1, 99, 100, 199, 200):
        trainer.state.global_step = step
        DnotitiaGRPOTrainer.log(trainer, {})

    assert seen == [(1, True), (99, False), (100, True), (199, False), (200, True)]
    assert trainer.log_completions is True
    assert trainer._last_completion_log_step == 200


def test_zero_interval_logs_every_call(monkeypatch):
    seen = []

    def fake_log(self, logs, start_time=None):
        seen.append(self.log_completions)

    monkeypatch.setattr(GRPOTrainer, "log", fake_log)
    trainer = _trainer(step=1, interval=0)
    DnotitiaGRPOTrainer.log(trainer, {})
    trainer.state.global_step = 2
    DnotitiaGRPOTrainer.log(trainer, {})
    assert seen == [True, True]


def test_disabled_completions_stay_disabled(monkeypatch):
    seen = []

    def fake_log(self, logs, start_time=None):
        seen.append(self.log_completions)

    monkeypatch.setattr(GRPOTrainer, "log", fake_log)
    trainer = _trainer(step=100, enabled=False)
    DnotitiaGRPOTrainer.log(trainer, {})
    assert seen == [False]
    assert trainer._last_completion_log_step == -1


def test_failed_log_does_not_consume_the_interval(monkeypatch):
    def boom(self, logs, start_time=None):
        raise RuntimeError("wandb down")

    monkeypatch.setattr(GRPOTrainer, "log", boom)
    trainer = _trainer(step=100)
    with pytest.raises(RuntimeError, match="wandb down"):
        DnotitiaGRPOTrainer.log(trainer, {})
    assert trainer.log_completions is True
    assert trainer._last_completion_log_step == -1


def test_unaligned_logging_steps_still_respect_the_gap(monkeypatch):
    """logging_steps=30 never lands on 100: upload the first call, then the next boundary."""
    seen = []

    def fake_log(self, logs, start_time=None):
        seen.append((self.state.global_step, self.log_completions))

    monkeypatch.setattr(GRPOTrainer, "log", fake_log)
    trainer = _trainer(step=0)
    for step in (30, 60, 90, 120, 150, 210):
        trainer.state.global_step = step
        DnotitiaGRPOTrainer.log(trainer, {})
    assert seen == [
        (30, True),
        (60, False),
        (90, False),
        (120, True),
        (150, False),
        (210, True),
    ]
