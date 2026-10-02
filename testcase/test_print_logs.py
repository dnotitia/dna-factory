"""`print_logs: false` drops the per-step metrics dict from stdout but keeps the bar."""

import contextlib
import io

import pytest
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from dna_factory.training_runner import silence_log_printing


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return {"loss": (self.w * x).sum()}


class _Data(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, i):
        return {"x": torch.tensor([1.0])}


def _train(tmp_path, quiet, disable_tqdm):
    args = TrainingArguments(
        str(tmp_path),
        max_steps=3,
        logging_steps=1,
        report_to=[],
        disable_tqdm=disable_tqdm,
        use_cpu=True,
        save_strategy="no",
    )
    trainer = Trainer(model=_Model(), args=args, train_dataset=_Data())
    if quiet:
        silence_log_printing(trainer)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        trainer.train()
    return out.getvalue() + err.getvalue()


@pytest.mark.parametrize("disable_tqdm", [False, True])
def test_default_prints_logs(tmp_path, disable_tqdm):
    assert _train(tmp_path, quiet=False, disable_tqdm=disable_tqdm).count("'loss'") == 3


@pytest.mark.parametrize("disable_tqdm", [False, True])
def test_silenced_drops_logs(tmp_path, disable_tqdm):
    assert "'loss'" not in _train(tmp_path, quiet=True, disable_tqdm=disable_tqdm)


def test_silenced_keeps_progress_bar(tmp_path):
    assert "3/3" in _train(tmp_path, quiet=True, disable_tqdm=False)
