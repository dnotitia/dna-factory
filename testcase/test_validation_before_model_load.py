"""
Test cases for dna_factory/training_runner.py::run_training argument validation order
"""

import dataclasses
import logging
import sys
from pathlib import Path

import pytest

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

import sft
from dna_factory import training_runner


@pytest.fixture(autouse=True)
def _skip_cuda_tf32_probe(monkeypatch):
    """TrainingArguments.__post_init__ probes CUDA whenever tf32 is set. These tests never train."""
    monkeypatch.setattr(
        "transformers.training_args.is_torch_tf32_available", lambda: False
    )


def parse_sft(monkeypatch, extra):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(training_runner, "run_training", lambda *args, **kwargs: args)
    return training_runner.cli_main(
        sft.SPEC, ["--use_cpu", "true", "--bf16", "false", *extra]
    )


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--periodic_save_seconds", "2x"], "periodic_save_seconds"),
        (["--eval_on_checkpoint", "true", "--eval_devices", ""], "eval_devices"),
    ],
)
def test_invalid_checkpoint_args_fail_before_the_model_loads(
    monkeypatch, tmp_path, extra, message
):
    spec, script, training, model, mixture, dna, *_ = parse_sft(monkeypatch, extra)
    # Exercise the real shared runner, but no network, GPU or training.
    monkeypatch.undo()
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(
        training_runner, "setup_logging", lambda *a: logging.getLogger(__name__)
    )
    training.output_dir = str(tmp_path)

    def load_models(*args, **kwargs):
        raise AssertionError("the model was loaded before the arguments were validated")

    spec = dataclasses.replace(spec, load_models=load_models)
    with pytest.raises(ValueError, match=message):
        training_runner.run_training(spec, script, training, model, mixture, dna)
