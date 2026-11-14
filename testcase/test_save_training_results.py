"""
Test cases for dna_factory/training_runner.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.training_runner import save_training_results


class _FakeTrainer:
    def __init__(self):
        self.calls = []
        self.accelerator = SimpleNamespace(is_main_process=True)
        config = SimpleNamespace(
            save_pretrained=lambda path: self.calls.append(
                ("config.save_pretrained", path)
            )
        )
        self.model = SimpleNamespace(config=config)

    def log_metrics(self, split, metrics):
        self.calls.append(("log_metrics", dict(metrics)))

    def save_metrics(self, split, metrics):
        self.calls.append(("save_metrics", dict(metrics)))

    def save_state(self):
        self.calls.append(("save_state",))

    def save_model(self, path):
        self.calls.append(("save_model", path))

    def create_model_card(self, **kwargs):
        self.calls.append(("create_model_card", kwargs))


def _run(train_split):
    trainer = _FakeTrainer()
    script_args = SimpleNamespace(dataset_train_split="train", dataset_name="ds")
    training_args = SimpleNamespace(output_dir="out", push_to_hub=False)
    result = SimpleNamespace(metrics={"loss": 1.0, "dropped": None})
    save_training_results(
        trainer, result, {"train": train_split}, script_args, training_args
    )
    return trainer


def _logged_metrics(trainer):
    return next(c[1] for c in trainer.calls if c[0] == "log_metrics")


def test_streaming_train_split_omits_train_samples_and_still_saves():
    iterable = Dataset.from_dict({"x": [1, 2, 3]}).to_iterable_dataset()
    trainer = _run(iterable)
    assert ("save_model", "out") in trainer.calls
    assert _logged_metrics(trainer) == {"loss": 1.0}


def test_sized_train_split_records_train_samples():
    trainer = _run(Dataset.from_dict({"x": [1, 2, 3]}))
    assert ("save_model", "out") in trainer.calls
    assert _logged_metrics(trainer) == {"loss": 1.0, "train_samples": 3}
