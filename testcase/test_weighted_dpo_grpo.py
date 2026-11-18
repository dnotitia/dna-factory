import sys
from pathlib import Path
from types import SimpleNamespace

import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dpo
import grpo
from dna_factory import dnotitia_dataset_mixture
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetConfig,
    WeightedDatasetMixtureConfig,
)


def test_dpo_applies_per_dataset_weight(monkeypatch):
    sources = {
        "first": datasets.Dataset.from_dict({"chosen": ["a", "b"], "unused": [1, 2]}),
        "second": datasets.Dataset.from_dict({"chosen": ["c", "d"], "unused": [3, 4]}),
    }
    monkeypatch.setattr(
        dnotitia_dataset_mixture,
        "load_dataset",
        lambda path, **kwargs: sources[path],
    )
    mixture = WeightedDatasetMixtureConfig(
        datasets=[
            WeightedDatasetConfig(path="first", columns=["chosen"], weight=2.0),
            WeightedDatasetConfig(path="second", columns=["chosen"], weight=0.5),
        ]
    )

    result = dpo.load_mixture(mixture, SimpleNamespace(seed=7), {}, None)

    assert result["train"].column_names == ["chosen"]
    assert result["train"]["chosen"] == ["a", "b", "a", "b", "c"]


def test_grpo_applies_weight_before_schema_alignment(monkeypatch):
    sources = {
        "first": datasets.Dataset.from_dict({"prompt": ["a", "b"]}),
        "second": datasets.Dataset.from_dict({"prompt": ["c", "d"]}),
    }
    monkeypatch.setattr(datasets, "load_dataset", lambda path, **kwargs: sources[path])
    mixture = grpo.LabeledDatasetMixtureConfig(
        datasets=[
            {"path": "first", "label": "one", "weight": 2.0},
            {"path": "second", "label": "two", "weight": 0.5},
        ]
    )

    result = grpo.get_dataset_with_schema_alignment(mixture, seed=7)

    assert result["train"]["label"] == ["one", "one", "one", "one", "two"]
