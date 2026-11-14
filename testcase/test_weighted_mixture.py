"""Per-dataset `weight` in the dataset mixture of each entry point."""

import sys
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpo
import opd
import sft
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetConfig,
    WeightedDatasetMixtureConfig,
)


@pytest.fixture
def prompt_sources(monkeypatch):
    sources = {
        "first": datasets.Dataset.from_dict({"prompt": ["a", "b"]}),
        "second": datasets.Dataset.from_dict({"prompt": ["c", "d"]}),
    }
    monkeypatch.setattr(datasets, "load_dataset", lambda path, **kwargs: sources[path])
    return sources


def _messages(text):
    return [
        {"role": "user", "content": text},
        {"role": "assistant", "content": text.upper()},
    ]


def test_sft_applies_per_dataset_weight(monkeypatch):
    sources = {
        "first": datasets.Dataset.from_dict(
            {"messages": [_messages("a"), _messages("b")], "unused": [1, 2]}
        ),
        "second": datasets.Dataset.from_dict(
            {"messages": [_messages("c"), _messages("d")], "unused": [3, 4]}
        ),
    }
    monkeypatch.setattr(datasets, "load_dataset", lambda path, **kwargs: sources[path])
    mixture = WeightedDatasetMixtureConfig(
        datasets=[
            WeightedDatasetConfig(path="first", columns=["messages"], weight=2.0),
            WeightedDatasetConfig(path="second", columns=["messages"], weight=0.5),
        ]
    )

    result = sft.load_mixture(mixture, SimpleNamespace(seed=7), {}, None)

    assert result["train"].column_names == ["messages"]
    users = [messages[0]["content"] for messages in result["train"]["messages"]]
    assert users == ["a", "b", "a", "b", "c"]


def test_grpo_applies_weight_before_schema_alignment(prompt_sources):
    mixture = grpo.LabeledDatasetMixtureConfig(
        datasets=[
            {"path": "first", "label": "one", "weight": 2.0},
            {"path": "second", "label": "two", "weight": 0.5},
        ]
    )

    result = grpo.get_dataset_with_schema_alignment(mixture, seed=7)

    assert result["train"]["label"] == ["one", "one", "one", "one", "two"]


def test_opd_applies_per_dataset_weight(prompt_sources):
    mixture = WeightedDatasetMixtureConfig(
        datasets=[
            WeightedDatasetConfig(path="first", weight=2.0),
            WeightedDatasetConfig(path="second", weight=0.5),
        ]
    )

    result = opd.load_mixture(mixture, SimpleNamespace(seed=7), {}, None)

    users = [prompt[0]["content"] for prompt in result["train"]["prompt"]]
    assert users == ["a", "b", "a", "b", "c"]
