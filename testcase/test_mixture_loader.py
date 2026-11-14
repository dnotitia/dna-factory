"""
The shared mixture loader (dna_factory/dnotitia_dataset_mixture.py::load_mixture) as seen through
each entry script's loader, run against real tiny local datasets.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict, IterableDatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dpo
import grpo
import opd
import sdft
import sft
from dna_factory.dnotitia_dataset_mixture import WeightedDatasetMixtureConfig

LOADERS = {
    "dpo": (dpo.get_weighted_dataset, WeightedDatasetMixtureConfig),
    "sft": (sft.get_dataset_with_schema_alignment, WeightedDatasetMixtureConfig),
    "grpo": (grpo.get_dataset_with_schema_alignment, grpo.LabeledDatasetMixtureConfig),
    "opd": (opd.get_dataset_with_schema_alignment, WeightedDatasetMixtureConfig),
    "sdft": (sdft.get_dataset_with_context, sdft.ContextDatasetMixtureConfig),
}


@pytest.fixture(params=list(LOADERS))
def load(request):
    loader, config_cls = LOADERS[request.param]

    def _load(entries, seed=42, **mixture_kwargs):
        return loader(config_cls(datasets=entries, **mixture_kwargs), seed=seed)

    return _load


def _rows(n, start=0):
    return [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ],
            "prompt": f"p{i}",
            "extra": i,
        }
        for i in range(start, start + n)
    ]


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


def _prompt_text(example):
    prompt = example["prompt"]
    return prompt if isinstance(prompt, str) else prompt[-1]["content"]


def test_data_files_json(load, tmp_path):
    file = _write_jsonl(tmp_path / "rows.jsonl", _rows(5))
    out = load([{"path": "json", "data_files": file, "split": "train"}])
    assert len(out["train"]) == 5


def test_save_to_disk_dataset_and_dataset_dict(load, tmp_path):
    Dataset.from_list(_rows(4)).save_to_disk(str(tmp_path / "ds"))
    DatasetDict({"train": Dataset.from_list(_rows(6))}).save_to_disk(
        str(tmp_path / "dd")
    )
    out = load([{"path": str(tmp_path / "ds")}, {"path": str(tmp_path / "dd")}])
    assert len(out["train"]) == 10


def test_plain_local_folder_of_jsonl(load, tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    _write_jsonl(folder / "train.jsonl", _rows(7))
    out = load([{"path": str(folder)}])
    assert len(out["train"]) == 7


def test_columns_are_honored(load, tmp_path):
    file = _write_jsonl(tmp_path / "rows.jsonl", _rows(3))
    out = load(
        [{"path": "json", "data_files": file, "split": "train", "columns": ["prompt"]}]
    )
    assert "extra" not in out["train"].column_names
    assert "prompt" in out["train"].column_names


def test_test_split_size(load, tmp_path):
    file = _write_jsonl(tmp_path / "rows.jsonl", _rows(20))
    entries = [{"path": "json", "data_files": file, "split": "train"}]
    out = load(entries, test_split_size=0.25)
    assert set(out) == {"train", "test"}
    assert (len(out["train"]), len(out["test"])) == (15, 5)


def test_fraction(load, tmp_path):
    a = _write_jsonl(tmp_path / "a.jsonl", _rows(10))
    b = _write_jsonl(tmp_path / "b.jsonl", _rows(6))
    out = load(
        [
            {"path": "json", "data_files": a, "split": "train", "fraction": 1.0},
            {"path": "json", "data_files": b, "split": "train", "fraction": 1.0},
        ]
    )
    assert len(out["train"]) == 12  # 6 + 6: capped by the smaller dataset


def test_weight(load, tmp_path):
    file = _write_jsonl(tmp_path / "rows.jsonl", _rows(4))
    out = load([{"path": "json", "data_files": file, "split": "train", "weight": 2.0}])
    assert len(out["train"]) == 8


def test_schema_mismatch_gives_clear_error(load, tmp_path):
    a = _write_jsonl(tmp_path / "a.jsonl", [{"prompt": "text"}])
    b = _write_jsonl(tmp_path / "b.jsonl", [{"prompt": 5}])
    with pytest.raises(ValueError, match="Could not align dataset mixture schemas"):
        load(
            [
                {"path": "json", "data_files": a, "split": "train"},
                {"path": "json", "data_files": b, "split": "train"},
            ]
        )


def test_grpo_rejects_existing_label_column(tmp_path):
    rows = [{"prompt": "p", "label": "own"}]
    path = _write_jsonl(tmp_path / "a.jsonl", rows)
    mixture = grpo.LabeledDatasetMixtureConfig(
        datasets=[{"path": "json", "data_files": path, "split": "train"}]
    )
    with pytest.raises(ValueError, match="already has a `label` column") as exc:
        grpo.get_dataset_with_schema_alignment(mixture)
    assert "json" in str(exc.value)
    # The dataset's own labels are untouched by the normalizer.
    with pytest.raises(ValueError, match="already has a `label` column"):
        grpo._normalize_dataset_for_grpo(Dataset.from_list(rows), "x")


def test_streaming_sft_mixture_postprocesses(tmp_path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "thinking": "t"},
            ]
        }
    ]
    file = _write_jsonl(tmp_path / "rows.jsonl", rows)
    mixture = WeightedDatasetMixtureConfig(
        datasets=[{"path": "json", "data_files": file, "split": "train"}],
        streaming=True,
    )
    out = sft.get_dataset_with_schema_alignment(mixture)
    assert isinstance(out, IterableDatasetDict)
    out = sft.postprocess_dataset(out, {}, None)
    first = next(iter(out["train"]))
    assert first["messages"][-1]["reasoning_content"] == "t"


def test_streaming_with_test_split_size_raises(tmp_path):
    file = _write_jsonl(tmp_path / "rows.jsonl", _rows(4))
    mixture = WeightedDatasetMixtureConfig(
        datasets=[{"path": "json", "data_files": file, "split": "train"}],
        streaming=True,
        test_split_size=0.25,
    )
    with pytest.raises(ValueError, match="test_split_size"):
        sft.get_dataset_with_schema_alignment(mixture)


def test_async_grpo_rejects_streaming_mixture(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("loader must not run")

    monkeypatch.setattr(grpo, "load_mixture_datasets", boom)
    mixture = grpo.LabeledDatasetMixtureConfig(
        datasets=[{"path": "json", "data_files": "x", "split": "train"}],
        streaming=True,
    )
    with pytest.raises(ValueError, match="non-streaming datasets only"):
        grpo.get_spec("async").load_mixture(
            mixture, SimpleNamespace(seed=1), {}, None
        )
