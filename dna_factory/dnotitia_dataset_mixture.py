"""Weighted dataset mixtures for DNA Factory.

TRL's stock `DatasetConfig` / `DatasetMixtureConfig` mix datasets by plain 1:1
concatenation — there is no per-dataset weight, so the only way to up-weight a
small dataset (e.g. persona vs. a larger uncensor set) was to list the same
`path` several times. This module adds an inline `weight:` field so the YAML can
say *how much* of each dataset to use as a single number instead:

    datasets:
      - path: dnotitia/persona...
        columns: [messages]
        weight: 4.0      # 4 full copies (replaces listing the entry 4x)
      - path: dnotitia/uncensor...
        columns: [messages]
        weight: 1.0      # used once (the default; omit for 1.0)

`weight` is a size multiplier on the dataset, generalizing the old "list it N
times" trick to fractional values:
  - integer part  -> that many full copies
  - fractional part -> a deterministic random subsample of that fraction
  - weight < 1     -> downsample (e.g. 0.25 -> a random 25% of the rows)
  - weight == 1.0  -> unchanged (backward compatible with existing configs)
  - weight == 0    -> dataset dropped entirely

The subsample is seeded so runs are reproducible.
"""

import logging
import os
from dataclasses import dataclass, field

from datasets import (
    Dataset,
    DatasetDict,
    IterableDatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from trl import DatasetMixtureConfig
from trl.scripts.utils import DatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class WeightedDatasetConfig(DatasetConfig):
    """A `DatasetConfig` with an extra `weight` (size multiplier, default 1.0)."""

    weight: float = field(
        default=1.0,
        metadata={
            "help": "Size multiplier for this dataset in the mixture (1.0 = use as-is)."
        },
    )


@dataclass
class WeightedDatasetMixtureConfig(DatasetMixtureConfig):
    """`DatasetMixtureConfig` whose entries are `WeightedDatasetConfig`.

    The annotation must stay `list[WeightedDatasetConfig]` (not bare `list`) so
    `HfArgumentParser` can introspect the element type.
    """

    datasets: list[WeightedDatasetConfig] = field(
        default_factory=list,
        metadata={
            "help": "List of (weighted) dataset configurations to include in the mixture."
        },
    )

    def __post_init__(self):
        # Convert dicts (from CLI/YAML parsing) into WeightedDatasetConfig objects.
        for idx, dataset in enumerate(self.datasets):
            if isinstance(dataset, dict):
                self.datasets[idx] = WeightedDatasetConfig(**dataset)


def resample_by_weight(dataset, weight, seed=42):
    """Resize `dataset` by `weight` (a size multiplier). See module docstring."""
    if weight == 1.0:
        return dataset
    if weight < 0:
        raise ValueError(f"dataset weight must be >= 0, got {weight}")

    n = len(dataset)
    full_copies = int(weight)
    frac = weight - full_copies

    parts = [dataset] * full_copies
    if frac > 1e-9 and n > 0:
        k = round(n * frac)
        if k > 0:
            idx = list(range(n))
            import random

            random.Random(seed).shuffle(idx)
            parts.append(dataset.select(sorted(idx[:k])))

    if not parts:  # weight == 0 (or rounded to nothing)
        return dataset.select([])
    return concatenate_datasets(parts) if len(parts) > 1 else parts[0]


def load_dataset_entry(dataset_config, streaming=False):
    """Load one mixture entry.

    A directory written by `save_to_disk` goes through `load_from_disk`, where only `split`
    applies (`name`, `data_dir` and `data_files` are not used). Anything else (hub id,
    builder name such as `json`, or a plain local folder of jsonl/parquet files) goes through
    `load_dataset`, which honors all of those fields.
    """
    path = dataset_config.path
    if os.path.isdir(path) and any(
        os.path.exists(os.path.join(path, marker))
        for marker in ("dataset_dict.json", "state.json")
    ):
        dataset = load_from_disk(path)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[dataset_config.split or "train"]
        return dataset.to_iterable_dataset() if streaming else dataset
    return load_dataset(
        path=path,
        name=dataset_config.name,
        data_dir=dataset_config.data_dir,
        data_files=dataset_config.data_files,
        split=dataset_config.split,
        streaming=streaming,
    )


def load_mixture(mixture_config, seed=42, normalize=None):
    """Load a TRL dataset mixture into a `DatasetDict`; shared by every entry script.

    Per entry: load, `columns`, optional `normalize(dataset, dataset_config)`, then the size
    multiplier. Then `fraction`, concatenation and (if set) a `test_split_size` split.
    """
    streaming = mixture_config.streaming
    if streaming and mixture_config.test_split_size is not None:
        raise ValueError(
            "Using `test_split_size` is not supported with streaming datasets."
        )
    datasets_list = []
    for dataset_config in mixture_config.datasets:
        logger.info(
            f"Loading dataset for mixture: {dataset_config.path} "
            f"(config name: {dataset_config.name})"
        )
        dataset = load_dataset_entry(dataset_config, streaming=streaming)
        if dataset_config.columns is not None:
            dataset = dataset.select_columns(dataset_config.columns)
        if normalize is not None:
            dataset = normalize(dataset, dataset_config)
        weight = getattr(dataset_config, "weight", 1.0)
        if weight != 1.0:
            if streaming:
                raise ValueError(
                    "Using a dataset `weight` is not supported with streaming datasets."
                )
            n_before = len(dataset)
            dataset = resample_by_weight(dataset, weight, seed=seed)
            logger.info(f"  weight={weight}: {n_before} -> {len(dataset)} examples")
        datasets_list.append(dataset)

    fractions = [dataset_config.fraction for dataset_config in mixture_config.datasets]
    if any(fraction is not None for fraction in fractions):
        if any(fraction is None for fraction in fractions):
            raise ValueError(
                "`fraction` must be set for either all datasets in the mixture or none of them."
            )
        if streaming:
            raise ValueError(
                "Using a dataset `fraction` is not supported with streaming datasets."
            )
        normalized = [fraction / sum(fractions) for fraction in fractions]
        total = min(
            len(dataset) / weight
            for dataset, weight in zip(datasets_list, normalized, strict=False)
            if weight > 0
        )
        datasets_list = [
            dataset.select(range(round(weight * total)))
            for dataset, weight in zip(datasets_list, normalized, strict=False)
        ]

    if not datasets_list:
        raise ValueError("No datasets were loaded from the mixture configuration")
    try:
        combined = concatenate_datasets(datasets_list)
    except Exception as e:
        raise ValueError(
            f"Could not align dataset mixture schemas for concatenation ({e}). This usually means two "
            "datasets define the same column with genuinely incompatible types (not just one of them "
            "missing it) — check the offending column above."
        ) from e
    if isinstance(combined, Dataset):
        logger.info(f"Created dataset mixture with {len(combined)} examples")
    if mixture_config.test_split_size is not None:
        logger.info(
            "Splitting dataset into train and test sets with test size: "
            f"{mixture_config.test_split_size}"
        )
        return combined.train_test_split(test_size=mixture_config.test_split_size)
    if streaming:
        return IterableDatasetDict({"train": combined})
    return DatasetDict({"train": combined})


def get_weighted_dataset(mixture_config, seed=42):
    """Load a TRL dataset mixture and apply each entry's size multiplier (no normalization)."""
    return load_mixture(mixture_config, seed=seed)
