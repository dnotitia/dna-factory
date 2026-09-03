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

from dataclasses import dataclass, field

from datasets import concatenate_datasets
from trl import DatasetMixtureConfig
from trl.scripts.utils import DatasetConfig


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
