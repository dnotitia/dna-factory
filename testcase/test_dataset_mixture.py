"""
Test cases for dna_factory/dnotitia_dataset_mixture.py::resample_by_weight
"""

import sys
from pathlib import Path

import pytest
from datasets import Dataset

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.dnotitia_dataset_mixture import resample_by_weight

N = 20


def _dataset():
    return Dataset.from_dict({"id": list(range(N))})


def _ids(dataset):
    return list(dataset["id"])


class TestResampleByWeight:
    """Test cases for resample_by_weight"""

    def test_weight_one_returns_dataset_unchanged(self):
        ds = _dataset()
        assert _ids(resample_by_weight(ds, 1.0)) == _ids(ds)

    def test_weight_zero_returns_empty_dataset(self):
        out = resample_by_weight(_dataset(), 0)
        assert len(out) == 0
        assert out.column_names == ["id"]

    def test_integer_weight_repeats_rows_in_order(self):
        out = resample_by_weight(_dataset(), 2.0)
        assert _ids(out) == list(range(N)) * 2

    def test_fractional_weight_downsamples_from_original_rows(self):
        ids = _ids(resample_by_weight(_dataset(), 0.25))
        assert len(ids) == round(N * 0.25)
        assert set(ids) <= set(range(N))
        assert len(set(ids)) == len(ids)

    def test_mixed_weight_adds_copies_and_subsample(self):
        ids = _ids(resample_by_weight(_dataset(), 2.5))
        assert len(ids) == 2 * N + round(N * 0.5)
        assert ids[: 2 * N] == list(range(N)) * 2
        tail = ids[2 * N :]
        assert set(tail) <= set(range(N))
        assert len(set(tail)) == len(tail)

    def test_negative_weight_raises_value_error(self):
        with pytest.raises(ValueError):
            resample_by_weight(_dataset(), -0.5)

    def test_same_seed_gives_identical_subsample(self):
        ds = _dataset()
        first = _ids(resample_by_weight(ds, 0.5, seed=7))
        second = _ids(resample_by_weight(ds, 0.5, seed=7))
        assert first == second

    def test_different_seed_gives_different_subsample(self):
        ds = _dataset()
        a = _ids(resample_by_weight(ds, 0.5, seed=1))
        b = _ids(resample_by_weight(ds, 0.5, seed=2))
        assert a != b
