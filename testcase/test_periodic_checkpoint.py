"""
Test cases for dna_factory/periodic_checkpoint.py::PeriodicCheckpointCallback
"""

import sys
from pathlib import Path

import pytest
from transformers import TrainerControl, TrainerState

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

import dna_factory.periodic_checkpoint as pc
from dna_factory.periodic_checkpoint import (
    PeriodicCheckpointCallback,
    parse_duration_to_seconds,
)


def _fresh():
    return TrainerState(), TrainerControl()


class TestPeriodicCheckpointCallback:
    """Test cases for PeriodicCheckpointCallback"""

    def test_rejects_non_positive_interval(self):
        with pytest.raises(ValueError):
            PeriodicCheckpointCallback(0)
        with pytest.raises(ValueError):
            PeriodicCheckpointCallback(-10)

    def test_no_save_before_interval(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(pc.time, "monotonic", lambda: now[0])
        cb = PeriodicCheckpointCallback(3600)
        state, control = _fresh()
        cb.on_train_begin(None, state, control)
        state.global_step = 5
        now[0] += 3599.0  # just under the interval
        out = cb.on_step_end(None, state, control)
        assert out.should_save is False

    def test_saves_once_per_interval(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(pc.time, "monotonic", lambda: now[0])
        cb = PeriodicCheckpointCallback(3600)
        state, control = _fresh()
        cb.on_train_begin(None, state, control)
        state.global_step = 5

        now[0] += 3600.0
        out = cb.on_step_end(None, state, control)
        assert out.should_save is True

        # Timer reset: the very next step must not save again.
        out.should_save = False  # as the trainer loop would after saving
        out = cb.on_step_end(None, state, out)
        assert out.should_save is False

    def test_step_save_resets_clock(self, monkeypatch):
        """A step-based save (on_save) also resets the wall-clock timer."""
        now = [1000.0]
        monkeypatch.setattr(pc.time, "monotonic", lambda: now[0])
        cb = PeriodicCheckpointCallback(3600)
        state, control = _fresh()
        cb.on_train_begin(None, state, control)
        state.global_step = 5

        now[0] += 3599.0
        cb.on_save(None, state, control)  # regular step-based save happened here
        now[0] += 3599.0  # 3599s since the last save of any kind
        out = cb.on_step_end(None, state, control)
        assert out.should_save is False

    def test_ignores_step_zero(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr(pc.time, "monotonic", lambda: now[0])
        cb = PeriodicCheckpointCallback(3600)
        state, control = _fresh()
        cb.on_train_begin(None, state, control)
        state.global_step = 0
        now[0] += 7200.0
        out = cb.on_step_end(None, state, control)
        assert out.should_save is False


class _FakeDist:
    """Stand-in for `torch.distributed`: `broadcast` hands out rank 0's verdict."""

    def __init__(self, world_size=8, rank0_due=True, backend="nccl"):
        self.world_size = world_size
        self.rank0_due = rank0_due
        self.backend = backend
        self.broadcasts = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_world_size(self):
        return self.world_size

    def get_backend(self):
        return self.backend

    def broadcast(self, tensor, src=0):
        self.broadcasts += 1
        tensor.fill_(int(self.rank0_due))


class TestRankAgreement:
    """The save step must come from rank 0, not from each rank's own clock."""

    @staticmethod
    def _callback(monkeypatch, fake, now):
        monkeypatch.setattr(pc.time, "monotonic", lambda: now[0])
        monkeypatch.setattr(pc, "dist", fake)
        cb = PeriodicCheckpointCallback(3600)
        state, control = _fresh()
        cb.on_train_begin(None, state, control)
        state.global_step = 5
        return cb, state, control

    def test_follower_does_not_save_when_rank0_is_not_due(self, monkeypatch):
        """The skew that deadlocked a run: a follower trips the interval first."""
        fake = _FakeDist(rank0_due=False)
        now = [1000.0]
        cb, state, control = self._callback(monkeypatch, fake, now)

        now[0] += 3600.0  # this rank is due, rank 0 (slower to finish its save) is not
        out = cb.on_step_end(None, state, control)
        assert out.should_save is False
        assert fake.broadcasts == 1

        # The clock was left alone, so the rank still saves on rank 0's step.
        fake.rank0_due = True
        out = cb.on_step_end(None, state, out)
        assert out.should_save is True

    def test_follower_saves_when_rank0_is_due(self, monkeypatch):
        """A lagging local clock must not keep a rank out of rank 0's save."""
        fake = _FakeDist(rank0_due=True)
        now = [1000.0]
        cb, state, control = self._callback(monkeypatch, fake, now)

        now[0] += 10.0  # nowhere near the interval on this rank
        out = cb.on_step_end(None, state, control)
        assert out.should_save is True
        assert cb._last_save == now[0]  # timer reset together with the others

    def test_broadcasts_on_every_step(self, monkeypatch):
        """The broadcast is a collective: it may not be skipped on quiet steps."""
        fake = _FakeDist(rank0_due=False)
        now = [1000.0]
        cb, state, control = self._callback(monkeypatch, fake, now)

        for _ in range(3):
            now[0] += 1.0
            cb.on_step_end(None, state, control)
        assert fake.broadcasts == 3

    def test_single_process_skips_the_broadcast(self, monkeypatch):
        fake = _FakeDist(world_size=1, rank0_due=False)
        now = [1000.0]
        cb, state, control = self._callback(monkeypatch, fake, now)

        now[0] += 3600.0
        out = cb.on_step_end(None, state, control)
        assert out.should_save is True  # local verdict stands
        assert fake.broadcasts == 0


class TestParseDurationToSeconds:
    """Test cases for parse_duration_to_seconds"""

    def test_human_units(self):
        assert parse_duration_to_seconds("6h") == 21600.0
        assert parse_duration_to_seconds("30m") == 1800.0
        assert parse_duration_to_seconds("90s") == 90.0
        assert parse_duration_to_seconds("1d") == 86400.0
        assert parse_duration_to_seconds("1w") == 604800.0
        assert parse_duration_to_seconds("1H") == 3600.0  # case-insensitive
        assert parse_duration_to_seconds("1.5h") == 5400.0

    def test_combined(self):
        assert parse_duration_to_seconds("1h30m") == 5400.0
        assert parse_duration_to_seconds("1h 30m") == 5400.0
        assert parse_duration_to_seconds("2hours15minutes") == 8100.0

    def test_numeric_passthrough_seconds(self):
        assert parse_duration_to_seconds(21600) == 21600.0
        assert parse_duration_to_seconds(0.5) == 0.5
        assert parse_duration_to_seconds("21600") == 21600.0  # bare string = seconds
        assert parse_duration_to_seconds("90") == 90.0

    def test_disabled_markers(self):
        for off in (None, 0, 0.0, "0", "off", "no", "never", "none", "disabled"):
            assert parse_duration_to_seconds(off) == 0.0

    def test_rejects_bad_input(self):
        for bad in (-5, "-3", "6x", "abc", True):
            with pytest.raises(ValueError):
                parse_duration_to_seconds(bad)
