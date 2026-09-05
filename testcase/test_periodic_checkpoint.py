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
from dna_factory.periodic_checkpoint import PeriodicCheckpointCallback


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
