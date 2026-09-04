"""Wall-clock periodic checkpointing for long training runs.

`transformers.Trainer` only saves on step/epoch counts (`save_steps`/`save_strategy`),
so a step-based interval can mean anything in wall-clock time on GRPO (generation
length varies per step) — or, as a ratio float like `save_steps: 0.2`, first save
tens of hours in. `PeriodicCheckpointCallback` fills that gap: every
`interval_seconds` of wall-clock time it sets `control.should_save`, and the
trainer's normal end-of-step path (`_maybe_log_save_evaluate` → `_save_checkpoint`)
does the actual save — DeepSpeed handling, `save_total_limit` rotation, and
main-process gating all apply unchanged.

Wired in `grpo.py` from `DnotitiaArguments.periodic_save_seconds` (`0` = off).
"""

import time

from transformers import TrainerCallback


class PeriodicCheckpointCallback(TrainerCallback):
    """Set `control.should_save` at most once per `interval_seconds` of wall time.

    The clock also resets on every `on_save`, so the interval means "at least this
    long between any two checkpoints", whether the other one came from this callback
    or from the regular step-based schedule.
    """

    def __init__(self, interval_seconds: float):
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {interval_seconds}"
            )
        self.interval_seconds = interval_seconds
        self._last_save = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_save = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or self._last_save is None:
            return control
        if time.monotonic() - self._last_save >= self.interval_seconds:
            control.should_save = True
            self._last_save = time.monotonic()
        return control

    def on_save(self, args, state, control, **kwargs):
        self._last_save = time.monotonic()
        return control
