"""Wall-clock periodic checkpointing for long training runs.

`transformers.Trainer` only saves on step/epoch counts (`save_steps`/`save_strategy`),
so a step-based interval can mean anything in wall-clock time on GRPO (generation
length varies per step). `PeriodicCheckpointCallback` fills that gap: every
`interval_seconds` of wall-clock time it sets `control.should_save`, and the
trainer's normal end-of-step path (`_maybe_log_save_evaluate` → `_save_checkpoint`)
does the actual save — DeepSpeed handling, `save_total_limit` rotation, and
main-process gating all apply unchanged.

This works even with `save_strategy="no"`: the default flow callback only *sets*
`should_save` for step/epoch schedules, but the save path itself honors
`should_save` from any callback regardless of strategy. So `"no"` cleanly disables
step-based saves while leaving periodic saves (and `save_total_limit` rotation and
`get_last_checkpoint` resume) intact.

Wired in `grpo.py` from `DnotitiaArguments.periodic_save_seconds`, parsed with
`parse_duration_to_seconds` (`0`/`"off"` = off).
"""

import re
import time

from transformers import TrainerCallback

_UNIT_TO_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
    "w": 604800,
    "week": 604800,
    "weeks": 604800,
}

_DISABLED_VALUES = {"", "0", "0.0", "off", "no", "never", "none", "disable", "disabled"}

_DURATION_TOKEN_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)?")


def parse_duration_to_seconds(value) -> float:
    """Parse a wall-clock duration into seconds.

    Accepts plain numbers (interpreted as seconds, for backward compatibility),
    human-friendly strings (`"6h"`, `"30m"`, `"90s"`, `"1d"`, combined `"1h30m"` /
    `"1h 30m"`), and disabled markers (`0`, `"off"`, `"no"`, `"never"`, `"none"`,
    `"disable"`/`"disabled"`, `None` → `0.0`). Units are case-insensitive and a
    bare number without a unit means seconds. Raises `ValueError` on negative or
    unparseable input.
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        raise ValueError(f"Invalid duration {value!r}: expected seconds or a string like '6h'.")
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"Invalid duration {value!r}: must be non-negative.")
        return float(value)
    text = str(value).strip().lower()
    if text in _DISABLED_VALUES:
        return 0.0
    pos = 0
    total = 0.0
    matched = False
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        match = _DURATION_TOKEN_RE.match(text, pos)
        if match is None:
            raise ValueError(
                f"Invalid duration {value!r}: could not parse at {text[pos:]!r}. "
                "Use plain seconds (21600) or a string like '6h', '30m', '1h30m'."
            )
        amount = float(match.group("value"))
        unit = (match.group("unit") or "s").lower()
        if unit not in _UNIT_TO_SECONDS:
            raise ValueError(
                f"Invalid duration {value!r}: unknown unit {unit!r}. "
                "Supported units: s, m, h, d, w (plus sec/min/hr/day/week spellings)."
            )
        total += amount * _UNIT_TO_SECONDS[unit]
        matched = True
        pos = match.end()
    if not matched:
        raise ValueError(
            f"Invalid duration {value!r}: use plain seconds (21600) or a string like '6h'."
        )
    return total


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
