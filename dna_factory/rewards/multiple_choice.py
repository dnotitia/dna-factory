"""
Multiple-choice reward family for GRPO on thinking models, graded the way `evals/_multiple_choice_ko.py`
(KMMLU / KMMLU-Redux / KMMLU-Pro) grades: a strict `^정답: X` line, then a lenient `정답: X` anywhere,
then a bare trailing letter. The parse runs on the answer section only (after the single
`</think>`), so a `정답: X` written inside the reasoning is ignored. The eval only matches this when
its `vllm serve` runs with `--reasoning-parser qwen3` (configs/GRPO/4b-kmmlu-grpo.yaml does):
without it, the opening `<think>` sits in the prompt, inspect never strips the reasoning, and an
earlier `정답: X` inside the reasoning wins the eval's parse.

- `make_mcq_accuracy_reward` -- 0/1: the letter parsed from the answer section equals the gold
  letter, on a finished completion whose reasoning closed exactly once.
- `make_mcq_format_penalty` -- non-positive: reasoning never closed, or `정답:` repeated in the
  answer section.

Gold is a letter (`A`..`E`) in the `solution` column; see utils/build-kmmlu-grpo-dataset.py.
Concrete instances live in my_rewards.py. Contract: docs/grpo-rewards.md.
"""

import functools
import re

from .generative import _to_text, _warn_once
from .reasoning import _answer_section

# Same patterns as evals/_multiple_choice_ko.py:parse_answers.
_STRICT_RE = re.compile(r"(?i)^정답\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)", re.MULTILINE)
_LENIENT_RE = re.compile(r"(?i)정답\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)")
_ANSWER_MARK_RE = re.compile(r"정답\s*:")


def _bare_answer(text):
    """Last non-empty line stripped of emphasis/parens (evals/_multiple_choice_ko.py:bare_answer)."""
    lines = [line.strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    stripped = lines[-1].strip("*_` \t").strip("()[].:：").strip("*_` \t")
    return stripped or None


def parse_mcq_answer(text, num_choices):
    """The single letter the KMMLU eval would extract from `text`, or None."""
    match = _STRICT_RE.search(text) or _LENIENT_RE.search(text)
    matched = match.group(1) if match is not None else _bare_answer(text)
    if matched is None:
        return None
    matched = matched.strip().rstrip(".")
    allowed = {chr(ord("A") + i) for i in range(num_choices)}
    return matched if matched in allowed else None


def _mcq_accuracy_reward(
    prompts,
    completions,
    completion_ids,
    solution=None,
    num_choices=None,
    log_extra=None,
    *,
    max_completion_len,
    default_num_choices,
    name,
    **kwargs,
):
    n = len(completions)
    if solution is None:
        _warn_once(
            f"{name}:missing_solution",
            f"{name}: no `solution` column in this dataset; scoring nothing (all None).",
        )
        return [None] * n
    if num_choices is None:
        num_choices = [default_num_choices] * n

    rewards, notes = [], []
    for completion, ids, gold, k in zip(
        completions, completion_ids, solution, num_choices, strict=True
    ):
        gold = (gold or "").strip().upper()
        if not gold:
            rewards.append(None)
            notes.append("[no gold]")
            continue
        # A completion cut at the cap never finished, whatever it wrote on the way.
        if len(ids) >= max_completion_len:
            rewards.append(0.0)
            notes.append("[truncated]")
            continue
        answer = _answer_section(_to_text(completion))
        if answer is None:
            rewards.append(0.0)
            notes.append("[reasoning not closed once]")
            continue
        # Only the answer section counts: a `정답: X` inside <think>...</think> is ignored.
        letter = parse_mcq_answer(answer, int(k))
        rewards.append(1.0 if letter == gold else 0.0)
        notes.append(f"{letter or '[unparsed]'} (gold {gold})")

    if log_extra is not None:
        log_extra(f"{name}_answer", notes)
    return rewards


def make_mcq_accuracy_reward(
    max_completion_len, default_num_choices=4, name="mcq_accuracy_reward"
):
    """
    Build a 0/1 multiple-choice accuracy reward with the KMMLU eval's answer parsing.

    1.0 when the letter `evals/_multiple_choice_ko.py`'s parsing extracts from the answer section
    (after `</think>`; the reasoning is ignored) equals the gold `solution` letter. 0.0 when it differs or nothing parses, when the completion hit
    `max_completion_len`, or when `</think>` does not appear exactly once. None when the row has
    no gold. The allowed letters come from an optional `num_choices` column (else
    `default_num_choices`), as the eval only accepts letters of existing options.

    Args:
        max_completion_len: the recipe's `max_completion_length`; not auto-synced.
        default_num_choices: option count when the dataset has no `num_choices` column.
        name: sets __name__ (the `rewards/<name>/mean` metric key).

    Returns:
        (prompts, completions, completion_ids, solution=None, num_choices=None, log_extra=None,
        **kwargs) -> list[float | None].
    """
    reward = functools.partial(
        _mcq_accuracy_reward,
        max_completion_len=max_completion_len,
        default_num_choices=default_num_choices,
        name=name,
    )
    reward.__name__ = name
    return reward


def _mcq_format_penalty(
    prompts,
    completions,
    completion_ids,
    *,
    max_completion_len,
    per_extra_answer,
    **kwargs,
):
    scores = []
    for completion, ids in zip(completions, completion_ids, strict=True):
        # Truncation is the overlong penalty's job; don't charge it twice.
        if len(ids) >= max_completion_len:
            scores.append(None)
            continue
        answer = _answer_section(_to_text(completion))
        if answer is None:
            scores.append(-1.0)
            continue
        extra = max(0, len(_ANSWER_MARK_RE.findall(answer)) - 1)
        scores.append(-min(1.0, per_extra_answer * extra))
    return scores


def make_mcq_format_penalty(
    max_completion_len, per_extra_answer=0.25, name="mcq_format_penalty"
):
    """
    Build a non-positive penalty for a finished multiple-choice completion's answer section.

    -1.0 when `</think>` does not appear exactly once; otherwise `-per_extra_answer` for every
    `정답:` beyond the first in the answer section (hedging across letters), floored at -1.0.
    None for a completion at `max_completion_len` (left to the overlong penalty).

    Returns:
        (prompts, completions, completion_ids, **kwargs) -> list[float | None].
    """
    reward = functools.partial(
        _mcq_format_penalty,
        max_completion_len=max_completion_len,
        per_extra_answer=per_extra_answer,
    )
    reward.__name__ = name
    return reward
