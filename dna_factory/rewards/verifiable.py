"""
Verifiable (string-match) reward family for GRPO — no model calls, no judge server: scores a
completion by extracting its final answer (`extractor`) and comparing it, normalized, against the
gold answer in `kwargs[answer_column]`. Faster/cheaper than a judge and dependency-free, but strict
— unlike `accuracy_reward` (math_verify), "0.5" and "1/2" won't match. `make_string_match_reward`
builds one reward; the concrete instances (e.g. `boxed_match_reward`) live in my_rewards.py.
Full contract: docs/grpo-rewards-full.md.
"""

import re

from .generative import _to_text, _warn_once


def _extract_boxed(text):
    """Last `\\boxed{...}` in text, scanned with balanced braces (so nested braces work)."""
    last = None
    for m in re.finditer(r"\\boxed\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end() : i - 1]
    return last


def _extract_gsm8k(text):
    if "####" not in text:
        return None
    return text.rsplit("####", 1)[-1].strip()


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _extract_last_number(text):
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def _extract_full(text):
    return text


_EXTRACTORS = {
    "boxed": _extract_boxed,
    "gsm8k": _extract_gsm8k,
    "last_number": _extract_last_number,
    "full": _extract_full,
}


def _normalize(text):
    """strip -> collapse whitespace -> lowercase -> drop trailing '.' -> drop thousands-separator
    commas inside numbers -> strip surrounding $/\\$ -> strip surrounding braces."""
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text.strip()).lower()
    if text.endswith("."):
        text = text[:-1]
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    for prefix in ("\\$", "$"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    for suffix in ("\\$", "$"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    return text.strip()


def make_string_match_reward(answer_column="solution", extractor="boxed", only_label=None, name=None):
    """
    Build a sync string-match GRPO reward function.

    Args:
        answer_column: dataset column holding the gold answer.
        extractor: "boxed" | "gsm8k" | "last_number" | "full" (see module docstring).
        only_label: if set, only samples whose dataset `label` column matches are scored; others -> None.
        name: sets __name__ (defaults to f"string_match_{extractor}").

    Returns:
        (prompts, completions, completion_ids, log_metric=None, **kwargs) -> list[float | None].

    See docs/grpo-rewards-full.md for the full contract.
    """
    if extractor not in _EXTRACTORS:
        raise ValueError(f"Unknown extractor {extractor!r}; choose from {sorted(_EXTRACTORS)}")
    extract = _EXTRACTORS[extractor]
    resolved_name = name or f"string_match_{extractor}"

    def reward(prompts, completions, completion_ids, log_metric=None, **kwargs):
        labels = kwargs.get("label")
        golds = kwargs.get(answer_column)
        if only_label is not None and labels is None:
            _warn_once(
                f"{resolved_name}:missing_label",
                f"{resolved_name}: only_label={only_label!r} is set but no dataset in this mixture "
                "has a 'label' column (grpo.py's mixture loader injects one automatically for every "
                "dataset — check how this dataset was loaded). Scoring nothing (all None).",
            )
            return [None] * len(prompts)

        scores = []
        for i in range(len(prompts)):
            if only_label is not None and labels[i] != only_label:
                scores.append(None)
                continue
            gold_raw = golds[i] if golds is not None else None
            gold_text = _to_text(gold_raw) if gold_raw is not None else None
            if gold_text is None or not gold_text.strip():
                scores.append(None)
                continue
            gold_extracted = extract(gold_text)
            gold_norm = _normalize(gold_extracted if gold_extracted is not None else gold_text)
            completion_norm = _normalize(extract(_to_text(completions[i])))
            scores.append(1.0 if completion_norm == gold_norm else 0.0)
        return scores

    reward.__name__ = resolved_name
    return reward
