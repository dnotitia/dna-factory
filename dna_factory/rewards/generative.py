"""
Generative (LLM-as-judge) reward framework for GRPO training.

    $ CUDA_VISIBLE_DEVICES=1 vllm serve <judge-model> --port 8001
    $ JUDGE_BASE_URL=http://localhost:8001/v1 python grpo.py --reward_funcs dna_factory.rewards.judge_reward

make_judge_reward(rubric_file, only_label, reference_column, name) builds one such async judge.
Quickstart: docs/grpo-rewards.md. Full contract, JUDGE_* reference, and usage examples:
docs/grpo-rewards-full.md.
This module is the framework only — all concrete instances (judge_reward, judge_reward_with_reference,
persona_judge, ccp_judge, rlvr_judge) live in my_rewards.py.
"""

import asyncio
import logging
import os
import re
import time
from pathlib import Path

# Initialize logger
logger = logging.getLogger(__name__)

# Packaged default rubric templates (see _load_default_prompt/_load_default_reference_prompt below).
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Only the judge-server connection is env-configurable; everything else is a fixed constant below.
JUDGE_BASE_URL = os.environ.get(
    "JUDGE_BASE_URL", "http://localhost:8001/v1"
)  # OpenAI-compatible judge endpoint
JUDGE_MODEL = os.environ.get("JUDGE_MODEL")  # None → auto-detect from the server
JUDGE_API_KEY = os.environ.get(
    "JUDGE_API_KEY", "EMPTY"
)  # API key (vLLM doesn't require one)


def _load_default_prompt():
    """Packaged direct-rubric default (dna_factory/rewards/prompts/example_judge_rubric_default.md), read lazily."""
    return (_PROMPTS_DIR / "example_judge_rubric_default.md").read_text()


def _load_default_reference_prompt():
    """Packaged reference-rubric default (dna_factory/rewards/prompts/example_judge_rubric_with_reference.md),
    read lazily."""
    return (_PROMPTS_DIR / "example_judge_rubric_with_reference.md").read_text()


# Lazy singletons: connect to the judge server only once training starts, not at import time.
_client = None
_resolved_model = None
_parse_failure_logged = False
_warned_once = set()  # dedup key set for "warn once" diagnostics (see _warn_once)


def _warn_once(key, message):
    if key not in _warned_once:
        _warned_once.add(key)
        logger.warning(message)


def _resolve_template_source(rubric_file, default_loader):
    """Load a rubric template: the explicit `rubric_file` if given, else the packaged
    `default_loader()` (so a pinned judge never touches the packaged default file)."""
    if rubric_file is not None:
        with open(rubric_file) as f:
            return f.read()
    return default_loader()


async def _get_client_and_model():
    global _client, _resolved_model
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(
            base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY, timeout=120
        )  # 120s per request
    if _resolved_model is None:
        if JUDGE_MODEL:
            _resolved_model = JUDGE_MODEL
        else:
            # Auto-detect: take the first model served by the endpoint (vLLM serves exactly one)
            models = await _client.models.list()
            _resolved_model = models.data[0].id
            logger.info(
                f"Auto-detected judge model from {JUDGE_BASE_URL}: {_resolved_model}"
            )
    return _client, _resolved_model


def _to_text(message_or_text):
    # Conversational format: a list of {"role": ..., "content": ...} dicts; standard format: a string
    if isinstance(message_or_text, list):
        return "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in message_or_text
            if msg["role"] != "system"
        )
    return message_or_text


def _parse_score(text: str):
    """Parse the judge's trailing 'Score: N' (0-10) and binarize at a fixed threshold of 7."""
    # Drop the judge's own thinking block, then expect a 'Score: N' line with N an integer 0-10.
    visible = text.split("</think>")[-1]
    match = re.search(r"Score:\s*(\d{1,2})", visible)
    if match is None:
        return None
    score = int(match.group(1))
    if not 0 <= score <= 10:
        return None
    # Binarize to RLVR pass/fail: N >= 7 → 1.0, else 0.0
    return 1.0 if score >= 7 else 0.0


async def _judge_one(semaphore, judge_input):
    """Score one rendered judge prompt; returns [0, 1], or None (skip / parse failure / API failure)."""
    if judge_input is None:
        return None

    async with semaphore:
        client, model = await _get_client_and_model()
        max_retries = (
            2  # retries after the first attempt (backoff 1s, 2s); not env-configurable
        )
        for attempt in range(max_retries + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": judge_input}],
                    temperature=0.0,
                    max_tokens=2048,  # includes the judge's own thinking
                )
                # Parse failures are not retried: at temperature 0 the judge would repeat itself.
                # Returning None excludes the sample from this reward (TRL converts it to NaN).
                score = _parse_score(response.choices[0].message.content)
                if score is None:
                    # Log the first unparseable output per process to make diagnosis possible
                    # (a common cause: the judge's own thinking exhausts the 2048-token cap,
                    # visible here as finish_reason='length').
                    global _parse_failure_logged
                    if not _parse_failure_logged:
                        _parse_failure_logged = True
                        output_tail = (response.choices[0].message.content or "")[-300:]
                        logger.warning(
                            f"Judge output could not be parsed (finish_reason="
                            f"{response.choices[0].finish_reason!r}). Output tail: {output_tail!r}"
                        )
                return score
            except Exception as e:
                if attempt == max_retries:
                    logger.warning(
                        f"Judge request failed after {max_retries + 1} attempts: {e}"
                    )
                    return None
                # Exponential backoff: 1s, 2s, 4s, ...
                await asyncio.sleep(2**attempt)


def _log_judge_metrics(log_metric, scores, elapsed, prefix):
    """Log judge health (parse-failure rate, mean score, batch latency) alongside training metrics."""
    if log_metric is None or not scores:
        return
    parsed = [s for s in scores if s is not None]
    log_metric(
        f"{prefix}/parse_failure_rate", (len(scores) - len(parsed)) / len(scores)
    )
    if parsed:
        log_metric(f"{prefix}/score_mean", sum(parsed) / len(parsed))
    log_metric(f"{prefix}/batch_latency_sec", elapsed)


def _build_judge_inputs(template, prompts, completions, references, labels, only_label):
    """Pure, network-free helper: decides per-sample whether this judge scores it (only_label /
    missing-reference checks) and renders the prompt. Returns (judge_inputs, skip_mask)."""
    has_reference = "{reference}" in template
    judge_inputs, skip_mask = [], []
    for i in range(len(prompts)):
        if only_label is not None:
            label = labels[i] if labels is not None else None
            if label != only_label:
                judge_inputs.append(None)
                skip_mask.append(True)
                continue

        if has_reference:
            ref = references[i] if references is not None else None
            reference_text = _to_text(ref) if ref is not None else ""
            if not reference_text.strip():
                judge_inputs.append(None)  # no gold answer for this sample → skip
                skip_mask.append(True)
                continue
            judge_inputs.append(
                template.format(
                    prompt=_to_text(prompts[i]),
                    completion=_to_text(completions[i]),
                    reference=reference_text,
                )
            )
        else:
            judge_inputs.append(
                template.format(
                    prompt=_to_text(prompts[i]), completion=_to_text(completions[i])
                )
            )
        skip_mask.append(False)

    return judge_inputs, skip_mask


def make_judge_reward(
    rubric_file=None, only_label=None, reference_column=None, name=None
):
    """
    Build an async LLM-as-judge GRPO reward function.

    Args:
        rubric_file: prompt template path; if None, uses a packaged default
            (dna_factory/rewards/prompts/example_judge_rubric_default.md or example_judge_rubric_with_reference.md).
        only_label: if set, only samples whose dataset `label` column matches are scored; others -> None.
        reference_column: if set, grades against this gold-answer dataset column (else "solution").
        name: sets __name__ — TRL logs metrics under rewards/<name>/*, so judges used together
            MUST have distinct names, or their metrics collide.

    Returns:
        async (prompts, completions, completion_ids, log_metric=None, **kwargs) -> list[float | None].

    See docs/grpo-rewards-full.md for the full contract.
    """
    resolved_name = name or (f"judge_{only_label}" if only_label else "judge_reward")
    state = {"template": None}

    def _template():
        if state["template"] is None:
            default_loader = (
                _load_default_reference_prompt
                if reference_column
                else _load_default_prompt
            )
            state["template"] = _resolve_template_source(rubric_file, default_loader)
        return state["template"]

    async def judge(prompts, completions, completion_ids, log_metric=None, **kwargs):
        template = _template()
        labels = kwargs.get("label")
        if only_label is not None and labels is None:
            _warn_once(
                f"{resolved_name}:missing_label",
                f"{resolved_name}: only_label={only_label!r} is set but no dataset in this mixture "
                "has a 'label' column (grpo.py's mixture loader injects one automatically for every "
                "dataset — check how this dataset was loaded). Scoring nothing (all None).",
            )
            return [None] * len(prompts)

        references = None
        if "{reference}" in template:
            ref_col = reference_column or "solution"
            references = kwargs.get(ref_col)
            if references is None:
                _warn_once(
                    f"{resolved_name}:missing_reference",
                    f"{resolved_name}: this rubric is reference-guided but no dataset in this mixture "
                    f"has a '{ref_col}' column (set reference_column to match your dataset). Columns "
                    f"forwarded by the trainer: {sorted(kwargs.keys())}. Scoring nothing (all None).",
                )
                return [None] * len(prompts)

        judge_inputs, _ = _build_judge_inputs(
            template, prompts, completions, references, labels, only_label
        )
        semaphore = asyncio.Semaphore(16)  # max in-flight judge requests per process

        start_time = time.monotonic()
        scores = await asyncio.gather(
            *[_judge_one(semaphore, ji) for ji in judge_inputs]
        )
        elapsed = time.monotonic() - start_time

        _log_judge_metrics(log_metric, scores, elapsed, prefix=resolved_name)
        return scores

    judge.__name__ = resolved_name
    return judge
