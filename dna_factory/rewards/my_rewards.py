"""
Concrete GRPO reward-function instances — every reward that is built by calling a factory
(`make_judge_reward` / `make_string_match_reward` / a `trl.rewards` factory) lives here, in one
place. The framework/factory code stays in its own modules (generative.py, verifiable.py); this is
the module you edit or copy to wire your own rewards. Configs reference these by dotted path, e.g.
`dna_factory.rewards.persona_judge`.

Full contract: docs/grpo-rewards-full.md.
"""

from pathlib import Path

from trl.rewards import (
    accuracy_reward as _hf_accuracy_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
)

from .generative import _warn_once, make_judge_reward
from .verifiable import make_string_match_reward

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# --- Generic judge instances (packaged default rubric; configured via JUDGE_* env vars) ---
judge_reward = make_judge_reward(name="judge_reward")
judge_reward_with_reference = make_judge_reward(
    reference_column="solution", name="judge_reward_with_reference"
)

# --- Per-label judges for configs/GRPO/qwen3-0.6B-rlvr*.yaml; each pins its own rubric_file ---
persona_judge = make_judge_reward(
    str(_PROMPTS_DIR / "example_judge_rubric_with_reference.md"),
    only_label="persona",
    reference_column="expected_output",
    name="persona_judge",
)
ccp_judge = make_judge_reward(
    str(_PROMPTS_DIR / "example_judge_rubric_safety.md"),
    only_label="ccp",
    name="ccp_judge",
)
rlvr_judge = make_judge_reward(
    str(_PROMPTS_DIR / "example_judge_rubric_reasoning.md"),
    only_label="rlvr",
    reference_column="expected_output",
    name="rlvr_judge",
)

# --- Verifiable (string-match) instance: last \boxed{...} vs the `solution` column ---
boxed_match_reward = make_string_match_reward(name="boxed_match_reward")


def safe_accuracy_reward(
    prompts, completions, completion_ids=None, solution=None, log_extra=None, **kwargs
):
    """
    Crash-safe drop-in for TRL's `accuracy_reward` (same call contract, same scores).

    Upstream computes every reward correctly but then stringifies the parsed sympy
    expressions for logging (`str(answer_parsed)`), which raises on rare model outputs
    (e.g. a `cot` at its singularity blows up inside sympy/mpmath with
    `ZeroDivisionError`) — killing the whole training run. This wrapper first tries the
    upstream call on the full batch (identical behavior when nothing is poisoned); if
    that raises, it retries sample-by-sample so only the poisoned samples score `None`
    (excluded from this reward, like an unparseable gold) while healthy samples keep
    their real scores. Reference with `- dna_factory.rewards.safe_accuracy_reward`.
    """
    n = len(completions)
    if solution is None:
        return [None] * n
    try:
        return _hf_accuracy_reward(
            completions=completions,
            solution=list(solution),
            log_extra=log_extra,
            **kwargs,
        )
    except Exception as e:
        _warn_once(
            "safe_accuracy_reward:batch_fallback",
            f"safe_accuracy_reward: upstream accuracy_reward raised {type(e).__name__} "
            f"({e}); retrying sample-by-sample, poisoned samples score None.",
        )
    rewards = []
    for completion, sol in zip(completions, solution, strict=True):
        try:
            rewards.append(
                _hf_accuracy_reward(completions=[completion], solution=[sol])[0]
            )
        except Exception:
            rewards.append(None)
    return rewards

# --- Other trl.rewards get_* factory instances ---
# requirements.txt pins trl==1.7.0, whose trl.rewards exposes three get_* factories; all three are
# instantiated here and referenced by dotted path (e.g. `- dna_factory.rewards.cosine_scaled_reward`).

# Cosine-scaled correctness reward: math-verified correctness scaled by completion length along a
# cosine schedule, so a shorter correct answer scores higher. Needs a `solution` column (like
# accuracy_reward) and math_verify. Rebuild with max_len matched to the recipe's max_completion_length.
cosine_scaled_reward = get_cosine_scaled_reward(max_len=4096)

# N-gram repetition penalty (anti-degeneration): penalizes repeated 3-grams in the completion.
repetition_penalty_reward = get_repetition_penalty_reward(
    ngram_size=3, max_penalty=-1.0
)

# Tuned for configs/_defaults-GRPO.yaml's max_completion_length: 4096 (soft_punish_cache = 4096 // 5).
# Used by configs/GRPO/qwen3-0.6B-rlvr-composed.yaml. If a recipe overrides max_completion_length,
# build a matching new instance rather than reusing this one — the length isn't auto-synced.
soft_overlong_penalty = get_soft_overlong_punishment(
    max_completion_len=4096, soft_punish_cache=819
)
