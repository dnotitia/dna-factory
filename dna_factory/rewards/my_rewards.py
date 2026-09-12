"""
Concrete GRPO reward instances. YAML: `- dna_factory.rewards.<name>`.

Framework stays in generative.py / verifiable.py; this is the file you edit.
Catalog and contract: docs/grpo-rewards.md.
"""

from pathlib import Path

from trl.rewards import (
    accuracy_reward as _hf_accuracy_reward,
)
from trl.rewards import (
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
)

from .generative import _warn_once, make_judge_reward
from .verifiable import make_string_match_reward

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# --- Generic judge instances (packaged default rubric; configured via JUDGE_* env vars) ---
judge_reward = make_judge_reward(name="judge_reward")
judge_reward.__doc__ = (
    "Default LLM-as-judge rubric, no reference. YAML: dna_factory.rewards.judge_reward"
)
judge_reward_with_reference = make_judge_reward(
    reference_column="solution", name="judge_reward_with_reference"
)
judge_reward_with_reference.__doc__ = (
    "Default reference-guided judge; column `solution`. "
    "YAML: dna_factory.rewards.judge_reward_with_reference"
)

# --- Per-label judges for configs/GRPO/qwen3-0.6B-rlvr*.yaml; each pins its own rubric_file ---
persona_judge = make_judge_reward(
    str(_PROMPTS_DIR / "example_judge_rubric_with_reference.md"),
    only_label="persona",
    reference_column="expected_output",
    name="persona_judge",
)
persona_judge.__doc__ = (
    "only_label=persona; column `expected_output`; "
    "example_judge_rubric_with_reference.md. YAML: dna_factory.rewards.persona_judge"
)
ccp_judge = make_judge_reward(
    str(_PROMPTS_DIR / "example_judge_rubric_safety.md"),
    only_label="ccp",
    name="ccp_judge",
)
ccp_judge.__doc__ = "only_label=ccp; example_judge_rubric_safety.md. YAML: dna_factory.rewards.ccp_judge"
rlvr_judge = make_judge_reward(
    str(_PROMPTS_DIR / "example_judge_rubric_reasoning.md"),
    only_label="rlvr",
    reference_column="expected_output",
    name="rlvr_judge",
)
rlvr_judge.__doc__ = (
    "only_label=rlvr; column `expected_output`; "
    "example_judge_rubric_reasoning.md. YAML: dna_factory.rewards.rlvr_judge"
)

# --- Verifiable (string-match) instance: last \boxed{...} vs the `solution` column ---
boxed_match_reward = make_string_match_reward(name="boxed_match_reward")
boxed_match_reward.__doc__ = (
    "Last \\boxed{...} vs `solution`. YAML: dna_factory.rewards.boxed_match_reward"
)


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


# --- Other trl.rewards get_* factory instances (length bound 4096; rebuild if max_completion_length changes) ---
cosine_scaled_reward = get_cosine_scaled_reward(max_len=4096)
cosine_scaled_reward.__doc__ = (
    "Correctness × length cosine; `solution` + math_verify. "
    "YAML: dna_factory.rewards.cosine_scaled_reward"
)

repetition_penalty_reward = get_repetition_penalty_reward(
    ngram_size=3, max_penalty=-1.0
)
repetition_penalty_reward.__doc__ = (
    "3-gram repetition penalty. YAML: dna_factory.rewards.repetition_penalty_reward"
)

# soft_punish_cache = max_completion_length // 5 (4096 // 5). Used by
# configs/GRPO/qwen3-0.6B-rlvr-composed.yaml. Length isn't auto-synced to the recipe.
soft_overlong_penalty = get_soft_overlong_punishment(
    max_completion_len=4096, soft_punish_cache=819
)
soft_overlong_penalty.__doc__ = (
    "Soft length punishment (max_completion_len=4096). "
    "YAML: dna_factory.rewards.soft_overlong_penalty"
)
