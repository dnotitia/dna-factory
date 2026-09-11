"""
dna_factory.rewards — GRPO reward functions. YAML: `- dna_factory.rewards.<name>`.

Framework / factory modules (no concrete instances):
- generative.py — LLM-as-judge framework: JUDGE_* connection config, make_judge_reward.
- verifiable.py — programmatic string-match framework: make_string_match_reward.

Concrete instances live in my_rewards.py (one-line __doc__ on each) and are
re-exported here:

  judge_reward · judge_reward_with_reference · persona_judge · ccp_judge · rlvr_judge
  boxed_match_reward · safe_accuracy_reward
  cosine_scaled_reward · repetition_penalty_reward · soft_overlong_penalty

Bare TRL names (`accuracy_reward`, `reasoning_accuracy_reward`, `think_format_reward`)
are resolved in grpo.py, not here. Catalog and contract: docs/grpo-rewards.md.
"""

from .generative import make_judge_reward
from .my_rewards import (
    boxed_match_reward,
    ccp_judge,
    cosine_scaled_reward,
    judge_reward,
    judge_reward_with_reference,
    persona_judge,
    repetition_penalty_reward,
    rlvr_judge,
    safe_accuracy_reward,
    soft_overlong_penalty,
)
from .verifiable import make_string_match_reward

__all__ = [
    "boxed_match_reward",
    "ccp_judge",
    "cosine_scaled_reward",
    "judge_reward",
    "judge_reward_with_reference",
    "make_judge_reward",
    "make_string_match_reward",
    "persona_judge",
    "repetition_penalty_reward",
    "rlvr_judge",
    "safe_accuracy_reward",
    "soft_overlong_penalty",
]
