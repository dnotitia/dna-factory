"""
dna_factory.rewards — GRPO reward functions.

Framework / factory modules (no concrete instances):
- generative.py — LLM-as-judge framework: JUDGE_* connection config, make_judge_reward.
- verifiable.py — programmatic string-match framework: make_string_match_reward.

Concrete instances — every reward built from those factories (and from trl.rewards factories) —
live together in my_rewards.py and are re-exported here so configs can reference them by the short
dotted path `dna_factory.rewards.<name>`.
Full contract: docs/grpo-rewards-full.md.
"""

from .generative import make_judge_reward
from .verifiable import make_string_match_reward
from .my_rewards import (
    judge_reward,
    judge_reward_with_reference,
    persona_judge,
    ccp_judge,
    rlvr_judge,
    boxed_match_reward,
    soft_overlong_penalty,
    cosine_scaled_reward,
    repetition_penalty_reward,
)

__all__ = [
    "make_judge_reward",
    "make_string_match_reward",
    "judge_reward",
    "judge_reward_with_reference",
    "persona_judge",
    "ccp_judge",
    "rlvr_judge",
    "boxed_match_reward",
    "soft_overlong_penalty",
    "cosine_scaled_reward",
    "repetition_penalty_reward",
]
