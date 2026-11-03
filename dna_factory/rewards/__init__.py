"""
dna_factory.rewards — GRPO reward functions. YAML: `- dna_factory.rewards.<name>`.

Framework / factory modules (no concrete instances):
- generative.py — LLM-as-judge framework: JUDGE_* connection config, make_judge_reward.
- verifiable.py — programmatic string-match framework: make_string_match_reward.
- reasoning.py — thinking-model answer checks: make_strict_accuracy_reward,
  make_answer_format_penalty, make_ngram_loop_penalty.
- multiple_choice.py — thinking-model multiple choice graded like evals/_multiple_choice_ko.py:
  make_mcq_accuracy_reward, make_mcq_format_penalty.

Concrete instances live in my_rewards.py (one-line __doc__ on each) and are
re-exported here:

  judge_reward · judge_reward_with_reference · persona_judge · ccp_judge · rlvr_judge
  boxed_match_reward · safe_accuracy_reward · strict_accuracy_reward
  cosine_scaled_reward · repetition_penalty_reward · soft_overlong_penalty
  answer_format_penalty · ngram_loop_penalty · soft_overlong_penalty_8k
  mcq_accuracy_reward · mcq_format_penalty

Bare TRL names (`accuracy_reward`, `reasoning_accuracy_reward`, `think_format_reward`)
are resolved in grpo.py, not here. Catalog and contract: docs/grpo-rewards.md.
"""

from .generative import make_judge_reward
from .multiple_choice import make_mcq_accuracy_reward, make_mcq_format_penalty
from .my_rewards import (
    answer_format_penalty,
    boxed_match_reward,
    ccp_judge,
    cosine_scaled_reward,
    judge_reward,
    judge_reward_with_reference,
    mcq_accuracy_reward,
    mcq_format_penalty,
    ngram_loop_penalty,
    persona_judge,
    repetition_penalty_reward,
    rlvr_judge,
    safe_accuracy_reward,
    soft_overlong_penalty,
    soft_overlong_penalty_8k,
    strict_accuracy_reward,
)
from .reasoning import (
    make_answer_format_penalty,
    make_ngram_loop_penalty,
    make_strict_accuracy_reward,
)
from .verifiable import make_string_match_reward

__all__ = [
    "answer_format_penalty",
    "boxed_match_reward",
    "ccp_judge",
    "cosine_scaled_reward",
    "judge_reward",
    "judge_reward_with_reference",
    "make_answer_format_penalty",
    "make_judge_reward",
    "make_mcq_accuracy_reward",
    "make_mcq_format_penalty",
    "make_ngram_loop_penalty",
    "make_strict_accuracy_reward",
    "make_string_match_reward",
    "mcq_accuracy_reward",
    "mcq_format_penalty",
    "ngram_loop_penalty",
    "persona_judge",
    "repetition_penalty_reward",
    "rlvr_judge",
    "safe_accuracy_reward",
    "soft_overlong_penalty",
    "soft_overlong_penalty_8k",
    "strict_accuracy_reward",
]
