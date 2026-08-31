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
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
)

from .generative import make_judge_reward
from .verifiable import make_string_match_reward

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# --- Generic judge instances (packaged default rubric; configured via JUDGE_* env vars) ---
judge_reward = make_judge_reward(name="judge_reward")
judge_reward_with_reference = make_judge_reward(reference_column="solution", name="judge_reward_with_reference")

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

# --- Other trl.rewards get_* factory instances ---
# requirements.txt pins trl==1.7.0, whose trl.rewards exposes three get_* factories; all three are
# instantiated here and referenced by dotted path (e.g. `- dna_factory.rewards.cosine_scaled_reward`).

# Cosine-scaled correctness reward: math-verified correctness scaled by completion length along a
# cosine schedule, so a shorter correct answer scores higher. Needs a `solution` column (like
# accuracy_reward) and math_verify. Rebuild with max_len matched to the recipe's max_completion_length.
cosine_scaled_reward = get_cosine_scaled_reward(max_len=256)

# N-gram repetition penalty (anti-degeneration): penalizes repeated 3-grams in the completion.
repetition_penalty_reward = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)

# Tuned for configs/_defaults-GRPO.yaml's max_completion_length: 256 (soft_punish_cache = 256 // 5).
# Used by configs/GRPO/qwen3-0.6B-rlvr-composed.yaml. If a recipe overrides max_completion_length,
# build a matching new instance rather than reusing this one — the length isn't auto-synced.
soft_overlong_penalty = get_soft_overlong_punishment(max_completion_len=256, soft_punish_cache=51)