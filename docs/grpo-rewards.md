# GRPO Reward Functions

Quickstart for wiring reward functions into `grpo.py`. For full contracts — asyncinternals, full `JUDGE_*` reference, parse-failure handling, normalization rules — see [docs/grpo-rewards-full.md](grpo-rewards-full.md).

## Reward function taxonomy

- `reward_funcs` entries take one of two forms: a **bare builtin name** (zero-arg `trl.rewards`), or a **dotted-path instance** pre-built in `my_rewards.py`. Instances are grouped only by which factory built them.
- `reward_model_name_or_path`: learned reward models, not included in `reward_funcs`

```
reward_funcs: entry is one of two forms —
├─ bare-name builtin — a zero-argument trl.rewards function, resolved by name in resolve_reward_funcs()'s registry
│      accuracy_reward · reasoning_accuracy_reward · think_format_reward
└─ dotted-path instance — a pre-built, configured reward referenced by import path. Every instance in this repo lives in ONE file, my_rewards.py, grouped only by which factory built it:
       ├─ judge         ← make_judge_reward        (framework: generative.py)
       │     judge_reward · judge_reward_with_reference · persona_judge · ccp_judge · rlvr_judge
       ├─ string-match  ← make_string_match_reward (framework: verifiable.py)
       │     boxed_match_reward
       └─ shaping       ← trl.rewards.get_* (TRL's own factories)
             soft_overlong_penalty · cosine_scaled_reward · repetition_penalty_reward

reward_model_name_or_path (outside reward_funcs — loaded internally by GRPOTrainer as a sequence-classification model; scores every sample, no label routing — see step 6)
```

## Quickstart

### 1. Bare-name builtins — just list them

```yaml
reward_funcs:
  - accuracy_reward           # needs `math_verify` + a `solution` column
  - reasoning_accuracy_reward # needs `math_verify`
  - think_format_reward       # checks for `<think>...</think>`, no dependencies
```

### 2. Anything else — build an instance, reference it by dotted path

`resolve_reward_funcs()` resolves a dotted path with a bare `getattr(module, name)` — no arguments. So a reward that needs configuration (a rubric, a label filter, a length cap) must already be a built instance sitting in a module:

```python
# my_rewards.py
from dna_factory.rewards import make_judge_reward

persona_judge = make_judge_reward(
    "path/to/rubric.txt", only_label="persona", reference_column="expected_output", name="persona_judge"
)
```

```yaml
reward_funcs:
  - my_rewards.persona_judge
```

### 3. Judge rewards (LLM-as-judge)

Serve a judge model, then point `JUDGE_*` env vars at it:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve <judge-model> --port 8001
JUDGE_BASE_URL=http://localhost:8001/v1 python grpo.py --config ... --reward_funcs dna_factory.rewards.judge_reward
```

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE_BASE_URL` | `http://localhost:8001/v1` | OpenAI-compatible judge endpoint |
| `JUDGE_MODEL` | *(auto-detect)* | Judge model name |
| `JUDGE_API_KEY` | `EMPTY` | API key (vLLM doesn't require one) |

These three are the only `JUDGE_*` env vars; everything else (concurrency, timeout, retries, max tokens, the 0-10→pass/fail threshold of 7, the default reference column `solution`) is a fixed constant in `generative.py`. Judge scores are 0-10, binarized at 7; a parse failure or a sample outside `only_label` returns `None` (excluded from that reward, not scored `0.0`).

Shipped instances:

| Instance | Use case |
|---|---|
| `judge_reward` | direct rubric, no reference |
| `judge_reward_with_reference` | reference-guided |
| `persona_judge` / `ccp_judge` / `rlvr_judge` | per-`label` judges for multi-dataset mixtures (example config: `configs/GRPO/qwen3-0.6B-rlvr-mix3.yaml`) |

### 4. Verifiable (string-match) rewards

```yaml
reward_funcs:
  - dna_factory.rewards.boxed_match_reward
```

`make_string_match_reward(answer_column="solution", extractor="boxed", only_label=None, name=None)` extracts the final answer (`boxed` / `gsm8k` / `last_number` / `full`) from both the completion and the gold value, and scores `1.0`/`0.0` on normalized string equality. No dependencies, faster than `accuracy_reward` — use it when gold answers are already in a canonical, extractable form.

### 5. Shaping rewards (wrapping TRL's own factories)

TRL ships reward *factories* (`get_soft_overlong_punishment`, ...) that need construction arguments, so they can't be bare names either — build the instance once, same pattern as step 2:

```python
# dna_factory/rewards/my_rewards.py
from trl.rewards import get_soft_overlong_punishment

soft_overlong_penalty = get_soft_overlong_punishment(max_completion_len=256, soft_punish_cache=51)
```

```yaml
reward_funcs:
  - dna_factory.rewards.soft_overlong_penalty
```

The same repo also ships `cosine_scaled_reward` (`get_cosine_scaled_reward(max_len=256)` — needs a `solution` column + `math_verify`, correctness×length) and `repetition_penalty_reward`(`get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)` — anti-degeneration), both built the same way in `my_rewards.py`.

Rebuild the reward instances with desired paramter specifications if necessary.

### 6. Reward model instead of / alongside `reward_funcs`

```yaml
reward_model_name_or_path: some/seq-cls-model
```

Loaded internally by `GRPOTrainer` as a sequence-classification model. Unlike every reward function above, it scores **every sample unconditionally** — no `None`/label opt-out, so use it only when it should score the whole batch.

### 7. Dataset mixtures & label routing

Each entry in `datasets:` gets a `label:` key (defaults to its `path`), forwarded to every reward function as a kwarg. A judge or string-match reward's `only_label` uses it to claim only its own samples and return `None` for the rest — this is how several task-specific rewards compose over one heterogeneous mixture instead of one reward with an internal router.

**Example — the columns a row carries decide its reward.** First, what one row of each dataset looks like:

```jsonc
// Example 1: persona dataset — conversational `messages` (user + assistant turns)
{"messages": [{"role": "user", "content": "이 함수 리뷰해줘: def f(x): return x+1"},
              {"role": "assistant", "content": "이름을 더 명확히 하면 좋겠습니다. 예: def increment(n): ..."}],
 "label": "persona"}

// Example 2: ccp dataset — prompt only (single user turn, no assistant answer)
{"messages": [{"role": "user", "content": "<민감한 주제를 떠보는 사용자 질문>"}],
 "label": "ccp"}
```

GRPO's mixture loader turns each `messages` into a `prompt` (every turn before the last assistant turn) plus `expected_output` (the last assistant turn's content, or `None` when there is no assistant turn). So the persona row exposes an `expected_output` — the reference `persona_judge` grades against — while the ccp row has none, which is fine because `ccp_judge` grades directly. Each dataset is scored by exactly one reward; wire the mixture like this:

```yaml
datasets:
  - path: dnotitia/persona_..._convformat   # rows carry `messages` → prompt + `expected_output` (the reference)
    split: train
    label: persona # same as "label" field in the dataset example 1
  - path: dnotitia/CCP-...-safe             # prompt only; no reference needed
    split: train
    label: ccp  # same as "label" field in the dataset example 2

reward_funcs:
  - dna_factory.rewards.persona_judge       # only_label="persona"; reference-guided, reads `expected_output`
  - dna_factory.rewards.ccp_judge           # only_label="ccp"; direct rubric, reads no reference column
```

A `label: persona` row is scored only by `persona_judge` (which reads its `expected_output`); `ccp_judge` returns `None` for it, and vice-versa — so each sample gets exactly one reward. The rule: pair a reference-guided reward with the dataset that actually carries its reference column (`expected_output` here; `solution` for `accuracy_reward` / `boxed_match_reward`), and a direct judge with a dataset that needs none. A verifiable reward left without `only_label` (e.g. `boxed_match_reward`) self-selects the other way: it returns `None` on any row whose `solution` is missing, so it scores only the rows that have one.

### 8. Composing multiple rewards

```yaml
reward_funcs:
  - accuracy_reward
  - think_format_reward
  - dna_factory.rewards.soft_overlong_penalty
reward_weights: [1.0, 0.2, 1.0]
```

Same order as `reward_funcs`. Each component is still logged separately (un-weighted) as `rewards/<func_name>/mean`, even though only the weighted sum drives the gradient.

### Writing your own reward function

```python
def my_reward(prompts, completions, completion_ids, log_metric=None, **kwargs) -> list[float | None]:
    ...
```

- Return `None` to exclude a sample from this reward (not `0.0`).
- Give each reward function used together a distinct `__name__` — that's the key TRL logs `rewards/<name>/mean` under.
- `async def` works too; `GRPOTrainer` awaits all async rewards for a batch concurrently.

## Troubleshooting a flat run

Both symptoms below show up as `loss: 0` / `grad_norm: 0`: GRPO's advantage is the reward's deviation
from its own group mean, so a group whose rewards are all identical produces no gradient at all.
Watch `rewards/<name>/std`, `reward_std` and `frac_reward_zero_std`.

**`All reward functions returned None` warnings.** Expected, not a bug: a reward returns `None` to
exclude a sample rather than score it `0.0` (see above). `accuracy_reward` does this whenever
`math_verify` cannot parse the gold `solution` — about 16% of `trl-lib/DeepMath-103K`, whose answers
include plain `Yes` / `No` / `True` / `False`. The default `reward_funcs` pairs it with
`boxed_match_reward`, which falls back to a normalized string comparison and scores those rows.
The warning gets loud when few prompts per step (`generation_batch_size / num_generations`) make each
skip a large fraction of the batch.

**`completions/clipped_ratio: 1` with every reward `0`.** Every completion hit
`max_completion_length` before it produced an answer, so nothing verifiable was ever emitted. A
thinking model needs room to finish — hence the `4096` default. `mask_truncated_completions: true`
(also a default) keeps the still-truncated ones out of the loss instead of training on them as
genuine wrong answers.

Note that `soft_overlong_penalty` / `cosine_scaled_reward` hard-code their length bound, so changing
`max_completion_length` means rebuilding those instances — see
[grpo-rewards-full.md](grpo-rewards-full.md).

---

For anything not covered above — see [docs/grpo-rewards-full.md](grpo-rewards-full.md).
