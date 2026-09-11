# GRPO Reward Functions

Wiring and contract for reward functions in `grpo.py`. Concrete instances live in `dna_factory/rewards/my_rewards.py` and are referenced as `- dna_factory.rewards.<name>`. The factories stay in `generative.py` (judge) and `verifiable.py` (string-match); `my_rewards.py` is the file you edit.

## Taxonomy

`reward_funcs` entries take one of two forms: a **bare builtin name** (zero-arg `trl.rewards`), or a **dotted-path instance** pre-built in `my_rewards.py`. `reward_model_name_or_path` is a learned sequence-classification model, not an entry in `reward_funcs`.

```
reward_funcs: entry is one of two forms —
├─ bare-name builtin — a zero-argument trl.rewards function, resolved by name in resolve_reward_funcs()'s registry
│      accuracy_reward · reasoning_accuracy_reward · think_format_reward
└─ dotted-path instance — a pre-built, configured reward referenced by import path. Every instance in this repo lives in ONE file, my_rewards.py:
       ├─ judge         ← make_judge_reward        (framework: generative.py)
       │     judge_reward · judge_reward_with_reference · persona_judge · ccp_judge · rlvr_judge
       ├─ string-match  ← make_string_match_reward (framework: verifiable.py)
       │     boxed_match_reward
       ├─ accuracy      ← crash-safe wrapper around TRL accuracy_reward
       │     safe_accuracy_reward
       └─ shaping       ← trl.rewards.get_* (TRL's own factories)
             soft_overlong_penalty · cosine_scaled_reward · repetition_penalty_reward

reward_model_name_or_path (outside reward_funcs — loaded internally by GRPOTrainer as a sequence-classification model; scores every sample, no label routing)
```

Every dotted-path entry shares the same reason it cannot be a bare name: `resolve_reward_funcs()` resolves a dotted path with a bare `getattr` — no arguments — so anything that needs configuration must already be a fully-built instance in a module.

### Shipped instances (`dna_factory.rewards.<name>`)

| Instance | Kind | Notes |
|---|---|---|
| `judge_reward` | judge | default rubric, no reference; needs a `JUDGE_*` server |
| `judge_reward_with_reference` | judge | default reference rubric; column `solution` |
| `persona_judge` | judge | `only_label=persona`; column `expected_output`; `example_judge_rubric_with_reference.md` |
| `ccp_judge` | judge | `only_label=ccp`; `example_judge_rubric_safety.md` |
| `rlvr_judge` | judge | `only_label=rlvr`; column `expected_output`; `example_judge_rubric_reasoning.md` |
| `boxed_match_reward` | string-match | last `\boxed{...}` vs `solution` |
| `safe_accuracy_reward` | accuracy | crash-safe TRL `accuracy_reward`; `solution` + `math_verify` (default GRPO config) |
| `cosine_scaled_reward` | shaping | correctness × length cosine; `solution` + `math_verify`; length bound 4096 |
| `repetition_penalty_reward` | shaping | 3-gram repetition penalty |
| `soft_overlong_penalty` | shaping | soft length punishment (`max_completion_len=4096`) |

Shaping instances hard-code a length bound of **4096**. Rebuild them if the recipe's `max_completion_length` changes (the GRPO default is `16384` and is not auto-synced). Bare TRL names (`accuracy_reward`, `reasoning_accuracy_reward`, `think_format_reward`) are resolved in `grpo.py`, not in this package.

## Reward function contract

A reward function is any callable with this signature:

```python
def my_reward(prompts, completions, completion_ids, log_metric=None, **kwargs) -> list[float | None]:
    ...
```

- `GRPOTrainer` (`_calculate_rewards` in `trl/trainer/grpo_trainer.py`) calls every entry in `reward_funcs` with `prompts`, `completions`, `completion_ids`, plus **every other column present in the dataset** (everything except `prompt`, `completion`, `completion_ids`) forwarded as a keyword argument, one value per sample. A reward that doesn't care about an extra column can simply not declare it — `**kwargs` swallows the rest.
- Two more kwargs are always forwarded:
  - `trainer_state` — the trainer's live `transformers.TrainerState` (e.g. for reward shaping keyed on training progress; none of the rewards in this repo currently use it).
  - `log_metric(name: str, value: float)` — logs a scalar that gets averaged over the logging window and reported alongside the built-in training metrics. This is a plain metric name with **no automatic `rewards/` prefix**.
  - (A third, `log_extra(column, values)`, exists for logging extra per-sample columns to the completions table; not used by anything in this repo today.)
- `async def` reward functions are supported and detected automatically (`inspect.iscoroutinefunction`). All async reward functions for a batch run concurrently via `asyncio.gather` on the trainer's dedicated background event loop (`self.async_loop`, a daemon thread started at trainer init) — the main training thread blocks on `asyncio.run_coroutine_threadsafe(...).result()` until they all finish, but multiple async reward functions (and, inside `make_judge_reward`, multiple concurrent per-sample judge calls under one `asyncio.Semaphore`) overlap rather than running serially.
- Return a `list[float | None]`, one entry per sample, in the same order as `prompts`/`completions`.
- **Returning `None` for a sample means "this reward doesn't apply to this sample"** — TRL converts it to `NaN` and excludes it from that sample's reward computation (`nansum`/`nanmean`) instead of scoring it `0.0`. This is how several task-specific rewards compose over one heterogeneous mixture: each reward looks at its own samples and returns `None` for everything else, so you list one reward per task/label in `reward_funcs:` instead of writing a single reward with an internal `if/elif` router.

Give each reward function used together a distinct `__name__` — that is the key TRL logs `rewards/<name>/mean` under.

### Metrics

Two separate metric families exist, and it's easy to conflate them:

- `rewards/<__name__>/mean` and `rewards/<__name__>/std` — logged **automatically** by TRL from the list of floats a reward function returns (`GRPOTrainer._metrics`). Because this is keyed by the function's `__name__`, **two reward functions used together must have distinct `__name__`s** — this is why `make_judge_reward`'s `name` argument exists.
- Anything logged via the `log_metric` kwarg — a **plain name you choose**, not automatically prefixed with `rewards/`. `make_judge_reward` uses this to report judge health as `<name>/score_mean`, `<name>/parse_failure_rate`, `<name>/batch_latency_sec` (e.g. `persona_judge/score_mean`, not `rewards/persona_judge/score_mean`).

## Wiring: `reward_funcs` and `resolve_reward_funcs`

`reward_funcs:` in YAML accepts a mix of:

- built-in names resolved by `grpo.py`'s `resolve_reward_funcs()` registry — these must be zero-argument reward functions: `accuracy_reward`, `reasoning_accuracy_reward`, `think_format_reward`, or
- any dotted import path (e.g. `dna_factory.rewards.persona_judge` or `my_lib.rewards.custom_reward`), resolved relative to the current working directory via `importlib.import_module` + `getattr`.

`script_args.reward_model_name_or_path`, if set, is prepended as a plain model-id string — loaded internally by `GRPOTrainer` as a sequence-classification reward model, not through this registry. **Unlike every reward function above, this path has no per-sample opt-out**: `GRPOTrainer` scores every sample in the batch unconditionally (`reward_func(**reward_inputs).logits[:, 0]`) — it cannot return `None`, so it can't be routed to a subset of a dataset mixture via `label`. Use a reward model only when it should score the entire batch.

```yaml
reward_model_name_or_path: some/seq-cls-model
```

**Important constraint:** for a dotted path, `resolve_reward_funcs()` does a bare `getattr(module, attr_name)` — it passes **no arguments**. Any reward that needs configuration (a rubric file, a label to filter on, a reference column, a length cap) must therefore already be a fully-configured module-level instance; you cannot point `reward_funcs:` at a factory function itself and expect it to be called with arguments.

Hence the pattern used throughout this repo: define configured instances in `my_rewards.py` (or your own module), and point YAML at *those* dotted paths:

```python
# my_rewards.py
from dna_factory.rewards import make_judge_reward

persona_judge = make_judge_reward(
    "path/to/rubric.txt", only_label="persona", reference_column="expected_output", name="persona_judge"
)
```

```yaml
reward_funcs:
  - dna_factory.rewards.persona_judge   # or my_rewards.persona_judge if you keep a local copy
```

This applies equally to TRL's own reward factories (`get_soft_overlong_punishment` and friends) — see [Shaping rewards](#shaping-rewards-trlrewards-factories) below.

### Bare-name builtins (`trl.rewards`)

Just list them. No configuration, no dotted path:

```yaml
reward_funcs:
  - accuracy_reward           # needs `math_verify` + a `solution` column
  - reasoning_accuracy_reward # needs `math_verify`
  - think_format_reward       # checks for `<think>...</think>`, no dependencies
```

- **`accuracy_reward`** — needs the `math_verify` package and a `solution` dataset column (this repo's default dataset, `trl-lib/DeepMath-103K`, already has one). Prefer `dna_factory.rewards.safe_accuracy_reward` in this repo — same scores, but a sympy stringify crash in upstream logging cannot kill the run.
- **`reasoning_accuracy_reward`** — also needs `math_verify`.
- **`think_format_reward`** — checks the completion is exactly `<think>...</think>` followed by an answer; binary `1.0`/`0.0`. Only meaningful for thinking-template models.

TRL also ships reward *factories* (`get_soft_overlong_punishment`, `get_cosine_scaled_reward`, `get_repetition_penalty_reward`) — these need construction arguments, so none of them are in this bare-name registry.

## Dataset mixtures & labels

`grpo.py`'s `get_dataset_with_schema_alignment()` (used whenever the YAML has a top-level `datasets:` list instead of `dataset_name`) mechanically normalizes each dataset in the mixture via `_normalize_dataset_for_grpo()`, then concatenates them:

- a `messages` column is split into `prompt` (every turn before the last assistant turn, role/content only) and `expected_output` (that last assistant turn's content, or `None` if there is no assistant turn at all); `messages` itself is dropped.
- a `prompt` column with no `messages` is normalized in place (a plain string becomes a single user turn; a list is reduced to role/content only). Every other column already on the dataset (`solution`, a pre-existing `expected_output`, anything else) is left untouched.
- neither `messages` nor `prompt` present → a hard error (nothing to build a GRPO prompt from).
- every dataset then gets a `label` column added: each entry's per-dataset `label:` key in the `datasets:` list (`LabeledDatasetConfig`, a `DatasetConfig` subclass), defaulting to that entry's `path` if `label:` is not set.

There is no closed routing enum — `label` is a plain, open string, and `datasets.concatenate_datasets` natively aligns the mismatched schemas across the mixture (a column missing from one dataset is filled with `None` for its rows), so adding a new dataset to a mixture never requires touching the loader. Reward functions decide what to do with `label` (and any other forwarded column) themselves — `only_label` on a judge or string-match reward is the usual hook.

**The columns a row carries decide its reward.** First, what one row of each dataset looks like:

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

Example config: `configs/GRPO/qwen3-0.6B-rlvr-mix3.yaml`.

## The judge family (`generative.py` → instances in `my_rewards.py`)

TRL has no first-class support for a *generative* reward model (the `reward_model_name_or_path` path only accepts a sequence-classification head via `.logits[:, 0]`), so a generative LLM-as-judge is integrated as a custom async reward function.

Serve a judge model, then point `JUDGE_*` env vars at it:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve <judge-model> --port 8001
JUDGE_BASE_URL=http://localhost:8001/v1 python grpo.py --config ... --reward_funcs dna_factory.rewards.judge_reward
```

`make_judge_reward(rubric_file=None, only_label=None, reference_column=None, name=None)` builds one such judge (an `async def` function matching the contract above):

- **`only_label`** — if set, this judge only scores samples whose `label` column equals this value; every other sample gets `None` (excluded). Composing several judges over a mixture means one `make_judge_reward(only_label=...)` instance per label, all listed in `reward_funcs:` — not one function with an internal switch.
- **`reference_column`** — if set, this judge is reference-guided: the gold answer is read from `kwargs[reference_column]` (falling back to `"solution"` if `reference_column` is `None`). Whether the *loaded* rubric text is actually reference-guided is decided by inspecting it for a `{reference}` placeholder at call time, not by this flag directly — the flag only steers which packaged default template is used when `rubric_file` is `None`.
- **`rubric_file`** — pins the prompt template (`{prompt}`/`{completion}`, plus `{reference}` for a reference-guided rubric) to a specific file, read and cached lazily on the judge's first call (not at import time — the judge server doesn't need to be reachable until training actually starts). When `rubric_file` is `None`, a packaged default template is used (direct, or reference-guided when `reference_column` is set) — there is no env-var rubric override.
- **`name`** — sets the returned function's `__name__` (defaults to `f"judge_{only_label}"` if `only_label` is set, else `"judge_reward"`). Required to be distinct across judges used together.

The packaged default rubrics live at `dna_factory/rewards/prompts/example_judge_rubric_default.md` (direct) and `dna_factory/rewards/prompts/example_judge_rubric_with_reference.md` (reference-guided) — loaded lazily via `Path(__file__).parent / "prompts" / ...`, not relative to the current working directory. Copy either file as a starting point for a custom rubric and point an instance's `rubric_file` at the copy.

### `JUDGE_*` environment variables

Only the judge-server *connection* is configured through environment variables (dotted-path `reward_funcs` resolution passes no arguments):

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE_BASE_URL` | `http://localhost:8001/v1` | OpenAI-compatible judge endpoint |
| `JUDGE_MODEL` | *(unset → auto-detect)* | Judge model name; if unset, the first model listed by the server is used |
| `JUDGE_API_KEY` | `EMPTY` | API key (vLLM doesn't require one) |

Everything else is a fixed constant in `generative.py`, no longer env-configurable: max in-flight concurrency `16`, per-request timeout `120s`, `2` retries (exponential backoff 1s, 2s), `2048` max generated tokens (including the judge's own thinking), the 0-10 → pass/fail binarize threshold `7`, and the default reference column `"solution"`.

### Score contract

The rubric always asks the judge for an integer **0-10** score, ending with a line in the exact format `Score: N`. `_parse_score()` strips the judge's own `<think>...</think>` block, regexes out `Score:\s*(\d{1,2})`, and then **binarizes**: `score >= 7` (a fixed threshold) → reward `1.0`, otherwise `0.0`. A coarse pass/fail signal is intentionally used instead of the raw 0-10 score, to be more robust against reward hacking and judge score-drift, while the 0-10 scale still gives the judge internal room to discriminate. **Any parse failure — an out-of-range score, a missing `Score:` line, or an API error after retries — returns `None`** for that sample (excluded from this reward, not scored `0.0`). The first unparseable judge output per process is logged with its `finish_reason` (a common cause: the judge's own thinking exhausts the `2048`-token cap before it reaches the score line).

Judge health is logged via `log_metric` under `<name>/score_mean`, `<name>/parse_failure_rate`, and `<name>/batch_latency_sec` — e.g. `persona_judge/score_mean`, `judge_reward/parse_failure_rate`.

### Shipped judges

| Instance | Config | Notes |
|---|---|---|
| `judge_reward` | direct rubric, no reference | `name="judge_reward"`, no `only_label`/`reference_column` |
| `judge_reward_with_reference` | reference-guided | `reference_column="solution"`; used by `configs/GRPO/qwen3-0.6B-judge.yaml` |
| `persona_judge` | `only_label="persona"`, `reference_column="expected_output"`, rubric `example_judge_rubric_with_reference.md` | used by `configs/GRPO/qwen3-0.6B-rlvr-mix3.yaml` |
| `ccp_judge` | `only_label="ccp"`, direct rubric `example_judge_rubric_safety.md` | used by `configs/GRPO/qwen3-0.6B-rlvr-mix3.yaml` |
| `rlvr_judge` | `only_label="rlvr"`, `reference_column="expected_output"`, rubric `example_judge_rubric_reasoning.md` | used by `configs/GRPO/qwen3-0.6B-rlvr-mix3.yaml` |

## Verifiable rewards (`verifiable.py`)

`make_string_match_reward(answer_column="solution", extractor="boxed", only_label=None, name=None)` builds a sync, no-dependency reward: extracts the final answer from the completion via `extractor`, extracts the same from `kwargs[answer_column]` (falling back to the raw gold string if extraction finds nothing there), and scores `1.0`/`0.0` on normalized string equality.

| Extractor | Extracts |
|---|---|
| `boxed` | last `\boxed{...}`, balanced-brace scan (nested braces work) |
| `gsm8k` | text after the last `####` |
| `last_number` | last numeric token (optional sign, optional decimal, commas allowed) |
| `full` | the whole completion, unchanged |

Normalization (both sides, before comparing): strip, collapse internal whitespace, lowercase, drop a trailing `.`, remove thousands-separator commas inside numbers, strip surrounding `$`/`\$`, strip surrounding `{}`.

`None` (sample excluded, not scored `0.0`) when: the gold value for that sample is missing/`None`/empty; `only_label` is set and the sample's `label` doesn't match; or `only_label` is set but no dataset in the mixture has a `label` column (warned once, same `_warn_once` pattern as the judges).

```yaml
reward_funcs:
  - dna_factory.rewards.boxed_match_reward
```

Trades `accuracy_reward`'s semantic equivalence (`math_verify` treats `0.5` and `1/2` as equal) for speed, determinism, and zero dependencies — pick string-match when gold answers are already in a canonical, extractable form, `accuracy_reward` / `safe_accuracy_reward` when they might not be.

## Accuracy: `safe_accuracy_reward`

The default GRPO config uses `dna_factory.rewards.safe_accuracy_reward` instead of the bare `accuracy_reward` name. Same call contract and same scores as TRL's `accuracy_reward` (`solution` column + `math_verify`), but crash-safe: upstream computes every reward correctly, then stringifies the parsed sympy expressions for logging (`str(answer_parsed)`), which raises on rare model outputs (e.g. a `cot` at its singularity blows up inside sympy/mpmath with `ZeroDivisionError`) and would kill the whole training run. The wrapper first tries the upstream call on the full batch; if that raises, it retries sample-by-sample so only the poisoned samples score `None` (excluded from this reward, like an unparseable gold) while healthy samples keep their real scores.

## Shaping rewards (`trl.rewards` factories)

TRL's own reward *factories* — `get_soft_overlong_punishment`, `get_cosine_scaled_reward`, `get_repetition_penalty_reward` — all need construction arguments, so none of them can be a bare name in `reward_funcs:`. They follow the same factory-in-module pattern as the judge and string-match rewards; `my_rewards.py` is where this repo builds them:

```python
# dna_factory/rewards/my_rewards.py
from trl.rewards import (
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
)

cosine_scaled_reward = get_cosine_scaled_reward(max_len=4096)  # needs `solution` + math_verify
repetition_penalty_reward = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
# soft_punish_cache = max_completion_length // 5
soft_overlong_penalty = get_soft_overlong_punishment(max_completion_len=4096, soft_punish_cache=819)
```

```yaml
reward_funcs:
  - dna_factory.rewards.soft_overlong_penalty
```

A length-derived argument (`max_completion_len`/`max_len`) is **not** auto-synced from `training_args`. The shipped instances are built for `4096`; the GRPO default `max_completion_length` is `16384`. If a recipe overrides that length, rebuild the matching instance by hand — `soft_overlong_penalty` is used by `configs/GRPO/qwen3-0.6B-rlvr-composed.yaml`.

## Composing multiple rewards: `reward_weights`

```yaml
reward_funcs:
  - accuracy_reward
  - think_format_reward
  - dna_factory.rewards.soft_overlong_penalty
reward_weights: [1.0, 0.2, 1.0]
```

Same order as `reward_funcs`. Each reward is evaluated per-sample, multiplied by its weight, and — under TRL's default `multi_objective_aggregation: sum_then_normalize` — summed into one scalar per sample *before* the group-relative advantage / `scale_rewards` step. Each component is still logged separately and un-weighted as `rewards/<func_name>/mean`, so individual components remain observable even though only their weighted sum drives the gradient. See `configs/GRPO/qwen3-0.6B-rlvr-composed.yaml` for a worked example.

## Writing your own

```python
def my_reward(prompts, completions, completion_ids, log_metric=None, **kwargs) -> list[float | None]:
    ...
```

- Return `None` to exclude a sample from this reward (not `0.0`).
- Give each reward function used together a distinct `__name__` — that's the key TRL logs `rewards/<name>/mean` under.
- `async def` works too; `GRPOTrainer` awaits all async rewards for a batch concurrently.
- Put configured instances in `my_rewards.py` (or your own module) and list the dotted path in `reward_funcs:`.

## Troubleshooting a flat run

Both symptoms below show up as `loss: 0` / `grad_norm: 0`: GRPO's advantage is the reward's deviation from its own group mean, so a group whose rewards are all identical produces no gradient at all. Watch `rewards/<name>/std`, `reward_std` and `frac_reward_zero_std`.

**`All reward functions returned None` warnings.** Expected, not a bug: a reward returns `None` to exclude a sample rather than score it `0.0`. `accuracy_reward` / `safe_accuracy_reward` do this whenever `math_verify` cannot parse the gold `solution` — about 16% of `trl-lib/DeepMath-103K`, whose answers include plain `Yes` / `No` / `True` / `False`. The default `reward_funcs` pairs it with `boxed_match_reward`, which falls back to a normalized string comparison and scores those rows. The warning gets loud when few prompts per step (`generation_batch_size / num_generations`) make each skip a large fraction of the batch.

**`completions/clipped_ratio: 1` with every reward `0`.** Every completion hit `max_completion_length` before it produced an answer, so nothing verifiable was ever emitted. A thinking model needs room to finish — hence the `16384` default. `mask_truncated_completions: true` (also a default) keeps the still-truncated ones out of the loss instead of training on them as genuine wrong answers.

`soft_overlong_penalty` / `cosine_scaled_reward` hard-code their length bound (4096), so changing `max_completion_length` means rebuilding those instances.
