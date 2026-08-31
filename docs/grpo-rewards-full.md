# GRPO Reward Functions

This is the full implementation guide for writing and wiring reward functions for `grpo.py`. `README.md`'s GRPO section and `CLAUDE.md` only summarize and link here — this is the one place that documents the contract in detail.

## Reward function taxonomy

- `reward_funcs` entries take one of two forms: a **bare builtin name** (zero-arg `trl.rewards`), or a **dotted path to a pre-built instance** in `my_rewards.py`. Every instance is built by a factory and grouped only by which factory made it — **judge** (`make_judge_reward`, generative.py) · **string-match** (`make_string_match_reward`, verifiable.py) · **shaping** (`trl.rewards.get_*`). Dnotitia authors the judge & string-match factories; shaping wraps TRL's own factories. All instances coexist in `my_rewards.py`.
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

reward_model_name_or_path (outside reward_funcs — loaded internally by GRPOTrainer as a sequence-classification model; scores every sample, no label routing — see the caveat in "Wiring" below)
```

Every dotted-path entry, whatever factory built it, shares the same reason it can't be a bare name: `resolve_reward_funcs()` resolves a dotted path with a bare `getattr` — no arguments — so anything needing configuration must already be a fully-built instance sitting in a module. See "Wiring" for the mechanism and "Dataset mixtures & labels" for how `label`-based routing lets several of these coexist over one heterogeneous dataset.

## Reward function contract

A reward function is any callable with this signature:

```python
def my_reward(prompts, completions, completion_ids, log_metric=None, **kwargs) -> list[float | None]:
    ...
```

- `GRPOTrainer` (`_calculate_rewards` in `trl/trainer/grpo_trainer.py`) calls every entry in `reward_funcs` with `prompts`, `completions`, `completion_ids`, plus **every other column present in the dataset** (everything except `prompt`, `completion`, `completion_ids`) forwarded as a keyword argument, one value per sample. A reward that doesn't care about an extra column can simply not declare it — the function just needs `**kwargs` to swallow the rest.
- Two more kwargs are always forwarded:
  - `trainer_state` — the trainer's live `transformers.TrainerState` (e.g. for reward shaping keyed on training progress; none of the rewards in this repo currently use it, but it's available).
  - `log_metric(name: str, value: float)` — logs a scalar that gets averaged over the logging window and reported alongside the built-in training metrics. This is a plain metric name with **no automatic `rewards/` prefix** — see the metrics section below for how this differs from the per-reward `rewards/<name>/mean|std` TRL logs automatically.
  - (A third, `log_extra(column, values)`, exists for logging extra per-sample columns to the completions table; not used by anything in this repo today.)
- `async def` reward functions are supported and detected automatically (`inspect.iscoroutinefunction`). All async reward functions for a batch run concurrently with each other via `asyncio.gather` on the trainer's own dedicated background event loop (`self.async_loop`, a daemon thread started at trainer init) — the main training thread blocks on `asyncio.run_coroutine_threadsafe(...).result()` until they all finish, but multiple async reward functions (and, inside `make_judge_reward`, multiple concurrent per-sample judge calls under one `asyncio.Semaphore`) overlap rather than running serially.
- Return a `list[float | None]`, one entry per sample, in the same order as `prompts`/`completions`.
- **Returning `None` for a sample means "this reward doesn't apply to this sample"** — TRL converts it to `NaN` and excludes it from that sample's reward computation (`nansum`/`nanmean`) instead of scoring it `0.0`. This is the mechanism for composing several task-specific rewards over one heterogeneous dataset mixture: each reward looks at its own samples and returns `None` for everything else, so you list one reward per task/label in `reward_funcs:` instead of writing a single reward with an internal `if/elif` router. See "Dataset mixtures & labels" below for how samples get tagged so a reward can decide which ones are "its own".

### Metrics

Two separate metric families exist, and it's easy to conflate them:

- `rewards/<__name__>/mean` and `rewards/<__name__>/std` — logged **automatically** by TRL from the list of floats a reward function returns (`GRPOTrainer._metrics`). Because this is keyed by the function's `__name__`, **two reward functions used together must have distinct `__name__`s** — this is exactly why `make_judge_reward`'s `name` argument exists (see below).
- Anything logged via the `log_metric` kwarg — a **plain name you choose**, not automatically prefixed with `rewards/`. `make_judge_reward` uses this to report judge health as `<name>/score_mean`, `<name>/parse_failure_rate`, `<name>/batch_latency_sec` (e.g. `persona_judge/score_mean`, not `rewards/persona_judge/score_mean`).

## Wiring: `reward_funcs` and `resolve_reward_funcs`

`reward_funcs:` in YAML accepts a mix of:
- built-in names resolved by `grpo.py`'s `resolve_reward_funcs()` registry — these must be zero-argument reward functions: `accuracy_reward`, `reasoning_accuracy_reward`, `think_format_reward`, or
- any dotted import path (e.g. `my_lib.rewards.custom_reward`), resolved relative to the current working directory via `importlib.import_module` + `getattr`.

`script_args.reward_model_name_or_path`, if set, is prepended as a plain model-id string — loaded internally by `GRPOTrainer` as a sequence-classification reward model, not through this registry. **Unlike every reward function above, this path has no per-sample opt-out**: `GRPOTrainer` (`_calculate_rewards`) scores every sample in the batch unconditionally (`reward_func(**reward_inputs).logits[:, 0]`) — it cannot return `None`, so it can't be routed to a subset of a dataset mixture via `label` the way a reward function can. Use a reward model only when it should score the entire batch.

**Important constraint:** for a dotted path, `resolve_reward_funcs()` does a bare `getattr(module, attr_name)` — it passes **no arguments**. Any reward that needs configuration (a rubric file, a label to filter on, a reference column, ...) must therefore already be a fully-configured module-level instance in your own module; you cannot point `reward_funcs:` at a factory function itself and expect it to be called with arguments.

Hence the pattern used throughout this repo: define configured instances in your own module, and point YAML at *those* dotted paths. Minimal example:

```python
# my_rewards.py
from dna_factory.rewards import make_judge_reward

persona_judge = make_judge_reward(
    "path/to/rubric.txt", only_label="persona", reference_column="expected_output", name="persona_judge"
)
```

```yaml
# your config.yaml
reward_funcs:
  - my_rewards.persona_judge
```

This applies equally to TRL's own reward factories, not just custom ones — see "Shaping rewards" below for the pattern applied to `get_soft_overlong_punishment` and friends.

## Dataset mixtures & labels

`grpo.py`'s `get_dataset_with_schema_alignment()` (used whenever the YAML has a top-level `datasets:` list instead of `dataset_name`) mechanically normalizes each dataset in the mixture via `_normalize_dataset_for_grpo()`, then concatenates them:

- a `messages` column is split into `prompt` (every turn before the last assistant turn, role/content only) and `expected_output` (that last assistant turn's content, or `None` if there is no assistant turn at all); `messages` itself is dropped.
- a `prompt` column with no `messages` is normalized in place (a plain string becomes a single user turn; a list is reduced to role/content only). Every other column already on the dataset (`solution`, a pre-existing `expected_output`, anything else) is left untouched.
- neither `messages` nor `prompt` present → a hard error (nothing to build a GRPO prompt from).
- every dataset then gets a `label` column added: each entry's per-dataset `label:` key in the `datasets:` list (`LabeledDatasetConfig`, a `DatasetConfig` subclass), defaulting to that entry's `path` if `label:` is not set.

There is no closed routing enum here — `label` is a plain, open string, and `datasets.concatenate_datasets` natively aligns the mismatched schemas across the mixture (a column missing from one dataset is filled with `None` for its rows), so adding a new dataset to a mixture never requires touching the loader. Reward functions decide what to do with `label` (and any other forwarded column) themselves — see `make_judge_reward`'s `only_label` below.

## The judge family (framework: `generative.py`; instances: `my_rewards.py`)

TRL has no first-class support for a *generative* reward model (the `reward_model_name_or_path` path only accepts a sequence-classification head via `.logits[:, 0]`), so a generative LLM-as-judge is integrated as a custom async reward function.

`make_judge_reward(rubric_file=None, only_label=None, reference_column=None, name=None)` builds one such judge (an `async def` function matching the contract above):

- **`only_label`** — if set, this judge only scores samples whose `label` column equals this value; every other sample gets `None` (excluded). Composing several judges over a mixture means one `make_judge_reward(only_label=...)` instance per label, all listed in `reward_funcs:` — not one function with an internal switch.
- **`reference_column`** — if set, this judge is reference-guided: the gold answer is read from `kwargs[reference_column]` (falling back to `"solution"` if `reference_column` is `None`). Whether the *loaded* rubric text is actually reference-guided is decided by inspecting it for a `{reference}` placeholder at call time, not by this flag directly — the flag only steers which packaged default template is used when `rubric_file` is `None`.
- **`rubric_file`** — pins the prompt template (`{prompt}`/`{completion}`, plus `{reference}` for a reference-guided rubric) to a specific file, read and cached lazily on the judge's first call (not at import time — the judge server doesn't need to be reachable until training actually starts). When `rubric_file` is `None`, a packaged default template is used (direct, or reference-guided when `reference_column` is set) — there is no env-var rubric override.

The packaged default rubrics live at `dna_factory/rewards/prompts/example_judge_rubric_default.md` (direct) and `dna_factory/rewards/prompts/example_judge_rubric_with_reference.md` (reference-guided) — loaded lazily via `Path(__file__).parent / "prompts" / ...`, not relative to the current working directory. Copy either file as a starting point for a custom rubric and point an instance's `rubric_file` at the copy.
- **`name`** — sets the returned function's `__name__` (defaults to `f"judge_{only_label}"` if `only_label` is set, else `"judge_reward"`). Required to be distinct across judges used together (see the metrics section above).

### `JUDGE_*` environment variables

Only the judge-server *connection* is configured through environment variables (dotted-path `reward_funcs` resolution passes no arguments — see "Wiring" above):

| Variable | Default | Meaning |
|---|---|---|
| `JUDGE_BASE_URL` | `http://localhost:8001/v1` | OpenAI-compatible judge endpoint |
| `JUDGE_MODEL` | *(unset → auto-detect)* | Judge model name; if unset, the first model listed by the server is used |
| `JUDGE_API_KEY` | `EMPTY` | API key (vLLM doesn't require one) |

Everything else is a fixed constant in `generative.py`, no longer env-configurable: max in-flight concurrency `16`, per-request timeout `120s`, `2` retries (exponential backoff 1s, 2s), `2048` max generated tokens (including the judge's own thinking), the 0-10 → pass/fail binarize threshold `7`, and the default reference column `"solution"`. A `make_judge_reward()` instance with no `rubric_file` uses the packaged default template (direct, or reference-guided when `reference_column` is set); there is no env-var rubric override.

### Score contract

The rubric always asks the judge for an integer **0-10** score, ending with a line in the exact format `Score: N`. `_parse_score()` strips the judge's own `<think>...</think>` block, regexes out `Score:\s*(\d{1,2})`, and then **binarizes**: `score >= 7` (a fixed threshold) → reward `1.0`, otherwise `0.0`. A coarse pass/fail signal is intentionally used instead of the raw 0-10 score, to be more robust against reward hacking and judge score-drift, while the 0-10 scale still gives the judge internal room to discriminate. **Any parse failure — an out-of-range score, a missing `Score:` line, or an API error after retries — returns `None`** for that sample (excluded from this reward, not scored `0.0`). The first unparseable judge output per process is logged with its `finish_reason` (a common cause: the judge's own thinking exhausts the `2048`-token cap before it reaches the score line).

Judge health is logged via `log_metric` (see the metrics section above) under `<name>/score_mean`, `<name>/parse_failure_rate`, and `<name>/batch_latency_sec` — e.g. `persona_judge/score_mean`, `judge_reward/parse_failure_rate`.

### Shipped instances

| Instance | Config | Notes |
|---|---|---|
| `judge_reward` | direct rubric, no reference | legacy name; `name="judge_reward"`, no `only_label`/`reference_column` |
| `judge_reward_with_reference` | reference-guided | `reference_column="solution"`; used by `configs/GRPO/qwen3-0.6B-judge.yaml` |
| `persona_judge` | `only_label="persona"`, `reference_column="expected_output"`, rubric `dna_factory/rewards/prompts/example_judge_rubric_with_reference.md` | used by `configs/GRPO/qwen3-0.6B-rlvr-mix3.yaml` |
| `ccp_judge` | `only_label="ccp"`, direct rubric `dna_factory/rewards/prompts/example_judge_rubric_safety.md` | used by `configs/GRPO/qwen3-0.6B-rlvr-mix3.yaml` |
| `rlvr_judge` | `only_label="rlvr"`, `reference_column="expected_output"`, rubric `dna_factory/rewards/prompts/example_judge_rubric_reasoning.md` | used by `configs/GRPO/qwen3-0.6B-rlvr-mix3.yaml` only |

## Built-in rewards (`trl.rewards`, bare-name)

Resolved directly by name in `resolve_reward_funcs()`'s registry — no configuration, no dotted path:

- **`accuracy_reward`** — needs the `math_verify` package and a `solution` dataset column (this repo's default dataset, `trl-lib/DeepMath-103K`, already has one).
- **`reasoning_accuracy_reward`** — also needs `math_verify`.
- **`think_format_reward`** — checks the completion is exactly `<think>...</think>` followed by an answer; binary `1.0`/`0.0`. Only meaningful for thinking-template models.

TRL also ships reward *factories* (`get_soft_overlong_punishment`, `get_cosine_scaled_reward`, `get_repetition_penalty_reward`) — these need construction arguments, so none of them are in this bare-name registry. See "Shaping rewards" below.

## Verifiable rewards (`dna_factory/rewards/verifiable.py`)

`make_string_match_reward(answer_column="solution", extractor="boxed", only_label=None, name=None)` builds a sync, no-dependency reward: extracts the final answer from the completion via `extractor`, extracts the same from `kwargs[answer_column]` (falling back to the raw gold string if extraction finds nothing there), and scores `1.0`/`0.0` on normalized string equality.

| Extractor | Extracts |
|---|---|
| `boxed` | last `\boxed{...}`, balanced-brace scan (nested braces work) |
| `gsm8k` | text after the last `####` |
| `last_number` | last numeric token (optional sign, optional decimal, commas allowed) |
| `full` | the whole completion, unchanged |

Normalization (both sides, before comparing): strip, collapse internal whitespace, lowercase, drop a trailing `.`, remove thousands-separator commas inside numbers, strip surrounding `$`/`\$`, strip surrounding `{}`.

`None` (sample excluded, not scored `0.0`) when: the gold value for that sample is missing/`None`/ empty; `only_label` is set and the sample's `label` doesn't match; or `only_label` is set but no dataset in the mixture has a `label` column (warned once, same `_warn_once` pattern as the judges).

```yaml
reward_funcs:
  - dna_factory.rewards.boxed_match_reward
```

Trades `accuracy_reward`'s semantic equivalence (`math_verify` treats `0.5` and `1/2` as equal) for speed, determinism, and zero dependencies — pick string-match when gold answers are already in a canonical, extractable form, `accuracy_reward` when they might not be.

## Shaping rewards (instances in `dna_factory/rewards/my_rewards.py`)

TRL's own reward *factories* — `get_soft_overlong_punishment`, `get_cosine_scaled_reward`, `get_repetition_penalty_reward`, etc. — all need construction arguments, so none of them can be a bare name in `reward_funcs:` (see "Wiring" above). They follow the exact same factory-in-user-module pattern as the judge and string-match rewards above; `dna_factory/rewards/my_rewards.py` is where this repo builds its `soft_overlong_penalty` instance of that pattern (plus `cosine_scaled_reward` and `repetition_penalty_reward`, below), alongside the judge and string-match instances:

```python
# dna_factory/rewards/my_rewards.py
from trl.rewards import get_soft_overlong_punishment

soft_overlong_penalty = get_soft_overlong_punishment(max_completion_len=256, soft_punish_cache=51)
```

```yaml
reward_funcs:
  - dna_factory.rewards.soft_overlong_penalty
```

The same pattern covers the other two factories in this repo's pinned `trl==1.7.0`, which are also built as instances in `my_rewards.py` (`cosine_scaled_reward`, `repetition_penalty_reward`):

```python
from trl.rewards import get_cosine_scaled_reward, get_repetition_penalty_reward

# cosine_scaled_reward needs a `solution` column + math_verify (correctness, like accuracy_reward).
cosine_scaled_reward = get_cosine_scaled_reward(max_len=256)
repetition_penalty_reward = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
```

A length-derived argument (`max_completion_len`/`max_len`) is not auto-synced from `training_args` — if you change a recipe's `max_completion_length`, update or replace the matching instance by hand.

## Composing multiple rewards: `reward_weights`

Multiple `reward_funcs` entries are combined via `GRPOConfig.reward_weights` (one float per entry, same order): each reward is evaluated per-sample, multiplied by its weight, and — under TRL's default `multi_objective_aggregation: sum_then_normalize` — summed into one scalar per sample *before* the group-relative advantage / `scale_rewards` step. Each component is still logged separately and un-weighted as `rewards/<func_name>/mean` (see the metrics section above), so individual components remain observable even though only their weighted sum drives the gradient. See `configs/GRPO/qwen3-0.6B-rlvr-composed.yaml` for a worked example (`accuracy_reward` + `think_format_reward` + `dna_factory.rewards.soft_overlong_penalty`, weights `[1.0, 0.2, 1.0]`).
