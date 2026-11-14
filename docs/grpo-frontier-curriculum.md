# GRPO frontier curriculum

A prompt none of whose `num_generations` rollouts is correct has advantage 0 on every row, so
GRPO learns nothing from it. [`dynamic_sampling`](grpo-dynamic-sampling.md) saves that group's
compute (`mask`) or replaces it with another prompt (`resample`); neither makes the problem
learnable. The frontier curriculum does: it shows the model the first steps of a verified
teacher solution as a hint, and withdraws them until the model solves the problem alone.

It implements the Monotone Frontier Curriculum (MFC) of "Unlocking the Unsolvable"
([arXiv:2609.13997](https://arxiv.org/abs/2609.13997)), Algorithm 1.

```yaml
frontier_curriculum: mfc                          # off | mfc
frontier_steps_column: steps
frontier_success_reward: strict_accuracy_reward   # must be one of reward_funcs
frontier_success_threshold: 0.5                   # tau
frontier_max_hint_ratio: 1.0                      # initial rho_max; 0 = control run
frontier_holdout_ratio: 0.1
```

Off by default. With `off` the trainer behaves exactly as before.

## The algorithm

Each problem `s` with a teacher trace of `n_steps` steps carries a frontier `g_curr`, the most
steps it may still be shown.

```
g_curr(s) <- n_steps(s)
on each visit to s:
    g_used ~ Uniform{0, 1, ..., g_curr(s)}
    roll out with the first g_used steps shown as a hint in the user message
    apply the GRPO update on the original problem's reward
    if mean verified reward >= tau and g_used < g_curr(s):
        g_curr(s) <- g_used
```

- The hint is auxiliary context in the prompt. The student writes a complete solution from
  scratch, so every token it is trained on is its own; the teacher's text is never a target.
- `g_curr` never rises. A failure changes nothing: with a policy that is still moving, a
  failure today says little about tomorrow, while a success proves less help is enough.
- The uniform draw puts `1 / (g_curr + 1)` of a problem's visits on the unguided prompt, and
  that share can only grow. This is the point of the method: a hinted prompt is not the prompt
  the model is evaluated on, so training has to be pulled towards the unguided one.
- At `g_curr = 0` the problem trains exactly like stock GRPO.

All rollouts of a prompt share one `g_used`. Otherwise the group advantage would compare a
rollout that saw half the solution against one that saw none.

Success is read from one named reward, `frontier_success_reward`, not from the advantage and
not from the weighted sum: a group in which every rollout is correct has advantage 0 and has
still solved the problem. Only full marks (1.0) count, so format, length and judge rewards
cannot move the frontier. A rollout that reward returns `None` for is left out of the mean.

The hint is appended to the last user message:

```
<the dataset's own prompt>

Below are some initial reasoning steps that may help you:
1. <step 1>
2. <step 2>

You should use the provided steps as a reference, but do NOT just copy them. Instead,
reconstruct the complete reasoning process in your own words, starting from the beginning, and
continue the reasoning to find the final answer.
```

`frontier_hint_template` replaces that wording; it must contain `{steps}`. The completion
table shows the hinted prompt and gets a `frontier_hint_steps` column (`-1`: the row has no
steps, `-2`: held out).

## What the paper reports

On Qwen3-1.7B and Qwen3-0.6B, trained on 128 problems the base model fails at pass@64, with
DeepSeek-V3.2 traces (8 rollouts, 1,200 steps, lr 1e-6, DAPO clip-higher):

| Qwen3-1.7B | 9-benchmark avg. pass@1 | Uns-22 (held-out unsolvable) |
| --- | --- | --- |
| base | 36.8 | 0.6 |
| GRPO on 2,000 mixed problems | 44.3 | 23.9 |
| SFT on the same 128 traces | 39.8 | n/a |
| AdaBack on the 128 | 45.8 | 29.5 |
| MFC on the 128 | 47.3 | 31.8 |

Those numbers are the paper's, from its own training stack on 8 GPUs for about 90 hours per
run. This implementation follows its algorithm, hint prompt and `tau`, but they have not been
reproduced here. The same paper shows plain GRPO leaves more than 75% of initially unsolvable
problems at zero successes after 1,200 steps, which is the gap this feature targets.

## Dataset

Rows need `prompt` (a list of messages), whatever the rewards need (`solution` for the
accuracy rewards), and `steps`: the teacher solution as a list of strings, or text with one
step per line. A row whose steps are empty or missing trains as stock GRPO, so a dataset with
traces can be mixed with one without. The steps column is removed from the rows before the
reward functions see them.

The paper's traces are public (Apache-2.0):

```bash
python utils/build-frontier-grpo-dataset.py --output_dir data/uns128_frontier
```

That writes the 128 problems as `prompt` / `solution` / `steps` and reports how many traces
state the gold answer before their last step (14 of 128).

For your own problems, which have a question and a gold answer but no trace, the same script
asks a stronger model for one through any OpenAI-compatible endpoint:

```bash
python utils/build-frontier-grpo-dataset.py --dataset <hub id or dir> --config <config> --list_columns
```

```bash
TEACHER_BASE_URL=https://api.example.com/v1 TEACHER_API_KEY=... python utils/build-frontier-grpo-dataset.py --dataset <hub id or dir> --config <config> --question_column <q> --answer_column <a> --teacher_model <model> --output_dir data/frontier
```

The teacher has to answer in `<step>` tags followed by one `<answer>` (the output contract of
the paper's teacher prompt). A problem is kept only when that answer verifies against the gold
answer with the same check the accuracy rewards use: a trace that ends in the wrong answer would
teach the wrong thing, so teacher failures are dropped (`--keep_unverified` keeps them without
steps, as plain GRPO rows). A source that already has a steps column is converted with
`--steps_column <s>` and no teacher.

Traces only pay off on problems the student cannot already solve, and each one costs a teacher
call, so narrow the source to those first. The paper keeps the problems its student fails at
pass@64 (about 7% of a 2,000-problem competition-math corpus). The method moves the teacher's
ability into the student; it does not exceed it.

`frontier_max_hint_ratio` below 1.0 keeps the last steps, which usually state the result, from
ever being shown. The paper starts at 1.0.

## Metrics

| metric | meaning |
| --- | --- |
| `frontier/heldout_solve_rate` | unguided solve rate on problems that are **never trained on**. The number to judge a run by. |
| `frontier/solve_rate_unguided` | solve rate on unguided visits to training problems |
| `frontier/solve_rate_guided` | solve rate on visits with a hint |
| `frontier/unguided_mass` | share of this batch's curriculum visits at `g_used = 0` (the paper's M₀) |
| `frontier/hint_ratio` | mean share of the trace shown in this batch (the paper's mean ρ used) |
| `frontier/unlocked_frac` | share of seen problems with `g_curr = 0` |
| `frontier/mean_help` | mean `g_curr / n_steps` over seen problems |
| `frontier/stalled_frac` | share of seen problems still needing help after 16 failed attempts with less |
| `frontier/unhelped_frac` | share of seen problems never solved in 16 visits, full trace included |
| `frontier/lowered` | frontiers lowered by this batch |

### Held-out problems

`frontier/unlocked_frac` and `frontier/solve_rate_unguided` rising is not the result. The
reward checks the final answer only and each problem is visited many times, so a model can
reach `g_curr = 0` by memorising answers. Nothing measured on the training problems can tell
that apart from learning to solve them.

`frontier_holdout_ratio` (default 0.1) therefore sets a share of the problems aside. They are
generated and scored without hints on every visit, their advantages are zeroed, and their
frontier is never touched. `frontier/heldout_solve_rate` is their solve rate: it can only rise
if what was learned on the other problems transfers. The held-out set is a function of the
prompt alone, so every run holds out the same problems. It costs their share of the generation
budget; `0` disables it. For a real result, also run a benchmark the run never saw
(`eval_on_checkpoint`, [checkpoint-eval.md](checkpoint-eval.md)); evaluation is never hinted.

### Stalled and unhelped problems

A problem counts as **stalled** when it still needs help and 16 attempts with fewer steps than
its frontier have failed since the frontier last moved. Usually the next step to withdraw is
the one the model cannot produce. RL has no signal there: the dead group this feature set out
to remove is back, one step further in. A problem is **unhelped** when it has never been solved
in 16 visits, the full trace included: the trace does not help this model.

Neither is fixed by more visits to the same prompt. The checkpoint's `frontier_state.json`
holds every problem's `frontier`, `steps`, `visits`, `wins` and `misses`, keyed by
`dna_factory.frontier_curriculum.problem_id(prompt)`, so those problems can be pulled out and
given a better trace or trained on with `sft.py` instead.

## Measuring whether it helps

`frontier_max_hint_ratio: 0` is the control run: no hint is ever shown, so training is stock
GRPO on the same problems, and `frontier/heldout_solve_rate` and
`frontier/solve_rate_unguided` are logged the same way. Run it and the curriculum from the same
model, dataset, seed and step count, and compare the held-out solve rate and a held-out
benchmark. If the curriculum does not beat the control there, it is not doing its job, whatever
`frontier/unlocked_frac` says.

The paper's two in-domain held-out sets are available as Inspect tasks, scored with the same
verifier and the same unguided prompt the run trains on:

```yaml
eval_on_checkpoint: true
eval_tasks:
  - evals/unsolvable_math.py@or1_200   # 200 held-out competition-math problems
  - evals/unsolvable_math.py@uns22     # 22 held-out problems a small model fails at pass@64
```

```bash
inspect eval evals/unsolvable_math.py@uns22 --model vllm/<checkpoint> --epochs 8 --epochs-reducer at_least_1
```

The second form is pass@8. `uns22` has 22 problems, so its pass@1 moves 4.5 points per problem;
read it over several seeds or as pass@k.

## Checkpoints and resume

Each checkpoint gets a `frontier_state.json`. A problem is identified by a hash of its
unhinted prompt, so the state survives reshuffling and a mixture `weight`. The step count drawn
for a visit is a hash of the seed, the problem and its visit number, so a resumed run continues
the same sequence. Resuming from a checkpoint without the file warns and restarts at full
hints.

## Multi-GPU

Each rank decides the hints of the groups it holds; after scoring, the ranks exchange what they
did and apply identical updates, so their states stay equal. Each rank's batch has to hold
whole rollout groups: `per_device_train_batch_size * steps_per_generation` must be a multiple
of `num_generations` (checked at startup; TRL only checks the global batch).

## Limits

Refused at startup:

- `dynamic_sampling: resample` (its refill generates groups outside the curriculum); `off` and
  `mask` are fine, and `mask` is useful here because unguided visits to unsolved problems and
  held-out rows are exactly the dead rows it skips;
- `use_unsloth: true`, `grpo_execution: async`, and `rollout_func` / tools / environments.

Prompts must be conversational with a text user message. The hint lengthens the prompt (the
paper allows 2,500 prompt tokens), so with vLLM raise `vllm_max_model_length` by the longest
trace.

Each problem has to be visited many times for its frontier to come down (the paper visits each
of its 128 problems 1,200 times), so a small set of hard problems needs many epochs.

## Verified here

- Unit tests of the algorithm and the trainer hooks (`testcase/test_frontier_curriculum.py`).
- End to end on the real `GRPOTrainer` with a tiny random model on CPU, single process and as a
  two-process distributed run: hints reach the prompt, held-out rows are never hinted or
  trained on, the ranks' states are identical, checkpoints carry the state.
- Not reproduced here: the paper's numbers. The implementation matches its Algorithm 1, hint
  prompt and `tau`; the training stack, optimiser settings and scale are this repository's.
