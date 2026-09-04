# GRPO dynamic sampling

A prompt whose `num_generations` rollouts all score the same has advantage 0 on every row. It
contributes nothing to the loss, but TRL still runs its forward/backward (`is_std_zero` is
computed for logging only).

```yaml
dynamic_sampling: off          # off | mask | resample
dynamic_sampling_max_rounds: 2 # resample only
```

| mode | effect | compute | effective batch | gradient |
| --- | --- | --- | --- | --- |
| `off` | stock TRL | — | unchanged | — |
| `mask` | all-dead micro-batches truncated to 1 token | less | shrinks | unchanged, under the conditions below |
| `resample` | informative groups kept, batch refilled from extra generation rounds | more generation | preserved | **deliberately different** |

## Gradient

`resample` changes the gradient on purpose — replacing zero-advantage rows with informative ones
is the point of the mode, so a run with it on is not reproducible against one with it off.

`mask` leaves the gradient untouched, but only when a dead row really contributes nothing.
That holds when:

- `beta: 0` — otherwise the row carries a KL term that truncation drops;
- no entropy bonus and no router auxiliary loss — both put gradient on rows whose advantage is
  zero.

All three are detected at init and warned about; the run continues.

The dapo normalizer is `inputs["num_items_in_batch"]`, a scalar fixed when the batch is scored,
so truncating dead rows does not rescale the survivors. `resample` recomputes it for the
refilled batch.

Not supported with multimodal or token-type inputs (`pixel_values`, `token_type_ids` etc.),
whose second axis is neither the prompt nor the completion; raises at the first batch.
`resample` is not supported with streaming datasets and raises at init.

## `mask`

Applied in `_prepare_inputs` on the micro-batch TRL returns, only when every row in it is dead
— a rectangular tensor cannot be shortened for some rows and not others. With
`per_device_train_batch_size: 1` that is every dead row. Generation cost is unchanged; a group's
deadness is not known until its rollouts exist.

Deadness is decided per group when the batch is scored, not per row in the micro-batch: a row
sitting exactly at its group's mean has advantage 0 while its group is informative, and masking
it would drop a real gradient.

## `resample`

Informative **groups** are kept whole, a fresh batch is drawn from a dedicated sampler
(`repeat_count=1`, `seed+1`, so it neither repeats within a step nor replays the main stream),
generated and scored, and its informative groups appended, until the local batch is full or
`dynamic_sampling_max_rounds` is hit. Filtering by group rather than by row matters when the
reward has few distinct levels: a three-tier abstention group averaging exactly 0.5 gives its
0.5 rows an advantage of 0, and a per-row filter would discard them from an otherwise useful
group. When the local batch does not line up with group boundaries — detectable because
group-normalized advantages sum to zero — it falls back to per-row. On the cap the original batch is used as-is; its dead rows
are then handled by the `mask` path. The loop exit is an all-reduce, since
`_generate_and_score_completions` gathers internally and every rank must call it the same
number of times.

## Metrics

| metric | meaning |
| --- | --- |
| `dyn/dead_frac` | fraction of local rows in a dead group, after any refill |
| `dyn/truncated_microbatches` | micro-batches `mask` shortened |
| `dyn/gen_rounds` | generation rounds the step used, including the first |
| `dyn/refilled` | 1 when the refill filled the batch, 0 when it fell back |

`dyn/dead_frac` is not `frac_reward_zero_std`: TRL measures that before filtering, over prompts,
and gathered across ranks.

The refill logs a warning the first time it falls short; after that `dyn/refilled` is the
per-step record. A run where it sits at 0 has a prompt pool too uniform for `resample` to help.

## References

Dynamic sampling is from [DAPO](https://arxiv.org/abs/2503.14476) §3.2. TRL ships the other DAPO
components (`loss_type="dapo"`, `epsilon_high`, `mask_truncated_completions`) but not this one
([trl#3708](https://github.com/huggingface/trl/issues/3708)). Comparable implementations: verl
[`FilterGroupsConfig`](https://verl.readthedocs.io/en/latest/algo/dapo.html), ms-swift
`GRPOTrainer._dynamic_sampling`.
