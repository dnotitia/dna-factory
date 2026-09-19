# AsyncGRPO

AsyncGRPO runs GRPO with generation decoupled from optimization. The student model trains on one GPU while an external vLLM server generates completions on another. Completions are consumed as they become available rather than in lockstep with optimizer updates.

The scope is intentionally narrow: text-only prompts and completions, full fine-tuning, and a single training process.

The implementation was verified on 8x B200 (sm100) using the repository's pinned dependencies: TRL 1.13.0, vLLM 0.28.0, and transformers 5.17.0. A twelve-step run reached a reward of 1.08, `reward_std` of 0.95, and `grad_norm` up to 1.49. Checkpointing and resume were also verified by resuming from step 6 and training to completion. Per-step weight synchronization took approximately 0.07 s.

The sections below document the configuration required for that run, including several settings whose defaults can otherwise fail silently.

## Entry point and configuration

AsyncGRPO uses the existing `grpo.py` entry point. Enable it either in YAML or from the command line:

```yaml
grpo_execution: async
```

```bash
python grpo.py \
  --config configs/GRPO/qwen3-0.6B-async.yaml \
  --grpo_execution async
```

The canonical example is `configs/GRPO/qwen3-0.6B-async.yaml`. AsyncGRPO has a separate defaults file, `configs/_defaults-AsyncGRPO.yaml`, so asynchronous scheduling parameters cannot affect ordinary synchronous GRPO behavior.

As elsewhere in the repository, CLI arguments override YAML values. `grpo.py` remains synchronous unless `grpo_execution` is explicitly set to `async`.

## Supported topology and exclusions

```text
training process + model + optimizer  ->  one training GPU
vLLM server                           ->  one separate generation GPU
```

Select the two GPUs using separate `CUDA_VISIBLE_DEVICES` settings. The assignments must not overlap. The vLLM server is an external process, not a second training process, so placing it on the training GPU causes both processes to contend for the same memory.

AsyncGRPO does not support DeepSpeed, FSDP, multiple training processes, evaluation runs, dynamic sampling, quantized models, PEFT, or a reward model hosted on a local GPU. `validate_async_args` in `dna_factory/async_grpo.py` rejects each of these configurations before loading the model or dataset.

The repository's general preference for DeepSpeed over FSDP applies only to training modes that support distributed execution. It does not imply DeepSpeed support for AsyncGRPO.

## vLLM server

Start vLLM on the GPU not used for training:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve dnotitia/Qwen3-0.6B \
  --logprobs-mode processed_logprobs \
  --weight-transfer-config '{"backend":"nccl"}' \
  --dtype bfloat16 \
  --max-model-len 16384
```

`VLLM_SERVER_DEV_MODE=1` is required. TRL's vLLM client calls `/pause`, `/resume`, `/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update`, `/get_world_size`, and `/server_info`; these routes are registered only in dev mode.

If another process on the same GPU changes its memory allocation while vLLM is starting, memory profiling can fail with:

```text
AssertionError: Error in memory profiling
```

To avoid this, pin the KV cache size explicitly, which bypasses profiling:

```text
--kv-cache-memory-bytes 34359738368
```

If the GPU is shared, also reduce `--gpu-memory-utilization`, because vLLM checks available memory against this value before loading the model.

### Completion length must fit the server context

vLLM does not truncate requests whose requested output exceeds the available context. Instead, it returns HTTP 400:

```text
This model's maximum context length is 8192 tokens. However, you requested
8192 output tokens and your prompt contains 12 characters ...
```

TRL retries the request thirty times before the rollout-worker child exits. The resulting top-level error is only:

```text
RuntimeError: AsyncRolloutWorker child exited during init
```

with the underlying `ClientResponseError: 400, message='Bad Request'`. In one run, this consumed eleven minutes without completing a single training step.

Both of the following conditions therefore need to hold:

```text
--max-model-len  >=  longest prompt + max_completion_length
token_budget     >=  longest prompt + max_completion_length
```

`validate_completion_length_against_server` in `dna_factory/async_grpo.py` checks this at startup. It reads the server's `max_model_len` from `GET /v1/models`, using the same lookup as TRL's `VLLMClient.get_max_model_len`.

The validation:

- raises if `max_completion_length` is greater than or equal to `max_model_len`,
- warns if a non-null `token_budget` is smaller than `max_model_len`, and
- logs the remaining prompt headroom.

`setup_async_training_args` performs this check before loading the tokenizer, dataset, or model, turning an otherwise multi-minute failure into a startup-time error.

The server check is best-effort. Training is often launched before vLLM has finished loading, so the validation waits only through a few short retries. If the server is still unavailable, it logs that the check was skipped and continues, leaving TRL's own `wait_for_server_ready` to handle a server that never becomes ready.

## Token budget

`token_budget` limits the number of non-padding tokens packed into a single row of one forward pass. Short sequences can share a row, while long sequences occupy a row by themselves. This bounds peak memory independently of the number of samples consumed in a step.

If a sample itself exceeds the budget, it cannot fit into any row. TRL then drops the sample with a warning and continues training. Because this failure is easy to miss, `configs/_defaults-AsyncGRPO.yaml` sets:

```yaml
token_budget: null
```

When left null, TRL reads the vLLM server's `max_model_len` at training startup. This satisfies the required context invariant by construction and avoids hardcoding a value that becomes stale whenever the server is launched with a different `--max-model-len`.

A budget that is too small primarily hurts throughput rather than correctness. With `token_budget: 4096` and 2048-token completions, each row held only one sample. As a result, each step consumed four samples and covered only half of a generation group. Over twelve steps, only seven groups were trained.

After increasing the server context to 16384 and leaving `token_budget` null, approximately two samples fit per row, and the same twelve steps trained eleven groups.

`num_generations` still determines the groups used for reward normalization, but these groups may span optimizer updates. The batcher does not preserve generation-group boundaries, so per-step `reward_std` and `grad_norm` should not be expected to correspond one-to-one.

## Staleness

A completion may be generated using policy weights that predate the optimizer update that eventually consumes it. `RolloutQueueDataset` drops any sample whose staleness exceeds `max_staleness` and pulls the next available sample instead.

Monitor:

```text
sample/staleness_mean
sample/staleness_max
sample/dropped_stale_total
perf/rollout_wait_s
sample/rollout_queue_size
```

Discarded stale samples are an inherent cost of decoupling generation from optimization. Their frequency depends on the balance between generation speed and training consumption speed.

Two measured twelve-step runs produced:

| Setting | Stale discards | Share of scored samples |
| --- | ---: | ---: |
| 2048-token completions, `max_inflight_tasks: 16` | 22 | 31% |
| 8192-token completions, `max_inflight_tasks: -1` (32) | 67 | 46% |

`max_inflight_tasks: -1` resolves to:

```text
max_staleness
x per_device_train_batch_size
x gradient_accumulation_steps
x num_processes
```

This formula assumes that an in-flight sample will be consumed within the configured staleness window. Long completions can violate that assumption: generation gets ahead of consumption, and the surplus samples age out before training reaches them.

If stale discards become frequent, increase `max_staleness` rather than reducing concurrency. Lowering concurrency tends to leave the rollout queue emptier without addressing the underlying consumption lag.

## Zero-advantage groups

GRPO normalizes rewards within each group of `num_generations` samples. If every sample in a group receives the same reward, every token receives zero advantage and the group contributes no gradient.

Synchronous GRPO addresses this with `dynamic_sampling` using either `mask` or `resample`. AsyncGRPO currently rejects dynamic sampling.

On difficult datasets, the resulting wasted steps can be significant. In the twelve-step run described above, six steps had `grad_norm` equal to zero. Some of these came from groups in which every generation was incorrect, which is ordinary behavior for a 0.6B model on competition-level math problems. AsyncGRPO currently has no mechanism to recover those groups.

`completions/clipped_ratio` helps distinguish between two common causes of zero-gradient steps:

- A zero-gradient step with a high clipped ratio indicates that completions were truncated before reaching an answer.
- A zero-gradient step with a clipped ratio of zero indicates that the completions finished normally but all received the same, typically incorrect, reward.

## Checkpoints and resume

`_save_checkpoint` writes `rollout_state.json` alongside the model checkpoint. On resume, this file is used to reposition the rollout worker's dataset cursor.

Resume itself works correctly: loading `checkpoint-6` from a twelve-step run resumed at global step 6 and completed the remaining steps.

However, `prompt_index` in `rollout_state.json` is a low-water mark, not a count of completed prompts. TRL computes it as the smallest group index absent from the set of groups that have reached the model.

This interacts poorly with stale-sample discards. The stale-discard path does not record the group of the discarded sample. If every sample from a particular group is discarded, that group never enters the completed-group set. A missing low-numbered group can therefore pin `prompt_index` indefinitely even as training continues far beyond it.

This occurred in two runs:

- one froze at `prompt_index = 3` while the trained-group count increased from 4 to 7;
- another froze at `prompt_index = 4` while the trained-group count increased from 6 to 11.

The consequence is duplicated work rather than lost data. Every group below the low-water mark did reach the model, but resume may regenerate and retrain groups above that mark that were already processed by the previous run.

Treat `prompt_index` as a lower bound on training progress, not as a precise progress counter.

This behavior originates upstream in `trl/experimental/async_grpo/async_grpo_trainer.py`; it is not introduced by this repository.

After resuming, reconnect to the external vLLM server and transfer the resumed policy weights before requesting new generations.

## Upstream workarounds

`dna_factory/async_grpo.py` contains two overrides for values currently hardcoded by TRL that should instead be configurable. Each override documents the condition under which it can be removed.

### Attention implementation

`AsyncGRPOTrainer.__init__` currently passes:

```text
attn_implementation="kernels-community/flash-attn3"
```

directly when constructing the model and does not read the value from configuration.

That kernel ships SASS only for CUDA architectures 8.0 and 9.0a, so it cannot load on sm100:

```text
CUDA capability 10.0 of the current device is not supported by the
architectures of the build: 8.0, 9.0a
```

`apply_attn_implementation_override` wraps the model construction so that the configured attention implementation is used instead.

This repository defaults to:

```text
kernels-community/flash-attn2
```

matching the synchronous default in `AGENTS.md`.

`sdpa` and `eager` are rejected. AsyncGRPO uses padding-free training, concatenating sequences into a single row and deriving `cu_seq_lens` from resets in `position_ids`. Neither backend handles this representation correctly.

The override can be removed once TRL reads `attn_implementation` from `AsyncGRPOConfig` or `model_init_kwargs`.

### Response template

The rollout-worker child calls `add_response_schema` to parse completions into content, `<think>` reasoning, and `<tool_call>` blocks.

That function identifies supported chat templates using exact string equality against a fixed table. Any other template raises:

```text
ValueError("Unrecognized chat template, ...")
```

`dnotitia/Qwen3-0.6B` fails this comparison because its template wraps the assistant turn in:

```text
{% generation %}...{% endgeneration %}
```

These are the assistant-token-mask markers used by transformers. TRL's bundled `qwen3.jinja` does not contain them, even though both templates render identical text.

`apply_response_template_override` checks for this render equivalence at runtime. When the templates are equivalent, it assigns TRL's own `qwen3_template` to the tokenizer before `AsyncGRPOTrainer` pickles the tokenizer across the rollout child's spawn boundary.

TRL already skips `add_response_schema` when a response template is present, so the child no longer raises the template error.

Templates that the equivalence check does not recognize are left untouched and continue through TRL's normal error path.

This workaround can be removed once the model ships a template TRL recognizes or TRL switches from exact-string matching to structural template matching.

## Minimal launch shape

Run the generation server and training process in separate terminals with non-overlapping GPUs:

```bash
# Terminal 1: generation server
CUDA_VISIBLE_DEVICES=1 VLLM_SERVER_DEV_MODE=1 vllm serve dnotitia/Qwen3-0.6B \
  --logprobs-mode processed_logprobs \
  --weight-transfer-config '{"backend":"nccl"}' \
  --dtype bfloat16 \
  --max-model-len 16384

# Terminal 2: one-process training
CUDA_VISIBLE_DEVICES=0 python grpo.py \
  --config configs/GRPO/qwen3-0.6B-async.yaml \
  --grpo_execution async \
  --max_completion_length 8192
```