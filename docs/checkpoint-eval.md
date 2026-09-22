# Evaluating checkpoints during training

`periodic_save_seconds` turns wall-clock time into checkpoints. `eval_on_checkpoint` turns those checkpoints into a quality curve: every checkpoint written during training is served with vLLM on a spare GPU, evaluated with [Inspect](https://inspect.aisi.org.uk/), and its scores logged to the **same** W&B run — so the benchmark curves sit next to the loss curve instead of arriving after the run is over.

Training never waits. The eval runs in a background thread on rank 0 while the optimizer keeps stepping, and an eval that fails is a warning in the log, never a failed run.

Off by default. Nothing changes for an existing config until `eval_on_checkpoint: true` is set.

## Configuration

Everything but the switch already has a working default, so turning it on is the whole
setup — as long as GPUs 0 and 1 are not the ones you train on:

```bash
python sft.py --config configs/SFT/qwen3-4bexpr.yaml --eval_on_checkpoint true
```

These are the defaults it runs with (`configs/_defaults-*.yaml`):

```yaml
eval_on_checkpoint: false    # master switch — the only one you have to set
eval_tasks:                  # Inspect registry names, or local task files
  - inspect_evals/mmlu_pro
  - inspect_evals/gpqa_diamond
  - evals/kmmlu_pro.py
  - evals/kmmlu_redux.py
eval_devices: "0,1"          # CUDA_VISIBLE_DEVICES for the eval vLLM server
eval_vllm_args: "--max-model-len 32768 --gpu-memory-utilization 0.85 --data-parallel-size 2"
eval_max_connections: 20     # inspect eval --max-connections
eval_max_tokens: 16000       # inspect eval --max-tokens
```

The fields live in `DnotitiaArguments` next to `periodic_save_seconds`, so YAML and CLI override work as everywhere else:

```bash
python sft.py --config configs/SFT/qwen3-4bexpr.yaml \
  --eval_on_checkpoint true \
  --eval_devices 6,7
```

All four entry points (`sft.py`, `dpo.py`, `grpo.py`, `distill.py`) get this, because it is wired in the shared runner.

`report_to` must include `wandb` for the curves; without it the scores still appear in the training log, and startup warns about it.

## What actually runs

Per checkpoint, the callback reproduces by hand exactly what used to be done by hand:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve <output_dir>/_eval_staging/checkpoint-1776 \
  --served-model-name Qwen3-4B-SFT-checkpoint-1776 \
  --port 8000 \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --data-parallel-size 2

# then, once /health answers, once per task:
OPENAI_API_KEY=dna-factory-local \
inspect eval evals/kmmlu_pro.py \
  --model openai/Qwen3-4B-SFT-checkpoint-1776 \
  --model-base-url http://127.0.0.1:8000/v1 \
  -M responses_api=false \
  --max-connections 20 --max-tokens 16000 \
  --log-dir <output_dir>/eval_logs/step-1776/kmmlu_pro
```

One server for all tasks, torn down when the last one finishes.

- **Port.** A free port is picked by scanning up from 8000, so an AsyncGRPO run — whose own rollout server sits on vLLM's default port — gets 8001 instead of a collision. Pin it with `--port` inside `eval_vllm_args` if you need a fixed one.
- **Served model name.** `<run_name or base model>-checkpoint-<step>`. It is also the model name recorded inside the Inspect log, which is what ties a log back to the checkpoint that produced it.
- **Task files.** `evals/kmmlu_pro.py` is resolved against the repo root when the working directory doesn't have it, so the eval works regardless of where training was launched from.
- **Artifacts.** Inspect logs and the server's stdout are kept under `<output_dir>/eval_logs/step-<N>/`, browsable afterwards with `inspect view --log-dir ...`. A failed task's warning quotes the tail of the relevant file.

## W&B

Each task becomes one metric, named after the task:

```
eval/mmlu_pro   eval/gpqa_diamond   eval/kmmlu_pro   eval/kmmlu_redux
```

with the standard error under `eval_stderr/<task>` so the four `eval/*` panels stay clean.

They are logged against a dedicated `eval/step` x-axis, declared once with `define_metric`:

```python
run.define_metric("eval/step")
run.define_metric("eval/*", step_metric="eval/step")
run.log({"eval/step": 1776, "eval/kmmlu_pro": 0.61})
```

This is not cosmetic. An eval finishes minutes to hours after the checkpoint it scores, by which time the run's internal `_step` has moved well past it, and W&B silently drops a `log(step=...)` that goes backwards — the score would vanish. Logging the checkpoint's `global_step` as a *metric* and declaring it the x-axis puts the eval curves on the training step axis without either side waiting for the other.

## GPU placement

The eval server needs devices of its own. Training already owns its GPUs, and an eval server sharing them is an OOM waiting to happen — several hours in, which is the worst time to find out. So an overlap between `eval_devices` and the training process's `CUDA_VISIBLE_DEVICES` is a loud warning at startup, and an empty `eval_devices` with the switch on is a startup error.

The default is two GPUs, `0,1`, with `--data-parallel-size 2`: one full replica per GPU, round-robining the samples. That is the right shape for an eval, which is many independent requests rather than one large one — use `--tensor-parallel-size` instead when the model doesn't fit on a single GPU.

**`eval_devices` and the parallel sizes have to agree.** vLLM multiplies its parallel dimensions out and expects exactly that many visible devices, so dropping to one eval GPU means dropping `--data-parallel-size` too:

```yaml
eval_devices: "7"
eval_vllm_args: "--max-model-len 32768 --gpu-memory-utilization 0.85"
```

Changing one and not the other is warned about at startup, rather than left for vLLM to reject at the first checkpoint hours later.

`eval_devices` is read as physical device ids: it is set verbatim in the server's environment, independent of whatever `CUDA_VISIBLE_DEVICES` the trainer was launched with.

## Guarantees

**Rank 0 only, no collective.** Everything is gated on `state.is_world_process_zero`; other ranks return from `on_save` untouched. Unlike `PeriodicCheckpointCallback` — which has to broadcast its verdict so every rank saves on the same step — there is nothing to agree on here, and no NCCL op is posted.

**`save_total_limit` cannot delete the checkpoint being evaluated.** Rotation (default: keep 3) runs from inside the *next* `_save_checkpoint`, which on a run that saves faster than it evaluates lands while the eval is still reading. So `on_save` immediately snapshots the checkpoint into `<output_dir>/_eval_staging/checkpoint-<N>` — the weights, tokenizer and configs, not `optimizer.pt` and friends — and serves that. The snapshot is `os.link`, so it costs no disk and no copy time (same filesystem), and the inodes outlive the `rmtree` that rotation does. The staging directory is removed when the eval ends, and its name matches neither `checkpoint-*` (rotation, `get_last_checkpoint` resume) nor anything `push_to_hub` uploads.

If linking is refused — a staging directory on another filesystem — it falls back to a real copy of those same files and warns, since that one does cost a model's worth of disk and makes the save slower.

**One eval at a time.** A checkpoint that arrives while the previous eval is still running is skipped with a warning rather than queued; queueing would only push the backlog further behind. If you see that warning, raise `periodic_save_seconds` or trim `eval_tasks`.

**No orphan server.** vLLM is started in its own session and torn down with `os.killpg` on the normal path, on exception, at `on_train_end`, and from an `atexit` hook — which covers `Ctrl-C` and an unhandled crash. Only `SIGKILL` on the trainer itself can leak a server. `on_train_end` waits up to an hour for an in-flight eval to finish before abandoning it, so the final checkpoint's scores usually land before the script exits.

**Failure is never fatal.** A task that crashes, a server that never becomes healthy (15-minute startup budget), an unreadable Inspect log, a W&B call that raises — each is caught and logged, and the remaining tasks still run.

## ZeRO-3 checkpoints

`zero3.yaml` and the other DeepSpeed configs set `zero3_save_16bit_model: true`, so a periodic save writes consolidated 16-bit safetensors at the top level of the checkpoint directory. That is what vLLM loads. DeepSpeed's sharded state lives in `global_step*/` subdirectories, which the snapshot skips along with the optimizer files.

## Cost

Serving is a fresh model load plus CUDA graph capture per checkpoint — minutes before the first sample — and then the tasks themselves. `inspect_evals/mmlu_pro` is ~12k samples; at 6-hour checkpoints, four full tasks per checkpoint is comfortable, but check the first cycle's timings against your interval before assuming it. Cut `eval_tasks` down, or lengthen `periodic_save_seconds`, if the "still evaluating an earlier checkpoint" warning starts showing up.
