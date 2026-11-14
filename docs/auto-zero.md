# Automatic ZeRO selection (single node)

```bash
python launch.py --zero auto --num-processes 4 \
  sampling_sft.py --config configs/SamplingSFT/qwen3-0.6B-sampling-comparison.yaml
```

The launcher runs Accelerate with `zero1.yaml`. If training raises a CUDA OOM
before any rank completes its first optimizer step, all launch-process-group workers
are stopped and it retries once with `zero3.yaml`. Model loading, forward/backward
and the first optimizer update are therefore exercised with the actual configuration.
No memory estimate is used. ZeRO-1 avoids parameter sharding's communication cost;
it is a starting preference, not a guarantee of the fastest configuration.

Batch size, accumulation, learning rate, seed, data and all other training arguments
are forwarded unchanged. This supports the repository's SFT, Sampling SFT, SDFT,
DPO, synchronous GRPO and OPD entry points. Put launcher options **before** the script
name and training options **after** it. Run from the repository root.

```bash
# Explicit selection, with no automatic retry:
python launch.py --zero 3 --num-processes 4 sft.py --config my-training.yaml

# Print the candidate commands without starting training (may fetch model config.json):
python launch.py --zero auto --num-processes 4 --dry-run sft.py
```

`CUDA_VISIBLE_DEVICES` is honored by the launched processes; select training GPUs
there, keeping checkpoint-evaluation GPUs separate. `--main-process-port` defaults
to 29500 and can be changed for concurrent jobs. The launcher reads the model's
config.json (including the requested revision) to choose architecture-specific configs.
Qwen3 MoE and Qwen3.5 MoE use their dedicated ZeRO-3 configs immediately. Qwen3.8's
`qwen4_exp` uses plain configs; its existing PEFT requirement still applies.

## Limits and failure behavior

- Single node only. For multi-node launch, use the existing explicit Accelerate configs.
- Automatic selection is for fresh runs. Explicit or automatically detected checkpoint
  resume is rejected in auto mode: specify the checkpoint's original ZeRO stage.
- After any completed optimizer step, OOM stops the job without retry. Variable-length
  batches can still OOM later; fitting the first update does not guarantee the full run.
- Non-CUDA errors, CPU OOM, import-time failures, SIGKILL and NCCL failures do not
  trigger a fallback. ZeRO-3 OOM also stops; there is no automatic CPU offload.
- Unsloth and AsyncGRPO retain their existing launch paths and are rejected here.
- A failed first attempt may leave logs, a failed W&B run, cache files and baseline
  evaluation artifacts. They are not deleted; the second attempt is a new process.
  Auto-generated output paths follow the existing naming logic for each attempt.
- The implementation uses explicit error/progress signals from the shared training
  runner, not log-text matching. It does not switch ZeRO stage inside a live Trainer.

Accelerate itself accepts a numeric `zero_stage`; `auto` here is a DNA Factory
launcher policy, not a new value in Accelerate YAML. See the
[Accelerate CLI reference](https://huggingface.co/docs/accelerate/en/package_reference/cli).
