# Documentation

Guides for the training scripts and the features around them. Setup and the SFT quick start are in the
[top-level README](../README.md). Every default is documented inline in `configs/_defaults-*.yaml`.

## Training methods

| Guide | Covers |
|---|---|
| [dpo.md](dpo.md) | DPO: dataset formats, per-dataset `weight`, and single/multi-GPU launch commands |
| [grpo.md](grpo.md) | GRPO: launch commands and the configuration variants (no vLLM, vLLM server mode, DAPO, Dr. GRPO, weighted rewards, LLM judges) |
| [grpo-rewards.md](grpo-rewards.md) | GRPO reward functions: the reward contract, `reward_funcs` wiring, labeled dataset mixtures, judge, string-match and shaping rewards, and troubleshooting a flat run |
| [grpo-dynamic-sampling.md](grpo-dynamic-sampling.md) | `dynamic_sampling` (`off` / `mask` / `resample`) for prompt groups with zero advantage |
| [async-grpo.md](async-grpo.md) | AsyncGRPO: generation on a separate vLLM server, token budget, staleness, checkpoints and resume |
| [distillation.md](distillation.md) | On-policy distillation with `distill.py`: teacher setup, divergence (`beta`), generation, LoRA, dataset format, and metrics |

## Training infrastructure

| Guide | Covers |
|---|---|
| [unsloth.md](unsloth.md) | The optional single-GPU unsloth backend (`use_unsloth: true`): supported models, benchmarks, and differences from the standard path |
| [checkpoint-eval.md](checkpoint-eval.md) | `eval_on_checkpoint`: serving each checkpoint with vLLM, running Inspect tasks, and logging scores to the same W&B run |
| [multi-nodes.md](multi-nodes.md) | Two-node training with `accelerate launch` |

## Models

| Guide | Covers |
|---|---|
| [gpt-oss.md](gpt-oss.md) | SFT of the dequantized gpt-oss-120b on two 8×H100 nodes |
| [qwen38-flash-next.md](qwen38-flash-next.md) | LoRA SFT of Qwen3.8-Flash-Next on ZeRO-3: the host-memory PLE table (`ple_host_table`), the vectorized QSA indexer (`qsa_indexer_fast`), non-reentrant checkpointing, and throughput |
