# unsloth backend (single GPU)

This guide covers `use_unsloth: true`. The option loads models through [unsloth](https://github.com/unslothai/unsloth)'s
`FastModel` and uses unsloth's patched versions of the TRL trainers in three scripts (`sft.py`, `grpo.py`, and
`opd.py`). It is a backend switch: the run follows the same configuration as with
`use_unsloth: false` (hyperparameters, model, which parameters train, LoRA rank and alpha, `max_length`, loss type),
and a setting unsloth cannot honour stops the run with an error or prints a warning instead of being changed
silently. How the work is computed does differ (kernels, packing groups, unsloth's own GRPO/OPD loss
code), so results match the standard path in what is trained, not bit for bit. See
[Known differences](#known-differences).

## Install and run

unsloth is an optional extra, pinned to `unsloth`/`unsloth-zoo` 2026.11.2:

```bash
$ uv sync --extra unsloth

$ python sft.py --config configs/SFT/qwen3-0.6B-sft-unsloth.yaml
$ python grpo.py --config grpo-unsloth.yaml  # saved from the example GRPO config below

# Or switch it on for any config
$ python opd.py --use_unsloth true --use_peft true --lora_dropout 0.0
```

unsloth officially caps `datasets`, `transformers`, and `trl` below this repo's pins. `pyproject.toml` keeps the
core pins through a `[tool.uv] override-dependencies` entry. As a result, `uv sync --extra unsloth` changes none
of the core pins. The `transformers` override also lifts vllm's cap for every sync, deliberately; see
`INTENTIONALLY_LIFTED` in `testcase/test_dependency_pins.py`. This combination was validated on GPU with all
three scripts on unsloth 2026.10.x. When you bump `datasets`, `transformers`, or `trl` in the core dependencies, bump the override too.

Write the flag out in full (`--use_unsloth true`, or `use_unsloth: true` in the YAML). The bootstrap reads it
before trl is imported and does not recognise abbreviated option names such as `--use_unsl`; the run then stops
with an error because the argument parser and the bootstrap disagree.

Run the script directly with `python`. Multi-process `accelerate launch` and DeepSpeed are rejected because
unsloth is single-GPU in this repository.

### Example GRPO config

Like the other [GRPO configuration variants](grpo.md#configuration-variants), the GRPO example is written out
here instead of being shipped as a file. Save this as `grpo-unsloth.yaml`:

```yaml
model_name_or_path: dnotitia/Qwen3-0.6B

# With `use_vllm: true` (the default), unsloth owns the colocated vLLM engine and shares the LoRA weights with
# it; `vllm_gpu_memory_utilization` sizes that engine. The engine is LoRA-only: full fine-tuning falls back to
# HF generate.
use_unsloth: true

use_peft: true
lora_r: 16
lora_alpha: 32
lora_dropout: 0.0  # unsloth's fast LoRA path needs 0 (TRL's default is 0.05)
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

num_generations: 8
per_device_train_batch_size: 8

# Short for math, as in qwen3-0.6B-grpo.yaml: many completions get truncated
max_completion_length: 4096
vllm_max_model_length: 8192
```

## Supported models

The switch is selected from the checkpoint's `config.json` `model_type`:

| `model_type` | Checkpoints | vLLM generation (GRPO / OPD) |
|---|---|---|
| `qwen3` | Qwen3 dense (`dnotitia/Qwen3-*`, `Qwen/Qwen3-*`) | unsloth's engine |
| `qwen3_moe` | Qwen3-30B-A3B, Qwen3-235B-A22B | HF generate (slower) |
| `qwen3_5` | Qwen3.5 dense, Qwen3.6-27B, Qwen3.8-27B | unsloth's engine |
| `qwen3_5_moe` | Qwen3.5-35B-A3B / 122B-A10B / 397B-A17B, Qwen3.6-35B-A3B | HF generate (slower) |

For any other model type, such as Qwen3.8-Flash-Next (`qwen4_exp`), the script logs a warning and uses the
standard path; unsloth is not imported. If the model's `config.json` cannot be read (a mistyped path, or the Hub
is unreachable with nothing cached), the run stops with an error instead of falling back. MoE models use HF
generate because unsloth's vLLM engine cannot serve LoRA on expert weights
([vllm-project/vllm#41754](https://github.com/vllm-project/vllm/issues/41754)).

## When it helps

These measurements use one B200 and this repo's defaults (FlashAttention-2 and Liger kernels for SFT):

| Workload (one B200) | `use_unsloth: false` | `use_unsloth: true` | Step time | Peak memory |
|---|---|---|---|---|
| SFT LoRA, Qwen3-8B, 4k, batch 2 | 0.82 s/step, 20.2 GiB | 0.55 s/step, 17.8 GiB | **33% faster** | **−12%** |
| SFT LoRA, Qwen3-8B, 16k | 10.25 s/step, 24.8 GiB | 7.92 s/step, 19.8 GiB | **23% faster** | **−20%** |
| SFT QLoRA (4-bit), Qwen3-8B, 4k, batch 2 | 0.73 s/step, 10.1 GiB | 0.50 s/step, 8.9 GiB | **32% faster** | **−12%** |
| SFT LoRA, Qwen3-30B-A3B (MoE), attention + experts, 2k | 0.44 s/step, 63.7 GiB ¹ | 0.33 s/step, 62.3 GiB | **25% faster** | −2% |
| SFT full fine-tuning, Qwen3-4B, 4k | 0.24 s/step, 31.0 GiB | 0.24 s/step, 30.9 GiB | same | same |
| GRPO LoRA + vLLM, Qwen3-4B, 8×1024 tokens | 4.14 s/step, 72.8 GiB ² | 7.14 s/step, 64.9 GiB ² | 72% slower | −11% |
| OPD LoRA + vLLM, Qwen3-0.6B ← 1.7B | 0.78 s/step, 49.5 GiB ² | 0.89 s/step, 48.2 GiB ² | 14% slower | −3% |

¹ The default config crashes on this model because Liger's MoE SwiGLU kernel and LoRA on expert weights do not
mix. The baseline therefore uses `use_liger_kernel: false`; the unsloth default works as is.
² Includes the colocated vLLM engine (`vllm_gpu_memory_utilization` 0.35 for GRPO, 0.25 for OPD).

Step times exclude the first three steps. Memory is peak allocated memory from single runs. All runs use `lora_r: 16`
on all linear layers (`lora_r: 8` for the MoE), `lora_dropout: 0.0`, and gradient checkpointing.

Why GRPO and OPD are slower: in both workloads, the two paths generated the same number of tokens.
unsloth's vLLM engine shares weights with the trained model and decodes with the LoRA adapter applied. The
standard path merges the adapter into the weights before each generation. Decoding is 32% slower for GRPO and 42%
slower for OPD, measured in tokens per second; this accounts for all of OPD's slowdown. For
GRPO it accounts for half of the slowdown (+1.4 s/step). The other half is training: the backward pass takes
1.5 s instead of 0.4 s.

Use unsloth for LoRA/QLoRA fine-tuning (SFT), MoE LoRA, and runs limited by memory. Full fine-tuning does
not benefit. For GRPO and OPD, it saves memory but is slower, so leave it off when throughput matters.

## What it changes, and what it keeps

**Import order.** unsloth patches `trl`/`transformers`/`peft` when imported, and the Dnotitia trainers subclass
TRL's trainers when they are defined. `import unsloth` must therefore run first. A small bootstrap at the top of
each entry script (`dna_factory/unsloth_bootstrap.py`) reads `use_unsloth` and the model's `model_type` from the
CLI/YAML and imports unsloth only when both select it. This is also why the switch works only when the script is
run directly.

**Model and LoRA.** The model is loaded by `FastModel.from_pretrained` with the requested repo, revision, dtype,
and attention implementation (`dna_factory/unsloth_integration.py`). unsloth's patched trainers ignore
`peft_config`, so LoRA is applied up front through `FastModel.get_peft_model` using the usual `lora_*` options.
When `lora_target_modules` is unset, PEFT's per-architecture default is used, as on the standard path (`q_proj`,
`v_proj` for Qwen3). Architectures without a PEFT default (Qwen3 MoE and Qwen3.5) use unsloth's default: every
linear layer in the language model, plus the MoE expert parameters. The script's tokenizer is used.

On `qwen3_moe` with both `gate_proj` and `up_proj` targeted (or `all-linear`), PEFT's checkpoint conversion gives the
fused expert `gate_up_proj` adapter rank `2*lora_r` and alpha `2*lora_alpha`; unsloth uses `lora_r` / `lora_alpha`
(same alpha/r scale, half the rank), and a warning is printed. For identical adapters on both paths, use
`lora_target_modules: [q_proj, k_proj, v_proj, o_proj]` with
`lora_target_parameters: [mlp.experts.gate_up_proj, mlp.experts.down_proj]`.

**Hyperparameters.** unsloth's patched TRL config classes rewrite some values during construction. For example,
GRPO with `loss_type: dapo` gets `epsilon_high: 0.28` and `top_k: -1`. The scripts re-parse the arguments with
TRL's own classes, restore every rewritten value, and print the changes:

```
[use_unsloth] restored 2 config value(s) rewritten by unsloth: top_k: -1 -> 0, epsilon_high: 0.28 -> None
```

The one value retained is `gradient_checkpointing_kwargs: {use_reentrant: true}`, which unsloth's gradient
checkpointing requires. `use_liger_kernel` is disabled with a warning because unsloth provides its own fused
kernels and Liger would patch the model a second time.

**SFT data.** unsloth's SFT trainer prepares datasets itself but does not support conversational `messages`
datasets. The integration tokenizes them first, exactly as TRL does, using the same chat template and assistant
masks. On `trl-lib/Capybara`, this was checked token for token: the sequences, tokens, and trained tokens match the
standard path. Rows that are not packed are truncated to `max_length` exactly as TRL does (`truncation_mode`, then
rows left with no trained token are dropped), including Qwen3.5-architecture models, which unsloth never packs.
Packed rows are chunked to `max_length` by packing itself. Two differences remain:

- Packing groups sequences slightly differently. unsloth packs the whole dataset rather than each
  `dataset_num_proc` shard: Capybara has 0.25% fewer packed rows, with the same number of trained tokens.
- unsloth does not pack Qwen3.5-architecture (hybrid linear-attention) models. The integration warns that these
  runs use unpacked rows and therefore have more steps per epoch.

**Checkpoints.** Saved checkpoints are ordinary PEFT adapters (LoRA) or Hugging Face checkpoints (full
fine-tuning) and load without unsloth. Qwen3.5-architecture checkpoints retain their
`*ForConditionalGeneration` architecture, as on the standard path.

## Known differences

The configuration is honoured on both paths; these are the ways the computation differs:

- **Numerics.** unsloth's fused kernels and gradient checkpointing replace the standard ones (Liger is off), so
  losses and weights track the standard path closely but not bit for bit.
- **SFT packing.** Packed rows are grouped differently (0.25% fewer on Capybara, same trained tokens), and
  Qwen3.5-architecture models are not packed at all (more steps per epoch). Data parity was checked for
  conversational `messages` datasets, the only format `sft.py` accepts.
- **GRPO and OPD loss.** The loss is computed by unsloth's own implementation of TRL's loss types
  (`unsloth_zoo`), not by TRL's code. It targets the same objective but was not compared numerically with TRL's.
- **Generation.** unsloth's vLLM engine decodes with the LoRA adapter applied instead of merging it into the
  weights. This is slower (see [When it helps](#when-it-helps)); rollouts were checked to match HF generate on the
  live model.
- **`qwen3_moe` gate/up adapter rank.** Half of PEFT's when `gate_proj` and `up_proj` are targeted; a warning is
  printed (see **Model and LoRA** above).

## Not supported with unsloth

- **AsyncGRPO** (`grpo_execution: async`): rejected with an error. It uses TRL's experimental
  `AsyncGRPOTrainer` and a separate `vllm serve` process, neither of which unsloth patches. Use
  `grpo_execution: sync` with unsloth. The error is raised even when the model type would otherwise make unsloth
  fall back to the standard path.
- **DeepSpeed / multi-process launches**: see above.

## Checkpoint evaluation

`eval_on_checkpoint: true` works unchanged. It reads only the checkpoints written by the trainer and serves them
with `vllm serve` on `eval_devices` in separate processes, so unsloth's engine and patches are not involved. The
checkpoints are ordinary PEFT/HF checkpoints (see above). Keep `eval_devices` off the GPU used for training, as
usual. As on the standard path, a LoRA run saves adapter weights only; merge the adapter first if the eval server
needs full weights.

## Per-trainer notes

- **SFT**: see above. Resuming from a checkpoint works as usual.
- **GRPO**: with `use_vllm: true`, unsloth's own vLLM engine is used (`fast_inference`). It is colocated, shares
  weights with the trained model, and is sized by `vllm_gpu_memory_utilization`. It requires LoRA; full
  fine-tuning falls back to HF generate with a warning, as do model types without engine support (see the table).
  `vllm_mode: server` is not supported. `dynamic_sampling` (`mask` / `resample`) works.
- **OPD**: only the student is loaded through unsloth; the teacher remains a plain Hugging Face model.
  Generation uses unsloth's engine as in GRPO, and the student's LoRA weights reach it before each generation
  (checked against HF generate on the live student).

## Errors you may hit

These combinations are rejected before the model is loaded rather than silently changed:

| Setting | Why | Fix |
|---|---|---|
| `lora_dropout > 0` on `qwen3_moe` / `qwen3_5_moe` | unsloth adapts MoE experts through PEFT's `ParamWrapper`, which has no dropout | `lora_dropout: 0.0` |
| GRPO on a MoE model with a non-zero `router_aux_loss_coef` (by default taken from the model config: `0.001` for Qwen3 MoE) | unsloth's GRPO step does not compute the router auxiliary loss | `router_aux_loss_coef: 0.0` |
| `use_dora: true` | unsloth has no DoRA kernels | `use_dora: false` |
| `load_in_4bit` / `load_in_8bit` without `use_peft` | quantized weights cannot be fully fine-tuned | `use_peft: true` |
| `model_revision` other than `main` with unsloth's vLLM engine | the engine always loads the default branch | drop the revision, or `use_vllm: false` |
| `vllm_mode: server` | unsloth manages the engine in-process | `vllm_mode: colocate` |

`lora_dropout` above 0 is otherwise allowed, but unsloth's fast LoRA path requires 0 (TRL's default is `0.05`),
so the example configs set `lora_dropout: 0.0`. unsloth also advises against 4-bit QLoRA on Qwen3.5-architecture
models; a warning is printed.

## Troubleshooting

- **`/tmp` mounted `noexec`** (as on our cluster): torch.compile caches kernels under
  `/tmp/torchinductor_<user>` by default, and compiled `.so` files cannot be loaded from there. unsloth then runs
  affected modules eagerly (`Unsloth: torch.compile could not generate code ...; running it eagerly`) or fails
  with `failed to map segment from shared object`. The bootstrap warns about this. Point the cache to an
  exec-capable filesystem:

  ```bash
  $ export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor  # any directory on an exec-capable filesystem
  ```
- The first steps are slow while unsloth compiles kernels. The cache makes later runs start faster.
- unsloth writes `unsloth_compiled_cache/` and, with its vLLM engine, `vllm_gen_lora_<gpu>/` into the working
  directory. Both are git-ignored.
- Two runs started at the same time on one machine can collide on `MASTER_PORT` (29500); set a different
  `MASTER_PORT` for each run.
