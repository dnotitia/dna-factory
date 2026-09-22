Guidance for Coding Assistant working in this repo.

## Setup

```bash
$ uv sync
$ source .venv/bin/activate
```

## Running Training

```bash
# SFT / DPO (single GPU)
python sft.py --config configs/SFT/qwen3-0.6B-sft.yaml
python dpo.py --config configs/DPO/qwen3-0.6B-dpo.yaml

# GRPO (single GPU); --use_vllm false for the slower HF generate path
python grpo.py --config configs/GRPO/qwen3-0.6B-grpo.yaml

# On-policy distillation (single GPU); teacher is set in YAML via teacher_model_name_or_path
python distill.py --config configs/Distill/qwen3-0.6B-distill.yaml

# Multi-GPU (see accelerate_configs/ for multi_gpu, zero1, zero3, zero3_cpuoffload)
accelerate launch --config_file accelerate_configs/zero3.yaml --num_processes 4 \
  sft.py --config configs/SFT/qwen3-0.6B-sft.yaml
```

CLI args override YAML. Multi-node flags: see docs/multi-nodes.md.

MoE models must launch with the matching MoE accelerate config
(`zero3-qwen3-moe.yaml`, `zero3-qwen3_5-moe.yaml`): it sets
`deepspeed_moe_layer_cls_names`, which is the only ZeRO-3 leaf handling
(the scripts contain no manual `set_z3_leaf_modules` call). Plain `zero3.yaml`
with a MoE model risks ZeRO-3 param-trace errors.

## Layout

- `sft.py` / `dpo.py` / `grpo.py` / `distill.py` — thin entry points. Each keeps only
  method-specific logic (model loading, dataset normalization, trainer wiring) as a
  `TrainingSpec`; the shared setup/train/save flow lives in
  `dna_factory/training_runner.py` (`run_training` + `cli_main`). Same flow across all
  four; DPO also loads `ref_model`; GRPO is online RL (generates completions and scores
  them with `reward_funcs`; no `ref_model` unless `beta != 0`); `distill.py` is on-policy
  distillation (student generates, a frozen `teacher_model` scores every token — no rewards,
  no `ref_model`).
- `dna_factory/` — custom trainers and utils (config merging, output-dir naming, colored
  logging). `rewards/` holds the GRPO reward framework (`generative.py` = judge,
  `verifiable.py` = string-match) and the concrete instances (`my_rewards.py`).
- `configs/` — `_defaults-{SFT,DPO,GRPO,Distill}.yaml` (base, **do not edit**) merged with
  model-specific overrides in `configs/{SFT,DPO,GRPO,Distill}/`.

## Gotchas

- `output_dir: auto` auto-generates a name from non-default args.
- Default attn is `kernels-community/flash-attn2` (required by `packing`/`padding_free`, and it has B200/sm100 kernels, unlike `kernels-community/vllm-flash-attn3`); default models are from the `dnotitia/` HF org.
- Preprocessing copies each message's `thinking` → `reasoning_content` for Qwen3's chat template (SFT/DPO only; GRPO/distillation datasets are prompt-only).
- DeepSpeed only (no FSDP).
- AsyncGRPO (`grpo.py` with `grpo_execution: async`) is the one exception to the distributed-training guidance: one training GPU, one separate vLLM GPU, one process, with DeepSpeed, FSDP, evaluation, dynamic sampling, PEFT and quantization all rejected. `dna_factory/async_grpo.py` overrides TRL's hardcoded flash-attn3 (no sm100 build) so the configured `attn_implementation` wins; `sdpa`/`eager` are rejected since training is padding-free. Leave `token_budget: null` and keep `max_completion_length` under the server's `--max-model-len`, which vLLM enforces with HTTP 400 rather than truncation. See docs/async-grpo.md.
- GRPO rewards: `reward_funcs:` takes `trl.rewards` names or dotted import paths; a reward that returns `None` excludes that sample instead of scoring it 0 — the mechanism for composing rewards over a mixture, keyed by the per-dataset `label`. `accuracy_reward` needs `math-verify` and a `solution` column. See docs/grpo-rewards.md.
- Checkpoint eval (`eval_on_checkpoint: true`, off by default, all four entry points):
  every checkpoint is served with `vllm serve` on `eval_devices` and scored with
  `inspect eval`, on rank 0 in a background thread — training never blocks and an eval
  failure is only a warning. `eval_devices` is required (startup error without it) since
  the eval server needs GPUs of its own. `on_save` hardlinks the checkpoint into
  `output_dir/_eval_staging/` first, so `save_total_limit` rotation can't delete it
  mid-eval. Scores go to the live W&B run as `eval/<task>` against an `eval/step` axis
  (`define_metric`), not `log(step=)` — that would be a backwards step and get dropped.
  See docs/checkpoint-eval.md.
- Distillation: prompt-only datasets (a conversational dataset's last assistant turn is dropped — the student writes the completion). The teacher must share the student's vocabulary or TRL raises. `beta` selects the divergence (1.0 = reverse KL, 0.0 = forward KL, 0.5 = JSD), **not** GRPO's reference-model KL penalty. See docs/distillation.md.
