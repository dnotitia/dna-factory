# CLAUDE.md

Guidance for Claude Code working in this repo.

## Setup

```bash
$ uv sync
$ CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install causal-conv1d --no-build-isolation --no-cache-dir --verbose  # Optional
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

CLI args override YAML. Multi-node flags: see README.md.

## Layout

- `sft.py` / `dpo.py` / `grpo.py` / `distill.py` — entry points. Same flow; DPO also loads `ref_model`; GRPO is online RL (generates completions and scores them with `reward_funcs`; no `ref_model` unless `beta != 0`); `distill.py` is on-policy distillation (student generates, a frozen `teacher_model` scores every token — no rewards, no `ref_model`).
- `dna_factory/` — custom trainers (colored token debug output) and utils (config merging, output-dir naming, colored logging). `rewards/` holds the GRPO reward framework (`generative.py` = judge, `verifiable.py` = string-match) and the concrete instances (`my_rewards.py`).
- `configs/` — `_defaults-{SFT,DPO,GRPO,Distill}.yaml` (base, **do not edit**) merged with model-specific overrides in `configs/{SFT,DPO,GRPO,Distill}/`.

## Gotchas

- `output_dir: auto` auto-generates a name from non-default args.
- Default attn is `sdpa` (`kernels-community/vllm-flash-attn3` is faster but has no B200/sm100 kernels yet); default models are from the `dnotitia/` HF org.
- Preprocessing copies each message's `thinking` → `reasoning_content` for Qwen3's chat template (SFT/DPO only; GRPO skips it — GRPO datasets are prompt-only).
- DeepSpeed only (no FSDP).
- GRPO rewards: `reward_funcs:` takes `trl.rewards` names or dotted import paths; a reward that returns `None` excludes that sample instead of scoring it 0 — the mechanism for composing rewards over a mixture, keyed by the per-dataset `label`. `accuracy_reward` needs `math-verify` and a `solution` column. See docs/grpo-rewards.md.
- Distillation: prompt-only datasets (a conversational dataset's last assistant turn is dropped — the student writes the completion). The teacher must share the student's vocabulary or TRL raises. `beta` selects the divergence (1.0 = reverse KL, 0.0 = forward KL, 0.5 = JSD), **not** GRPO's reference-model KL penalty. See docs/distillation.md.
