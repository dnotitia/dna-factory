- [DNA Factory](#dna-factory)
- [Key Features](#key-features)
- [Design Principles](#design-principles)
- [News](#news)
- [How to Run](#how-to-run)
  - [Advanced Usage](#advanced-usage)
  - [Multi-GPU](#multi-gpu)
  - [Multi-Node](#multi-node)
- [Supported backends](#supported-backends)

# DNA Factory

![](./assets/dna-factory.png)

*"It Just Works!"*

LLM Post-Training (SFT / DPO / GRPO / On-Policy Distillation) on HuggingFace TRL + DeepSpeed.

# Key Features

1. **Color-coded logging** across huggingface_hub, datasets, tokenizers, transformers, torch, accelerate, and trl.
1. **Auto-generated output directory** from the model name and CLI args (e.g. `Qwen3-0.6B-SFT-num_train_epochs-2-...`).
1. **Commented default YAMLs** (`configs/_defaults-*.yaml`), overridable by custom YAML or CLI args.
1. **Ready-made accelerate configs** for DDP and DeepSpeed ZeRO-1/3 (+ CPU offload).

<img width="80%" src="https://github.com/user-attachments/assets/f58514c2-004f-46cd-9545-0a9b69e85ecd" />

# Design Principles

- **One Way Approach** — one proven choice per decision (e.g. DeepSpeed over FSDP).
- **Lightweight Design** — no bloat beyond core functionality.
- **Clean and Readable Code** — every line understandable by any developer.

# News
- Sep/02/2026 - **On-Policy Distillation** support.
- Jun/05/2026 - **GRPO (Group Relative Policy Optimization)** support.
- Oct/29/2025 - **DPO (Direct Preference Optimization)** support.
- Sep/21/2025 - **DNA Factory** is born! 🎉

# How to Run

```bash
$ uv sync
$ source .venv/bin/activate
$ python sft.py
```

```bash
# DPO (same accelerate configs as SFT; see docs/dpo.md)
$ python dpo.py

# GRPO — online RL: completions are generated during training and scored by
# reward functions. Datasets are prompt-only; extra columns (e.g. `solution`)
# are forwarded to reward functions. See docs/grpo-rewards.md.
$ python grpo.py

# On-Policy Distillation — the student trains on its own completions, scored
# token by token by a frozen teacher (per-token reverse KL).
# See docs/distillation.md.
$ python distill.py
```

## Advanced Usage

CLI options, custom YAML, or both (CLI wins):

```bash
$ python sft.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name dnotitia/Reasoning_R1_Kor_completion_25k_sharegpt_v1 \
  --num_train_epochs 2

$ python sft.py --config configs/SFT/qwen3-0.6B-sft.yaml

$ python sft.py \
  --config configs/SFT/qwen3-0.6B-sft.yaml \
  --num_train_epochs 2
```

## Multi-GPU

DDP for speed, DeepSpeed ZeRO to save memory (only stages 1 and 3 are supported):

```bash
$ accelerate launch --config_file accelerate_configs/multi_gpu.yaml \
    --num_processes 2 \
    sft.py

$ accelerate launch --config_file accelerate_configs/zero1.yaml \
    --num_processes 2 \
    sft.py \
    --config configs/SFT/qwen3-0.6B-sft.yaml
```

MoE models must use the matching MoE config (`zero3-qwen3-moe.yaml`,
`zero3-qwen3_5-moe.yaml`), not plain `zero3.yaml`.

## Multi-Node

See [multi-nodes.md](docs/multi-nodes.md) for the master/worker launch commands.

# Supported backends
- HuggingFace TRL <https://github.com/huggingface/trl>
