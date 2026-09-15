
# DNA Factory

![](./assets/dna-factory.png)

*"It Just Works!"*

A lightweight, opinionated stack for LLM Post-Training on Hugging Face TRL + DeepSpeed.
Four trainers share one setup/train/save path: **SFT** on instruction or conversation data,
**DPO** on chosen vs rejected pairs, **GRPO** as online RL scored by reward functions,
and **On-Policy Distillation** where a frozen teacher grades the student's own tokens.

- [News](#news)
- [Design Principles](#design-principles)
- [Key Features](#key-features)
- [How to Run](#how-to-run)
  - [Advanced Usage](#advanced-usage)
  - [Multi-GPUs](#multi-gpus)
  - [Multi-Nodes](#multi-nodes)
- [Development](#development)
- [Acknowledgments](#acknowledgments)

# News

- Sep/02/2026 - **On-Policy Distillation** support.
- Jun/05/2026 - **GRPO (Group Relative Policy Optimization)** support.
- Oct/29/2025 - **DPO (Direct Preference Optimization)** support.
- Sep/21/2025 - **DNA Factory** is born! 🎉

<img width="80%" src="https://github.com/user-attachments/assets/f58514c2-004f-46cd-9545-0a9b69e85ecd" />

# Design Principles

- **One Way Approach** — one proven choice per decision (e.g. DeepSpeed over FSDP).
- **Lightweight Design** — no bloat beyond core functionality.
- **Clean and Readable Code** — every line understandable by any developer.

# Key Features

1. **Color-coded logging** across huggingface_hub, datasets, tokenizers, transformers, torch, accelerate, and trl.
1. **Auto-generated output directory** from the model name and CLI args (e.g. `Qwen3-0.6B-SFT-num_train_epochs-2-...`).
1. **Commented default YAMLs** (`configs/_defaults-*.yaml`), overridable by custom YAML or CLI args.
1. **Ready-made accelerate configs** for DDP and DeepSpeed ZeRO-1/3 (+ CPU offload).
1. **Flexible checkpoint interval** — save on a wall-clock schedule (e.g. `6h`) instead of a step count.
1. **Weighted dataset mixtures** via a per-dataset `weight:` (upsample, downsample, or drop) instead of listing the same path N times.
1. **Composable GRPO rewards** — TRL builtins or dotted paths (judge / string-match / shaping); returning `None` skips a sample so several rewards can share one labeled mixture.
1. **GRPO dynamic sampling** (`off` / `mask` / `resample`) — drop or refill zero-advantage groups so they don't waste a backward pass.

# How to Run

```bash
$ uv sync
$ source .venv/bin/activate

# SFT — supervised fine-tuning on labeled instruction/response (or conversation) data.
$ python sft.py
```

Beyond SFT, DPO, GRPO, and On-Policy Distillation are also available:

```bash
# DPO — preference learning on chosen vs rejected responses.
$ python dpo.py

# GRPO — online RL: generate completions and score them with reward functions.
$ python grpo.py

# On-Policy Distillation — student completions scored token-wise by a frozen teacher.
$ python distill.py
```

## Advanced Usage

CLI options, custom YAML, or both (CLI wins):

```bash
$ python sft.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name dnotitia/Reasoning_R1_Kor_completion_25k_sharegpt_v1 \
  --num_train_epochs 2

$ python sft.py \
  --config configs/SFT/qwen3-0.6B-sft.yaml

$ python sft.py \
  --config configs/SFT/qwen3-0.6B-sft.yaml \
  --num_train_epochs 2
```

## Multi-GPUs

DDP for speed, DeepSpeed ZeRO to save memory (ZeRO-1 and ZeRO-3 are supported):

```bash
$ accelerate launch \
  --config_file accelerate_configs/multi_gpu.yaml \
  --num_processes 2 \
  sft.py

$ accelerate launch \
  --config_file accelerate_configs/zero1.yaml \
  --num_processes 2 \
  sft.py \
  --config configs/SFT/qwen3-0.6B-sft.yaml
```

## Multi-Nodes

See [multi-nodes.md](docs/multi-nodes.md) for the master/worker launch commands.

# Development

```bash
$ uv sync
$ source .venv/bin/activate

# Lint / format
$ ruff check .
$ ruff format .

# Tests
$ pytest testcase
```

Do not edit `configs/_defaults-*.yaml`; put model-specific overrides in `configs/{SFT,DPO,GRPO,Distill}/`.

# Acknowledgments

This project is made possible thanks to:

- HuggingFace TRL <https://github.com/huggingface/trl>

<img width="40%" src="./assets/nipaLogo.png" alt="NIPA 정보통신산업진흥원" />

- NIPA 오픈소스 지원 프로그램 <https://www.nipa.kr>


### check-GRPO-dataset.py

Validates prompt-only datasets used by `grpo.py` and `distill.py`.

```bash
python utils/check-GRPO-dataset.py --dataset_name <name_or_path> [--config <name>] [--split <split>] [--require-solution]
```

The dataset must expose either a `prompt` column (string or conversational) or a
`messages` column whose last turn has `role == "user"`. With `--require-solution`,
the script also checks for the `solution` column used by `accuracy_reward`
(see `docs/grpo-rewards.md`).
