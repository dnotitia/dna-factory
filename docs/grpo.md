# GRPO

Rewards are configured in YAML via `reward_funcs` (built-in names from `trl.rewards`, dotted import paths — including this repo's own judge and string-match rewards in `dna_factory.rewards`) and/or `reward_model_name_or_path`. Dataset entries can combine reward-routing labels with per-dataset size multipliers:

```yaml
datasets:
  - path: org/math-prompts
    label: math
    weight: 2.0
  - path: org/general-prompts
    label: general
    weight: 0.5
```

```bash
# Single GPU
$ python grpo.py --config configs/GRPO/qwen3-0.6B-grpo.yaml

# Multiple GPUs
$ accelerate launch --config_file accelerate_configs/zero3.yaml \
    --num_processes 4 \
    grpo.py
```

Notes:
- The effective generation batch size (`per_device_train_batch_size` × number of processes × `steps_per_generation`) must be divisible by `num_generations`.
- A reward returning `None` excludes that sample instead of scoring it `0.0`, so `All reward functions returned None` warnings are normal. `max_completion_length` is `8192` in our defaults because a truncated completion never reaches an answer, which flattens the whole group to `grad_norm: 0`. Both are covered in [grpo-rewards.md](grpo-rewards.md#troubleshooting-a-flat-run).

Reward functions (built-in `trl.rewards`, LLM-as-judge, and string-match) are documented in [grpo-rewards.md](grpo-rewards.md).

## Configuration variants

`configs/GRPO/` keeps one small example, `qwen3-0.6B-grpo.yaml`: Qwen3-0.6B on the default math dataset, with a 4096-token completion cap that is meant for a quick look rather than real training. `qwen3-0.6B-async.yaml` is the example for [AsyncGRPO](async-grpo.md), which runs a different trainer with its own defaults file.

The variants below change only a few keys each, so they are described here instead of being shipped as separate files. Put the keys in your own config, or pass them on the command line on top of any config. Only one `--config` file is merged over `configs/_defaults-GRPO.yaml`, so a variant cannot be added as a second `--config`. A `datasets:` mixture can only be written in YAML.

| Variant | Keys | Use it to |
|---|---|---|
| [Without vLLM](#without-vllm) | `use_vllm: false` | debug, or run where vLLM is not installed |
| [vLLM server mode](#vllm-server-mode) | `vllm_mode: server` | generate on GPUs separate from training |
| [DAPO](#dapo) | `loss_type: dapo`, `epsilon_high` | train long reasoning with token-level loss and clip-higher |
| [Dr. GRPO](#dr-grpo) | `loss_type: dr_grpo` | remove GRPO's length and difficulty biases |
| [Weighted rewards](#weighted-rewards) | `reward_funcs`, `reward_weights` | combine correctness, format and length rewards |
| [LLM judge with a reference](#llm-judge-with-a-reference) | `reward_funcs` | grade answers that `math_verify` cannot parse |
| [Per-label judges on a mixture](#per-label-judges-on-a-mixture) | `datasets` with `label`, `reward_funcs` | score each dataset of a mixture with its own judge |
| [unsloth](unsloth.md#example-grpo-config) | `use_unsloth: true`, LoRA | LoRA GRPO on one GPU with less memory |

### Without vLLM

Generation goes through `transformers.generate()` instead of vLLM. It is much slower, so use it for debugging, CI, or a machine without vLLM installed.

```bash
$ python grpo.py --config configs/GRPO/qwen3-0.6B-grpo.yaml --use_vllm false
```

### vLLM server mode

By default vLLM runs colocated on the training GPUs (`vllm_mode: colocate`, `vllm_gpu_memory_utilization: 0.3`). Server mode moves generation to a separate `trl vllm-serve` process on its own GPUs, which leaves the training GPUs' memory to training. The server must serve the same model as `model_name_or_path`.

```yaml
use_vllm: true
vllm_mode: server
vllm_server_base_url: http://localhost:8000
```

```bash
# Terminal 1: vLLM server on a dedicated GPU
$ CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model dnotitia/Qwen3-0.6B --port 8000

# Terminal 2: training on the remaining GPUs
$ CUDA_VISIBLE_DEVICES=1 python grpo.py --config configs/GRPO/qwen3-0.6B-grpo.yaml --vllm_mode server
```

### DAPO

[DAPO](https://arxiv.org/abs/2503.14476) averages the loss over all tokens in the batch instead of over each sequence first. With per-sequence averaging (`loss_type: grpo`, the default), a long wrong completion gets a smaller penalty per token than a short one. DAPO also raises the upper clip bound ("clip-higher") so that low-probability tokens can still grow, which works against entropy collapse.

```yaml
loss_type: dapo
epsilon: 0.2        # lower clip bound (the default)
epsilon_high: 0.28  # upper clip bound
```

DAPO's overlong filtering, `mask_truncated_completions: true`, is already on in the defaults. For DAPO's soft overlong penalty, add `dna_factory.rewards.soft_overlong_penalty` to `reward_funcs`; it is built for a 4096-token cap (see [Shaping rewards](grpo-rewards.md#shaping-rewards-trlrewards-factories)).

```bash
$ python grpo.py --config configs/GRPO/qwen3-0.6B-grpo.yaml --loss_type dapo --epsilon_high 0.28
```

### Dr. GRPO

[Dr. GRPO](https://arxiv.org/abs/2503.20783) removes two biases of the original GRPO objective. Dividing each sequence's loss by its own length favours long wrong answers, and dividing advantages by the group's reward standard deviation gives extra weight to questions whose rollouts nearly all agree.

```yaml
loss_type: dr_grpo
scale_rewards: none  # already the default
```

```bash
$ python grpo.py --config configs/GRPO/qwen3-0.6B-grpo.yaml --loss_type dr_grpo
```

### Weighted rewards

Several rewards are summed into one score per sample with a weight each. `reward_weights` follows the order of `reward_funcs`, and each reward is still logged separately under `rewards/<name>/*`. How the sum feeds the advantage is described in [Composing multiple rewards](grpo-rewards.md#composing-multiple-rewards-reward_weights).

```yaml
reward_funcs:
  - accuracy_reward
  - think_format_reward
  - dna_factory.rewards.soft_overlong_penalty
reward_weights: [1.0, 0.2, 1.0]

max_completion_length: 4096  # soft_overlong_penalty is built for 4096; rebuild it for another cap
```

### LLM judge with a reference

An LLM judge grades each completion against the gold `solution` column, instead of `math_verify`. Use it when gold answers are free-form and a verifier cannot parse them. Serve a judge model first. The `JUDGE_*` variables and the score contract are described in [grpo-rewards.md](grpo-rewards.md#judge_-environment-variables).

```yaml
reward_funcs:
  - dna_factory.rewards.judge_reward_with_reference
```

```bash
# Terminal 1: judge server
$ CUDA_VISIBLE_DEVICES=1 vllm serve <judge-model> --port 8001

# Terminal 2: training
$ CUDA_VISIBLE_DEVICES=0 JUDGE_BASE_URL=http://localhost:8001/v1 python grpo.py \
    --config configs/GRPO/qwen3-0.6B-grpo.yaml \
    --reward_funcs dna_factory.rewards.judge_reward_with_reference
```

### Per-label judges on a mixture

Each dataset in a mixture gets a `label`, and each judge scores only the rows with its own label (`only_label`) and returns `None` for the others, so every row gets exactly one reward. The public datasets below stand in for your own data and rubrics. How labels, `expected_output` and `None` fit together is explained in [Dataset mixtures & labels](grpo-rewards.md#dataset-mixtures--labels).

```yaml
dataset_name: null  # use the mixture below instead of the default dataset
datasets:
  - path: trl-lib/Capybara              # label persona -> persona_judge (reference-guided)
    split: train
    label: persona
  - path: trl-lib/tldr                  # label ccp -> ccp_judge (direct)
    split: train
    label: ccp
  - path: HuggingFaceH4/ultrachat_200k  # label rlvr -> rlvr_judge (reference-guided)
    split: train_sft
    label: rlvr

reward_funcs:
  - dna_factory.rewards.persona_judge
  - dna_factory.rewards.ccp_judge
  - dna_factory.rewards.rlvr_judge
```

The `persona` and `rlvr` datasets need a `messages` column: the loader turns its last assistant turn into `expected_output`, which the two reference-guided judges grade against. Serve a judge and set `JUDGE_BASE_URL` as in the previous section. Each entry also accepts `weight:`, as shown at the top of this page.
