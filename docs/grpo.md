# GRPO

Rewards are configured in YAML via `reward_funcs` (built-in names from `trl.rewards`, dotted import paths — including this repo's own judge and string-match rewards in `dna_factory.rewards`) and/or `reward_model_name_or_path`:

```bash
# Without vLLM (slower generation through transformers)
$ python grpo.py --use_vllm false

# Multiple GPUs
$ accelerate launch --config_file accelerate_configs/zero3.yaml \
    --num_processes 4 \
    grpo.py
```

If you prefer dedicating separate GPUs to generation, use vLLM server mode instead of colocate:

```bash
# Terminal 1: vLLM server on a dedicated GPU
$ CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model dnotitia/Qwen3-0.6B

# Terminal 2: training on the remaining GPUs
$ CUDA_VISIBLE_DEVICES=1 python grpo.py --vllm_mode server
```

Notes:
- The effective generation batch size (`per_device_train_batch_size` × number of processes × `steps_per_generation`) must be divisible by `num_generations`.
- A reward returning `None` excludes that sample instead of scoring it `0.0`, so `All reward functions returned None` warnings are normal. `max_completion_length` is `16384` in our defaults because a truncated completion never reaches an answer, which flattens the whole group to `grad_norm: 0`. Both are covered in [grpo-rewards.md](grpo-rewards.md#troubleshooting-a-flat-run).

Reward functions (built-in `trl.rewards`, LLM-as-judge, and string-match) are documented in [grpo-rewards.md](grpo-rewards.md).
