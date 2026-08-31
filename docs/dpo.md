# DPO

DPO training follows almost the same approach as SFT. You can train with a single GPU or multiple GPUs using the same accelerate configurations:

```bash
# Single GPU
$ python dpo.py \
  --config configs/DPO/qwen3-0.6B-dpo.yaml

# Single GPU - DDP
$ accelerate launch --config_file accelerate_configs/multi_gpu.yaml \
  --num_processes 1 \
  dpo.py \
  --config configs/DPO/qwen3-0.6B-dpo.yaml

# Multiple GPUs
$ accelerate launch --config_file accelerate_configs/zero3.yaml \
    --num_processes 4 \
    dpo.py \
    --config configs/DPO/qwen3-0.6B-dpo.yaml
```
