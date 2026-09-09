# DPO

DPO trains a policy against chosen/rejected pairs with a frozen reference model
for the KL term, so it loads **two models** (`model` + `ref_model`) — budget ~2×
the memory of an equivalent SFT run.

Dataset: string `prompt`/`chosen`/`rejected` columns (e.g.
`trl-lib/ultrafeedback_binarized`) or conversational `messages`. Qwen `thinking`
fields are mapped to `reasoning_content` the same way as SFT.

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
