# gpt-oss

You can train the dequantized model on a 8xH100 2-node setup as shown below:

```bash
# Master
$ accelerate launch --config_file accelerate_configs/zero3_cpuoffload-gpt-oss.yaml \
    --num_machines 2 \
    --num_processes 16 \
    --main_process_ip 10.233.92.191 \
    --main_process_port 6000 \
    --machine_rank 0 \
    sft.py \
    --config configs/SFT/gpt-oss-120b.yaml

# Worker
$ accelerate launch --config_file accelerate_configs/zero3_cpuoffload-gpt-oss.yaml \
    --num_machines 2 \
    --num_processes 16 \
    --main_process_ip 10.233.92.191 \
    --main_process_port 6000 \
    --machine_rank 1 \
    sft.py \
    --config configs/SFT/gpt-oss-120b.yaml
```

After training is complete, you need to re-quantize the model using NVIDIA Model Optimizer.

LEGACY: If you want to build flash-attn directly instead of using the kernel, you can do so with the command below. However, this is not recommended as it requires a lengthy build process and may encounter compatibility issues.

```bash
$ uv pip install flash-attn --no-build-isolation --verbose
```
