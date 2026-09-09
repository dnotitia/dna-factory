# Multi Nodes

Two-machine training over plain TCP: run the same command on both hosts with
different `--machine_rank`, pointing at the master's IP/port.

```bash
# Master
$ accelerate launch --config_file accelerate_configs/zero1.yaml \
    --num_machines 2 \
    --num_processes 16 \
    --main_process_ip 10.233.71.18 \
    --main_process_port 6000 \
    --machine_rank 0 \
    sft.py \
    --config configs/SFT/smollm3-sft.yaml

# Worker
$ accelerate launch --config_file accelerate_configs/zero1.yaml \
    --num_machines 2 \
    --num_processes 16 \
    --main_process_ip 10.233.71.18 \
    --main_process_port 6000 \
    --machine_rank 1 \
    sft.py \
    --config configs/SFT/smollm3-sft.yaml
```
