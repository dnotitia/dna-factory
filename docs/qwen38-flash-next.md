# Qwen3.8-Flash-Next (LoRA SFT)

Qwen3.8-Flash-Next (`qwen4_exp`, ~180B MoE: 512 experts / 10 active, 36 Gated DeltaNet + 12 full-attention
layers with a QSA sparse-attention indexer) requires a few specialized training changes. This page describes
those changes and how to run [configs/SFT/qwen3.8-flash-next-lora.yaml](../configs/SFT/qwen3.8-flash-next-lora.yaml).

## Requirements

- transformers with `qwen4_exp` (>= 5.16), attention `sdpa` (the model sets `_supports_flash_attn = False`).
- **`fla-core`** next to `flash-linear-attention`. Since fla 0.5 the kernels live in `fla-core`; without it,
  `import fla.ops` fails and transformers silently falls back to the pure-torch `chunk_gated_delta_rule`
  (fp32, far slower, and out of memory without gradient checkpointing). Install it with the same version
  and `--no-deps` so torch is left alone.
- `causal-conv1d` for the GDN short convolution.

## Changes

| Argument (`DnotitiaArguments`) | What it does | Why |
|---|---|---|
| `qsa_indexer_fast: true` | [dna_factory/qsa_indexer_fast.py](../dna_factory/qsa_indexer_fast.py) vectorizes the QSA indexer for causal (possibly right-padded) rows without cached indexer keys (training, and the prefill of generation); other batches, such as decoding steps, use upstream's loop | Upstream selects blocks in a Python loop over every query token (with a host-syncing `nonzero`): ~78% of a training step. Under ZeRO-3 that loop also deadlocks: it calls `k_layernorm` once per query, so ranks with different lengths issue different all-gathers; the fast path calls it once per row. The mask is identical to upstream's on real tokens (up to top-k ties); see [testcase/test_qsa_indexer_fast.py](../testcase/test_qsa_indexer_fast.py) |
| `ple_host_table: <dir>` | [dna_factory/ple_host_offload.py](../dna_factory/ple_host_offload.py) keeps the 95 GiB PLE n-gram embedding table in a host memmap shared by all processes and gathers rows on the CPU | As a parameter, ZeRO-3 gathers the whole table onto one GPU at load (OOM). The table is frozen under LoRA |

`DnotitiaSFTTrainer` also keeps an explicit `gradient_checkpointing_kwargs.use_reentrant: false`. TRL forces it
to `true` for PEFT + DeepSpeed ZeRO-3 (trl#2514), and with reentrant checkpointing DeepSpeed does not release
the weights it re-gathers for the backward recompute ([deepspeed#8481](https://github.com/deepspeedai/DeepSpeed/issues/8481)):
here the frozen MoE experts (4.7 GiB per layer once gathered) pile up from the last layer down and OOM the
first backward. Non-reentrant checkpointing has no such leak, and the recompute check that motivated TRL's
override passed on this stack.

Point `ple_host_table` at a directory in `/dev/shm` (RAM). If it contains no table, or one built from another
checkpoint, the first process on the node builds it from the checkpoint shards while the others wait on a lock
file; subsequent runs reuse it.

The table is read through a memmap with random row lookups, so it is only fast while its pages are in
memory. On a network filesystem, once other workloads push it out of the page cache, every step waits on
storage and throughput can drop substantially. `/dev/shm` is not subject to ordinary filesystem cache eviction
and uses ~95 GiB of host RAM per node, shared by all ranks. As a fallback,
the table is prefetched with `posix_fadvise(WILLNEED)` when first opened, but that cache can still be evicted.

Why not transformers' own placement: forcing the table to `"cpu"` with a `device_map` makes Accelerate attach
an offload hook that copies the whole 95 GiB table to the GPU on every forward (OOM), and keeps a 96 GiB copy
in every process; `_no_placement_params` only applies to `device_map` loading, not to ZeRO-3.

## Run: DeepSpeed ZeRO-3

```bash
accelerate launch --config_file accelerate_configs/zero3.yaml --num_processes 4 sft.py \
    --config configs/SFT/qwen3.8-flash-next-lora.yaml \
    --model_name_or_path path/to/Qwen3.8-Flash-Next \
    --ple_host_table /dev/shm/ple-host \
    --output_dir output/qwen3.8-flash-next-lora
```

ZeRO-3 all-gathers every weight once per micro-batch, so larger micro-batches amortize the communication:
the config uses `per_device_train_batch_size` 4. Keep the global batch at 128: `gradient_accumulation_steps`
8 on 4 GPUs, 4 on 8 GPUs.

## Representative throughput (transformers 5.16, LoRA r16, gradient checkpointing enabled)

| Setup | Throughput |
|---|---|
| Stock model code (QSA loop, torch GDN fallback) | ~110 tok/s |
| ZeRO-3, `fla-core` + `qsa_indexer_fast`, non-reentrant, micro-batch 1 x ga4 | 868 tok/s |
| ZeRO-3, same, **micro-batch 4 x ga1** | **4,279 tok/s** |

## Known limitations

- Packing is not supported: under `sdpa` the GDN layers do not receive sequence boundaries (`cu_seq_lens_q`),
  so packed sequences would leak state across each other.
- Left padding falls back to upstream's indexer loop, which deadlocks under ZeRO-3 (see above); TRL pads on
  the right.
