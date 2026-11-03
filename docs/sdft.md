# SDFT: Self-Distillation Fine-Tuning

Guide for `sdft.py`, built on TRL's experimental `SDFTTrainer` (`trl.experimental.sdft`). The method is
described in [Self-Distillation Enables Continual Learning](https://huggingface.co/papers/2601.19897)
(arXiv:2601.19897), the paper TRL cites for this trainer. The loss family is the same
generalized-divergence distillation that [distillation.md](distillation.md) covers, based on
[On-Policy Distillation of Language Models](https://huggingface.co/papers/2306.13649).

## What it does

There is one model and no external teacher. For each prompt:

1. **The student samples a completion** from the plain prompt (on-policy).
2. **The teacher is the same model, shown extra context**: the *privileged context*, such as a system
   prompt, a demonstration or a hint, that the student never sees. By default the teacher is a frozen copy of
   the initial student (`teacher_model_kind: base`).
3. **The student is trained to match the teacher's next-token distribution** over its own completion.

After training, the model produces without the context what it would have produced with it. Typical uses are
baking a system prompt into the weights (context distillation) and adding a behavior while staying close to
the original model, since the targets are the model's own distributions rather than external text.

When to use `sdft.py` rather than `distill.py`: `distill.py` needs a separate (usually larger) teacher;
`sdft.py` needs only the model and a context.

## Data format

Each training row has two fields:

| Column | Meaning |
|---|---|
| `prompt` | What the student sees: a list of chat messages, or a string. `messages` is accepted too, and its last assistant turn is dropped. |
| `privileged_context` | Text the teacher sees in addition. A string; a list of messages is flattened. |

The context can be a column in the data, or set per dataset in YAML. With a `datasets:` mixture, each entry
takes (besides the usual `path`, `split`, `columns`, `weight`):

```yaml
datasets:
  - path: my-org/my-prompts
    columns: [prompt]
    privileged_context: "You are a concise assistant."          # inline text, or
    # privileged_context_file: path/to/system_prompt.txt          # a file, for long prompts
```

A YAML value overrides a `privileged_context` column of the same dataset. With neither, the rows have an empty
context.

### Anchor rows

An **empty context leaves the teacher prompt identical to the student prompt** (under both placements below).
The teacher is then the plain frozen model, so the loss on those rows pulls the student toward its own initial
behavior. Mix a few such rows in from a general dataset to keep behavior on unrelated prompts from drifting while
the context rows teach the new behavior. `weight` sets the ratio between the two.

## Where the teacher sees the context

`teacher_context_placement` (a `DnotitiaArguments` field, default `user`):

- `user`: TRL's default. The context is appended to the last user turn via `teacher_prompt_template`
  (default `"{prompt}\n\n{privileged_context}"`).
- `system`: the context becomes a system message in front of the conversation, which matches how a deployed
  system prompt is seen. If the prompt already starts with a system message, the context is placed before its
  text in the same message. Needs conversational prompts.

`generate_from_teacher: true` makes the student's completions come from the teacher prompt instead of the
student prompt. It follows the same placement.

## Running it

```bash
# Single GPU (vLLM colocate shares the GPU with the frozen teacher copy)
$ python sdft.py --config configs/SDFT/qwen3-0.6B-sdft.yaml

# Bake a system prompt into the weights, with anchor rows from a general dataset
$ python sdft.py --config configs/SDFT/qwen3-0.6B-sdft-system-prompt.yaml

# Without vLLM (slower generation through transformers)
$ python sdft.py --config configs/SDFT/qwen3-0.6B-sdft.yaml --use_vllm false

# Multiple GPUs
$ accelerate launch --config_file accelerate_configs/zero3.yaml --num_processes 4 \
  sdft.py --config configs/SDFT/qwen3-0.6B-sdft-system-prompt.yaml
```

Key options (all documented in `configs/_defaults-SDFT.yaml`):

| Option | Meaning |
|---|---|
| `teacher_model_kind` | `base` (frozen initial model; with LoRA, the adapter is disabled for the teacher forward), `live` or `ema` |
| `distillation_mode` | `topk_logits` (default), `full_logits` (exact, more memory) or `sampled_token` |
| `distillation_alpha` | 0 = forward KL, 1 = reverse KL, 0.5 = JSD |
| `num_loss_tokens_to_skip` | Leave the first N completion tokens out of the loss |
| `num_generations` | Samples per prompt; `generation_batch_size` must be divisible by it |
| `teacher_context_placement` | `user` or `system`, see above |

LoRA works as in the other scripts (`use_peft: true`); then no second model copy is held. `use_unsloth` is not
supported.

## Multi-GPU notes

- Each rank handles its own micro-batch. If a rank's batch has no loss tokens (all completions empty or inside
  `num_loss_tokens_to_skip`), TRL 1.16.0 returns from the loss before the `accelerator.gather` that the other
  ranks run, and NCCL hangs until its watchdog timeout. `DnotitiaSDFTTrainer` makes such a rank join the gather
  with a zero first (see `_compute_self_distillation_loss`).
- The frozen teacher copy costs one extra model's memory per GPU (ZeRO-3 shards it). Lower
  `vllm_gpu_memory_utilization` if colocated vLLM runs out of memory.

## Metrics and debugging

`self_distillation/distillation_loss` is the mean per-token divergence; the `completions/*` metrics track
length and the clipped ratio. `debug_first_n_batches` prints the student prompt, the teacher prompt (with the
context in the chosen placement) and the completion for the first few batches, so you can check the context
landed where you expect.
