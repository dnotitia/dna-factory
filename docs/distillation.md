# On-Policy Distillation

Guide for `distill.py`. For the method itself, see Thinking Machines' [On-Policy
Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) and the
[On-Policy Distillation of Language Models](https://huggingface.co/papers/2306.13649) paper the
`DistillationTrainer` implements.

## What it does

Two models, one dataset of prompts:

1. **The student samples its own completions** (on-policy). Off-policy distillation — plain SFT on teacher
   outputs — only ever shows the student states the *teacher* visits, so at inference the student
   compounds errors from states it was never trained on (exposure bias). Sampling from the student removes
   that mismatch.
2. **The teacher grades every token of those completions.** Each token gets its own supervision signal.
   RL, by contrast, collapses a whole trajectory into one scalar reward — which is why distillation buys
   an order of magnitude in compute efficiency over RL for the same behavior.
3. **The loss is the per-token reverse KL**, `KL(student ‖ teacher)`. It is *mode-seeking*: the student
   commits to one teacher behavior instead of blurring several together. It is also unhackable — low KL
   can only mean the student is reproducing teacher behavior.

Where this fits next to the other trainers in this repo:

| | supervision | trajectories | signal density |
|---|---|---|---|
| `sft.py` | fixed reference answers | off-policy (dataset) | per token |
| `dpo.py` | preference pairs | off-policy (dataset) | per sequence |
| `grpo.py` | reward functions | on-policy (student) | per sequence (scalar reward) |
| `distill.py` | a teacher model | on-policy (student) | **per token** |

Typical uses: compressing a large teacher into a small student; recovering instruction-following after
domain mid-training (distill from the *pre-mid-training* checkpoint on chat prompts); and re-teaching a
capability cheaply, since prompts can be reused over many epochs without the memorization RL suffers from.

## Running it

```bash
# Single GPU (vLLM colocate is on by default; it shares the training GPU with the teacher,
# so headroom is set via `vllm_gpu_memory_utilization: 0.25`)
$ python distill.py \
  --config configs/Distill/qwen3-0.6B-distill.yaml

# Without vLLM (slower generation through transformers)
$ python distill.py \
  --config configs/Distill/qwen3-0.6B-distill.yaml \
  --use_vllm false

# Multiple GPUs
$ accelerate launch --config_file accelerate_configs/zero3.yaml \
    --num_processes 4 \
    distill.py \
    --config configs/Distill/qwen3-0.6B-distill.yaml

# A larger pair: 1.7B student <- 4B teacher
$ python distill.py \
  --config configs/Distill/qwen3-1.7B-distill.yaml
```

Dedicating separate GPUs to generation works the same way as GRPO:

```bash
# Terminal 1: vLLM server on a dedicated GPU (serves the *student*, which is what generates)
$ CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model dnotitia/Qwen3-0.6B

# Terminal 2: training on the remaining GPUs
$ CUDA_VISIBLE_DEVICES=1 python distill.py \
  --config configs/Distill/qwen3-0.6B-distill.yaml \
  --vllm_mode server
```

## Configuration

### Teacher

```yaml
model_name_or_path: dnotitia/Qwen3-0.6B          # student — the model being trained
teacher_model_name_or_path: dnotitia/Qwen3-1.7B  # teacher — frozen, provides the supervision
# teacher_model_revision: v1.0                   # optional: pin the teacher's branch/tag/commit
```

**The teacher must share the student's vocabulary.** The loss compares full next-token distributions, so a
`vocab_size` mismatch is a hard error, not a warning. Stay inside one model family (e.g. any two Qwen3
checkpoints) — a cross-tokenizer pair needs a different method entirely.

The teacher runs a forward pass on every training step and stays resident in GPU memory, so budget for
`student (trained, with optimizer states) + teacher (inference only) + vLLM cache`. Under DeepSpeed the
teacher is prepared with ZeRO-3 inference sharding automatically.

### `beta` — which divergence

```yaml
beta: 1.0   # reverse KL, KL(student || teacher)  <- on-policy distillation (default)
# beta: 0.0   forward KL, KL(teacher || student)  <- mass-covering
# beta: 0.5   Jensen-Shannon divergence
```

> **Not GRPO's `beta`.** In `grpo.py`, `beta` is the coefficient of a KL *penalty* against a reference
> model. Here it interpolates the loss function itself, and there is no reference model.

Leave it at `1.0` unless you specifically want the student to cover all of the teacher's modes.

### Generation

`temperature`, `top_p`, `top_k`, `min_p` and `repetition_penalty` control the student's sampling, and
`temperature` additionally softens the distributions the divergence is computed over. Keep
`temperature: 1.0` to train on the student's true policy; lowering it narrows the states you visit.

`max_completion_length` is the main lever on step cost. Unlike GRPO there is no `num_generations` and no
divisibility constraint — one completion per prompt — so batch sizing is ordinary.

### Loss path

`use_liger_kernel: false` (default) runs TRL's chunked divergence loss: the lm_head projection is done a
chunk at a time so the full `(batch, completion, vocab)` logits are never materialized, and it reports a
per-token `entropy` metric. `use_liger_kernel: true` swaps in Liger's fused JSD kernel — faster and
lighter, no entropy metric, and rejected for models with `logit_scale` / `final_logit_softcapping`
(Cohere, Gemma) because the fused kernel cannot apply them.

### LoRA

`--use_peft --lora_r 32 ...` works, with two restrictions the trainer enforces: `lm_head` must not be in
`target_modules` (the loss reads `lm_head.weight` directly, so an adapter there would never be trained),
and prompt-learning methods (PromptTuning / PrefixTuning / P-Tuning) are unsupported. Use plain LoRA.

## Dataset

Prompt-only, like GRPO:

```json
{"prompt": [{"role": "user", "content": "What color is the sky?"}]}
```

A plain-text `prompt` column works too. If the dataset is conversational with a `messages` column, the
loader keeps the turns *before* the last assistant turn and discards the reference answer — the student
writes the completion and the teacher grades it, so a reference answer is never used. Every other column
is dropped (there are no reward functions to forward them to).

Mixtures support the same `weight:` size multiplier as `sft.py`:

```yaml
datasets:
  - path: dnotitia/some-chat-prompts
    weight: 2.0
  - path: dnotitia/some-domain-prompts
    weight: 1.0
```

## Reading the metrics

- **`loss`** — the mean per-token divergence, in nats. This *is* the objective; it should fall steadily.
  It is directly interpretable: `loss ≈ 0` means the student reproduces the teacher's distribution on its
  own rollouts.
- **`entropy`** — mean per-token student entropy (chunked loss path only). A collapse toward 0 means the
  student stopped exploring; it usually accompanies too high a learning rate.
- **`log_completions: true`** writes sampled (prompt, completion) pairs under `<output_dir>/completions`
  every `logging_steps` — the cheapest way to watch the student drift toward the teacher's style.

The first `debug_first_n_batches` samples are also dumped token-by-token in color: cyan for tokens the
loss is computed on, dark gray for masked/padding tokens.

## Tuning notes

- Start from `learning_rate: 1e-6` (the RL-scale default). Distillation's dense signal tolerates more than
  RL does, but a too-large step shows up immediately as collapsing `entropy`.
- Prompts can be reused across epochs far more safely than in RL — the per-token teacher signal keeps
  supervising even on a prompt the student has already seen.
- If the student is already close to the teacher, the loss starts low; judge progress by the *relative*
  drop, and confirm on held-out prompts rather than on the training curve alone.

To confirm on held-out prompts, `testcase/expr_distillation_reverse_kl.py` measures exactly the training
objective — the student samples its own completions, and the per-token reverse KL against the teacher is
averaged over them — on a fixed prompt set and seed. Run it on the base student and on the checkpoint:

```bash
$ python testcase/expr_distillation_reverse_kl.py \
    --student dnotitia/Qwen3-0.6B --teacher dnotitia/Qwen3-1.7B
$ python testcase/expr_distillation_reverse_kl.py \
    --student ./my-distilled-checkpoint --tokenizer dnotitia/Qwen3-1.7B \
    --teacher dnotitia/Qwen3-1.7B
```
