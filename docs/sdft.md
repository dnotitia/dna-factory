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

The context is either the same text on every row of a dataset (set in YAML) or a different text per row (a
column in the data). With a `datasets:` mixture, each entry takes, besides the usual `path`, `name`, `split`,
`data_files`, `columns` and `weight`:

| Key | Meaning |
|---|---|
| `privileged_context` | Same context on every row of this dataset (inline text). |
| `privileged_context_file` | Same, read from a text file (for long system prompts). |
| `privileged_context_column` | Column with a per-row context. Default `privileged_context`. |
| `prompt_column` | Column with the student prompt (string or messages). Default `prompt`, else `messages`. |

A YAML value (`privileged_context` / `privileged_context_file`) **replaces** the column on every row, so leave both
unset on an entry whose context should vary per row. With no YAML value and no context column, the rows have an
empty context (anchor rows). Only the columns listed in `columns:` are kept, so list the context column there too.

### Same context on every row (e.g. a system prompt)

```yaml
datasets:
  - path: my-org/my-prompts
    columns: [prompt]
    privileged_context: "You are a concise assistant."          # inline text, or
    # privileged_context_file: path/to/system_prompt.txt          # a file, for long prompts
```


### A different context on every row (e.g. demonstrations)

Put the context in a column. With the default column names, a local JSONL file looks like:

```jsonl
{"prompt": [{"role": "user", "content": "Write a function that returns the n-th Fibonacci number."}], "privileged_context": "Reference solution:\ndef fib(n): ..."}
{"prompt": "What is the capital of Australia?", "privileged_context": "Answer in one sentence."}
{"prompt": "Hi!", "privileged_context": ""}
```

```yaml
datasets:
  - path: json
    data_files: my_rows.jsonl
    split: train
    columns: [prompt, privileged_context]
```

A cell may be a string or a list of messages (their contents are joined); an empty or null cell makes that row an
anchor row. Datasets with other column names need no preprocessing, for example GSM8K's question and reference
solution:

```yaml
datasets:
  - path: openai/gsm8k
    name: main
    split: train
    columns: [question, answer]
    prompt_column: question
    privileged_context_column: answer
```

See `configs/SDFT/qwen3-0.6B-sdft.yaml`, which also sets a `teacher_prompt_template` that
introduces the demonstration.

### Anchor rows

An **empty context leaves the teacher prompt identical to the student prompt** (under both placements below).
The teacher is then the plain frozen model, so the loss on those rows pulls the student toward its own initial
behavior. Mix a few such rows in from a general dataset to keep behavior on unrelated prompts from drifting while
the context rows teach the new behavior. `weight` sets the ratio between the two.

## Where the teacher sees the context

`teacher_context_placement` (a `DnotitiaArguments` field, default `user`):

- `user`: TRL's default. The context is appended to the last user turn via `teacher_prompt_template`
  (TRL's default is `"{prompt}\n\n{privileged_context}"`; `_defaults-SDFT.yaml` uses a wording for its GSM8K
  reference solutions, so set your own for other data).
- `system`: the context becomes a system message in front of the conversation, which matches how a deployed
  system prompt is seen. If the prompt already starts with a system message, the context is placed before its
  text in the same message. Needs conversational prompts.

`generate_from_teacher: true` makes the student's completions come from the teacher prompt instead of the
student prompt. It follows the same placement.

## Running it

```bash
# Single GPU: GSM8K reference solutions as per-row demonstrations, the default
# (vLLM colocate shares the GPU with the frozen teacher copy)
$ python sdft.py

# Without vLLM (slower generation through transformers)
$ python sdft.py --use_vllm false

# Multiple GPUs
$ accelerate launch --config_file accelerate_configs/zero3.yaml --num_processes 4 sdft.py
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
  `num_loss_tokens_to_skip`), TRL 1.17.1 returns from the loss before the `accelerator.gather` that the other
  ranks run, and NCCL hangs until its watchdog timeout. `DnotitiaSDFTTrainer` makes such a rank join the gather
  with a zero first (see `_compute_self_distillation_loss`).
- The frozen teacher copy costs one extra model's memory per GPU (ZeRO-3 shards it). Lower
  `vllm_gpu_memory_utilization` if colocated vLLM runs out of memory.

## Metrics and debugging

`self_distillation/distillation_loss` is the mean per-token divergence; the `completions/*` metrics track
length and the clipped ratio. `debug_first_n_batches` prints the student prompt, the teacher prompt (with the
context in the chosen placement) and the completion for the first few batches, so you can check the context
landed where you expect.
