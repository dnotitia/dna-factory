# Sampling SFT (Finetuning with Sampling)

This implements the offline projection sampling procedure in the official
`aakaran/finetuning-with-sampling` repository (`boost_math.py` / `boost_sci.py`).
The reference checkout used for implementation was `../finetuning-with-sampling`.

For each question, the initial model generates a response in blocks, conditioned on
the reference solution and current partial response. Each refinement selects a random
cut and regenerates the suffix. Accept it only if **mean token log-probability under
the original question without the solution** strictly increases. Check EOS after each
block's refinements. This follows the upstream greedy acceptance rule, not a
Metropolis–Hastings probability. There is no correctness filter or reward model.

Train ordinary SFT on the resulting question/response pairs. `sampling_sft.py` reuses
the SFT trainer, model loader, preprocessing, shared runner and `_defaults-SFT.yaml`
directly: learning rate `2e-5`, cosine scheduler with minimum ratio `0.1`, warmup
`0.05`, two epochs, per-device batch one, accumulation eight, seed 42. No new loss
or online generation is introduced into the training loop. The original project's
learning rate `5e-5` and batch 16 are deliberately not adopted.

## Run a matched comparison

Run from the repository root after the normal installation. Sampling uses one CUDA
GPU and vLLM; finish it before starting training so its GPU memory is released.

```bash
# Optional smoke run: append --limit 8 and use a separate --output-dir.
python prepare_sampling_sft.py

python sft.py --config configs/SFT/qwen3-0.6B-sampling-comparison.yaml
python sampling_sft.py --config configs/SamplingSFT/qwen3-0.6B-sampling-comparison.yaml
python sdft.py --config configs/SDFT/qwen3-0.6B-sampling-comparison.yaml
```

Preparation defaults to GSM8K and `dnotitia/Qwen3-0.6B`. It writes the same ordered
questions to three files under `data/sampling-sft/`: reference-answer SFT,
sampled-answer SFT, and SDFT with the reference answer as privileged context.
All rows are kept. The manifest records generation settings and `complete: true`
only after all rows finish; do not train from an incomplete run. Existing output
directories must be empty, preventing accidental overwrite or duplicate rows.
There is no automatic resume; use a fresh output directory after a failed run.

Generation defaults: 512 response tokens, 16 blocks, 10 refinements per block,
temperature `0.6`, repetition penalty `1.1`, context window 8192. These are sampling
parameters, separate from optimization parameters. This can require hundreds of
generation/scoring calls per example. The generic proposal prompt follows upstream's
structure; use `--proposal-template-file` with `{PROBLEM}`, `{SOLUTION}`, `{RESPONSE}`
for domain wording. The chemistry reference uses repetition penalty `1.05`; these
defaults are not a reproduction of the paper's dataset-specific experiments.

For other data, use `--dataset`, `--dataset-name`, `--split`, `--question-column`
and `--solution-column`. Local JSONL example:

```bash
python prepare_sampling_sft.py --dataset json --dataset-name '' \
  --data-files my_data.jsonl --question-column question --solution-column answer
```

## What is controlled

The three comparison configs explicitly match the initial model, optimizer training
arguments, learning rate, scheduler, warmup, epochs, per-device batch, accumulation,
precision and seed. SFT and Sampling SFT use exactly the same training configuration.
Packing is disabled in these comparison configs so row counts per optimizer update
match; ordinary SFT defaults remain unchanged. Use the same GPU count for all runs.
SDFT uses one generation per prompt and a frozen base teacher, with reference context.

SFT uses a total sequence cap of 1024; SDFT has separate prompt/completion caps of
512/512. These are not identical truncation rules: check token lengths for your data.
Long reference answers can truncate differently from generated answers. SDFT's online
generation and distribution loss inherently differ from offline SFT, so equal epochs
do not imply equal trained tokens, wall time or compute. Prompt loss masking follows
ordinary SFT defaults; SDFT's loss is on completions. Liger remains an SFT optimization,
while SDFT uses its own trainer. Report these distinctions with comparison results.

Use identical checkpoint evaluation settings across runs to compare benchmark scores;
see [checkpoint-eval.md](checkpoint-eval.md). No GPU training result is implied by the
CPU algorithm tests.
