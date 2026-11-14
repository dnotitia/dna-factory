"""Generate matched offline SFT / Sampling SFT / SDFT datasets on one GPU."""

import argparse
import json
import os
import random
from contextlib import ExitStack
from pathlib import Path

from dna_factory.projection_sampling import (
    PROPOSAL_TEMPLATE,
    VLLMProjectionBackend,
    comparison_rows,
    projection_sample,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dnotitia/Qwen3-0.6B")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-name", default="main")
    parser.add_argument(
        "--data-files",
        default=None,
        help="For local data: --dataset json --dataset-name ''",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--solution-column", default="answer")
    parser.add_argument("--output-dir", default="data/sampling-sft")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-num", type=int, default=16)
    parser.add_argument("--mcmc-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, help="Use the same first N rows for all three methods"
    )
    parser.add_argument(
        "--proposal-template-file",
        help="Template with {PROBLEM}, {SOLUTION}, {RESPONSE}",
    )
    args = parser.parse_args()
    if (
        args.max_new_tokens <= 0
        or args.block_num <= 0
        or args.max_new_tokens % args.block_num
    ):
        parser.error("--max-new-tokens must be positive and divisible by --block-num")
    if args.mcmc_steps < 0 or (args.limit is not None and args.limit <= 0):
        parser.error("--mcmc-steps must be nonnegative and --limit positive")
    template = PROPOSAL_TEMPLATE
    if args.proposal_template_file:
        template = Path(args.proposal_template_file).read_text(encoding="utf-8")
        if any(
            "{" + key + "}" not in template
            for key in ("PROBLEM", "SOLUTION", "RESPONSE")
        ):
            parser.error(
                "proposal template must contain {PROBLEM}, {SOLUTION}, {RESPONSE}"
            )
        template.format(PROBLEM="", SOLUTION="", RESPONSE="")

    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        parser.error(
            "output directory must be empty; use a new directory for each sampling run"
        )

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from datasets import load_dataset
    from vllm import LLM

    dataset = load_dataset(
        args.dataset,
        name=args.dataset_name or None,
        data_files=args.data_files,
        split=args.split,
    )
    for column in (args.question_column, args.solution_column):
        if column not in dataset.column_names:
            parser.error(f"Missing dataset column: {column}")
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if not len(dataset):
        parser.error("dataset is empty")
    for row in dataset:
        if any(
            not isinstance(row[c], str) or not row[c].strip()
            for c in (args.question_column, args.solution_column)
        ):
            parser.error("question and solution must be nonempty strings")

    llm = LLM(
        model=args.model,
        seed=args.seed,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    backend = VLLMProjectionBackend(
        llm,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        template=template,
    )
    rng = random.Random(args.seed)
    metadata = {
        **vars(args),
        "proposal_template": template,
        "rows": len(dataset),
        "complete": False,
    }
    manifest = destination / "manifest.json"
    manifest.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with ExitStack() as stack:
        files = {
            key: stack.enter_context(
                (destination / f"{key}.jsonl").open("x", encoding="utf-8")
            )
            for key in ("sft", "sampling_sft", "sdft", "sampling_metrics")
        }
        for index, row in enumerate(dataset):
            question, solution = row[args.question_column], row[args.solution_column]
            tokens, metrics = projection_sample(
                backend,
                question,
                solution,
                max_new_tokens=args.max_new_tokens,
                block_num=args.block_num,
                mcmc_steps=args.mcmc_steps,
                rng=rng,
            )
            response = backend.tokenizer.decode(tokens, skip_special_tokens=True)
            if not response.strip():
                raise ValueError(
                    f"Empty generated response at row {index}; sampling run incomplete"
                )
            records = comparison_rows(question, solution, response)
            records["sampling_metrics"] = {
                "index": index,
                "token_ids": tokens,
                **metrics,
            }
            for key, record in records.items():
                files[key].write(json.dumps(record, ensure_ascii=False) + "\n")
                files[key].flush()
            print(f"Sampled {index + 1}/{len(dataset)}", flush=True)
    metadata["complete"] = True
    manifest.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
