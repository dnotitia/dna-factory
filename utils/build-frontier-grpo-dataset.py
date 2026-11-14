"""
Build a GRPO dataset for the frontier curriculum (docs/grpo-frontier-curriculum.md).

Writes the three columns the curriculum needs: `prompt` (system + user message), `solution`
(`\\boxed{...}`, for the accuracy rewards) and `steps` (the teacher solution as a list of
reasoning steps). Two ways to get the steps:

1. The source already has them. The defaults read the public traces of "Unlocking the
   Unsolvable" (arXiv:2609.13997, Apache-2.0): 128 problems a small model fails at pass@64.

       python utils/build-frontier-grpo-dataset.py --output_dir data/uns128_frontier

2. The source has only questions and gold answers. `--teacher_model` asks a stronger model,
   through any OpenAI-compatible endpoint, for a step-by-step solution, and keeps a problem
   only when the teacher's own final answer verifies against the gold answer, with the same
   check the accuracy rewards use. A wrong trace would teach the wrong thing, so teacher
   failures are dropped, not kept as plain rows (`--keep_unverified` keeps them without steps).

       TEACHER_BASE_URL=https://api.example.com/v1 TEACHER_API_KEY=... \\
           python utils/build-frontier-grpo-dataset.py --dataset <hub id or dir> --config <config> \\
           --question_column <q> --answer_column <a> --teacher_model <model> --output_dir data/frontier

`--list_columns` prints the source's columns. A source without steps and without a teacher (an
eval split) is written with empty steps, i.e. as plain GRPO rows.

Traces only pay off on problems the student cannot already solve, and every teacher call costs
money, so filter the source to those problems first when you can.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Prompt of the student model in arXiv:2609.13997, Appendix G.2.
DEFAULT_SYSTEM = "You are an expert mathematician with strong problem-solving skills. Think step by step."
DEFAULT_SUFFIX = (
    "Please solve the problem step by step and put your final answer within \\boxed{}."
)
# Shorter answers ("2", "-1") occur in any worked solution by chance and say nothing about a leak.
MIN_LEAK_LENGTH = 3

# Same output contract as the teacher prompt of arXiv:2609.13997, Appendix G.1: discrete
# <step> tags inside one <solution> block, then a single <answer>.
TEACHER_SYSTEM = """You are an expert teacher preparing a worked solution for a student. Explain the solution so that it is clear, logical and easy to follow; the student is capable but will miss intermediate steps that are not written out.

Formatting rules:
- Your entire response must be one <solution>...</solution> block.
- Reason step by step. Wrap each logical reasoning step in its own <step>...</step> tag.
- Wrap only the final answer in a single <answer>...</answer> tag, with no explanation inside it.
- Output nothing outside these tags.

Example
User: What is the sum of the first 10 positive odd integers?
Assistant: <solution>
<step>The first 10 positive odd integers are 1, 3, 5, ..., 19, an arithmetic sequence with first term 1 and last term 19.</step>
<step>The sum of an arithmetic sequence is the number of terms times the average of the first and last term: 10 * (1 + 19) / 2.</step>
<step>That is 10 * 10 = 100.</step>
<answer>100</answer>
</solution>"""


def boxed(answer):
    answer = str(answer).strip()
    return answer if "\\boxed" in answer else f"\\boxed{{{answer}}}"


def as_step_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.splitlines()
    return [str(step).strip() for step in value if str(step).strip()]


def leaks_early(steps, answer):
    """Does a step before the last one already contain the gold answer?"""
    needle = str(answer).strip()
    if len(needle) < MIN_LEAK_LENGTH:
        return False
    return any(needle in step for step in steps[:-1])


def build_row(question, answer, steps, system, suffix):
    content = str(question).strip()
    if suffix:
        content = f"{content}\n\n{suffix}"
    prompt = [{"role": "user", "content": content}]
    if system:
        prompt.insert(0, {"role": "system", "content": system})
    return {"prompt": prompt, "solution": boxed(answer), "steps": as_step_list(steps)}


def parse_teacher_output(text):
    """(steps, answer) from a teacher response; ([], None) when it does not follow the format."""
    steps = [
        step.strip()
        for step in re.findall(r"<step>(.*?)</step>", text or "", flags=re.DOTALL)
    ]
    answers = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.DOTALL)
    if not steps or len(answers) != 1:
        return [], None
    return [step for step in steps if step], answers[0].strip()


def answer_verifies(teacher_answer, gold):
    """Is the teacher's final answer the gold answer, by the check the accuracy rewards use?"""
    from dna_factory.rewards import safe_accuracy_reward

    completion = [{"role": "assistant", "content": boxed(teacher_answer)}]
    score = safe_accuracy_reward(None, [completion], solution=[boxed(gold)])[0]
    if score is None:  # gold not parseable as math: fall back to the exact string
        return str(teacher_answer).strip() == str(gold).strip()
    return score >= 1.0


def ask_teacher(question, model, base_url, api_key, temperature, timeout=600):
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": TEACHER_SYSTEM},
                {"role": "user", "content": str(question).strip()},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def teach(question, gold, ask, attempts):
    """Verified teacher steps for one problem, or [] after `attempts` failed tries."""
    for _ in range(attempts):
        try:
            steps, answer = parse_teacher_output(ask(question))
        except Exception as error:  # noqa: BLE001 - one failed call must not lose the batch
            print(f"  teacher call failed: {type(error).__name__}: {error}")
            continue
        if steps and answer_verifies(answer, gold):
            return steps
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset",
        default="yukangzhu/unlocking-the-unsolvable",
        help="hub id or save_to_disk dir",
    )
    parser.add_argument("--config", default="uns128")
    parser.add_argument("--split", default="train")
    parser.add_argument("--question_column", default="question")
    parser.add_argument("--answer_column", default="ground_truth")
    parser.add_argument("--steps_column", default="steps")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM, help="'' for none")
    parser.add_argument(
        "--prompt_suffix",
        default=DEFAULT_SUFFIX,
        help="appended to every question; '' to keep the question as is",
    )
    parser.add_argument(
        "--teacher_model",
        default=None,
        help="generate the steps with this model at TEACHER_BASE_URL (key: TEACHER_API_KEY)",
    )
    parser.add_argument("--teacher_temperature", type=float, default=0.0)
    parser.add_argument("--teacher_attempts", type=int, default=2)
    parser.add_argument("--teacher_concurrency", type=int, default=8)
    parser.add_argument(
        "--keep_unverified",
        action="store_true",
        help="keep problems the teacher failed on, without steps (they train as plain GRPO)",
    )
    parser.add_argument("--limit", type=int, default=None, help="first N rows only")
    parser.add_argument("--list_columns", action="store_true")
    parser.add_argument("--output_dir", default=None, help="save_to_disk dir")
    parser.add_argument("--push_to_hub", default=None, help="hub repo id (private)")
    args = parser.parse_args()

    if os.path.isdir(args.dataset):
        source = load_from_disk(args.dataset)
        if isinstance(source, DatasetDict):
            source = source[args.split]
    else:
        source = load_dataset(args.dataset, name=args.config, split=args.split)

    if args.list_columns:
        print(f"{len(source)} rows, columns: {source.column_names}")
        return
    for column in (args.question_column, args.answer_column):
        if column not in source.column_names:
            raise SystemExit(
                f"column `{column}` not in the source (columns: {source.column_names})"
            )
    if args.limit:
        source = source.select(range(min(args.limit, len(source))))
    questions = source[args.question_column]
    answers = source[args.answer_column]

    if args.teacher_model:
        base_url = os.environ.get("TEACHER_BASE_URL", "http://localhost:8001/v1")
        api_key = os.environ.get("TEACHER_API_KEY", "EMPTY")

        def ask(question):
            return ask_teacher(
                question,
                args.teacher_model,
                base_url,
                api_key,
                args.teacher_temperature,
            )

        print(
            f"asking {args.teacher_model} at {base_url} for {len(questions)} solutions"
        )
        with ThreadPoolExecutor(max_workers=args.teacher_concurrency) as pool:
            all_steps = list(
                pool.map(
                    lambda pair: teach(pair[0], pair[1], ask, args.teacher_attempts),
                    zip(questions, answers, strict=True),
                )
            )
        verified = sum(bool(steps) for steps in all_steps)
        print(
            f"{verified}/{len(questions)} teacher solutions verified against the gold answer; "
            f"the rest are {'kept without steps' if args.keep_unverified else 'dropped'}."
        )
    elif args.steps_column in source.column_names:
        all_steps = source[args.steps_column]
    else:
        all_steps = [None] * len(questions)

    rows, early, counts = [], 0, []
    for question, answer, steps in zip(questions, answers, all_steps, strict=True):
        row = build_row(question, answer, steps, args.system_prompt, args.prompt_suffix)
        if args.teacher_model and not row["steps"] and not args.keep_unverified:
            continue
        counts.append(len(row["steps"]))
        early += leaks_early(row["steps"], answer)
        rows.append(row)

    with_steps = sum(count > 0 for count in counts)
    print(f"{len(rows)} rows; {with_steps} with teacher steps", end="")
    if with_steps:
        print(
            f" ({min(c for c in counts if c)}-{max(counts)} steps, mean "
            f"{sum(counts) / with_steps:.1f}); {early} state the gold answer before the last step."
        )
    else:
        print(": no steps, written as plain GRPO rows.")
    if not rows:
        raise SystemExit("nothing to write")

    dataset = Dataset.from_list(rows)
    if args.output_dir:
        DatasetDict({args.split: dataset}).save_to_disk(args.output_dir)
        print(f"saved to {args.output_dir}")
    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed to {args.push_to_hub}")
    if not args.output_dir and not args.push_to_hub:
        print("dry run: pass --output_dir or --push_to_hub to write the dataset.")


if __name__ == "__main__":
    main()
