"""
Build a prompt-only GRPO dataset from the HAERAE-HUB/KMMLU train split (all 45 subjects).

Each row's prompt asks for an explanation of how the answer was reached, ending in a
`정답: $LETTER` line (`A) ...` options, as in evals/_multiple_choice_ko.py). Unlike the eval's
non-CoT template ("the whole response must be '정답: $LETTER'"), the answer section then carries the
reasoning too, so a correct rollout's advantage reinforces the explanation as well as the letter.
mcq_accuracy_reward parses that last line with the eval's own patterns.
Columns: `prompt` (one user message), `solution` (gold letter), `num_choices`, `category`.

Rows whose question also appears in LGAI-EXAONE/KMMLU-Redux (the eval set, drawn from the KMMLU
test split) are dropped, by whitespace-normalized question text.

    python utils/build-kmmlu-grpo-dataset.py --push_to_hub dnotitia/kmmlu_grpo_data  # private
    python utils/build-kmmlu-grpo-dataset.py --output_dir data/kmmlu_grpo  # local only
"""

import argparse
import re
from collections import Counter

from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

LETTERS = "ABCD"

EXPLAIN_TEMPLATE = """
다음 객관식 질문에 답하십시오. 왜 그렇게 정답을 도출하게 되었는지 충분히 설명해주시고 마지막에 '정답: $LETTER' (따옴표 없이)를 출력하세요. 여기서 LETTER는 {letters} 중 하나입니다.

{question}

{choices}
""".strip()


def normalize_question(text):
    return re.sub(r"\s+", " ", (text or "").strip())


def render_prompt(question, options):
    choices = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
    return EXPLAIN_TEMPLATE.format(
        choices=choices, letters=",".join(LETTERS[: len(options)]), question=question
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--output_dir", default=None, help="save_to_disk dir (optional)"
    )
    parser.add_argument("--push_to_hub", default=None, help="Hub repo id (optional)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (args.output_dir or args.push_to_hub):
        parser.error("give --push_to_hub and/or --output_dir")

    redux = load_dataset("LGAI-EXAONE/KMMLU-Redux", split="test")
    redux_questions = {normalize_question(q) for q in redux["question"]}

    parts = [
        load_dataset("HAERAE-HUB/KMMLU", subject, split="train")
        for subject in get_dataset_config_names("HAERAE-HUB/KMMLU")
    ]
    raw = concatenate_datasets(parts)
    print(f"KMMLU train: {len(raw)} rows")

    dropped = Counter()

    def keep(row):
        options = [row[c] for c in LETTERS]
        if not normalize_question(row["question"]) or any(
            not str(o).strip() for o in options
        ):
            dropped["empty question/option"] += 1
            return False
        if row["answer"] not in (1, 2, 3, 4):
            dropped["answer not in 1..4"] += 1
            return False
        if normalize_question(row["question"]) in redux_questions:
            dropped["in KMMLU-Redux"] += 1
            return False
        return True

    # num_proc=1 so the Counter above sees every drop.
    filtered = raw.filter(keep)
    print(f"dropped: {dict(dropped)}")

    def to_grpo(row):
        options = [str(row[c]) for c in LETTERS]
        return {
            "prompt": [
                {"role": "user", "content": render_prompt(row["question"], options)}
            ],
            "solution": LETTERS[row["answer"] - 1],
            "num_choices": len(options),
            "category": row["Category"],
        }

    dataset = filtered.map(to_grpo, remove_columns=filtered.column_names)
    dataset = dataset.shuffle(seed=args.seed)
    print(dataset)
    print(f"gold letter balance: {Counter(dataset['solution'])}")
    print(dataset[0]["prompt"][0]["content"], "\n->", dataset[0]["solution"])

    if args.output_dir:
        dataset.save_to_disk(args.output_dir)
        print(f"saved to {args.output_dir}")
    if args.push_to_hub:
        dataset.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed to {args.push_to_hub}")


if __name__ == "__main__":
    main()
