"""
Dataset validator for prompt-only (GRPO / Distill) datasets.

Usage:
  Check mode:
    python utils/check-GRPO-dataset.py --dataset_name <name_or_path> [--config <name>] [--split <split>] [--require-solution]

The dataset must provide either:
  - a `prompt` column: a string, or a list of messages
    (each message a dict with `role` and `content`), or
  - a `messages` column: a list of messages whose last turn has role == "user"
    (i.e. prompt-only data — assistant turns are not expected).

With --require-solution, the script also checks for a `solution` column
(used by `accuracy_reward`, see docs/grpo-rewards.md) and reports how many
rows would be excluded because they lack it.
"""

import argparse
import json
import os
import sys

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

# Color codes (kept consistent with check-SFT-dataset.py)
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def load_dataset_any(dataset_name, config_name=None, split="train"):
    """Load a dataset from the Hub or a local directory (mirrors grpo.py)."""
    if os.path.isdir(dataset_name):
        dataset = load_from_disk(dataset_name)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split or "train"]
        return dataset
    return load_dataset(path=dataset_name, name=config_name, split=split)


def is_conversational(column):
    """True if the value looks like a list of {role, content} dicts."""
    if not isinstance(column, list) or len(column) == 0:
        return False
    for message in column:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            return False
    return True


def validate_prompt_schema(dataset):
    """
    Validate that the dataset can be normalized into the prompt-only schema
    used by grpo.py / distill.py.

    Returns (is_valid, message, example).
    """
    columns = dataset.column_names
    if "messages" in columns:
        for idx, example in enumerate(dataset):
            messages = example.get("messages")
            if not isinstance(messages, list) or len(messages) == 0:
                print(f"{YELLOW}Example data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                return False, f"Row {idx}: 'messages' is not a valid list", example
            last = messages[-1]
            if not isinstance(last, dict) or last.get("role") != "user":
                print(f"{YELLOW}Example data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                return False, f"Row {idx}: last message role is not 'user'", example
        return True, "All checks passed (messages schema)", None

    if "prompt" in columns:
        for idx, example in enumerate(dataset):
            prompt = example.get("prompt")
            if isinstance(prompt, str):
                if not prompt.strip():
                    print(f"{YELLOW}Example data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                    return False, f"Row {idx}: 'prompt' is empty", example
                continue
            if is_conversational(prompt):
                if prompt[-1].get("role") != "user":
                    print(f"{YELLOW}Example data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
                    return False, f"Row {idx}: last prompt message role is not 'user'", example
                continue
            print(f"{YELLOW}Example data:{RESET} {json.dumps(example, indent=2, ensure_ascii=False)}")
            return False, f"Row {idx}: 'prompt' is neither a string nor a list of dicts", example
        return True, "All checks passed (prompt schema)", None

    return False, f"Dataset has neither `messages` nor `prompt` (columns: {columns})", None


def check_solution_column(dataset):
    """Report how many rows have a usable `solution` column."""
    if "solution" not in dataset.column_names:
        return False, "Missing 'solution' column (required by accuracy_reward)", None
    missing = 0
    for example in dataset:
        solution = example.get("solution")
        if solution is None or (isinstance(solution, str) and not solution.strip()):
            missing += 1
    return True, f"'solution' column present; {missing}/{len(dataset)} rows would be excluded", None


def print_examples(dataset, n=2):
    """Print a couple of example rows in yellow, matching the SFT validator style."""
    print(f"\n{YELLOW}Example rows:{RESET}")
    for i in range(min(n, len(dataset))):
        print(f"{YELLOW}--- row {i} ---{RESET}")
        print(json.dumps(dataset[i], indent=2, ensure_ascii=False))


def check_mode(dataset_name, config_name, split, require_solution):
    print(f"Loading dataset: {dataset_name} (config: {config_name}, split: {split})")
    dataset = load_dataset_any(dataset_name, config_name=config_name, split=split)
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")
    print("Validating prompt schema...")
    is_valid, message, _ = validate_prompt_schema(dataset)
    if not is_valid:
        print(f"\n{RED}x Not Valid{RESET}")
        print(f"Reason: {message}")
        sys.exit(1)
    print(f"{GREEN}v Prompt schema valid{RESET} - {message}")
    if require_solution:
        ok, sol_message, _ = check_solution_column(dataset)
        print(f"{GREEN}v {sol_message}{RESET}" if ok else f"{YELLOW}[WARNING] {sol_message}{RESET}")
    print_examples(dataset)
    print(f"\n{GREEN}v Valid{RESET}")
    print(f"Total examples validated: {len(dataset)}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a prompt-only (GRPO / Distill) dataset."
    )
    parser.add_argument(
        "dataset_name",
        help="Dataset name on the Hub or local path to a saved dataset.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config name (passed to load_dataset).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--require-solution",
        action="store_true",
        help="Also check for a `solution` column (used by accuracy_reward).",
    )
    args = parser.parse_args()
    check_mode(args.dataset_name, args.config, args.split, args.require_solution)


if __name__ == "__main__":
    main()
