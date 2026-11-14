"""Held-out math evals for the GRPO frontier curriculum (docs/grpo-frontier-curriculum.md).

The two in-domain test sets of "Unlocking the Unsolvable" (arXiv:2609.13997), from its public
dataset (Apache-2.0): `or1_200`, 200 held-out competition-math problems, and `uns22`, 22
held-out problems a small model fails at pass@64. Neither overlaps the Uns-128 training set.

The prompt is the unguided training prompt that utils/build-frontier-grpo-dataset.py writes
(same system message and solving instruction, never a hint), and the answer is checked with
the accuracy rewards' own verifier, so a checkpoint is scored exactly as it was trained.

    inspect eval evals/unsolvable_math.py@or1_200 --model vllm/<checkpoint>
    inspect eval evals/unsolvable_math.py@uns22 --model vllm/<checkpoint> --epochs 8 --epochs-reducer at_least_1

The second form is pass@8: `uns22` has only 22 problems, so its pass@1 moves 4.5 points per
problem.
"""

import sys
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, system_message

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASET = "yukangzhu/unlocking-the-unsolvable"
# Keep in sync with DEFAULT_SYSTEM / DEFAULT_SUFFIX in utils/build-frontier-grpo-dataset.py.
SYSTEM = "You are an expert mathematician with strong problem-solving skills. Think step by step."
SUFFIX = (
    "Please solve the problem step by step and put your final answer within \\boxed{}."
)


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=f"{str(record['question']).strip()}\n\n{SUFFIX}",
        target=str(record["ground_truth"]).strip(),
        id=str(record["index"]),
    )


def answer_section(completion: str) -> str:
    """The text after the reasoning block; a model served without a reasoning parser returns both."""
    return completion.rsplit("</think>", 1)[-1]


@scorer(metrics=[accuracy(), stderr()])
def verified_answer():
    """Correct when the final answer verifies against the gold answer (math-verify)."""
    from dna_factory.rewards import safe_accuracy_reward

    async def score(state: TaskState, target: Target) -> Score:
        answer = answer_section(state.output.completion or "")
        gold = target.text if "\\boxed" in target.text else f"\\boxed{{{target.text}}}"
        reward = safe_accuracy_reward(
            None, [[{"role": "assistant", "content": answer}]], solution=[gold]
        )[0]
        return Score(
            value=CORRECT if reward and reward >= 1.0 else INCORRECT,
            answer=answer[-200:],
        )

    return score


def _task(config: str) -> Task:
    return Task(
        dataset=hf_dataset(
            path=DATASET, name=config, split="test", sample_fields=record_to_sample
        ),
        solver=[system_message(SYSTEM), generate()],
        scorer=verified_answer(),
    )


@task
def or1_200() -> Task:
    return _task("or1_200")


@task
def uns22() -> Task:
    return _task("uns22")
