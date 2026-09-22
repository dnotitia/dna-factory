from collections.abc import Callable
from typing import Any

from _multiple_choice_ko import multiple_choice_ko
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import choice


@task
def kmmlu_redux(
    cot: bool = False,
    input_postfix: str | None = None,
) -> Task:
    dataset = get_kmmlu_redux_dataset()

    return Task(
        dataset=dataset,
        solver=multiple_choice_ko(
            cot=cot,
            input_postfix=input_postfix,
        ),
        scorer=choice(),
        config=GenerateConfig(temperature=0.0),
    )


def record_to_sample_kmmlu_redux(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        choices=record["options"],
        # converts 1 -> A, 2 -> B, etc.
        target=("ABCDE"[int(record["solution"]) - 1]),
        metadata={
            "license_name": record["license_name"],
            "category": record["category"],
        },
    )


def get_kmmlu_redux_dataset(
    sampling_function: Callable[
        [dict[str, Any]], Sample
    ] = record_to_sample_kmmlu_redux,
) -> Dataset:
    ds_path = "LGAI-EXAONE/KMMLU-Redux"

    dataset = hf_dataset(
        path=ds_path,
        split="test",
        sample_fields=sampling_function,
    )

    return dataset
