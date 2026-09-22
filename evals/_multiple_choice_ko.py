"""Korean-prompt multiple choice solver for inspect_ai.

Port of dna_eval's ``multiple_choice_ko`` onto the public ``inspect_ai`` API.
Differences from the original: the deprecated ``shuffle`` kwarg and the
``with_passages`` option (which relied on a dna_eval-only ``TaskState.passages``
field) are dropped. Shuffle choices at dataset load time instead.
"""

import re
from enum import Enum

from inspect_ai.solver import Choices, Generate, Solver, TaskState, solver
from inspect_ai.util import resource

SINGLE_ANSWER_TEMPLATE = r"""
다음 객관식 질문에 답하십시오. 당신의 응답 전체 내용은 반드시 다음 형식이어야 합니다: '정답: $LETTER' (따옴표 없이)이며, 여기서 LETTER는 {letters} 중 하나입니다.

{question}

{choices}
""".strip()

SINGLE_ANSWER_TEMPLATE_COT = r"""
다음 객관식 질문에 답하십시오. 당신의 응답의 마지막 줄은 반드시 다음 형식이어야 합니다: '정답: $LETTER' (따옴표 없이)이며, 여기서 LETTER는 {letters} 중 하나입니다. 답하기 전에 단계별로 생각하십시오.

{question}

{choices}
""".strip()

MULTIPLE_ANSWER_TEMPLATE = r"""
다음 객관식 질문에 답하십시오. 여러 개의 정답이 있을 수 있습니다. 당신의 응답 전체 내용은 반드시 다음 형식이어야 합니다: '정답: $LETTERS' (따옴표 없이)이며, 여기서 LETTERS는 {letters} 중 하나 이상입니다.

{question}

{choices}
""".strip()

MULTIPLE_ANSWER_TEMPLATE_COT = r"""
다음 객관식 질문에 답하십시오. 여러 개의 정답이 있을 수 있습니다. 응답의 마지막 줄은 반드시 다음 형식이어야 합니다: '정답: $LETTERS' (따옴표 없이)이며, 여기서 LETTERS는 {letters} 중 하나 이상입니다. 답하기 전에 단계별로 생각하십시오.

{question}

{choices}
""".strip()


class MultipleChoiceTemplateKo(str, Enum):
    """Korean templates for multiple choice questions."""

    SINGLE_ANSWER = SINGLE_ANSWER_TEMPLATE
    SINGLE_ANSWER_COT = SINGLE_ANSWER_TEMPLATE_COT
    MULTIPLE_ANSWER = MULTIPLE_ANSWER_TEMPLATE
    MULTIPLE_ANSWER_COT = MULTIPLE_ANSWER_TEMPLATE_COT


def answer_character(index: int) -> str:
    """0 -> 'A', 1 -> 'B', ..."""
    return chr(ord("A") + index)


def answer_index(char: str) -> int:
    """'A' -> 0, 'B' -> 1, ..."""
    return ord(char.upper()) - ord("A")


def answer_options(choices: Choices) -> str:
    r"""["c1", "c2"] -> "A) c1\nB) c2"."""
    return "\n".join(
        f"{answer_character(i)}) {choice.value}" for i, choice in enumerate(choices)
    )


def prompt(question: str, choices: Choices, template: str) -> str:
    return template.format(
        choices=answer_options(choices),
        letters=",".join(answer_character(i) for i in range(len(choices))),
        question=question,
    )


def parse_answers(state: TaskState, multiple_correct: bool) -> set[str]:
    """Extract '정답: <letters>' from the completion; empty set if not found."""
    # Strict: a line that is exactly '정답: X' (optionally followed by a period).
    match = re.search(
        r"(?i)^정답\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
        state.output.completion,
        flags=re.MULTILINE,
    )
    # Lenient fallback.
    if match is None:
        match = re.search(
            r"(?i)정답\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)",
            state.output.completion,
        )
    if match is None:
        return set()

    matched = match.group(1).strip().rstrip(".")
    allowed_options = {answer_character(i) for i in range(len(state.choices))}

    if multiple_correct:
        matched = matched.replace(" ", "")
        split_comma = set(matched.split(","))
        if split_comma.issubset(allowed_options):
            return split_comma
        split_nothing = set(matched)
        if split_nothing.issubset(allowed_options):
            return split_nothing
    elif matched in allowed_options:
        return {matched}

    return set()


def set_choices_based_on_generated_response(
    state: TaskState, answers: set[str]
) -> None:
    true_answers = [answer_index(letter) for letter in answers]
    for i in range(len(state.choices)):
        state.choices.mark_choice(i, i in true_answers)


def valid_template(template: str) -> bool:
    return bool(
        re.search(r"\{question\}", template) and re.search(r"\{choices\}", template)
    )


@solver
def multiple_choice_ko(
    *,
    template: str | None = None,
    cot: bool = False,
    multiple_correct: bool = False,
    max_tokens: int | None = None,
    input_postfix: str | None = None,
) -> Solver:
    """Korean multiple choice solver. Formats the prompt, calls `generate()`, marks choices.

    Constraints (same as inspect_ai's `multiple_choice`):

    1. The `Sample` must have `choices` set.
    2. Use with the `choice()` scorer.
    3. Calls `generate()` internally.

    Args:
      template: Custom template with `{question}` and `{choices}` (and optional
        `{letters}`) placeholders. Defaults come from `MultipleChoiceTemplateKo`.
      cot: Ask the model to reason step by step before answering.
        No effect with a custom template.
      multiple_correct: Allow more than one correct letter.
        No effect with a custom template.
      max_tokens: Passed through to `generate()`.
      input_postfix: Text appended (space-separated) to the end of the user prompt.
    """
    postfix = "" if input_postfix is None else " " + input_postfix

    if template and not valid_template(template):
        raise ValueError(
            "The template must contain '{question}' and '{choices}' placeholders for string substitution."
        )

    if template is None:
        if multiple_correct:
            template = MULTIPLE_ANSWER_TEMPLATE_COT if cot else MULTIPLE_ANSWER_TEMPLATE
        else:
            template = SINGLE_ANSWER_TEMPLATE_COT if cot else SINGLE_ANSWER_TEMPLATE

    template = resource(template)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError(
                "The multiple_choice_ko solver requires samples with choices"
            )

        state.user_prompt.text = (
            prompt(
                question=state.user_prompt.text,
                choices=state.choices,
                template=str(template),
            )
            + postfix
        )

        state = await generate(state, max_tokens=max_tokens)

        answers = parse_answers(state, multiple_correct)
        if answers:
            set_choices_based_on_generated_response(state=state, answers=answers)

        return state

    return solve
