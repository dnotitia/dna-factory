"""
Test cases for evals/_multiple_choice_ko.py::parse_answers

The '정답: X' formats must keep parsing exactly as before, and a completion that
skips the prefix and answers with a bare letter ('A', 'A)', '**A**') must fall
back to that letter instead of scoring as invalid_response_format.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from _multiple_choice_ko import answers_from_letters, bare_answer, parse_answers


def _state(completion, num_choices=4):
    """Duck-typed TaskState: parse_answers only reads .output.completion/.choices."""
    return SimpleNamespace(
        output=SimpleNamespace(completion=completion),
        choices=[None] * num_choices,
    )


class TestParseAnswersPrefixed:
    """The '정답:' paths must be untouched by the fallback."""

    @pytest.mark.parametrize(
        "completion",
        [
            "정답: A",
            "정답:A",
            "정답 : A",
            "정답: A.",
            "**정답: A**",
            "단계별로 생각해보면 ...\n\n정답: A",
        ],
    )
    def test_prefixed_forms(self, completion):
        assert parse_answers(_state(completion), False) == {"A"}

    def test_prefix_wins_over_trailing_line(self):
        """A stray letter after the marker must not override it."""
        assert parse_answers(_state("정답: B\n\nC"), False) == {"B"}

    def test_out_of_range_letter(self):
        assert (
            parse_answers(
                _state("정답: E"),
                False,
            )
            == set()
        )


class TestParseAnswersBareLetter:
    """New fallback: no '정답:' anywhere, the last line is just the letter."""

    @pytest.mark.parametrize(
        "completion", ["A", "A)", "(A)", "A.", "**A**", "`A`", " A \n", "A:"]
    )
    def test_bare_forms(self, completion):
        assert parse_answers(_state(completion), False) == {"A"}

    def test_last_line_of_reasoning(self):
        completion = "B는 조선 후기에 해당하고, C는 시기가 맞지 않는다.\n\nA"
        assert parse_answers(_state(completion), False) == {"A"}

    @pytest.mark.parametrize(
        "completion",
        [
            "",
            "   \n\n",
            "A) 서울",  # the choice text was echoed, not an answer
            "따라서 정답은 A입니다",  # '정답:' colon missing, line is not bare
            "잘 모르겠습니다",
            "Z",  # beyond the sample's 4 choices
        ],
    )
    def test_no_answer(self, completion):
        assert parse_answers(_state(completion), False) == set()

    def test_multiple_correct(self):
        assert parse_answers(_state("A, C"), True) == {"A", "C"}
        assert parse_answers(_state("AC"), True) == {"A", "C"}


class TestHelpers:
    def test_bare_answer_strips_decoration(self):
        assert bare_answer("**(A).**") == "A"
        assert bare_answer("풀이\n- 첫째\nD") == "D"
        assert bare_answer("\n\n") is None

    def test_answers_from_letters_bounds(self):
        assert answers_from_letters("C", 4, False) == {"C"}
        assert answers_from_letters("C", 2, False) == set()
