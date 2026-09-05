"""
Test cases for dna_factory/rewards/my_rewards.py::safe_accuracy_reward

The wrapper must behave exactly like TRL's accuracy_reward on healthy batches,
and isolate poisoned samples (ones that make upstream raise, e.g. the observed
sympy `cot` singularity ZeroDivisionError in str(answer_parsed)) as None instead
of killing the run.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

import dna_factory.rewards as rewards
import dna_factory.rewards.my_rewards as my_rewards
from dna_factory.rewards import safe_accuracy_reward


def _completions(*contents):
    return [[{"role": "assistant", "content": c}] for c in contents]


class TestSafeAccuracyReward:
    """Test cases for safe_accuracy_reward"""

    def test_referenced_by_dotted_path(self):
        """grpo.py resolves dotted paths via getattr: dna_factory.rewards.X must exist."""
        assert getattr(rewards, "safe_accuracy_reward") is safe_accuracy_reward
        assert safe_accuracy_reward.__name__ == "safe_accuracy_reward"

    def test_fast_path_matches_upstream(self, monkeypatch):
        """Healthy batch: single upstream call, scores (and log_extra) passed through."""
        calls = []

        def fake_upstream(completions, solution, log_extra=None, **kwargs):
            calls.append((completions, list(solution), log_extra))
            return [1.0, 0.0]

        monkeypatch.setattr(my_rewards, "_hf_accuracy_reward", fake_upstream)
        logged = []
        out = safe_accuracy_reward(
            prompts=["p1", "p2"],
            completions=_completions("a", "b"),
            completion_ids=[[1], [2]],
            solution=["s1", "s2"],
            log_extra=logged.append,
        )
        assert out == [1.0, 0.0]
        assert len(calls) == 1  # one full-batch call, no per-sample retry
        assert calls[0][0] == _completions("a", "b")
        assert calls[0][1] == ["s1", "s2"]
        assert calls[0][2] is not None

    def test_poison_sample_isolated(self, monkeypatch):
        """Batch raising ZeroDivisionError (the observed cot crash) falls back to
        per-sample scoring: healthy samples keep scores, poison scores None."""

        def fake_upstream(completions, solution, log_extra=None, **kwargs):
            contents = [c[0]["content"] for c in completions]
            if any("POISON" in c for c in contents):
                raise ZeroDivisionError  # what sympy/mpmath raised on cot
            return [1.0] * len(completions)

        monkeypatch.setattr(my_rewards, "_hf_accuracy_reward", fake_upstream)
        out = safe_accuracy_reward(
            prompts=["p1", "p2"],
            completions=_completions("fine", "POISON \\\\cot"),
            completion_ids=[[1], [2]],
            solution=["s1", "s2"],
        )
        assert out == [1.0, None]

    def test_missing_solution_scores_all_none(self, monkeypatch):
        """No solution column: nothing to verify against, upstream never called."""
        called = []
        monkeypatch.setattr(
            my_rewards, "_hf_accuracy_reward", lambda *a, **k: called.append(1)
        )
        out = safe_accuracy_reward(
            prompts=["p1", "p2"],
            completions=_completions("a", "b"),
            completion_ids=[[1], [2]],
            solution=None,
        )
        assert out == [None, None]
        assert called == []

    def test_real_upstream_parity(self):
        """End-to-end against the real math_verify backend (no mocks)."""
        pytest.importorskip("math_verify")
        out = safe_accuracy_reward(
            prompts=["p1", "p2"],
            completions=_completions(
                r"My answer is \boxed{\frac{1}{3}}",
                r"My answer is \boxed{\frac{1}{2}}",
            ),
            completion_ids=[[1], [2]],
            solution=[r"\frac{1}{3}", r"\frac{1}{3}"],
        )
        assert out == [1.0, 0.0]
