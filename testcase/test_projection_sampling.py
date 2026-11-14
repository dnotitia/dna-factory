"""CPU tests of the projection search and comparison data contract."""

import random
import unittest
from types import SimpleNamespace

from dna_factory.projection_sampling import (
    VLLMProjectionBackend,
    comparison_rows,
    projection_sample,
)


class Backend:
    eos_token_id = 99

    def __init__(self, proposals, scores):
        self.proposals = iter(proposals)
        self.scores = scores
        self.calls = []

    def propose(self, question, solution, prefix, length):
        self.calls.append((list(prefix), length))
        return next(self.proposals)

    def score(self, question, tokens):
        return self.scores[tuple(tokens)]


class ProjectionTests(unittest.TestCase):
    def test_backend_scores_only_response_with_mean_not_sum(self):
        backend = object.__new__(VLLMProjectionBackend)
        prompts = []
        calls = []

        def prompt_ids(text):
            prompts.append(text)
            return [10, 11]

        def generate(ids, **kwargs):
            calls.append((ids, kwargs))
            return SimpleNamespace(
                prompt_logprobs=[
                    None,
                    {11: SimpleNamespace(logprob=-100)},
                    {20: SimpleNamespace(logprob=-2)},
                    {21: SimpleNamespace(logprob=-4)},
                ]
            )

        backend.prompt_ids = prompt_ids
        backend._generate = generate
        self.assertEqual(backend.score("question only", [20, 21]), -3)
        self.assertEqual(prompts, ["question only"])
        self.assertEqual(calls[0][0], [10, 11, 20, 21])
        self.assertEqual(calls[0][1]["prompt_logprobs"], 0)

    def test_accepts_only_strict_improvements(self):
        backend = Backend(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            {(1, 2): -3, (3, 4): -4, (5, 6): -2, (7, 8): -2},
        )
        tokens, stats = projection_sample(
            backend,
            "q",
            "a",
            max_new_tokens=2,
            block_num=1,
            mcmc_steps=3,
            rng=random.Random(9),
        )
        self.assertEqual(tokens, [5, 6])
        self.assertEqual(stats, {"mean_logprob": -2, "accepted": 1, "attempts": 3})
        self.assertTrue(
            all(len(prefix) + length == 2 for prefix, length in backend.calls)
        )

    def test_eos_checked_after_refinement(self):
        backend = Backend([[1, 99], [2, 99]], {(1, 99): -3, (2, 99): -2})
        tokens, stats = projection_sample(
            backend, "q", "a", max_new_tokens=4, block_num=2, mcmc_steps=1
        )
        self.assertEqual(tokens, [2, 99])
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(stats["accepted"], 1)

    def test_blocks_extend_current_prefix(self):
        backend = Backend([[1, 2], [1, 2, 3, 4]], {(1, 2): -2, (1, 2, 3, 4): -3})
        tokens, _ = projection_sample(
            backend, "q", "a", max_new_tokens=4, block_num=2, mcmc_steps=0
        )
        self.assertEqual(tokens, [1, 2, 3, 4])
        self.assertEqual(backend.calls, [([], 2), ([1, 2], 2)])

    def test_invalid_budget(self):
        for kwargs in (
            {"max_new_tokens": 3, "block_num": 2},
            {"block_num": 0},
            {"mcmc_steps": -1},
        ):
            with self.assertRaises(ValueError):
                projection_sample(None, "q", "a", **kwargs)

    def test_matched_rows_do_not_leak_reference_into_student_prompt(self):
        rows = comparison_rows("question", "secret reference", "sampled answer")
        prompt = [{"role": "user", "content": "question"}]
        self.assertEqual(rows["sdft"]["prompt"], prompt)
        for key, answer in (
            ("sft", "secret reference"),
            ("sampling_sft", "sampled answer"),
        ):
            self.assertEqual(rows[key]["messages"][:-1], prompt)
            self.assertEqual(rows[key]["messages"][-1]["content"], answer)
        self.assertEqual(rows["sdft"]["privileged_context"], "secret reference")


if __name__ == "__main__":
    unittest.main()
