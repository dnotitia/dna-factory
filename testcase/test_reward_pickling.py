"""Pickle compatibility for the reward factories used by spawned GRPO workers."""

import asyncio
import inspect
import multiprocessing
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.rewards import my_rewards


def _run_reward_in_spawned_child(reward, kwargs, result_queue):
    """Execute a reward received through multiprocessing's spawn pickle boundary."""
    is_async = inspect.iscoroutinefunction(reward)
    result = reward(["prompt"], [r"The answer is \\boxed{42}"], [[1]], **kwargs)
    if is_async:
        result = asyncio.run(result)
    result_queue.put((is_async, result))


def test_default_rewards_pickle_round_trip():
    """Concrete factory defaults can cross a spawned-worker pickle boundary."""
    rewards = [
        my_rewards.judge_reward,
        my_rewards.judge_reward_with_reference,
        my_rewards.persona_judge,
        my_rewards.ccp_judge,
        my_rewards.rlvr_judge,
        my_rewards.boxed_match_reward,
    ]

    for reward in rewards:
        restored = pickle.loads(pickle.dumps(reward))
        assert restored.__name__ == reward.__name__


def test_round_tripped_rewards_keep_sync_and_async_behavior():
    """Judge remains async while string matching remains sync and callable."""
    judge = pickle.loads(pickle.dumps(my_rewards.judge_reward_with_reference))
    string_match = pickle.loads(pickle.dumps(my_rewards.boxed_match_reward))

    assert inspect.iscoroutinefunction(judge)
    assert not inspect.iscoroutinefunction(string_match)
    assert asyncio.run(judge(["prompt"], ["completion"], [[1]])) == [None]
    assert string_match(
        ["prompt"], [r"The answer is \\boxed{42}"], [[1]], solution=["42"]
    ) == [1.0]


def test_default_rewards_execute_in_spawned_child():
    """Spawned workers can unpickle and execute concrete sync and async defaults."""
    context = multiprocessing.get_context("spawn")
    cases = [
        (my_rewards.boxed_match_reward, {"solution": ["42"]}, (False, [1.0])),
        # Missing label makes this concrete async judge skip before creating an HTTP client.
        (my_rewards.persona_judge, {}, (True, [None])),
    ]

    for reward, kwargs, expected in cases:
        result_queue = context.Queue()
        process = context.Process(
            target=_run_reward_in_spawned_child,
            args=(reward, kwargs, result_queue),
        )
        process.start()
        is_async, result = result_queue.get(timeout=10)
        process.join(timeout=10)

        assert process.exitcode == 0
        assert (is_async, result) == expected
