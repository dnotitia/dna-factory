import inspect
import json
import random
import re
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dna_factory.frontier_curriculum import (
    DEFAULT_HINT_TEMPLATE,
    HELD_OUT,
    HINT_COLUMN,
    NO_TRACE,
    STALL_MISSES,
    STATE_FILE,
    FrontierController,
    FrontierCurriculumMixin,
    add_hint,
    as_steps,
    is_held_out,
    normalize_mode,
    problem_id,
    render_hint,
)

GROUP = 4
STEPS = [f"step number {i}" for i in range(1, 9)]  # 8 teacher steps


def shown_steps(prompt):
    """How many teacher steps a (possibly hinted) prompt shows."""
    return len(
        re.findall(r"^\d+\. step number", prompt[-1]["content"], flags=re.MULTILINE)
    )


class FakeGRPOTrainer:
    """The slice of GRPOTrainer the mixin touches, with the same call order.

    `solver(row_index, steps_shown)` decides whether a rollout is correct; `solved_reward` is
    the verified 0/1 reward and `format_reward` a shaping reward that is always positive.
    `exchange`, when set, stands in for the cross-rank gather of a distributed run.
    """

    def __init__(
        self, solver=lambda i, shown: False, num_processes=1, exchange=None, **attrs
    ):
        self.num_generations = GROUP
        self.reward_func_names = ["format_reward", "solved_reward"]
        self.accelerator = SimpleNamespace(num_processes=num_processes)
        self.model = SimpleNamespace(training=True)
        self.args = SimpleNamespace(
            seed=7,
            should_save=True,
            per_device_train_batch_size=GROUP,
            steps_per_generation=1,
        )
        self.state = SimpleNamespace(global_step=10)
        self.train_dataset = SimpleNamespace(
            column_names=["prompt", "solution", "steps"]
        )
        self.rollout_func = None
        self.tools = None
        self.environment_factories = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._pending_extra_logs = defaultdict(list)
        self.solver = solver
        self.exchange = exchange
        self.output_dir = None
        self.seen_rows = None
        self.reward_prompts = None
        for key, value in attrs.items():
            setattr(self, key, value)

    def _log_completion_extra(self, column, values):
        self._pending_extra_logs[column].extend(values)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        self.reward_prompts = prompts
        rows = []
        for index, prompt in enumerate(prompts):
            verdict = self.solver(index, shown_steps(prompt))
            solved = float("nan") if verdict is None else float(bool(verdict))
            rows.append([0.5, solved])
        if self.exchange is not None:  # TRL returns the rewards of every rank
            rows = self.exchange("rewards", rows)
        return torch.tensor(rows)

    def _generate_and_score_completions(self, inputs):
        self.seen_rows = inputs
        prompts = [x["prompt"] for x in inputs]
        self._calculate_rewards(inputs, prompts, [""] * len(inputs), None)
        return {"advantages": torch.ones(len(inputs))}

    def _get_output_dir(self, trial=None):
        return self.output_dir

    def _save_checkpoint(self, model, trial):
        Path(self.output_dir, f"checkpoint-{self.state.global_step}").mkdir(
            parents=True
        )

    def _load_optimizer_and_scheduler(self, checkpoint):
        pass


class Trainer(FrontierCurriculumMixin, FakeGRPOTrainer):
    def _frontier_gather(self, plan):
        if self.exchange is not None:
            return self.exchange("plan", plan)
        return super()._frontier_gather(plan)


def make_trainer(**kwargs):
    kwargs.setdefault("frontier_curriculum", "mfc")
    kwargs.setdefault("frontier_success_reward", "solved_reward")
    kwargs.setdefault("frontier_holdout_ratio", 0.0)
    return Trainer(**kwargs)


def question(q):
    return [
        {"role": "system", "content": "You are a mathematician."},
        {"role": "user", "content": f"Solve {q}."},
    ]


def batch(*questions, steps=STEPS):
    return [
        {"prompt": question(q), "solution": "42", "steps": steps}
        for q in questions
        for _ in range(GROUP)
    ]


def pid_of(q):
    return problem_id(question(q))


def drive(controller, pid, steps, solves, visits):
    """Run `visits` visits of one problem; `solves(revealed)` is the whole group's outcome."""
    frontiers, shown = [], []
    for _ in range(visits):
        revealed = controller.reveal(pid, steps)
        shown.append(revealed)
        controller.update(pid, revealed, [solves(revealed)] * GROUP, steps)
        frontiers.append(controller.frontier(pid))
    return frontiers, shown


# --- controller: Algorithm 1 ------------------------------------------------------------------


def test_frontier_never_increases_under_any_outcome_sequence():
    rng = random.Random(0)
    controller = FrontierController()
    frontiers, shown = drive(controller, "p", 10, lambda g: rng.random() < 0.5, 300)
    assert frontiers == sorted(frontiers, reverse=True)
    assert frontiers[-1] == 0
    # No visit ever shows more than the frontier that was in force for it.
    assert all(g <= before for g, before in zip(shown, [10, *frontiers[:-1]]))


def test_draw_is_uniform_over_zero_to_the_frontier():
    controller = FrontierController(seed=1)
    _, shown = drive(controller, "p", 4, lambda g: False, 5000)  # frontier stays at 4
    counts = Counter(shown)
    assert sorted(counts) == [0, 1, 2, 3, 4]
    assert all(850 < counts[g] < 1150 for g in range(5))  # 1 / (g_curr + 1) each


def test_frontier_commits_only_on_success_strictly_below_it():
    controller = FrontierController(success_threshold=0.5)
    assert controller.update("p", 8, [True] * GROUP, 8) is False  # at the frontier
    assert (
        controller.update("p", 3, [True, False, False, False], 8) is False
    )  # mean < tau
    assert controller.frontier("p") == 8
    assert controller.update("p", 3, [True, True, False, False], 8) is True
    assert controller.frontier("p") == 3
    assert (
        controller.update("p", 1, [False] * GROUP, 8) is False
    )  # failures change nothing
    assert controller.frontier("p") == 3


def test_fixed_learner_frontier_settles_on_what_it_needs_and_is_reported_stalled():
    # Solvable only from 3 steps up and never improving: a step RL cannot cross.
    controller = FrontierController()
    frontiers, _ = drive(controller, "p", 8, lambda g: g >= 3, 200)
    assert frontiers[-1] == 3
    summary = controller.summary()
    assert summary["stalled_frac"] == 1.0
    assert summary["unlocked_frac"] == 0.0
    assert summary["unhelped_frac"] == 0.0
    assert summary["mean_help"] == 3 / 8


def test_problem_never_solved_even_with_every_step_is_reported_unhelped():
    controller = FrontierController()
    frontiers, _ = drive(controller, "p", 8, lambda g: False, 3 * STALL_MISSES)
    assert frontiers[-1] == 8
    assert controller.summary()["unhelped_frac"] == 1.0


def test_unguided_share_of_visits_grows_as_the_frontier_falls():
    # The property the paper builds on: a problem at frontier g spends 1 / (g + 1) of its
    # visits on the unguided prompt, and the frontier only falls.
    need = {"steps": 8}

    def solves(revealed):
        if revealed >= need["steps"]:
            need["steps"] = max(need["steps"] - 1, 0)
            return True
        return False

    controller = FrontierController(seed=3)
    frontiers, shown = drive(controller, "p", 8, solves, 400)
    assert frontiers[-1] == 0
    first, last = shown[:100], shown[-100:]
    assert sum(g == 0 for g in first) < sum(g == 0 for g in last) == 100


def test_unscored_rollouts_are_excluded_not_counted_as_failures():
    controller = FrontierController(success_threshold=1.0)
    assert controller.update("p", 1, [True, None, None, None], 8) is True
    assert controller.update("q", 1, [None] * GROUP, 8) is False
    assert controller.frontier("q") == 8


def test_state_round_trip_continues_the_same_sequence():
    first = FrontierController(seed=5)
    drive(first, "p", 8, lambda g: g >= 2, 9)
    restored = FrontierController(seed=5)
    restored.load_state_dict(json.loads(json.dumps(first.state_dict())))
    assert restored.state_dict() == first.state_dict()
    assert drive(restored, "p", 8, lambda g: g >= 2, 20) == drive(
        first, "p", 8, lambda g: g >= 2, 20
    )


def test_state_from_another_version_is_refused():
    with pytest.raises(ValueError, match="version"):
        FrontierController().load_state_dict({"version": 1, "problems": {}})


def test_held_out_set_is_a_fixed_function_of_the_problem():
    pids = [f"problem-{i}" for i in range(2000)]
    held = [pid for pid in pids if is_held_out(pid, 0.1)]
    assert 150 < len(held) < 250
    assert held == [pid for pid in pids if is_held_out(pid, 0.1)]
    assert set(held) <= {pid for pid in pids if is_held_out(pid, 0.3)}
    assert not any(is_held_out(pid, 0.0) for pid in pids)


# --- hint rendering ---------------------------------------------------------------------------


def test_steps_come_from_a_list_or_from_lines_of_text():
    assert as_steps(["a", " b ", ""]) == ["a", "b"]
    assert as_steps("a\n\n b \n") == ["a", "b"]
    assert as_steps(None) == []
    assert as_steps([]) == []


def test_hint_goes_into_the_last_user_message_and_leaves_the_original_alone():
    prompt = question("x")
    hinted = add_hint(prompt, render_hint(STEPS[:2], DEFAULT_HINT_TEMPLATE))
    assert prompt == question("x")  # not mutated
    assert hinted[0] == prompt[0]
    assert hinted[1]["role"] == "user"
    assert hinted[1]["content"].startswith("Solve x.\n\nBelow are some initial")
    assert "1. step number 1\n2. step number 2" in hinted[1]["content"]
    assert "step number 3" not in hinted[1]["content"]
    assert "do NOT just copy them" in hinted[1]["content"]


def test_prompt_without_a_text_user_message_is_refused():
    with pytest.raises(ValueError, match="no user message"):
        add_hint([{"role": "system", "content": "s"}], "h")
    with pytest.raises(NotImplementedError, match="text prompts"):
        add_hint([{"role": "user", "content": [{"type": "text", "text": "q"}]}], "h")
    with pytest.raises(ValueError, match="conversational"):
        add_hint("plain string prompt", "h")


def test_normalize_mode_accepts_yaml_off():
    assert normalize_mode(False) == "off"
    assert normalize_mode("MFC") == "mfc"
    with pytest.raises(ValueError):
        normalize_mode("on")


# --- mixin ------------------------------------------------------------------------------------


def test_group_shares_one_hint_and_the_steps_column_never_reaches_rewards():
    trainer = make_trainer()
    trainer._generate_and_score_completions(batch("q1", "q2", "q3"))
    for start in range(0, 3 * GROUP, GROUP):
        prompts = [row["prompt"] for row in trainer.seen_rows[start : start + GROUP]]
        assert all(prompt == prompts[0] for prompt in prompts)
    assert all("steps" not in row for row in trainer.seen_rows)
    assert all(row["solution"] == "42" for row in trainer.seen_rows)


def test_hint_is_the_first_steps_of_the_trace_in_order():
    trainer = make_trainer()
    seen = set()
    for _ in range(60):
        trainer._generate_and_score_completions(batch("q"))
        prompt = trainer.seen_rows[0]["prompt"]
        count = shown_steps(prompt)
        seen.add(count)
        logged = trainer._pending_extra_logs[HINT_COLUMN][-1]
        assert logged == count
        if count:
            expected = "\n".join(f"{i}. {s}" for i, s in enumerate(STEPS[:count], 1))
            assert expected in prompt[-1]["content"]
            assert f"{count + 1}. " not in prompt[-1]["content"]
        else:
            assert prompt == question("q")  # unguided: the stock prompt, untouched
    assert seen == set(range(9))  # 0..8: the full trace is allowed, as in the paper


def test_max_hint_ratio_caps_the_steps_ever_shown():
    trainer = make_trainer(frontier_max_hint_ratio=0.5)
    for _ in range(60):
        trainer._generate_and_score_completions(batch("q"))
        assert shown_steps(trainer.seen_rows[0]["prompt"]) <= 4


def test_rows_without_steps_train_as_stock_grpo():
    trainer = make_trainer()
    rows = batch("q", steps=None) + batch("r", steps=[])
    output = trainer._generate_and_score_completions(rows)
    assert [row["prompt"] for row in trainer.seen_rows] == [
        row["prompt"] for row in rows
    ]
    assert trainer.frontier.state_dict()["problems"] == {}
    assert trainer._pending_extra_logs[HINT_COLUMN] == [NO_TRACE] * (2 * GROUP)
    assert output["advantages"].tolist() == [1.0] * (2 * GROUP)


def test_frontier_moves_on_the_verified_reward_not_on_shaping_rewards():
    # format_reward is 0.5 on every row; only solved_reward may move the frontier.
    never = make_trainer(solver=lambda i, shown: False)
    for _ in range(40):
        never._generate_and_score_completions(batch("q"))
    assert never.frontier.frontier(pid_of("q")) == 8

    # Every rollout correct: advantage is 0 for the whole group, the problem is still solved.
    always = make_trainer(solver=lambda i, shown: True)
    for _ in range(40):
        always._generate_and_score_completions(batch("q"))
    assert always.frontier.frontier(pid_of("q")) == 0


def test_problem_solvable_only_with_help_is_unlocked_and_reported():
    need = {"steps": 6}

    def solver(index, shown):
        if shown >= need["steps"]:
            if index == 0:  # once per group
                need["steps"] = max(need["steps"] - 1, 0)
            return True
        return False

    trainer = make_trainer(solver=solver)
    seen = []
    for _ in range(300):
        trainer._generate_and_score_completions(batch("hard"))
        seen.append(trainer.frontier.frontier(pid_of("hard")))
    assert seen == sorted(seen, reverse=True)
    assert seen[-1] == 0
    metrics = trainer._metrics["train"]
    assert metrics["frontier/unlocked_frac"][-1] == 1.0
    assert metrics["frontier/mean_help"][-1] == 0.0
    assert metrics["frontier/unguided_mass"][-1] == 1.0
    assert metrics["frontier/solve_rate_unguided"][-1] == 1.0


def test_stalled_problem_shows_up_in_the_metrics():
    trainer = make_trainer(solver=lambda i, shown: shown >= 3)
    for _ in range(200):
        trainer._generate_and_score_completions(batch("cliff"))
    assert trainer.frontier.frontier(pid_of("cliff")) == 3
    assert trainer._metrics["train"]["frontier/stalled_frac"][-1] == 1.0


def test_zero_ratio_is_a_control_run_with_stock_prompts():
    trainer = make_trainer(frontier_max_hint_ratio=0.0)
    for _ in range(10):
        trainer._generate_and_score_completions(batch("q"))
        assert trainer.seen_rows[0]["prompt"] == question("q")
    metrics = trainer._metrics["train"]
    assert metrics["frontier/unguided_mass"] == [1.0] * 10
    assert "frontier/solve_rate_unguided" in metrics
    assert "frontier/solve_rate_guided" not in metrics


def held_out_question(ratio=0.3):
    return next(
        q for q in (f"q{i}" for i in range(200)) if is_held_out(pid_of(q), ratio)
    )


def trained_question(ratio=0.3):
    return next(
        q for q in (f"q{i}" for i in range(200)) if not is_held_out(pid_of(q), ratio)
    )


def test_held_out_problems_are_scored_unguided_and_never_trained_on():
    held, kept = held_out_question(), trained_question()
    trainer = make_trainer(solver=lambda i, shown: True, frontier_holdout_ratio=0.3)
    for _ in range(3):
        output = trainer._generate_and_score_completions(batch(held, kept))
        assert trainer.seen_rows[0]["prompt"] == question(held)  # never hinted
    assert output["advantages"].tolist() == [0.0] * GROUP + [1.0] * GROUP
    logged = trainer._pending_extra_logs[HINT_COLUMN]
    assert logged[-2 * GROUP : -GROUP] == [HELD_OUT] * GROUP
    assert pid_of(held) not in trainer.frontier.state_dict()["problems"]
    assert pid_of(kept) in trainer.frontier.state_dict()["problems"]
    assert trainer._metrics["train"]["frontier/heldout_solve_rate"] == [1.0] * 3


def test_held_out_rows_are_marked_dead_for_dynamic_sampling_mask():
    class DeadFlagBase(FakeGRPOTrainer):
        """Adds the per-row flags DynamicSamplingMixin attaches below the curriculum."""

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            output["_dyn_dead"] = torch.zeros(len(inputs), dtype=torch.bool)
            return output

    class Stack(FrontierCurriculumMixin, DeadFlagBase):
        pass

    trainer = Stack(
        frontier_curriculum="mfc",
        frontier_success_reward="solved_reward",
        frontier_holdout_ratio=0.3,
    )
    output = trainer._generate_and_score_completions(
        batch(held_out_question(), trained_question())
    )
    assert output["_dyn_dead"].tolist() == [True] * GROUP + [False] * GROUP


def test_custom_hint_template_is_used_and_must_have_the_placeholder():
    trainer = make_trainer(frontier_hint_template="HINT>>\n{steps}\n<<HINT")
    for _ in range(20):
        trainer._generate_and_score_completions(batch("q"))
        content = trainer.seen_rows[0]["prompt"][-1]["content"]
        if shown_steps(trainer.seen_rows[0]["prompt"]):
            assert content.endswith("<<HINT") and "HINT>>\n1. step number 1" in content
    with pytest.raises(ValueError, match="steps"):
        make_trainer(frontier_hint_template="no placeholder")


def test_mixed_prompts_in_one_group_are_refused():
    trainer = make_trainer()
    rows = batch("a")[:2] + batch("b")[:2]
    with pytest.raises(ValueError, match="different prompts"):
        trainer._generate_and_score_completions(rows)


def test_evaluation_is_never_hinted_and_does_not_touch_the_frontier():
    trainer = make_trainer()
    trainer.model.training = False
    trainer._generate_and_score_completions(batch("q"))
    assert all(row["prompt"] == question("q") for row in trainer.seen_rows)
    assert all("steps" not in row for row in trainer.seen_rows)
    assert trainer.frontier.state_dict()["problems"] == {}


def test_off_leaves_the_batch_untouched():
    trainer = Trainer(frontier_curriculum=False)
    rows = batch("q")
    trainer._generate_and_score_completions(rows)
    assert trainer.seen_rows is rows


def test_state_is_written_to_the_checkpoint_and_restored(tmp_path):
    trainer = make_trainer(solver=lambda i, shown: shown >= 2)
    trainer.output_dir = str(tmp_path)
    for _ in range(15):
        trainer._generate_and_score_completions(batch("q"))
    trainer._save_checkpoint(None, None)
    checkpoint = tmp_path / "checkpoint-10"
    assert (checkpoint / STATE_FILE).exists()

    resumed = make_trainer(solver=lambda i, shown: shown >= 2)
    resumed._load_optimizer_and_scheduler(str(checkpoint))
    assert resumed.frontier.state_dict() == trainer.frontier.state_dict()
    # Both continue identically: nothing but the state file is needed to resume.
    for _ in range(15):
        trainer._generate_and_score_completions(batch("q"))
        resumed._generate_and_score_completions(batch("q"))
        assert resumed.seen_rows[0]["prompt"] == trainer.seen_rows[0]["prompt"]
    assert resumed.frontier.state_dict() == trainer.frontier.state_dict()


# --- distributed ------------------------------------------------------------------------------


class Exchange:
    """Stands in for the cross-rank gather: every rank contributes, all get the concatenation."""

    def __init__(self, ranks):
        self.ranks = ranks
        self.barrier = threading.Barrier(ranks)
        self.slots = {}
        self.local = threading.local()

    def __call__(self, stage, value):
        self.slots[(stage, self.local.rank)] = value
        self.barrier.wait(timeout=10)
        merged = [
            item for rank in range(self.ranks) for item in self.slots[(stage, rank)]
        ]
        self.barrier.wait(timeout=10)  # all have read before the next round overwrites
        return merged


def test_ranks_of_a_distributed_run_stay_identical_and_match_one_process():
    questions = [f"q{i}" for i in range(6)]

    def solver(index, shown):
        return shown % 3 != 1  # depends on the hint alone, so every rank agrees

    single = make_trainer(solver=solver, frontier_holdout_ratio=0.3)
    exchange = Exchange(2)
    ranks = [
        make_trainer(
            solver=solver,
            frontier_holdout_ratio=0.3,
            num_processes=2,
            exchange=exchange,
        )
        for _ in range(2)
    ]
    halves = [questions[:3], questions[3:]]
    errors = []

    def run(rank):
        exchange.local.rank = rank
        try:
            ranks[rank]._generate_and_score_completions(batch(*halves[rank]))
        except Exception as error:  # noqa: BLE001 - surfaced through `errors`
            errors.append(error)
            exchange.barrier.abort()

    for _ in range(40):
        single._generate_and_score_completions(batch(*questions))
        threads = [threading.Thread(target=run, args=(rank,)) for rank in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, errors
        assert ranks[0].frontier.state_dict() == ranks[1].frontier.state_dict()
        assert ranks[0].frontier.state_dict() == single.frontier.state_dict()
    assert len(single.frontier.state_dict()["problems"]) > 0


def test_distributed_run_needs_whole_groups_per_rank():
    trainer_args = SimpleNamespace(
        seed=7, should_save=True, per_device_train_batch_size=3, steps_per_generation=1
    )
    with pytest.raises(ValueError, match="whole rollout groups"):
        make_trainer(num_processes=2, args=trainer_args)
    make_trainer(num_processes=2)  # 4 rows per rank, groups of 4


# --- configuration ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"dynamic_sampling": "resample"}, ValueError, "resample"),
        ({"rollout_func": object()}, NotImplementedError, "rollout_func"),
        ({"frontier_success_reward": "missing"}, ValueError, "frontier_success_reward"),
        ({"frontier_success_reward": None}, ValueError, "frontier_success_reward"),
        ({"frontier_steps_column": "absent"}, ValueError, "frontier_steps_column"),
        ({"frontier_max_hint_ratio": 1.5}, ValueError, "frontier_max_hint_ratio"),
        ({"frontier_holdout_ratio": 1.0}, ValueError, "frontier_holdout_ratio"),
        ({"frontier_success_threshold": 0.0}, ValueError, "frontier_success_threshold"),
    ],
)
def test_unsupported_configurations_are_refused_at_init(kwargs, error, match):
    with pytest.raises(error, match=match):
        make_trainer(**kwargs)


def test_hinted_prompt_that_cannot_fit_the_vllm_context_is_refused_at_init():
    class WordTokenizer:
        def apply_chat_template(self, prompt, add_generation_prompt, tokenize):
            return [0] * sum(len(message["content"].split()) for message in prompt)

    def build(limit):
        vllm_args = SimpleNamespace(
            seed=7,
            should_save=True,
            per_device_train_batch_size=GROUP,
            steps_per_generation=1,
            use_vllm=True,
            vllm_max_model_length=limit,
            max_completion_length=100,
        )
        return make_trainer(
            args=vllm_args,
            processing_class=WordTokenizer(),
            train_dataset=FakeDataset(batch("q")[:1] + batch("r", steps=[])[:1]),
        )

    # The fully hinted prompt is 86 words; the row without steps does not count.
    with pytest.raises(ValueError, match="at least 186"):
        build(limit=185)
    build(limit=186)


class FakeDataset(list):
    column_names = ("prompt", "solution", "steps")


def test_dynamic_sampling_mask_is_accepted():
    make_trainer(dynamic_sampling="mask")


# --- entry point ------------------------------------------------------------------------------


def test_grpo_dataset_path_keeps_the_steps_column_as_a_list():
    import datasets

    import grpo

    source = datasets.Dataset.from_list(
        [
            {"prompt": question("a"), "solution": "\\boxed{1}", "steps": ["s1", "s2"]},
            {"prompt": question("b"), "solution": "\\boxed{2}", "steps": []},
        ]
    )
    normalized = grpo._normalize_dataset_for_grpo(source, "frontier")
    assert normalized[0]["steps"] == ["s1", "s2"]
    assert normalized[1]["steps"] == []
    assert normalized[0]["prompt"] == question("a")


def test_grpo_passes_every_frontier_option_to_the_trainer():
    import grpo
    from dna_factory.dnotitia_arguments import DnotitiaArguments

    dna = DnotitiaArguments(frontier_curriculum="mfc", frontier_holdout_ratio=0.2)
    ctx = {"reward_funcs": [], "quantization_config": None}
    kwargs = grpo.extra_trainer_kwargs(None, None, None, dna, ctx, None)
    accepted = inspect.signature(FrontierCurriculumMixin.__init__).parameters
    passed = {key for key in kwargs if key.startswith("frontier_")}
    assert passed == {key for key in accepted if key.startswith("frontier_")}
    assert kwargs["frontier_curriculum"] == "mfc"
    assert kwargs["frontier_holdout_ratio"] == 0.2


def test_grpo_refuses_the_curriculum_with_unsloth():
    import grpo

    script = SimpleNamespace(
        reward_model_name_or_path=None,
        reward_funcs=["dna_factory.rewards.boxed_match_reward"],
    )
    dna = SimpleNamespace(frontier_curriculum="mfc")
    with pytest.raises(ValueError, match="use_unsloth"):
        grpo.setup_training_args(script, None, None, dna, {"use_unsloth": True}, None)


# --- coupling to TRL --------------------------------------------------------------------------


def test_trl_still_exposes_the_hooks_the_mixin_overrides():
    """A TRL bump that renames one of these would silently switch the curriculum off."""
    from trl import GRPOTrainer

    expected = {
        "_generate_and_score_completions": ["self", "inputs"],
        "_calculate_rewards": [
            "self",
            "inputs",
            "prompts",
            "completions",
            "completion_ids_list",
        ],
        "_log_completion_extra": ["self", "column", "values"],
        "_save_checkpoint": ["self", "model", "trial"],
        "_load_optimizer_and_scheduler": ["self", "checkpoint"],
    }
    for name, params in expected.items():
        signature = inspect.signature(inspect.unwrap(getattr(GRPOTrainer, name)))
        assert list(signature.parameters) == params, name


def test_dnotitia_trainer_puts_the_curriculum_ahead_of_dynamic_sampling():
    from trl import GRPOTrainer

    from dna_factory.dnotitia_grpo_trainer import DnotitiaGRPOTrainer
    from dna_factory.dynamic_sampling import DEAD_KEY, DynamicSamplingMixin
    from dna_factory.frontier_curriculum import _DEAD_KEY

    mro = DnotitiaGRPOTrainer.__mro__
    assert (
        mro.index(FrontierCurriculumMixin)
        < mro.index(DynamicSamplingMixin)
        < mro.index(GRPOTrainer)
    )
    assert _DEAD_KEY == DEAD_KEY
