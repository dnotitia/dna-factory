"""Frontier-guided curriculum for GRPO: teacher hints that are only ever withdrawn.

A prompt none of whose rollouts is correct has advantage 0 on every row, so GRPO learns nothing
from it. `dynamic_sampling` saves that group's compute or swaps it for another prompt; it does
not make the problem learnable. This module does, with the Monotone Frontier Curriculum of
"Unlocking the Unsolvable" (arXiv:2609.13997, Algorithm 1):

    g_curr(s) <- n_steps(s)                       for every problem s with a teacher trace
    on each visit to s:
        g_used ~ Uniform{0, 1, ..., g_curr(s)}
        roll out with the first g_used teacher steps shown as a hint in the prompt
        apply the GRPO update on the original problem's reward
        if mean verified reward >= tau and g_used < g_curr(s):  g_curr(s) <- g_used

The teacher's steps go into the user message as auxiliary context and the student writes a
complete solution from scratch, so every token it is trained on is its own. The frontier g_curr
never rises, and the uniform draw puts 1 / (g_curr + 1) of a problem's visits on the unguided
prompt, a share that can only grow. That unguided share is what the paper identifies as the
thing a curriculum has to maximise, because a hinted prompt is not the prompt the model is
evaluated on. At g_curr = 0 a problem trains exactly like stock GRPO.

Added here, because the frontier alone does not show them:

  - stalled / unhelped problems. A problem whose probes below the frontier keep failing gets
    no signal from RL at all; one never solved even with every step shown has a trace that does
    not help. Both are counted, and the state file names them.
  - a held-out share of the problems, generated and scored unguided but never trained on. Their
    solve rate is the one number here that memorising the training problems cannot move.
  - distributed runs: ranks exchange what they did and apply identical updates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

MODES = ("off", "mfc")
STATE_FILE = "frontier_state.json"
STATE_VERSION = 3
# Appended to the last user message when at least one step is shown. Wording follows the hint
# prompt of arXiv:2609.13997, Appendix G.2.
DEFAULT_HINT_TEMPLATE = (
    "Below are some initial reasoning steps that may help you:\n"
    "{steps}\n\n"
    "You should use the provided steps as a reference, but do NOT just copy them. Instead, "
    "reconstruct the complete reasoning process in your own words, starting from the "
    "beginning, and continue the reasoning to find the final answer."
)
# Completion-table column: teacher steps shown to the row.
HINT_COLUMN = "frontier_hint_steps"
NO_TRACE = -1  # the row has no teacher steps and trains as stock GRPO
HELD_OUT = -2  # the row is held out: generated and scored unguided, never trained on
# A rollout counts as solved when the designated reward gives full marks.
_SOLVED = 1.0 - 1e-6
# Failed probes below the frontier since it last moved before a problem counts as stalled;
# also the visits without a single success before it counts as unhelped.
STALL_MISSES = 16
# `dynamic_sampling` marks zero-advantage rows with this key (dna_factory/dynamic_sampling.py).
_DEAD_KEY = "_dyn_dead"


def normalize_mode(mode) -> str:
    """Resolve the configured mode, accepting the False that YAML 1.1 makes of a bare `off`."""
    if mode is False or mode is None:
        return "off"
    if not isinstance(mode, str) or mode.lower() not in MODES:
        raise ValueError(f"frontier_curriculum must be one of {MODES}, got {mode!r}")
    return mode.lower()


def problem_id(prompt) -> str:
    """Stable id of a problem, derived from its (unhinted) prompt.

    Hashing the prompt rather than using the row index keeps the id valid across a resume, a
    reshuffle, and a mixture whose `weight` duplicates or drops rows.
    """
    text = json.dumps(prompt, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _draw(*parts) -> int:
    key = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha1(key).digest()[:8], "big")


def is_held_out(pid: str, ratio: float) -> bool:
    """Whether a problem belongs to the held-out share.

    A function of the problem alone, not of the seed, so a control run and a curriculum run hold
    out the same problems and their held-out solve rates are comparable.
    """
    return ratio > 0.0 and _draw("holdout", pid) / 2**64 < ratio


def as_steps(value) -> list[str]:
    """Teacher steps from a dataset cell: a list of strings, or text with one step per line."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.splitlines()
    return [str(step).strip() for step in value if str(step).strip()]


def render_hint(steps: list[str], template: str) -> str:
    return template.format(
        steps="\n".join(f"{index}. {step}" for index, step in enumerate(steps, 1))
    )


def add_hint(prompt, hint: str):
    """A copy of a conversational prompt with `hint` appended to its last user message."""
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(
            "frontier_curriculum needs conversational prompts (a list of messages)."
        )
    for index in range(len(prompt) - 1, -1, -1):
        message = prompt[index]
        if message.get("role") != "user":
            continue
        if not isinstance(message.get("content"), str):
            raise NotImplementedError(
                "frontier_curriculum supports text prompts only, not multimodal content."
            )
        hinted = {**message, "content": f"{message['content']}\n\n{hint}"}
        return [*prompt[:index], hinted, *prompt[index + 1 :]]
    raise ValueError("frontier_curriculum found no user message to attach the hint to.")


class FrontierController:
    """Per-problem frontier state. Pure Python: no model, tokenizer or trainer involved.

    The step count drawn for a visit is a hash of (seed, problem, visit number), so a resumed
    run and every rank of a distributed run draw the same sequence from the state alone.
    """

    def __init__(self, success_threshold: float = 0.5, seed: int = 0):
        if not 0.0 < float(success_threshold) <= 1.0:
            raise ValueError(
                f"frontier_success_threshold must be in (0, 1], got {success_threshold}"
            )
        self.success_threshold = float(success_threshold)
        self.seed = int(seed)
        # problem id -> {steps, frontier, visits, wins, misses}
        #   steps     steps revealable when the problem was first seen (the initial frontier)
        #   frontier  g_curr: most steps it may still be shown; never increases
        #   wins      visits that met the success threshold, at any hint length
        #   misses    failed visits below the frontier since it last moved
        self._state: dict[str, dict[str, int]] = {}

    def _entry(self, pid: str, steps: int) -> dict[str, int]:
        entry = self._state.get(pid)
        if entry is None:
            steps = max(int(steps), 0)
            entry = {
                "steps": steps,
                "frontier": steps,
                "visits": 0,
                "wins": 0,
                "misses": 0,
            }
            self._state[pid] = entry
        return entry

    def frontier(self, pid: str) -> int | None:
        entry = self._state.get(pid)
        return None if entry is None else entry["frontier"]

    def reveal(self, pid: str, steps: int) -> int:
        """g_used for the next visit, uniform over 0..g_curr; `steps` seeds a new problem."""
        entry = self._entry(pid, steps)
        return _draw(self.seed, pid, entry["visits"]) % (entry["frontier"] + 1)

    def update(
        self, pid: str, revealed: int, solved: list[bool | None], steps: int
    ) -> bool:
        """Record one visit that showed `revealed` steps; lower the frontier on a success below it.

        `solved` holds one entry per rollout: True/False from the verified reward, None for a
        rollout that reward did not score. Returns True when the frontier moved.
        """
        entry = self._entry(pid, steps)
        entry["visits"] += 1
        scored = [s for s in solved if s is not None]
        success = bool(scored) and sum(scored) / len(scored) >= self.success_threshold
        entry["wins"] += success
        if revealed >= entry["frontier"]:
            return False
        if not success:
            entry["misses"] += 1
            return False
        entry["frontier"] = max(int(revealed), 0)
        entry["misses"] = 0
        return True

    @staticmethod
    def is_stalled(entry: dict[str, int]) -> bool:
        """Still needs help, and attempts with less of it keep failing."""
        return entry["frontier"] > 0 and entry["misses"] >= STALL_MISSES

    @staticmethod
    def is_unhelped(entry: dict[str, int]) -> bool:
        """Never solved at any hint length, the full trace included."""
        return entry["wins"] == 0 and entry["visits"] >= STALL_MISSES

    def summary(self) -> dict[str, float]:
        """Shares over the problems that had help to withdraw (`steps > 0`)."""
        entries = [e for e in self._state.values() if e["steps"] > 0]
        if not entries:
            return {
                "mean_help": 1.0,
                "unlocked_frac": 0.0,
                "stalled_frac": 0.0,
                "unhelped_frac": 0.0,
            }
        count = len(entries)
        return {
            # rho_max averaged over problems: share of the trace a problem may still be shown.
            "mean_help": sum(e["frontier"] / e["steps"] for e in entries) / count,
            "unlocked_frac": sum(e["frontier"] == 0 for e in entries) / count,
            "stalled_frac": sum(self.is_stalled(e) for e in entries) / count,
            "unhelped_frac": sum(self.is_unhelped(e) for e in entries) / count,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "seed": self.seed,
            "problems": {pid: dict(entry) for pid, entry in self._state.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != STATE_VERSION:
            raise ValueError(
                f"frontier state version {state.get('version')!r} is not {STATE_VERSION}; it was "
                "written by a different version of the curriculum and cannot be resumed."
            )
        restored = {}
        for pid, saved in state.get("problems", {}).items():
            entry = {
                key: max(int(saved[key]), 0)
                for key in ("steps", "frontier", "visits", "wins", "misses")
            }
            entry["frontier"] = min(entry["frontier"], entry["steps"])
            restored[pid] = entry
        self._state = restored


class FrontierCurriculumMixin:
    """Wires `FrontierController` into a GRPOTrainer.

    Mix in ahead of the trainer (and of DynamicSamplingMixin) so its hooks wrap theirs:

        class MyTrainer(FrontierCurriculumMixin, DynamicSamplingMixin, GRPOTrainer): ...

    Rows without teacher steps are anchor rows and train as stock GRPO, so a dataset with traces
    can be mixed with one without. Evaluation is never hinted.

    Distributed runs: every rank decides its own groups' hints from the controller, then all
    ranks exchange what they did (`_frontier_gather`) and apply the same updates in the same
    order, so the controllers stay identical. The exchange is unconditional, once per generation.
    """

    def __init__(
        self,
        *args,
        frontier_curriculum="off",
        frontier_steps_column: str = "steps",
        frontier_success_reward: str | None = None,
        frontier_success_threshold: float = 0.5,
        frontier_max_hint_ratio: float = 1.0,
        frontier_holdout_ratio: float = 0.1,
        frontier_hint_template: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.frontier_curriculum = normalize_mode(frontier_curriculum)
        self.frontier_steps_column = frontier_steps_column
        self.frontier_max_hint_ratio = float(frontier_max_hint_ratio)
        self.frontier_holdout_ratio = float(frontier_holdout_ratio)
        self.frontier_hint_template = frontier_hint_template or DEFAULT_HINT_TEMPLATE
        self._frontier_hint_column = (
            None  # per-row steps shown while a generation is in flight
        )
        self._frontier_solved = None
        if self.frontier_curriculum == "off":
            return

        if not 0.0 <= self.frontier_max_hint_ratio <= 1.0:
            raise ValueError(
                f"frontier_max_hint_ratio must be in [0, 1], got {frontier_max_hint_ratio}"
            )
        if not 0.0 <= self.frontier_holdout_ratio < 1.0:
            raise ValueError(
                f"frontier_holdout_ratio must be in [0, 1), got {frontier_holdout_ratio}"
            )
        if "{steps}" not in self.frontier_hint_template:
            raise ValueError(
                "frontier_hint_template must contain the `{steps}` placeholder."
            )
        self.frontier = FrontierController(
            success_threshold=frontier_success_threshold,
            seed=getattr(self.args, "seed", 0) or 0,
        )
        self._frontier_success_idx = self._frontier_validate(frontier_success_reward)
        logger.info(
            "frontier curriculum: success=%s>=%.2f max_hint_ratio=%.2f holdout_ratio=%.2f "
            "steps_column=%s",
            frontier_success_reward,
            self.frontier.success_threshold,
            self.frontier_max_hint_ratio,
            self.frontier_holdout_ratio,
            self.frontier_steps_column,
        )

    def _frontier_validate(self, success_reward: str | None) -> int:
        """Refuse configurations the hooks below do not cover, before any GPU time is spent."""
        if self.accelerator.num_processes > 1:
            # Each rank plans hints for the rows it holds, so those rows must be whole groups.
            # TRL only checks the global generation batch (see dynamic_sampling.py).
            rows = (
                self.args.per_device_train_batch_size * self.args.steps_per_generation
            )
            if rows % self.num_generations:
                raise ValueError(
                    "frontier_curriculum needs each rank's batch to hold whole rollout groups, "
                    f"but per_device_train_batch_size({self.args.per_device_train_batch_size}) * "
                    f"steps_per_generation({self.args.steps_per_generation}) = {rows} is not a "
                    f"multiple of num_generations({self.num_generations})."
                )
        if getattr(self, "dynamic_sampling", "off") == "resample":
            raise ValueError(
                "frontier_curriculum is incompatible with dynamic_sampling='resample': the refill "
                "generates extra groups outside the curriculum, which would be scored without a "
                "hint. Use dynamic_sampling='off' or 'mask'."
            )
        for attr in ("rollout_func", "tools", "environment_factories"):
            if getattr(self, attr, None):
                raise NotImplementedError(
                    f"frontier_curriculum does not support `{attr}`: the hint is written into the "
                    "single-turn prompt before TRL generates from it."
                )
        names = list(getattr(self, "reward_func_names", []))
        if success_reward not in names:
            raise ValueError(
                f"frontier_success_reward={success_reward!r} must name one of reward_funcs "
                f"{names}: the 0/1 verified-answer reward that decides whether a problem was "
                "solved. Format, length and judge rewards must not move the frontier."
            )
        columns = getattr(self.train_dataset, "column_names", None)
        if columns is not None and self.frontier_steps_column not in columns:
            raise ValueError(
                f"frontier_steps_column `{self.frontier_steps_column}` not in the training dataset "
                f"(columns: {columns})."
            )
        self._frontier_check_context(columns)
        return names.index(success_reward)

    def _frontier_check_context(self, columns) -> None:
        """Fail at startup if the longest hinted prompt cannot fit the vLLM context.

        vLLM rejects an over-long request instead of truncating it, which would otherwise
        surface hours into a run, on the first visit that draws a long hint for a long problem.
        """
        limit = getattr(self.args, "vllm_max_model_length", None)
        if not limit or not getattr(self.args, "use_vllm", False) or columns is None:
            return
        tokenizer = getattr(self, "_tokenizer", None) or self.processing_class
        kwargs = getattr(self, "chat_template_kwargs", None) or {}
        longest = 0
        for row in self.train_dataset:
            steps = as_steps(row.get(self.frontier_steps_column))
            allowed = round(self.frontier_max_hint_ratio * len(steps))
            if not allowed:
                continue
            prompt = add_hint(
                row["prompt"], render_hint(steps[:allowed], self.frontier_hint_template)
            )
            ids = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=True, **kwargs
            )
            ids = ids["input_ids"] if isinstance(ids, dict) else ids
            longest = max(longest, len(ids))
        needed = longest + (getattr(self.args, "max_completion_length", 0) or 0)
        if needed > int(limit):
            raise ValueError(
                f"frontier_curriculum: the longest hinted prompt is {longest} tokens, so with "
                f"max_completion_length={self.args.max_completion_length} a rollout needs "
                f"{needed} tokens, more than vllm_max_model_length={limit}. Raise "
                f"vllm_max_model_length to at least {needed}, or lower frontier_max_hint_ratio."
            )
        logger.info(
            "frontier curriculum: longest hinted prompt %d tokens (%d with the completion, "
            "vllm_max_model_length %d)",
            longest,
            needed,
            int(limit),
        )

    def _frontier_gather(self, plan: list) -> list:
        """Every rank's plan, concatenated in rank order: the order TRL gathers rewards in."""
        if self.accelerator.num_processes == 1:
            return plan
        from accelerate.utils import gather_object

        return gather_object(plan)

    def _frontier_log(self, metrics: dict[str, float]) -> None:
        for key, val in metrics.items():
            self._metrics["train"][key].append(val)

    def _generate_and_score_completions(self, inputs):
        if self.frontier_curriculum == "off":
            return super()._generate_and_score_completions(inputs)
        column = self.frontier_steps_column
        # The steps are the answer key: they never reach the reward functions as a column, in
        # training or eval.
        rows = [{k: v for k, v in x.items() if k != column} for x in inputs]
        if not self.model.training:
            return super()._generate_and_score_completions(rows)

        group = self.num_generations
        if len(inputs) % group:
            raise ValueError(
                f"frontier_curriculum cannot fold {len(inputs)} rows into groups of {group}."
            )

        # One entry per group, in row order. `revealed` is NO_TRACE, HELD_OUT or a step count.
        plan: list[dict[str, Any]] = []
        for start in range(0, len(inputs), group):
            block = inputs[start : start + group]
            pids = {problem_id(x["prompt"]) for x in block}
            if len(pids) != 1:
                # A group must share one hint, or its advantages would compare rollouts that saw
                # different amounts of the solution.
                raise ValueError(
                    "frontier_curriculum found different prompts inside one rollout group; rows "
                    "must arrive as consecutive blocks of num_generations copies of a prompt."
                )
            pid = pids.pop()
            steps = as_steps(block[0].get(column))
            if not steps:
                entry = {"pid": pid, "revealed": NO_TRACE}
            elif is_held_out(pid, self.frontier_holdout_ratio):
                entry = {"pid": pid, "revealed": HELD_OUT}
            else:
                allowed = round(self.frontier_max_hint_ratio * len(steps))
                revealed = self.frontier.reveal(pid, allowed)
                if revealed:
                    hint = render_hint(steps[:revealed], self.frontier_hint_template)
                    prompt = add_hint(block[0]["prompt"], hint)
                    for row in rows[start : start + group]:
                        row["prompt"] = prompt
                entry = {
                    "pid": pid,
                    "revealed": revealed,
                    "allowed": allowed,
                    "steps": len(steps),
                }
            plan.append(entry)

        self._frontier_hint_column = [
            entry["revealed"] for entry in plan for _ in range(group)
        ]
        self._frontier_solved = None
        try:
            output = super()._generate_and_score_completions(rows)
        finally:
            self._frontier_hint_column = None

        # Rewards come back gathered over all ranks, so the plan has to be as well.
        global_plan = self._frontier_gather(plan)
        solved = self._frontier_solved
        if solved is None or len(solved) != len(global_plan) * group:
            raise RuntimeError(
                "frontier_curriculum did not see this batch's rewards: `_calculate_rewards` was "
                "not called once with one row per completion. The TRL hooks this mixin relies on "
                "have changed."
            )
        self._frontier_update(global_plan, solved)
        self._frontier_exclude_held_out(plan, output)
        return output

    def _frontier_update(self, plan, solved) -> None:
        group = self.num_generations
        lowered = 0
        hint_share, unguided, guided, held_out = [], [], [], []
        for index, entry in enumerate(plan):
            revealed = entry["revealed"]
            if revealed == NO_TRACE:
                continue
            outcomes = solved[index * group : (index + 1) * group]
            scored = [s for s in outcomes if s is not None]
            rate = sum(scored) / len(scored) if scored else None
            if revealed == HELD_OUT:
                if rate is not None:
                    held_out.append(rate)
                continue
            lowered += self.frontier.update(
                entry["pid"], revealed, outcomes, entry["allowed"]
            )
            hint_share.append(revealed / entry["steps"])
            if rate is not None:
                (guided if revealed else unguided).append(rate)
        metrics = {}
        if hint_share:
            summary = self.frontier.summary()
            metrics.update(
                {
                    # M_0(t) of the paper: share of this batch's curriculum visits at g = 0.
                    "frontier/unguided_mass": sum(share == 0 for share in hint_share)
                    / len(hint_share),
                    # mean_rho_used of the paper: share of the trace actually shown.
                    "frontier/hint_ratio": sum(hint_share) / len(hint_share),
                    "frontier/mean_help": summary["mean_help"],
                    "frontier/unlocked_frac": summary["unlocked_frac"],
                    "frontier/stalled_frac": summary["stalled_frac"],
                    "frontier/unhelped_frac": summary["unhelped_frac"],
                    "frontier/lowered": float(lowered),
                }
            )
        if unguided:
            metrics["frontier/solve_rate_unguided"] = sum(unguided) / len(unguided)
        if guided:
            metrics["frontier/solve_rate_guided"] = sum(guided) / len(guided)
        # The number memorising the training problems cannot move: unguided solve rate on
        # problems that are never trained on.
        if held_out:
            metrics["frontier/heldout_solve_rate"] = sum(held_out) / len(held_out)
        if metrics:
            self._frontier_log(metrics)

    def _frontier_exclude_held_out(self, plan, output) -> None:
        """Zero the advantage of held-out rows, so they are scored but never trained on.

        With `beta != 0` or an entropy bonus those rows still carry that regularisation gradient,
        which does not depend on whether the answer was right.
        """
        group = self.num_generations
        rows = [
            index * group + offset
            for index, entry in enumerate(plan)
            if entry["revealed"] == HELD_OUT
            for offset in range(group)
        ]
        if not rows:
            return
        output["advantages"][rows] = 0
        if _DEAD_KEY in output:
            # dynamic_sampling='mask' may now skip their forward pass as well.
            output[_DEAD_KEY][rows] = True

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        column = self._frontier_hint_column
        if column is not None:
            # Before super(): TRL flushes extra columns to the completion table right after
            # scoring.
            self._log_completion_extra(HINT_COLUMN, list(column))
        rewards = super()._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        if column is not None:
            # Read the verified reward itself, not the advantage: a group in which every rollout
            # is correct has advantage 0 and has still solved the problem.
            values = rewards[:, self._frontier_success_idx].tolist()
            self._frontier_solved = [
                None if math.isnan(value) else value >= _SOLVED for value in values
            ]
        return rewards

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.frontier_curriculum == "off" or not self.args.should_save:
            return
        folder = os.path.join(
            self._get_output_dir(trial=trial), f"checkpoint-{self.state.global_step}"
        )
        if not os.path.isdir(folder):
            return
        with open(os.path.join(folder, STATE_FILE), "w", encoding="utf-8") as handle:
            json.dump(self.frontier.state_dict(), handle)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if self.frontier_curriculum == "off" or checkpoint is None:
            return
        path = os.path.join(checkpoint, STATE_FILE)
        if not os.path.exists(path):
            logger.warning(
                "frontier curriculum: %s has no %s, so every problem restarts at full hints. "
                "Expected only when the checkpoint was written with frontier_curriculum off.",
                checkpoint,
                STATE_FILE,
            )
            return
        with open(path, encoding="utf-8") as handle:
            self.frontier.load_state_dict(json.load(handle))
        summary = self.frontier.summary()
        logger.info(
            "frontier curriculum: restored %d problems (mean help %.2f, unlocked %.2f, "
            "stalled %.2f)",
            len(self.frontier.state_dict()["problems"]),
            summary["mean_help"],
            summary["unlocked_frac"],
            summary["stalled_frac"],
        )
