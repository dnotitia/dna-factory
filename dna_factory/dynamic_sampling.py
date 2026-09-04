"""Zero-advantage group handling for GRPO (DAPO dynamic sampling).

A prompt whose rollouts all score the same has advantage 0 on every row and contributes
nothing to the loss, but TRL still runs its forward/backward.

  mask      truncate all-dead micro-batches to one token; drops their compute
  resample  keep informative groups, generate more until the batch is full

`resample` changes the gradient by design — swapping dead rows for informative ones is the
point. `mask` leaves it untouched only while a dead row truly contributes nothing: beta == 0,
no entropy bonus, no router auxiliary loss. The dapo normalizer is `num_items_in_batch`, a
scalar fixed when the batch is scored, so truncation does not rescale the surviving rows.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

MODES = ("off", "mask", "resample")


def normalize_mode(mode) -> str:
    """Resolve the configured mode, accepting the False that YAML 1.1 makes of a bare `off`."""
    if mode is False:
        return "off"
    if not isinstance(mode, str) or mode.lower() not in MODES:
        raise ValueError(f"dynamic_sampling must be one of {MODES}, got {mode!r}")
    return mode.lower()

# TRL left-pads prompts and right-pads completions. Padding on the wrong side inserts a gap
# between prompt and completion that shifts positions and breaks the stored logps.
_PROMPT_KEYS = ("prompt_ids", "prompt_mask")
_COMPLETION_KEYS = (
    "completion_ids",
    "completion_mask",
    "tool_mask",
    "old_per_token_logps",
    "ref_per_token_logps",
    "sampling_per_token_logps",
    "importance_sampling_ratio",
)
# Keys whose second axis is neither the prompt nor the completion — image patches, or a mask
# spanning prompt and completion at once. Slicing the completion axis would leave them at the
# old width, so a batch carrying one is refused rather than reshaped.
_UNSUPPORTED_KEYS = (
    "pixel_values",
    "pixel_attention_mask",
    "image_grid_thw",
    "image_sizes",
    "image_position_ids",
    "num_images",
    "token_type_ids",
    "mm_token_type_ids",
)


# Deadness is decided once, where group context exists, and carried on the batch so the
# micro-batch hook does not have to re-derive it from a single row's advantage. TRL's
# split/shuffle helpers move any batch-dim entry along with its row, so it stays aligned.
DEAD_KEY = "_dyn_dead"
# Per-row token count behind the dapo normalizer, taken before a refill replaces the rows.
LEN_KEY = "_dyn_len"


def row_lengths(batch: dict[str, Any]) -> torch.Tensor:
    """Per-row token count of the loss mask TRL builds `num_items_in_batch` from.

    That mask is `completion_mask`, intersected with `tool_mask` when a rollout supplies one so
    that tokens the model did not generate stay out of the denominator.
    """
    cm = batch["completion_mask"]
    tool = batch.get("tool_mask")
    loss_mask = cm * tool if isinstance(tool, torch.Tensor) and tool.dim() >= 2 else cm
    return loss_mask.sum(dim=1)


def dead_row_mask(advantages: torch.Tensor) -> torch.Tensor:
    """Rows with zero advantage — the fallback for when group context is unavailable.

    Only exact zeros count. A group whose rollouts all scored the same centres on its own mean,
    so its advantages are exactly 0 whatever the reward scale, and a tolerance buys nothing while
    costing real gradients: under `scale_rewards: none` an informative group can sit entirely
    below any fixed epsilon.
    """
    return advantages == 0


def groups_are_aligned(advantages: torch.Tensor, num_generations: int) -> bool:
    """Can this local batch be reshaped into whole groups?

    Advantages are mean-centred within their group, so a correctly aligned reshape has every row
    of every group summing to zero. A local batch that starts mid-group — which happens when
    per_device_train_batch_size is not a multiple of num_generations, since TRL splits a group
    across ranks — fails that check.
    """
    n = advantages.shape[0]
    if num_generations < 2 or n < num_generations or n % num_generations:
        return False
    sums = advantages.view(-1, num_generations).sum(dim=1).abs()
    scale = advantages.abs().max().clamp(min=1.0)
    return bool((sums <= 1e-3 * scale).all())


def dead_group_mask(advantages: torch.Tensor, num_generations: int) -> torch.Tensor:
    """Per-row mask marking every row of a group whose rollouts all scored the same.

    Groups, not rows, are the unit DAPO filters on. A row's own advantage is zero whenever its
    reward equals the group mean, which happens inside perfectly informative groups when the
    reward has few distinct levels — a three-tier abstention group averaging exactly 0.5 zeroes
    all of its 0.5 rows. Judging by group std keeps those rows and keeps groups intact.

    Rows arrive here in sampler order, before `_prepare_inputs` shuffles them. Falls back to
    per-row when the local batch does not line up with group boundaries.
    """
    if not groups_are_aligned(advantages, num_generations):
        return dead_row_mask(advantages)
    std = advantages.view(-1, num_generations).std(dim=1)
    return (std <= 1e-6).repeat_interleave(num_generations)


def informative_group_mask(advantages: torch.Tensor, num_generations: int) -> torch.Tensor:
    return ~dead_group_mask(advantages, num_generations)


def _is_iterable_dataset(dataset) -> bool:
    """True for streaming datasets, which have no sampler and no length."""
    if dataset is None:
        return False
    if isinstance(dataset, torch.utils.data.IterableDataset):
        return True
    try:
        from datasets import IterableDataset, IterableDatasetDict
        return isinstance(dataset, (IterableDataset, IterableDatasetDict))
    except ImportError:
        return False


def check_supported(batch: dict[str, Any]) -> None:
    for key in _UNSUPPORTED_KEYS:
        if key in batch:
            raise NotImplementedError(
                f"dynamic sampling cannot reshape a batch carrying {key!r}; "
                "multimodal and token-type inputs are not supported."
            )


def truncate_if_all_dead(inputs: dict[str, Any], stub_len: int = 1) -> tuple[dict[str, Any], bool]:
    """Shorten the completion axis of a micro-batch whose rows are all dead.

    Only when every row is dead — a rectangular tensor cannot be shortened for some rows and
    not others. With per_device_train_batch_size=1 that is every dead row.

    Deadness comes from `DEAD_KEY`, decided per group when the batch was scored; a row's own
    advantage is consulted only if that flag is missing. The returned dict is always a new one
    with `DEAD_KEY` removed, and the tensors are never edited in place: TRL buffers these
    micro-batches and hands the same objects back on every inner iteration.
    """
    stripped = {k: v for k, v in inputs.items() if k != DEAD_KEY}
    cm = inputs.get("completion_mask")
    if cm is None or cm.dim() < 2:
        return stripped, False
    flags = inputs.get(DEAD_KEY)
    if flags is not None:
        all_dead = bool(flags.bool().all())
    else:
        adv = inputs.get("advantages")
        all_dead = adv is not None and bool(dead_row_mask(adv).all())
    keep = max(int(stub_len), 1)
    if not all_dead or keep >= cm.shape[1]:
        return stripped, False
    for key in _COMPLETION_KEYS:
        t = stripped.get(key)
        if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[1] == cm.shape[1]:
            stripped[key] = t[:, :keep].clone()
    stripped["completion_mask"] = torch.zeros_like(cm[:, :keep])
    return stripped, True


def take_rows(batch: dict[str, Any], idx: torch.Tensor) -> dict[str, Any]:
    """Row-select every row-indexed tensor/list; pass scalars through."""
    out: dict[str, Any] = {}
    n = None
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 1:
            if n is None:
                n = val.shape[0]
            if val.shape[0] == n:
                out[key] = val[idx]
                continue
        if isinstance(val, list) and n is not None and len(val) == n:
            out[key] = [val[i] for i in idx.tolist()]
            continue
        out[key] = val
    return out


def concat_batches(a: dict[str, Any], b: dict[str, Any], pad_token_id: int = 0) -> dict[str, Any]:
    """Concatenate two scored batches along rows.

    Prompt tensors are left-padded, completion tensors right-padded, matching TRL's layout so
    that prompt_ids + completion_ids stays contiguous and positions are preserved.
    """
    out: dict[str, Any] = {}
    for key in a:
        if key not in b:
            continue
        va, vb = a[key], b[key]
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor) and va.dim() >= 1:
            if va.dim() >= 2 and vb.dim() >= 2 and va.shape[1] != vb.shape[1]:
                width = max(va.shape[1], vb.shape[1])
                left = key in _PROMPT_KEYS
                fill = pad_token_id if key == "prompt_ids" else 0
                va, vb = _pad_to(va, width, fill, left), _pad_to(vb, width, fill, left)
            out[key] = torch.cat([va, vb], dim=0)
        elif isinstance(va, list) and isinstance(vb, list):
            out[key] = va + vb
        else:
            out[key] = va
    return out


def _pad_to(t: torch.Tensor, width: int, value: int, left: bool) -> torch.Tensor:
    if t.shape[1] >= width:
        return t
    shape = list(t.shape)
    shape[1] = width - t.shape[1]
    pad = torch.full(shape, value, dtype=t.dtype, device=t.device)
    return torch.cat([pad, t] if left else [t, pad], dim=1)


class DynamicSamplingMixin:
    """Wires the helpers above into a GRPOTrainer.

    Mix in ahead of the trainer so its `super()` calls reach the base implementation:

        class MyTrainer(DynamicSamplingMixin, GRPOTrainer): ...
    """

    def __init__(self, *args, dynamic_sampling="off", dynamic_sampling_max_rounds=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_sampling = normalize_mode(dynamic_sampling)
        self.dynamic_sampling_max_rounds = max(int(dynamic_sampling_max_rounds), 0)
        self._resample_iter = None
        self._warned_unaligned = False
        self._warned_short = False
        self._lengths_verified = False
        if self.dynamic_sampling != "off":
            logger.info("dynamic sampling: mode=%s max_rounds=%d",
                        self.dynamic_sampling, self.dynamic_sampling_max_rounds)
            # Terms that put gradient on a zero-advantage row. Truncating such a row drops that
            # gradient, so `mask` stops being a pure compute saving whenever one is enabled.
            extra = []
            if getattr(self.args, "beta", 0.0):
                extra.append(f"beta={self.args.beta}")
            if getattr(self, "_entropy_bonus_enabled", False):
                extra.append("entropy bonus")
            if getattr(self, "aux_loss_enabled", False):
                extra.append("router auxiliary loss")
            if extra:
                logger.warning(
                    "dynamic sampling with %s: dead rows still carry gradient that masking "
                    "drops, so the result is not identical to a run with it off.",
                    ", ".join(extra),
                )
        if self.dynamic_sampling == "resample" and _is_iterable_dataset(self.train_dataset):
            # The refill dataloader is built from a RepeatSampler, and samplers do not apply to
            # IterableDataset. TRL instead wraps iterable data with repeat_iterable_dataset so each
            # prompt appears num_generations times; without that the refill batch would group
            # different prompts together and normalize their rewards as one rollout group.
            raise NotImplementedError(
                "dynamic_sampling='resample' does not support streaming datasets. "
                "Use dynamic_sampling='mask', or load the dataset without streaming."
            )

    def _log_dyn(self, metrics):
        mode = "train" if self.model.training else "eval"
        for key, val in metrics.items():
            self._metrics[mode][key].append(val)

    def _all_ranks_agree(self, value: int) -> int:
        """All-reduce sum. _generate_and_score_completions gathers internally, so every rank
        must call it the same number of times; the refill loop's exit is decided globally."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return value
        t = torch.tensor([value], device=self.accelerator.device, dtype=torch.long)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return int(t.item())

    def _get_resample_sampler(self, dataset=None):
        # repeat_count=1: the training sampler repeats each chunk num_iterations*steps_per_generation
        # times, which would hand the refill the same prompts round after round. seed+1: a different
        # permutation from the main stream, so the refill does not replay batches already trained on.
        from trl.trainer.utils import RepeatSampler

        return RepeatSampler(
            data_source=dataset if dataset is not None else self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=1,
            shuffle=self.shuffle_dataset,
            seed=(self.args.seed or 0) + 1,
        )

    def _next_resample_batch(self):
        if self._resample_iter is None:
            def cyclic():
                while True:
                    yield from self._get_dataloader(
                        dataset=self.train_dataset,
                        description="Resample",
                        batch_size=self._train_batch_size * self.args.steps_per_generation,
                        sampler_fn=self._get_resample_sampler,
                        is_training=True,
                    )
            self._resample_iter = cyclic()
        return next(self._resample_iter)

    def _keep_informative_groups(self, scored):
        adv = scored["advantages"]
        if not groups_are_aligned(adv, self.num_generations) and not self._warned_unaligned:
            logger.warning(
                "dynamic sampling: local batch of %d does not align with num_generations=%d, "
                "falling back to per-row filtering; groups may be split across ranks.",
                adv.shape[0], self.num_generations,
            )
            self._warned_unaligned = True
        keep = informative_group_mask(adv, self.num_generations)
        return take_rows(scored, keep.nonzero(as_tuple=True)[0])

    def _resample_until_full(self, scored):
        target = scored["advantages"].shape[0]
        pool = self._keep_informative_groups(scored)
        pad_id = self.processing_class.pad_token_id or 0
        rounds = 0

        while rounds < self.dynamic_sampling_max_rounds:
            have = pool["advantages"].shape[0]
            if self._all_ranks_agree(int(have >= target)) == self.accelerator.num_processes:
                break
            extra = self._score(self._next_resample_batch())
            good = self._keep_informative_groups(extra)
            if good["advantages"].shape[0]:
                pool = concat_batches(pool, good, pad_token_id=pad_id)
            rounds += 1

        # Every rank must agree on the branch: the refilled path runs a gather, and a rank that
        # took the fallback would not join it. Deciding per rank hangs the run whenever ranks end
        # up with different numbers of informative rows, which is the normal case.
        have = pool["advantages"].shape[0]
        all_filled = self._all_ranks_agree(int(have >= target)) == self.accelerator.num_processes
        if not all_filled:
            # Use the original batch; its dead rows are truncated per micro-batch in
            # _prepare_inputs and contribute nothing either way.
            if not self._warned_short:
                # Once only: on a pool this hits often it would be one line per step, and
                # dyn/refilled already records every occurrence.
                self._warned_short = True
                logger.warning(
                    "dynamic sampling: %d/%d informative rows after %d rounds, using the batch "
                    "as-is. Logged once; dyn/refilled tracks it per step.",
                    have, target, rounds,
                )
            self._log_dyn({"dyn/gen_rounds": float(rounds + 1), "dyn/refilled": 0.0})
            return scored

        out = take_rows(pool, torch.arange(target, device=pool["advantages"].device))
        # The dapo normalizer counts the loss mask over the whole batch, and the refilled batch
        # holds different rows than the one it was computed for, so it has to be recomputed from
        # the per-row counts taken when each row was scored.
        local = out.pop(LEN_KEY).sum()
        out["num_items_in_batch"] = self.accelerator.gather(local.reshape(1)).sum()
        self._log_dyn({"dyn/gen_rounds": float(rounds + 1), "dyn/refilled": 1.0})
        return out

    def _score(self, batch):
        """Score a batch and record each row's token count for the dapo normalizer."""
        scored = super()._generate_and_score_completions(batch)
        scored[LEN_KEY] = row_lengths(scored)
        self._verify_lengths(scored)
        return scored

    def _verify_lengths(self, scored):
        """Check once that LEN_KEY sums to TRL's own `num_items_in_batch`.

        A refill has to rebuild that denominator for its new rows, and getting it wrong rescales
        the whole loss. What goes into it is TRL's decision and has changed across versions, so
        it is checked against the value TRL computed for the same rows rather than assumed.
        """
        if self._lengths_verified:
            return
        self._lengths_verified = True
        want = scored.get("num_items_in_batch")
        if want is None:
            return
        got = self.accelerator.gather(scored[LEN_KEY].sum().reshape(1)).sum()
        if int(got) != int(want):
            logger.warning(
                "dynamic sampling: completion-token count %d does not match TRL's %d; "
                "the dapo normalizer will be off by that ratio after a refill.",
                int(got), int(want),
            )

    def _generate_and_score_completions(self, generation_batch):
        if self.dynamic_sampling == "off" or not self.model.training:
            return super()._generate_and_score_completions(generation_batch)
        scored = self._score(generation_batch)
        check_supported(scored)
        if self.dynamic_sampling == "resample":
            scored = self._resample_until_full(scored)
        scored.pop(LEN_KEY, None)
        dead = dead_group_mask(scored["advantages"], self.num_generations)
        scored[DEAD_KEY] = dead
        self._log_dyn({"dyn/dead_frac": dead.float().mean().item()})
        return scored

    def _prepare_inputs(self, generation_batch):
        inputs = super()._prepare_inputs(generation_batch)
        if self.dynamic_sampling == "off" or not self.model.training:
            return inputs
        inputs, did = truncate_if_all_dead(inputs)
        if did:
            self._log_dyn({"dyn/truncated_microbatches": 1.0})
        return inputs
