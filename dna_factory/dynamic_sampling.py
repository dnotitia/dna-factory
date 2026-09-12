"""Zero-advantage group handling for GRPO (DAPO dynamic sampling).

A prompt whose rollouts all score the same has advantage 0 on every row and contributes
nothing to the policy loss, but TRL still runs its forward and backward.

  mask      truncate all-dead micro-batches to a two-token stub, dropping their compute
  resample  keep informative groups and generate more until the batch is full

`resample` changes the gradient by design: replacing dead rows with informative ones is the
point. `mask` leaves it untouched only while a dead row truly contributes nothing, which
requires beta == 0, no entropy bonus and no router auxiliary loss; the trainer refuses the
other combinations rather than silently dropping their gradient. The dapo normalizer
`num_items_in_batch` is a scalar fixed when the batch is scored, so truncation does not
rescale the surviving rows.

Every collective this module adds is gated on `_all_ranks_agree`, so all ranks enter it or
none do. Ranks routinely disagree about how many of their rows are dead, so an ungated
collective here deadlocks against the parameter all-gathers in `compute_loss`.
"""

from __future__ import annotations

import logging
import os
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
# Only `resample` writes and reads it; `mask` never needs it.
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


def groups_are_aligned(advantages: torch.Tensor, num_generations: int) -> bool:
    """Can this local batch be reshaped into whole groups?

    Advantages are mean-centred within their group, so a correctly aligned reshape has every row
    of every group summing to zero. A local batch that starts mid-group — which happens when
    `per_device_train_batch_size * steps_per_generation` is not a multiple of num_generations,
    since TRL splits a group across ranks — fails that check.
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
    reward takes few distinct values. Judging by group std keeps those rows.

    Rows arrive here in sampler order, before `_prepare_inputs` shuffles them.

    There is deliberately no per-row fallback for a batch that does not fold into whole groups:
    filtering on `advantages == 0` would discard the informative rows described above, silently
    and for the rest of the run. `_require_rank_batch_holds_whole_groups` settles the divisibility
    half of the precondition at construction, so reaching the raise below means a startup
    invariant was violated. That half is a pure function of config and fires identically on every
    rank, so the process group dies together rather than hanging.
    """
    if not groups_are_aligned(advantages, num_generations):
        n = advantages.shape[0]
        raise ValueError(
            f"dynamic sampling cannot fold {n} rows into groups of {num_generations}: "
            + (f"{n} % {num_generations} != 0"
               if n % num_generations or n < num_generations
               else "the reshape divides, but the groups do not sum to zero, so rows from "
                    "different prompts are being folded into the same group")
            + ". Group-normalised advantages must sum to zero within each group; see "
              "`_require_rank_batch_holds_whole_groups` for the configuration this is supposed "
              "to have guaranteed."
        )
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


def truncate_if_all_dead(
    inputs: dict[str, Any], stub_len: int = 2, zero_mask: bool = False
) -> tuple[dict[str, Any], bool]:
    """Shrink a micro-batch whose rows are all dead to a stub, prompt and completion alike.

    Only when EVERY row is dead, because a rectangular tensor cannot be shortened for some rows
    and not others. Skipping the forward instead is not an option: deadness differs per rank, so
    a rank returning early would leave ZeRO-3's collectives without their partners, and agreeing
    globally to skip almost never fires (`dead_fraction ** num_processes`). Every rank runs a
    forward; the aim is to make it cheap.

    Deadness comes from `DEAD_KEY`, decided per group when the batch was scored. The returned
    dict is always new, with `DEAD_KEY` removed, and no tensor is edited in place: TRL buffers
    these micro-batches and hands the same objects back on every inner iteration.

    `stub_len` is 2 rather than 1 to avoid a TRL branch, not to save compute (cutting to 1 saves
    no more than cutting to 2). TRL reads a token axis of length 1 as the signal that
    `importance_sampling_level == "sequence"` and skips mask normalization, which would put the
    stub's meaningless position straight into `entropy`, and into `kl` when beta != 0. As of TRL
    1.12.0 the branch lives in the `masked_seq_mean` and `global_masked_mean` closures of
    `_compute_loss`; it has been renamed before, so check for the behaviour rather than the name.

    `zero_mask` is off by default because zeroing the mask is neither necessary nor free. Every
    row here is dead, so `advantages` is exactly 0 and `per_token_loss` is 0 whatever the mask
    says; meanwhile TRL builds the forward's attention mask as
    `cat([prompt_mask, completion_mask])`, so a zeroed completion mask drops those positions out
    of attention and leaves the logits there undefined. Turn it on only when `beta != 0`, where
    `beta * per_token_kl` is added AFTER the advantage multiply and a live mask would let it
    reach the loss.

    With the mask live, `entropy` and `sampling_logp_difference` are measured over the surviving
    tokens instead of the whole completion: a truncated sample of a real quantity rather than a
    well-formed average over an undefined forward.
    """
    stripped = {k: v for k, v in inputs.items() if k != DEAD_KEY}
    cm = inputs.get("completion_mask")
    if cm is None or cm.dim() < 2:
        return stripped, False
    flags = inputs.get(DEAD_KEY)
    if flags is None:
        # Deriving deadness from `advantages == 0` here would be wrong for the same reason the
        # per-row rule is: a row inside an informative group can sit exactly on the group mean.
        raise KeyError(
            f"{DEAD_KEY!r} is missing from this micro-batch, so its rows carry no group-level "
            f"deadness. It is set once per generation in `_generate_and_score_completions` and "
            f"rides along through TRL's shuffle and split; a batch without it did not come "
            f"through that path."
        )
    all_dead = bool(flags.bool().all())
    keep = max(int(stub_len), 1)
    if not all_dead or keep >= cm.shape[1]:
        return stripped, False
    for key in _COMPLETION_KEYS:
        t = stripped.get(key)
        if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[1] == cm.shape[1]:
            stripped[key] = t[:, :keep].clone()
    # The prompt is forwarded too: `_compute_loss` builds `input_ids` as
    # cat([prompt_ids, completion_ids]) and attends over the whole thing, keeping logits only for
    # the last `logits_to_keep` positions. Shortening the completion alone therefore still pays
    # for every prompt token. Nothing about a dead row's output is read, so the prompt carries no
    # more meaning here than the completion does.
    #
    # Prompts are LEFT-padded, so the real tokens sit at the end and the tail is what to keep.
    # Keeping the tail also leaves prompt and completion contiguous, which is the layout the
    # position ids assume.
    pm = inputs.get("prompt_mask")
    if isinstance(pm, torch.Tensor) and pm.dim() >= 2 and pm.shape[1] > keep:
        for key in _PROMPT_KEYS:
            t = stripped.get(key)
            if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[1] == pm.shape[1]:
                stripped[key] = t[:, -keep:].clone()
    if zero_mask:
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


def concat_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Undo `split_tensor_dict` for a run of consecutive chunks.

    No padding is needed, unlike `concat_batches`: these chunks came from one split of one already
    padded batch, so every row-indexed tensor shares its width. Scalars are per-batch values that
    the split copied into every chunk (`num_items_in_batch` is the one that matters), so the first
    chunk's copy is the batch's value and is carried through unchanged.
    """
    out: dict[str, Any] = {}
    first = chunks[0]
    n = first["advantages"].shape[0]
    for key, val in first.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == n:
            out[key] = torch.cat([c[key] for c in chunks], dim=0)
        elif isinstance(val, list) and len(val) == n:
            out[key] = [row for c in chunks for row in c[key]]
        else:
            out[key] = val
    return out


def split_rows(batch: dict[str, Any], num_chunks: int, chunk_size: int) -> list[dict[str, Any]]:
    """Split back into `num_chunks` dicts of `chunk_size` rows, matching split_tensor_dict's shape.

    Scalars and anything not row-indexed are shared by reference, exactly as TRL's own split does.
    """
    n = batch["advantages"].shape[0]
    out = []
    for i in range(num_chunks):
        lo, hi = i * chunk_size, (i + 1) * chunk_size
        chunk: dict[str, Any] = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[0] == n:
                chunk[key] = val[lo:hi]
            elif isinstance(val, list) and len(val) == n:
                chunk[key] = val[lo:hi]
            else:
                chunk[key] = val
        out.append(chunk)
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
        self._truncation_is_lossless = True
        self._warned_regroup = False
        self._warned_short = False
        if self.dynamic_sampling != "off":
            logger.info("dynamic sampling: mode=%s max_rounds=%d",
                        self.dynamic_sampling, self.dynamic_sampling_max_rounds)
            self._require_rank_batch_holds_whole_groups()
            # Terms that put gradient on a zero-advantage row. `mask` drops that gradient, so it
            # stops being a pure compute saving the moment one of them is on.
            extra = []
            if getattr(self.args, "beta", 0.0):
                extra.append(f"beta={self.args.beta}")
            if getattr(self, "_entropy_bonus_enabled", False):
                extra.append("entropy bonus")
            if getattr(self, "aux_loss_enabled", False):
                extra.append("router auxiliary loss")
            if extra and self.dynamic_sampling == "mask":
                raise ValueError(
                    f"dynamic_sampling='mask' is incompatible with {', '.join(extra)}. `mask` "
                    f"rests on a dead row contributing exactly 0 to the loss, which holds for the "
                    f"policy term alone: beta * per_token_kl is added AFTER the advantage "
                    f"multiply, and the entropy and router terms do not go through the advantage "
                    f"at all. A dead row still carries their gradient, so truncating it would drop "
                    f"that gradient on whatever fraction of the batch is dead. Use "
                    f"dynamic_sampling='resample', where every row surviving the refill keeps its "
                    f"full gradient, or set the terms above to zero."
                )
            # `resample` normally leaves no dead row to truncate, but its refill can come up short
            # and then it falls back to the original batch. Truncating there would drop the same
            # gradient `mask` was just refused for, so switch truncation off entirely instead.
            self._truncation_is_lossless = not extra
            if extra:
                logger.info(
                    "dynamic sampling 'resample' with %s: truncation of dead micro-batches is "
                    "disabled, because a dead row still carries gradient from these terms. A "
                    "short refill costs its full forward instead of dropping that gradient.",
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

    def _require_rank_batch_holds_whole_groups(self) -> None:
        """Refuse a configuration whose per-rank scored batch cannot be reshaped into groups.

        `dead_group_mask` folds one rank's scored batch with `view(-1, num_generations)`, and that
        rank holds `per_device_train_batch_size * steps_per_generation` rows. TRL validates only the
        GLOBAL generation batch (`generation_batch_size % num_generations == 0`), which does not
        imply the per-rank slice divides: `pd=3, spg=9, procs=7, num_generations=7` gives a global
        189 that TRL accepts while each rank holds 27, and 27 % 7 != 0. Folding that batch mixes
        rows from different prompts into one group, whose std then means nothing.

        The condition is a pure function of config, so settle it here, before any GPU time, rather
        than leaving it to `dead_group_mask` after a generation has been paid for.
        """
        pd = self.args.per_device_train_batch_size
        spg = self.args.steps_per_generation
        rows_per_rank = pd * spg
        if rows_per_rank % self.num_generations == 0:
            return
        raise ValueError(
            f"dynamic_sampling={self.dynamic_sampling!r} needs each rank's scored batch to hold "
            f"whole rollout groups, but per_device_train_batch_size({pd}) * "
            f"steps_per_generation({spg}) = {rows_per_rank} is not a multiple of "
            f"num_generations({self.num_generations}). TRL only checks the global generation batch, "
            f"so this configuration is accepted upstream and would then fold rows from different "
            f"prompts into the same group here. Raise or lower one of the three so the product "
            f"divides."
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
        keep = informative_group_mask(scored["advantages"], self.num_generations)
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
                # Once only; `dyn/refilled` records every occurrence per step.
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
        """Score a batch, recording each row's token count when the refill will need it.

        LEN_KEY is only consumed by `_resample_until_full`, so `mask` does not pay for it.
        """
        scored = super()._generate_and_score_completions(batch)
        if self.dynamic_sampling == "resample":
            scored[LEN_KEY] = row_lengths(scored)
        return scored

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
        if os.environ.get("DYN_LOG_PER_RANK"):
            # `dyn/dead_frac` goes through `self._metrics`, which only rank 0 reports, so it says
            # nothing about whether ranks disagree. Set this to see each rank's own count, which
            # is what a test of the distributed paths has to exercise.
            #
            # WARNING and not INFO on purpose: `TrainingArguments.log_level_replica` defaults to
            # "warning", so INFO is dropped on every rank but 0, precisely the ranks this line
            # exists to observe.
            logger.warning("dyn/per-rank rank=%d dead=%d/%d",
                           self.accelerator.process_index, int(dead.sum()), dead.numel())
        return scored

    def _regroup_dead_rows(self) -> bool:
        """Rearrange the freshly built buffer so dead rows land in whole micro-batches.

        Truncation needs EVERY row of a micro-batch to be dead, because a rectangular tensor cannot
        be shortened per row. TRL shuffles the scored batch before splitting it, which scatters dead
        rows, so at per_device_train_batch_size > 1 an all-dead micro-batch is a coincidence:
        `dead_fraction ** pd`, i.e. 12.5% at pd=4 and 0.4% at pd=8 for a dead fraction of 0.5.

        Sorting within one optimizer step's slice fixes that without changing what that step trains
        on. A step consumes `gradient_accumulation_steps` consecutive buffer entries, and under
        `loss_type='dapo'` its loss is `(per_token_loss * mask).sum() / num_items_in_batch` with a
        normalizer fixed at scoring time: a sum over exactly those rows, so permuting them inside
        the slice leaves the accumulated gradient identical. ('grpo' averages per row and divides
        by ga, the same uniform average, so it is invariant too.)

        Sorting across the WHOLE generation batch would not be. Dead rows would pile into the first
        slices and a step could come out entirely dead, taking an optimizer step on a zero gradient
        while the optimizer's moments decay and the LR schedule advances.

        Returns True when the buffer was rearranged.
        """
        buf = self._buffered_inputs
        ga = self.current_gradient_accumulation_steps
        if not buf or ga <= 1 or len(buf) % ga:
            # Windows that straddle generations cannot be reasoned about this way; leave them be.
            if not self._warned_regroup and buf and len(buf) % ga:
                self._warned_regroup = True
                logger.warning(
                    "dynamic sampling: steps_per_generation(%d) is not a multiple of "
                    "gradient_accumulation_steps(%d), so an optimizer step straddles two "
                    "generations and dead rows cannot be regrouped; truncation stays coincidental.",
                    len(buf), ga,
                )
            return False

        moved = False
        for start in range(0, len(buf), ga):
            window = buf[start:start + ga]
            flags = [c.get(DEAD_KEY) for c in window]
            if any(f is None for f in flags):
                continue
            dead = torch.cat([f.reshape(-1) for f in flags])
            if bool(dead.all()) or not bool(dead.any()):
                continue  # already uniform; nothing to gain
            order = torch.argsort(dead.int(), descending=True, stable=True)
            merged = concat_chunks(window)
            rows = merged["advantages"].shape[0]
            regrouped = take_rows(merged, order.to(merged["advantages"].device))
            buf[start:start + ga] = split_rows(regrouped, ga, rows // ga)
            moved = True
        return moved

    def _prepare_inputs(self, generation_batch):
        if self.dynamic_sampling == "off" or not self.model.training:
            return super()._prepare_inputs(generation_batch)

        # Read the same condition TRL uses, before it runs, so we know whether the call below
        # rebuilds the buffer. Reaching into `_buffered_inputs` afterwards is the only coupling to
        # TRL's internals here; its own body stays untouched, which is what keeps this working
        # across versions where a copied `_prepare_inputs` would silently drift.
        generate_every = self.args.steps_per_generation * self.num_iterations
        fresh = self._step % generate_every == 0 or self._buffered_inputs is None

        inputs = super()._prepare_inputs(generation_batch)

        if fresh and self.args.per_device_train_batch_size > 1 and self._truncation_is_lossless:
            if self._regroup_dead_rows():
                # super() already handed back the pre-sort chunk; re-read the sorted one.
                inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
                self._log_dyn({"dyn/regrouped_generations": 1.0})

        if self._truncation_is_lossless:
            inputs, did = truncate_if_all_dead(inputs)
            if did:
                self._log_dyn({"dyn/truncated_microbatches": 1.0})
        return inputs
