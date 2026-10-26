"""Vectorized QSA indexer for Qwen4Exp (Qwen3.8-Flash-Next).

Upstream `Qwen4ExpTextQSAIndexer.forward` selects each query's key blocks in a Python loop over every
(batch, query) pair: ~20 small kernels per query, including a host-syncing `nonzero`. For one 2k-token
sequence that is 2,048 iterations in each of the 12 full-attention layers, and it makes the training step
CPU-bound (profile: 7.5 s wall vs 3.1 s of CUDA time per fwd+bwd).

Upstream blocks a query's visible keys in key order: complete blocks of `compress_ratio` keys, of which the
`block_topk` best by relu(q . k_block) summed over heads are kept, plus the trailing partial block. Whenever
every query of a batch row sees a prefix of the same list of keys (the keys any of the row's queries sees),
all those queries share that list's blocks. That holds for causal masks with right or left padding, for a
cache of earlier keys (decoding, chunked prefill) and for static caches, i.e. for everything but masks that
give queries different key sets, such as packed sequences with per-document masks. For such batches this
module packs each row's keys, pools the blocks once and selects blocks for every query of every row with one
[batch, query, block] score tensor and a masked top-k, the same pattern as transformers' DSA indexers (e.g.
GLM5-Next). The selection matches upstream for every query, up to ties in the block scores (top-k
tie-breaking). Other batches fall back to upstream's loop; in training under ZeRO-3 they raise instead,
since the loop calls k_layernorm once per query and would deadlock the ranks.

Enabled with qsa_indexer_fast: true (DnotitiaArguments; see training_runner.create_model_kwargs). This is a
monkeypatch of a reimplementation, validated against the transformers versions in VALIDATED_TRANSFORMERS by
testcase/test_qsa_indexer_fast.py; enable() warns on any other version.
"""

import logging
import math

import torch
import transformers
from transformers.integrations import is_deepspeed_zero3_enabled

logger = logging.getLogger(__name__)

# transformers versions whose Qwen4ExpTextQSAIndexer testcase/test_qsa_indexer_fast.py checked this patch against.
# Add a version here after running those tests on it.
VALIDATED_TRANSFORMERS = ("5.20.2",)

_ORIG = {}


def _shared_prefixes(visible):
    """(valid, rank, lengths) if every query sees a prefix of its row's keys, else None.

    valid [batch, kv]: keys any of the row's queries sees.
    rank [batch, kv]: a valid key's position in that list.
    lengths [batch, query]: how many of those keys each query sees.
    """
    rows = visible[:, 0]
    valid = rows.any(dim=1)
    rank = valid.cumsum(dim=-1) - 1
    lengths = rows.sum(dim=-1)
    expected = valid[:, None, :] & (rank[:, None, :] < lengths[:, :, None])
    if not torch.equal(rows, expected):
        return None
    return valid, rank, lengths


def _select_blocks(
    self, m, q, raw_keys, full_cos, full_sin, visible, valid, rank, lengths
):
    """Token selection mask [batch, query, kv] for rows whose queries see prefixes of the row's keys."""
    batch_size, seq_length = q.shape[:2]
    c, device = self.compress_ratio, q.device
    kv_length = valid.shape[-1]
    packed_length = int(lengths.max())
    num_blocks = packed_length // c

    # Original key position of each row's r-th valid key (r < packed_length); invalid keys go to a dropped slot.
    positions = torch.arange(kv_length, device=device).expand(batch_size, -1)
    slots = torch.where(valid, rank, packed_length)
    packed_positions = torch.zeros(
        batch_size, packed_length + 1, dtype=torch.long, device=device
    )
    packed_positions = packed_positions.scatter(1, slots, positions)[:, :packed_length]

    # One k_layernorm call per forward whatever the data: under ZeRO-3 every module call gathers its
    # parameters, so a data-dependent number of calls (upstream: one per query) desyncs the ranks.
    block_positions = packed_positions[:, : num_blocks * c]
    keys = raw_keys.gather(
        1, block_positions[..., None].expand(-1, -1, raw_keys.shape[-1])
    )
    pooled = self.k_layernorm(
        keys.view(batch_size, num_blocks, c, raw_keys.shape[-1])
        .float()
        .mean(dim=2)
        .to(raw_keys.dtype)
    )
    # Every query keeps all its blocks: the selection is exactly what it sees.
    if num_blocks <= self.block_topk:
        return visible[:, 0]
    starts = block_positions[:, ::c]
    block_keys = m.apply_rotary_pos_emb(
        pooled.unsqueeze(2),
        cos=full_cos.gather(1, starts[..., None].expand(-1, -1, full_cos.shape[-1])),
        sin=full_sin.gather(1, starts[..., None].expand(-1, -1, full_sin.shape[-1])),
        unsqueeze_dim=2,
    ).squeeze(2)

    # [batch, query * head, dim] @ [batch, dim, block]: one batched GEMM (matmul would broadcast the keys to
    # every query first)
    num_heads, head_dim = q.shape[2:]
    scores = torch.bmm(
        q.float().reshape(batch_size, seq_length * num_heads, head_dim),
        block_keys.float().transpose(1, 2),
    ).view(batch_size, seq_length, num_heads, num_blocks)
    scores = torch.relu(scores).sum(dim=2) / math.sqrt(self.index_head_dim)
    complete_blocks = lengths // c
    block_ids = torch.arange(num_blocks, device=device)
    scores = scores.masked_fill(block_ids >= complete_blocks[..., None], float("-inf"))
    top = scores.topk(min(self.block_topk, num_blocks), dim=-1)
    chosen = torch.zeros(
        batch_size, seq_length, num_blocks, dtype=torch.bool, device=device
    )
    chosen = chosen.scatter(-1, top.indices, top.values > float("-inf"))

    # Selection over the packed keys (selected blocks, then the trailing partial block), mapped back to the
    # original key positions. The extra last slot stays False and absorbs the invalid keys.
    packed = torch.arange(packed_length + 1, device=device)
    selected = (packed >= (complete_blocks * c)[..., None]) & (
        packed < lengths[..., None]
    )
    selected[..., : num_blocks * c] |= chosen.repeat_interleave(c, dim=-1)
    return selected.gather(-1, slots[:, None, :].expand(-1, seq_length, -1))


def _fast_forward(
    self, hidden_states, position_embeddings, attention_mask, past_key_values
):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    batch_size, seq_length, _ = hidden_states.shape
    visible = (
        attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    )
    # Decided from the mask alone, before touching the cache, so a fallback leaves it for upstream to update.
    prefixes = _shared_prefixes(visible)
    if prefixes is None:
        if self.training and is_deepspeed_zero3_enabled():
            raise RuntimeError(
                "qsa_indexer_fast: this batch gives queries of a row different key sets (e.g. packing with "
                "per-document masks), and upstream transformers' indexer loop that would handle it "
                "deadlocks under ZeRO-3. Train with packing: false."
            )
        return _ORIG["forward"](
            self, hidden_states, position_embeddings, attention_mask, past_key_values
        )

    # Same projections as upstream; the cos/sin cover every key, the queries are the last seq_length of them.
    hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
    full_cos, full_sin = position_embeddings
    current_cos, current_sin = (
        full_cos[:, -seq_length:, :],
        full_sin[:, -seq_length:, :],
    )
    qk = self.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk,
        [
            self.index_n_heads * self.index_head_dim,
            self.index_kv_heads * self.index_head_dim,
        ],
        dim=-1,
    )
    q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
    q = self.q_layernorm(q)
    q = m.apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

    selected = _select_blocks(
        self, m, q, raw_keys, full_cos, full_sin, visible, *prefixes
    ).unsqueeze(1)
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected = torch.where(selected, attention_mask.new_zeros(()), min_dtype)
    return selected


def enable():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextQSAIndexer
    if "forward" in _ORIG:
        return
    if transformers.__version__ not in VALIDATED_TRANSFORMERS:
        logger.warning(
            "qsa_indexer_fast reimplements the QSA indexer of transformers "
            f"{', '.join(VALIDATED_TRANSFORMERS)}, not {transformers.__version__}, whose indexer may have "
            "changed. Run testcase/test_qsa_indexer_fast.py, which compares it with the installed indexer, "
            "and add the version to VALIDATED_TRANSFORMERS."
        )
    _ORIG["forward"] = cls.forward
    cls.forward = _fast_forward
