"""Vectorized QSA indexer for Qwen4Exp (Qwen3.8-Flash-Next) training forwards.

Upstream `Qwen4ExpTextQSAIndexer.forward` (transformers 5.16) selects each query's key blocks in a Python
loop over every (batch, query) pair: ~20 small kernels per query, including a host-syncing `nonzero`.
For one 2k-token sequence that is 2,048 iterations in each of the 12 full-attention layers, and it makes
the training step CPU-bound (profile: 7.5 s wall vs 3.1 s of CUDA time per fwd+bwd).

Its output is only a boolean mask (no gradient flows through it). When a batch row's mask is causal (possibly
right-padded) and there is no cache or the cache holds no indexer keys for the layer yet (a prompt's first
forward, which writes its keys to the cache as upstream does), i.e. every training forward and every
prefill without left padding, real query i sees
tokens 0..i, so all queries share the same complete blocks [b*c, (b+1)*c). This module then computes the pooled block keys
once and selects blocks for all queries with one [T, H, num_blocks] score tensor and a masked top-k:
  * blocks: top min(block_topk, (i+1)//c) visible blocks by relu(q_i . k_b) summed over heads;
  * tail: tokens from the last complete visible block boundary up to i are always selected;
  * if every visible block fits the budget (T <= token_budget + c - 1), the mask is all-selected.
Padding queries keep their visible mask. Any other batch (left padding, a cache with earlier keys, i.e.
decoding, a non-causal mask) falls back to upstream's loop. The selection matches upstream up to ties in the block scores (top-k tie-breaking).

Enabled with qsa_indexer_fast: true (DnotitiaArguments; see training_runner.create_model_kwargs).
"""

import math

import torch

_ORIG = {}


def _fresh_cache(past_key_values, layer_idx):
    """Whether the cache holds no indexer keys for this layer yet (the prompt's first forward)."""
    try:
        layer = past_key_values.layers[layer_idx]
    except AttributeError, IndexError, TypeError:
        return False
    if not getattr(layer, "is_indexer_initialized", False):
        return True
    keys = getattr(layer, "indexer_keys", None)
    # A static layer keeps a preallocated buffer once initialized, so only an empty dynamic one counts.
    return keys is not None and keys.numel() == 0


def _real_lengths(visible, seq_length):
    """Per batch row, its number of real tokens L if the row is causal with right padding, else -1.

    Such a row's real queries (diagonal True, positions < L) see exactly keys 0..i; its padding queries
    (diagonal False, positions >= L) may see anything, as they never reach the loss. A row without
    padding is the L == seq_length case. Returns None when the kv length differs from the query length.
    """
    if visible.shape[-1] != seq_length:
        return None
    rows = visible[:, 0]
    positions = torch.arange(seq_length, device=visible.device)
    causal = positions[None, :] <= positions[:, None]
    diagonal = rows.diagonal(dim1=-2, dim2=-1)
    lengths = diagonal.sum(dim=-1)
    real = positions[None, :] < lengths[:, None]
    contiguous = (diagonal == real).all(dim=-1)
    causal_real = ((rows == causal) | ~real[:, :, None]).flatten(1).all(dim=-1)
    ok = contiguous & causal_real
    return [int(n) if good else -1 for n, good in zip(lengths.tolist(), ok.tolist())]


def _select_causal(self, m, q, raw_keys, full_cos, full_sin):
    """Token selection mask [T, T] for one causal row. q: [T, H, D], raw_keys: [T, D]."""
    seq_length = q.shape[0]
    c, budget = self.compress_ratio, self.block_topk
    device = q.device
    positions = torch.arange(seq_length, device=device)
    causal = positions[None, :] <= positions[:, None]
    num_blocks = seq_length // c
    # k_layernorm runs exactly once per row regardless of length: under ZeRO-3 every module call gathers
    # its parameters, so a length-dependent number of calls (upstream: one per query) desyncs the ranks'
    # collectives and deadlocks them.
    keys = raw_keys[: num_blocks * c].view(num_blocks, c, raw_keys.shape[-1])
    pooled = self.k_layernorm(keys.float().mean(dim=1).to(raw_keys.dtype))
    if (
        num_blocks <= budget
    ):  # every visible block fits the budget: selection == causal mask
        return causal
    starts = positions[: num_blocks * c : c]
    block_keys = m.apply_rotary_pos_emb(
        pooled.unsqueeze(1),
        cos=full_cos.index_select(0, starts),
        sin=full_sin.index_select(0, starts),
    ).squeeze(1)
    scores = torch.einsum("thd,bd->tbh", q.float(), block_keys.float())
    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(
        self.index_head_dim
    )  # [T, num_blocks]
    visible_blocks = (positions + 1) // c  # complete visible blocks per query
    block_ids = torch.arange(num_blocks, device=device)
    valid = block_ids[None, :] < visible_blocks[:, None]
    scores = scores.masked_fill(~valid, float("-inf"))
    top = scores.topk(budget, dim=-1)
    chosen = torch.zeros(seq_length, num_blocks, dtype=torch.bool, device=device)
    chosen.scatter_(1, top.indices, top.values > float("-inf"))
    selected = torch.zeros(seq_length, seq_length, dtype=torch.bool, device=device)
    selected[:, : num_blocks * c] = chosen.repeat_interleave(c, dim=1)
    tail = (positions[None, :] >= (visible_blocks * c)[:, None]) & causal
    return selected | tail


def _fast_forward(
    self, hidden_states, position_embeddings, attention_mask, past_key_values
):
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    batch_size, seq_length, _ = hidden_states.shape
    visible = (
        attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
    )
    # Decided before touching the cache, so a fallback leaves it for upstream to update exactly once.
    fresh = past_key_values is None or _fresh_cache(past_key_values, self.layer_idx)
    # Per row: real length L if causal with right padding, -1 otherwise.
    lengths = _real_lengths(visible, seq_length) if fresh else None
    # If any row is not causal-with-right-padding (e.g. left padding, a cache with earlier keys,
    # query length != kv length), fall back to the original (slower) forward pass.
    if lengths is None or -1 in lengths:
        return _ORIG["forward"](
            self, hidden_states, position_embeddings, attention_mask, past_key_values
        )

    hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
    full_cos, full_sin = position_embeddings
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
    if past_key_values is not None:
        # Fresh cache: store this prompt's keys for the decode steps, as upstream does (it then holds
        # exactly these keys, so the returned tensor is raw_keys itself).
        raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)
    q = self.q_layernorm(q)
    q = m.apply_rotary_pos_emb(q, cos=full_cos, sin=full_sin, unsqueeze_dim=2)
    # Real queries get the causal selection over their row's L real tokens; padding queries keep their
    # visible mask (upstream would score them too, but their outputs never reach the loss).
    selected = visible.clone()
    for b, length in enumerate(lengths):
        selected[b, 0, :length, :length] = _select_causal(
            self,
            m,
            q[b, :length],
            raw_keys[b, :length],
            full_cos[b, :length],
            full_sin[b, :length],
        )
    if attention_mask.is_floating_point():
        min_dtype = torch.finfo(attention_mask.dtype).min
        selected = torch.where(selected, attention_mask.new_zeros(()), min_dtype)
    return selected


def enable():
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    cls = m.Qwen4ExpTextQSAIndexer
    if "forward" not in _ORIG:
        _ORIG["forward"] = cls.forward
        cls.forward = _fast_forward
