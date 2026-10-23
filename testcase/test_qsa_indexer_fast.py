"""The vectorized Qwen4Exp QSA indexer must select exactly the tokens upstream's per-query loop selects."""

from types import SimpleNamespace

import pytest
import torch

m = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

from dna_factory import qsa_indexer_fast


def _indexer():
    # Only the fields the indexer reads (the full config validates a real model's layout).
    config = SimpleNamespace(
        hidden_size=64,
        rms_norm_eps=1e-6,
        indexer_n_heads=16,  # relu-sum over many heads: an exact 0.0 score (a top-k tie) is vanishingly rare
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=16,  # 4 blocks of 4 tokens: sequences longer than ~19 tokens exercise top-k
        indexer_compress_ratio=4,
    )
    torch.manual_seed(0)
    indexer = m.Qwen4ExpTextQSAIndexer(config, layer_idx=0).eval()
    for p in indexer.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return indexer


def _rope(batch_size, seq_length, dim=8):
    """Partial-rotary cos/sin of shape [batch, seq, dim] (dim < indexer_head_dim, as in the model)."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(torch.arange(seq_length).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)[None].expand(batch_size, -1, -1)
    return emb.cos(), emb.sin()


@pytest.mark.parametrize("seq_length", [3, 16, 19, 20, 37, 64, 129])
@pytest.mark.parametrize("bool_mask", [True, False])
def test_fast_matches_upstream_on_causal_masks(seq_length, bool_mask):
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    hidden = torch.randn(2, seq_length, 64)
    rope = _rope(hidden.shape[0], seq_length)
    causal = (
        torch.ones(seq_length, seq_length, dtype=torch.bool)
        .tril()[None, None]
        .expand(2, 1, -1, -1)
    )
    mask = (
        causal
        if bool_mask
        else torch.where(causal, 0.0, torch.finfo(torch.float32).min)
    )
    with torch.no_grad():
        ref = upstream(indexer, hidden, rope, mask, None)
        out = _patched(indexer, hidden, rope, mask)
    assert out.dtype == ref.dtype and out.shape == ref.shape
    assert torch.equal(out, ref)


def test_non_causal_rows_fall_back_to_upstream():
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    seq_length = 40
    hidden = torch.randn(1, seq_length, 64)
    rope = _rope(hidden.shape[0], seq_length)
    mask = (
        torch.ones(seq_length, seq_length, dtype=torch.bool).tril()[None, None].clone()
    )
    mask[..., :5] = False  # left padding: not plain causal
    with torch.no_grad():
        ref = upstream(indexer, hidden, rope, mask, None)
        out = _patched(indexer, hidden, rope, mask)
    assert torch.equal(out, ref)


def _patched(indexer, hidden, rope, mask):
    qsa_indexer_fast._ORIG.setdefault("forward", m.Qwen4ExpTextQSAIndexer.forward)
    return qsa_indexer_fast._fast_forward(indexer, hidden, rope, mask, None)


def _right_padded_mask(lengths, seq_length):
    """Causal mask with right padding as transformers builds it: query i sees real keys j <= i."""
    positions = torch.arange(seq_length)
    causal = positions[None, :] <= positions[:, None]
    rows = [causal & (positions[None, :] < n) for n in lengths]
    return torch.stack(rows)[:, None]


@pytest.mark.parametrize("lengths", [(40, 23), (64, 64, 7), (129, 90)])
def test_right_padded_rows_match_upstream_on_real_queries(lengths):
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    seq_length = max(lengths)
    hidden = torch.randn(len(lengths), seq_length, 64)
    rope = _rope(hidden.shape[0], seq_length)
    mask = _right_padded_mask(lengths, seq_length)
    with torch.no_grad():
        ref = upstream(indexer, hidden, rope, mask, None)
        out = _patched(indexer, hidden, rope, mask)
    for b, n in enumerate(lengths):
        assert torch.equal(
            out[b, 0, :n], ref[b, 0, :n]
        )  # padding queries never reach the loss


def test_k_layernorm_called_once_per_row_whatever_the_length():
    # Under ZeRO-3 each module call gathers parameters, so the call count must not depend on the data.
    indexer = _indexer()
    calls = []
    indexer.k_layernorm.register_forward_hook(lambda *_: calls.append(1))
    for lengths in [(3, 3), (40, 23), (129, 90)]:
        calls.clear()
        seq_length = max(lengths)
        hidden = torch.randn(len(lengths), seq_length, 64)
        with torch.no_grad():
            _patched(
                indexer,
                hidden,
                _rope(len(lengths), seq_length),
                _right_padded_mask(lengths, seq_length),
            )
        assert len(calls) == len(lengths)


class _IndexerCache:
    """Minimal cache over transformers' real indexer cache layer, as `Cache.update_indexer` dispatches."""

    def __init__(self):
        from transformers.cache_utils import DynamicIndexedLayer

        self.layers = [DynamicIndexedLayer()]

    def update_indexer(self, keys, layer_idx):
        return self.layers[layer_idx].update_indexer(keys)


def test_fresh_cache_prefill_matches_upstream_and_fills_the_cache(monkeypatch):
    # use_cache=True (e.g. training without gradient checkpointing, or a generation prefill) passes an
    # empty cache: the fast path must still apply and leave the cache as upstream does.
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    lengths, seq_length = (40, 23), 40
    hidden = torch.randn(len(lengths), seq_length, 64)
    rope = _rope(len(lengths), seq_length)
    mask = _right_padded_mask(lengths, seq_length)
    ref_cache, fast_cache = _IndexerCache(), _IndexerCache()
    with torch.no_grad():
        ref = upstream(indexer, hidden, rope, mask, ref_cache)
        # Any fallback to upstream's loop fails the test.
        monkeypatch.setitem(
            qsa_indexer_fast._ORIG,
            "forward",
            lambda *_: pytest.fail("fell back to upstream"),
        )
        out = qsa_indexer_fast._fast_forward(indexer, hidden, rope, mask, fast_cache)
    for b, n in enumerate(lengths):
        assert torch.equal(out[b, 0, :n], ref[b, 0, :n])
    assert torch.equal(
        fast_cache.layers[0].indexer_keys, ref_cache.layers[0].indexer_keys
    )


def test_decode_step_with_earlier_keys_falls_back_and_updates_the_cache_once():
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    qsa_indexer_fast._ORIG.setdefault("forward", upstream)
    prompt = 40
    hidden = torch.randn(1, prompt + 1, 64)
    rope = _rope(1, prompt + 1)
    prefill_mask = _right_padded_mask((prompt,), prompt)
    decode_mask = torch.ones(1, 1, 1, prompt + 1, dtype=torch.bool)
    caches = [_IndexerCache(), _IndexerCache()]
    with torch.no_grad():
        for cache in caches:
            upstream(
                indexer,
                hidden[:, :prompt],
                (rope[0][:, :prompt], rope[1][:, :prompt]),
                prefill_mask,
                cache,
            )
        ref = upstream(indexer, hidden[:, prompt:], rope, decode_mask, caches[0])
        out = qsa_indexer_fast._fast_forward(
            indexer, hidden[:, prompt:], rope, decode_mask, caches[1]
        )
    assert torch.equal(out, ref)
    assert caches[1].layers[0].indexer_keys.shape[1] == prompt + 1
    assert torch.equal(
        caches[1].layers[0].indexer_keys, caches[0].layers[0].indexer_keys
    )
