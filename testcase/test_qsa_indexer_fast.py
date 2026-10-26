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


def _patched(indexer, hidden, rope, mask, cache=None):
    qsa_indexer_fast._ORIG.setdefault("forward", m.Qwen4ExpTextQSAIndexer.forward)
    return qsa_indexer_fast._fast_forward(indexer, hidden, rope, mask, cache)


def _no_fallback(monkeypatch):
    # Any fallback to upstream's loop fails the test.
    monkeypatch.setitem(
        qsa_indexer_fast._ORIG,
        "forward",
        lambda *_: pytest.fail("fell back to upstream"),
    )


def _padded_mask(lengths, seq_length, left=False):
    """Causal mask with right (or left) padding as transformers builds it: query i sees real keys j <= i."""
    positions = torch.arange(seq_length)
    causal = positions[None, :] <= positions[:, None]
    rows = []
    for n in lengths:
        real = positions >= seq_length - n if left else positions < n
        rows.append(causal & real[None, :])
    return torch.stack(rows)[:, None]


def _document_mask(boundaries, seq_length):
    """Causal mask of packed documents: a query sees only the earlier keys of its own document."""
    doc = torch.zeros(seq_length, dtype=torch.long)
    for start in boundaries:
        doc[start:] += 1
    positions = torch.arange(seq_length)
    causal = positions[None, :] <= positions[:, None]
    return (causal & (doc[None, :] == doc[:, None]))[None, None]


@pytest.mark.parametrize("left", [False, True])
@pytest.mark.parametrize("lengths", [(40, 23), (64, 64, 7), (129, 90), (5, 2)])
def test_padded_rows_match_upstream_on_every_query(lengths, left, monkeypatch):
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    seq_length = max(lengths)
    hidden = torch.randn(len(lengths), seq_length, 64)
    rope = _rope(hidden.shape[0], seq_length)
    mask = _padded_mask(lengths, seq_length, left=left)
    with torch.no_grad():
        ref = upstream(indexer, hidden, rope, mask, None)
        _no_fallback(monkeypatch)
        out = _patched(indexer, hidden, rope, mask)
    assert torch.equal(out, ref)


def test_rows_padded_on_different_sides_match_upstream(monkeypatch):
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    seq_length = 50
    mask = torch.cat(
        [
            _padded_mask((50,), seq_length),
            _padded_mask((31,), seq_length),
            _padded_mask((44,), seq_length, left=True),
        ]
    )
    hidden = torch.randn(3, seq_length, 64)
    rope = _rope(3, seq_length)
    with torch.no_grad():
        ref = upstream(indexer, hidden, rope, mask, None)
        _no_fallback(monkeypatch)
        out = _patched(indexer, hidden, rope, mask)
    assert torch.equal(out, ref)


def test_document_masks_fall_back_to_upstream():
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    seq_length = 60
    hidden = torch.randn(1, seq_length, 64)
    rope = _rope(1, seq_length)
    mask = _document_mask([25], seq_length)
    assert qsa_indexer_fast._shared_prefixes(mask) is None
    with torch.no_grad():
        ref = upstream(indexer, hidden, rope, mask, None)
        out = _patched(indexer, hidden, rope, mask)
    assert torch.equal(out, ref)


def test_k_layernorm_called_once_per_forward_whatever_the_data():
    # Under ZeRO-3 each module call gathers parameters, so the call count must not depend on the data.
    indexer = _indexer()
    calls = []
    indexer.k_layernorm.register_forward_hook(lambda *_: calls.append(1))
    for lengths, left in [((3, 3), False), ((40, 23), False), ((129, 90), True)]:
        calls.clear()
        seq_length = max(lengths)
        hidden = torch.randn(len(lengths), seq_length, 64)
        with torch.no_grad():
            _patched(
                indexer,
                hidden,
                _rope(len(lengths), seq_length),
                _padded_mask(lengths, seq_length, left=left),
            )
        assert len(calls) == 1


class _IndexerCache:
    """Minimal cache over transformers' real indexer cache layer, as `Cache.update_indexer` dispatches."""

    def __init__(self):
        from transformers.cache_utils import DynamicIndexedLayer

        self.layers = [DynamicIndexedLayer()]

    def update_indexer(self, keys, layer_idx):
        return self.layers[layer_idx].update_indexer(keys)


@pytest.mark.parametrize("left", [False, True])
def test_cached_prefill_and_decode_match_upstream_and_fill_the_cache(left, monkeypatch):
    # Prefill, a multi-token chunk and single-token decode steps, all with a cache of earlier keys.
    indexer = _indexer()
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    lengths, total = (40, 23), 52
    chunks = [(0, 40), (40, 48), (48, 49), (49, 50), (50, 52)]
    hidden = torch.randn(len(lengths), total, 64)
    rope = _rope(len(lengths), total)
    # Padding (to the prompt length 40) stays in the cache; the tokens generated after it are real.
    prompt_mask = _padded_mask(lengths, 40, left=left)[:, 0, -1]
    real = torch.cat(
        [prompt_mask, torch.ones(len(lengths), total - 40, dtype=torch.bool)], dim=-1
    )
    positions = torch.arange(total)
    full_mask = ((positions[None, :] <= positions[:, None])[None] & real[:, None, :])[
        :, None
    ]
    ref_cache, fast_cache = _IndexerCache(), _IndexerCache()
    with torch.no_grad():
        for start, end in chunks:
            args = (
                hidden[:, start:end],
                (rope[0][:, :end], rope[1][:, :end]),
                full_mask[:, :, start:end, :end],
            )
            ref = upstream(indexer, *args, ref_cache)
            with monkeypatch.context() as patch:
                _no_fallback(patch)
                out = qsa_indexer_fast._fast_forward(indexer, *args, fast_cache)
            assert torch.equal(out, ref)
    assert torch.equal(
        fast_cache.layers[0].indexer_keys, ref_cache.layers[0].indexer_keys
    )


def test_fallback_raises_in_training_under_zero3(monkeypatch):
    # Upstream's loop calls k_layernorm once per query, which deadlocks ZeRO-3 ranks: fail loudly instead.
    indexer = _indexer().train()
    monkeypatch.setattr(qsa_indexer_fast, "is_deepspeed_zero3_enabled", lambda: True)
    qsa_indexer_fast._ORIG.setdefault("forward", m.Qwen4ExpTextQSAIndexer.forward)
    seq_length = 40
    hidden = torch.randn(1, seq_length, 64)
    with torch.no_grad(), pytest.raises(RuntimeError, match="ZeRO-3"):
        qsa_indexer_fast._fast_forward(
            indexer,
            hidden,
            _rope(1, seq_length),
            _document_mask([15], seq_length),
            None,
        )


def _fresh_enable(monkeypatch, validated):
    import transformers

    monkeypatch.setattr(qsa_indexer_fast, "_ORIG", {})
    monkeypatch.setattr(
        qsa_indexer_fast,
        "VALIDATED_TRANSFORMERS",
        (transformers.__version__,) if validated else ("0.0.0",),
    )
    monkeypatch.setattr(
        m.Qwen4ExpTextQSAIndexer, "forward", m.Qwen4ExpTextQSAIndexer.forward
    )


@pytest.mark.parametrize("validated", [True, False])
def test_enable_patches_and_warns_only_on_an_unvalidated_version(
    validated, monkeypatch, caplog
):
    upstream = m.Qwen4ExpTextQSAIndexer.forward
    _fresh_enable(monkeypatch, validated=validated)
    with caplog.at_level("WARNING"):
        qsa_indexer_fast.enable()
    assert m.Qwen4ExpTextQSAIndexer.forward is qsa_indexer_fast._fast_forward
    assert qsa_indexer_fast._ORIG["forward"] is upstream
    assert ("VALIDATED_TRANSFORMERS" in caplog.text) is not validated
