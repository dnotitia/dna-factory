"""The host-offloaded PLE n-gram table must give exactly upstream's embeddings, and be (re)built on demand."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

m = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
safetensors_torch = pytest.importorskip("safetensors.torch")

from dna_factory import ple_host_offload

PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
NUM_SHARDS = 4
EMBEDDING_DIM = 32


def _config():
    # Only the fields the n-gram embedding reads (the full config validates a real model's layout).
    return SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=2,
        vocab_size=500,
        ngram_vocab_size_base=101,
        seed=0,
        eos_token_id=[2],
        make_ngram_vocab_size_divisible_by=8,
    )


def _write_checkpoint(model_dir, weight):
    """Store `weight` the way the released checkpoint does: row shards under `<PREFIX>.shard_<i>.weight`."""
    os.makedirs(model_dir, exist_ok=True)
    weight_map = {}
    for i, shard in enumerate(weight.chunk(NUM_SHARDS, dim=0)):
        key, file = f"{PREFIX}.shard_{i}.weight", f"model-{i:05d}.safetensors"
        safetensors_torch.save_file(
            {key: shard.contiguous()}, os.path.join(model_dir, file)
        )
        weight_map[key] = file
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f)


@pytest.fixture(scope="module")
def upstream_and_table(tmp_path_factory):
    torch.manual_seed(0)
    upstream = m.Qwen4ExpTextNGramEmbedding(
        _config(), EMBEDDING_DIM, layer_idx=1, ple_layer_index=0
    )
    upstream = upstream.to(torch.bfloat16).eval()
    weight = upstream.ngram_embedding.weight.detach()
    assert weight.shape[0] % NUM_SHARDS == 0
    model_dir = tmp_path_factory.mktemp("checkpoint")
    _write_checkpoint(model_dir, weight)
    table_dir = tmp_path_factory.mktemp("ple-host")
    # Builds the table from the shards and patches the class.
    ple_host_offload.enable(str(table_dir), model_dir=str(model_dir))
    return upstream, weight, model_dir, table_dir


def test_offloaded_lookup_matches_upstream_embedding(upstream_and_table):
    upstream, _, _, _ = upstream_and_table
    patched = m.Qwen4ExpTextNGramEmbedding(
        _config(), EMBEDDING_DIM, layer_idx=1, ple_layer_index=0
    ).eval()
    # No table parameter left for ZeRO-3 to gather.
    assert not list(patched.parameters())
    input_ids = torch.randint(0, _config().vocab_size, (2, 37))
    input_ids[0, 10] = 2  # an EOS resets the n-gram context
    with torch.no_grad():
        expected = upstream(input_ids, None)
        actual = patched(input_ids, None)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_built_table_is_the_concatenated_shards(upstream_and_table):
    _, weight, _, table_dir = upstream_and_table
    table = ple_host_offload._table(str(table_dir))
    assert torch.equal(table, weight)


def test_table_is_reused_then_rebuilt_when_the_checkpoint_changes(upstream_and_table):
    _, weight, model_dir, table_dir = upstream_and_table
    table_file = os.path.join(table_dir, "ngram_embedding.bf16")
    built = os.stat(table_file).st_mtime_ns
    ple_host_offload.ensure_table(str(table_dir), str(model_dir))
    assert os.stat(table_file).st_mtime_ns == built  # same checkpoint: no rebuild

    index = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index) as f:
        data = json.load(f)
    data["metadata"]["note"] = "changed"
    with open(index, "w") as f:
        json.dump(data, f)
    ple_host_offload.ensure_table(str(table_dir), str(model_dir))
    assert os.stat(table_file).st_mtime_ns != built  # checkpoint changed: rebuilt
    with open(os.path.join(table_dir, "meta.json")) as f:
        assert json.load(f)["rows"] == weight.shape[0]
