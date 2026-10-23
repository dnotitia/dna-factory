"""Host-memory offload of Qwen4Exp's PLE n-gram embedding table (Qwen3.8-Flash-Next).

The table (`...ple.ple_embedding.ngram_embedding`, ~320M x 160 bf16 = 95 GiB, ~29% of the model) is a
pure lookup: its row ids are a hash of the input token n-grams, so they are known as soon as the
batch is, and each 8k-token sequence touches only ~130k rows (~42 MB). Keeping it as a GPU parameter
breaks ZeRO-3 (loading gathers the 95 GiB parameter on one GPU and OOMs; every forward would gather
it again) and pins 95 GiB on one GPU under layer-wise model parallelism. It is frozen under LoRA.

With ple_host_table: <dir> (DnotitiaArguments), <dir> holds the table as one flat file (ngram_embedding.bf16 +
meta.json). If it is missing or was built from another checkpoint, the first process on the node builds it
from the checkpoint's shards; the other processes wait on a file lock. Then:
  * the module is built without its nn.Embedding, so ZeRO-3 never creates or partitions it;
  * the table is an np.memmap of the flat file, so all processes on the node share one copy of the
    pages instead of each holding 95 GiB;
  * forward computes the ids exactly as upstream and gathers the rows on the CPU, then copies them
    to the activations' device (pinned, non-blocking);
  * the checkpoint's 128 shard keys are dropped from the weight conversion and ignored on load.
The memmap is only fast while the file is in the page cache: rows are random small reads, so when the
pages are evicted, every step waits on storage. Put the table on /dev/shm (RAM, never evicted); on first use the file is also prefetched
with posix_fadvise(WILLNEED) as a best effort.
Written against transformers 5.16 (Qwen4ExpTextNGramEmbedding); the __init__ below mirrors upstream's
minus the embedding, and forward reuses upstream's by swapping in a lookup shim.
"""

import fcntl
import json
import logging
import math
import os
import re

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)

_TABLE = {}
_TABLE_FILE = "ngram_embedding.bf16"
_SHARD_KEY = re.compile(
    r"^(model\.language_model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding)\.shard_(\d+)\.weight$"
)


def _source(model_dir):
    """Identity of the checkpoint a table is built from (rebuild when it changes)."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    stat = os.stat(index)
    return {
        "model_dir": os.path.abspath(model_dir),
        "index_size": stat.st_size,
        "index_mtime_ns": stat.st_mtime_ns,
    }


def build_table(model_dir, out_dir):
    """Concatenate the checkpoint's PLE n-gram shards (dim 0, natural order, as transformers does) into one
    raw row-major bf16 file. Writes to temporary names and renames, so an interrupted build leaves no table."""
    from safetensors import safe_open

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(
        (int(m.group(2)), key, m.group(1))
        for key in weight_map
        if (m := _SHARD_KEY.match(key))
    )
    if [s[0] for s in shards] != list(range(len(shards))) or len(
        {s[2] for s in shards}
    ) != 1:
        raise ValueError(
            f"{model_dir}: expected one PLE table stored as contiguous shard_0..shard_N"
        )

    rows, dim = 0, None
    tmp = os.path.join(out_dir, _TABLE_FILE + ".tmp")
    with open(tmp, "wb") as f:
        for idx, key, _ in shards:
            with safe_open(os.path.join(model_dir, weight_map[key]), "pt") as handle:
                shard = handle.get_tensor(key)
            if shard.dtype != torch.bfloat16 or (
                dim is not None and shard.shape[1] != dim
            ):
                raise ValueError(
                    f"{key}: unexpected dtype/shape {shard.dtype} {tuple(shard.shape)}"
                )
            dim = shard.shape[1]
            f.write(shard.contiguous().view(torch.int16).numpy().tobytes())
            rows += shard.shape[0]
            if idx % 16 == 0:
                logger.info(f"PLE table: shard {idx}/{len(shards)}, {rows} rows")
    os.replace(tmp, os.path.join(out_dir, _TABLE_FILE))
    meta = {
        "param": shards[0][2] + ".weight",
        "rows": rows,
        "dim": dim,
        "dtype": "bfloat16",
        "shards": len(shards),
        "source": _source(model_dir),
    }
    with open(os.path.join(out_dir, "meta.json.tmp"), "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(
        os.path.join(out_dir, "meta.json.tmp"), os.path.join(out_dir, "meta.json")
    )
    return meta


def _is_current(out_dir, model_dir):
    try:
        with open(os.path.join(out_dir, "meta.json")) as f:
            meta = json.load(f)
        size = os.path.getsize(os.path.join(out_dir, _TABLE_FILE))
    except OSError, ValueError:
        return False
    return size == meta["rows"] * meta["dim"] * 2 and meta.get("source") == _source(
        model_dir
    )


def ensure_table(out_dir, model_dir):
    """Build the table under out_dir unless a complete one from this checkpoint is already there. Processes on
    the same node serialize on a lock file, so exactly one builds and the rest reuse its result."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, ".build.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if not _is_current(out_dir, model_dir):
                logger.warning(
                    f"Building the PLE host table in {out_dir} from {model_dir}"
                )
                build_table(model_dir, out_dir)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _prefetch(file):
    """Ask the kernel to read the whole table into the page cache (asynchronous; a hint, not a pin)."""
    if not hasattr(os, "posix_fadvise"):
        return
    fd = os.open(file, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_WILLNEED)
    finally:
        os.close(fd)


def _table(path):
    if path not in _TABLE:
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        file = os.path.join(path, _TABLE_FILE)
        _prefetch(file)
        mm = np.memmap(
            file, dtype=np.int16, mode="r", shape=(meta["rows"], meta["dim"])
        )
        _TABLE[path] = torch.from_numpy(mm).view(
            torch.bfloat16
        )  # read-only; never written
    return _TABLE[path]


class _HostLookup(nn.Module):
    """Parameter-free stand-in for the nn.Embedding; upstream forward calls it with the id tensor."""

    def __init__(self, path, num_embeddings, embedding_dim):
        super().__init__()
        self.path, self.num_embeddings, self.embedding_dim = (
            path,
            num_embeddings,
            embedding_dim,
        )
        # upstream does `ids.to(self.ngram_embedding.weight.device)`: keep ids on CPU for the gather.
        # Explicit device: from_pretrained builds modules under a meta default device.
        self.weight = torch.empty(0, device="cpu")

    def forward(self, ids):
        table = _table(self.path)
        assert table.shape == (self.num_embeddings, self.embedding_dim), (
            tuple(table.shape),
            self.num_embeddings,
        )
        rows = table.index_select(0, ids.reshape(-1).cpu())
        return rows.view(*ids.shape, self.embedding_dim)


def _materialize_id_buffers(module, m, device):
    """Rebuild the id-hashing buffers if loading left them on meta.

    They are deterministic in the config (upstream `_init_weights` fills them with these same calls), so
    recomputing them is exact. They stay meta here because the module no longer owns a parameter.
    """
    values = {
        "layer_multipliers": m._build_layer_multipliers(
            module.unigram_vocab_size,
            module.ngram_size,
            module.ple_layer_index,
            module.seed,
        ),
        "ngram_heads_vocab_sizes": torch.tensor(
            module.head_vocab_sizes, dtype=torch.long
        ),
        "ngram_heads_offsets": torch.tensor(module.head_offsets, dtype=torch.long),
    }
    for name, value in values.items():
        if getattr(module, name).is_meta:
            module._buffers[name] = value.to(device)


def _patched_forward(orig_forward, m):
    def forward(self, input_ids, past_key_values):
        _materialize_id_buffers(self, m, input_ids.device)
        out = orig_forward(
            self, input_ids, past_key_values
        )  # CPU rows (ids were sent to weight.device)
        if out.device != input_ids.device:
            out = out.pin_memory().to(input_ids.device, non_blocking=True)
        return out

    return forward


def enable(path, model_dir=None):
    """Patch transformers' qwen4_exp before the model is constructed. With model_dir, first make sure the
    table under `path` exists and matches that checkpoint (building it if needed)."""
    import transformers.conversion_mapping as cm
    from transformers.models.qwen4_exp import modeling_qwen4_exp as m

    if model_dir is not None:
        ensure_table(path, model_dir)
    elif not os.path.exists(os.path.join(path, _TABLE_FILE)):
        raise FileNotFoundError(
            f"No PLE host table in {path}; pass model_dir to build it"
        )

    cls = m.Qwen4ExpTextNGramEmbedding
    if getattr(cls, "_dna_host_offload", False):
        return

    def __init__(self, config, embedding_dim, layer_idx, ple_layer_index=0):
        # Mirrors upstream __init__ except the final `self.ngram_embedding = nn.Embedding(...)`.
        nn.Module.__init__(self)
        self.layer_idx = layer_idx
        self.ngram_size = config.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.unigram_vocab_size = config.vocab_size
        self.ngram_vocab_size_base = config.ngram_vocab_size_base
        head_dim_per_ngram = embedding_dim // self.ngram_heads
        self.seed = config.seed
        self.eos_token_id = (
            config.eos_token_id[0]
            if isinstance(config.eos_token_id, list)
            else config.eos_token_id
        )
        self.head_vocab_sizes, self.head_offsets, self.total_vocab_size = [], [], 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = m._find_nth_prime_after(
                self.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            self.head_vocab_sizes.append(size)
            self.head_offsets.append(self.total_vocab_size)
            self.total_vocab_size += size
        self.layer_multipliers = nn.Buffer(
            m._build_layer_multipliers(
                self.unigram_vocab_size,
                self.ngram_size,
                self.ple_layer_index,
                self.seed,
            )
        )
        self.ngram_heads_vocab_sizes = nn.Buffer(
            torch.tensor(self.head_vocab_sizes, dtype=torch.long)
        )
        self.ngram_heads_offsets = nn.Buffer(
            torch.tensor(self.head_offsets, dtype=torch.long)
        )
        divisor = config.make_ngram_vocab_size_divisible_by
        padded_vocab_size = math.ceil(self.total_vocab_size / divisor) * divisor
        self.ngram_embedding = _HostLookup(path, padded_vocab_size, head_dim_per_ngram)

    cls.__init__ = __init__
    cls.forward = _patched_forward(cls.forward, m)
    cls._dna_host_offload = True

    # Drop the shard-concatenation converter (it would materialize the 95 GiB table in every process)
    # from the cached mapping itself (get_checkpoint_conversion_mapping returns deep copies).
    cm.get_checkpoint_conversion_mapping("qwen4_exp_text")
    cache = cm._checkpoint_conversion_mapping_cache
    for key, convs in cache.items():
        if key.startswith("qwen4_exp") and convs:
            cache[key] = [
                c
                for c in convs
                if "ngram_embedding" not in str(getattr(c, "source_patterns", ""))
            ]
    for model_cls in (
        m.Qwen4ExpPreTrainedModel,
        m.Qwen4ExpTextModel,
        m.Qwen4ExpModel,
        m.Qwen4ExpForCausalLM,
        m.Qwen4ExpForConditionalGeneration,
    ):
        ignore = set(
            getattr(model_cls, "_keys_to_ignore_on_load_unexpected", None) or []
        )
        model_cls._keys_to_ignore_on_load_unexpected = ignore | {
            r"ngram_embedding\.shard_\d+\.weight"
        }
