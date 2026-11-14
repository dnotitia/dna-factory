# Dataset mixtures

Every trainer (`sft.py`, `grpo.py`, `opd.py`, `sdft.py`) takes either a single `dataset_name` or a
`datasets:` mixture in YAML. A mixture concatenates its entries into one training set. Each entry names a
dataset (`path`, plus optional `name` and `split`) and can carry a per-dataset `weight`.

## Per-dataset `weight`

`weight` is a size multiplier (default `1.0`), applied to each dataset before the entries are concatenated.
Use it to set the ratio between datasets instead of listing the same `path` several times:

```yaml
datasets:
  - path: org/small-specialized-data
    weight: 4.0   # four full copies
  - path: org/large-general-data
    weight: 0.5   # a random half of the rows
```

| `weight` | Effect |
|---|---|
| `1.0` (default) | The dataset is used as is. |
| integer part | That many full copies, in the original row order. |
| fractional part | A random subsample of that fraction of the rows, without replacement. |
| below `1.0` | Downsampling, e.g. `0.25` keeps a random 25% of the rows. |
| `0` | The dataset is dropped. |
| negative | Rejected with an error. |

For example, `2.5` on a 1,000-row dataset gives 2,500 rows: two full copies plus a random 500. The subsample
is seeded with the training `seed`, so a run is reproducible. The implementation is `resample_by_weight` in
`dna_factory/dnotitia_dataset_mixture.py`.

## Per-method keys

Each entry script loads the mixture itself, because the schema it needs differs per method, and adds its own
keys next to `weight`:

| Script | Extra keys | Guide |
|---|---|---|
| `sft.py` | none | [top-level README](../README.md) |
| `grpo.py` | `label` (routes rewards per dataset) | [grpo.md](grpo.md), [grpo-rewards.md](grpo-rewards.md) |
| `opd.py` | none | [opd.md](opd.md) |
| `sdft.py` | `privileged_context`, `privileged_context_file` and the context/prompt column names | [sdft.md](sdft.md) |
