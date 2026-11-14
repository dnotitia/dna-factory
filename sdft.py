"""
Self-Distillation Fine-Tuning (SDFT) training script for DNA Factory.

Method-specific logic only (prompt + per-row privileged-context dataset mixture, trainer wiring); the shared
setup/train/save flow lives in dna_factory/training_runner.py.
Key SDFT-specific difference: the teacher is the model itself, shown a privileged context (e.g. a system prompt)
that the student does not see, and the student is trained on its own samples to match that teacher.
"""

import logging
from dataclasses import dataclass, field

from datasets import Dataset
from trl import ModelConfig, ScriptArguments, get_quantization_config
from trl.experimental.sdft import SDFTConfig

from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetConfig,
    WeightedDatasetMixtureConfig,
)
from dna_factory.dnotitia_dataset_mixture import load_mixture as load_mixture_datasets
from dna_factory.dnotitia_sdft_trainer import DnotitiaSDFTTrainer
from dna_factory.training_runner import (
    TrainingSpec,
    cli_main,
    resolve_trust_remote_code,
    run_training,
)

logger = logging.getLogger(__name__)


@dataclass
class ContextDatasetConfig(WeightedDatasetConfig):
    """WeightedDatasetConfig + the privileged context the teacher sees on every row of this dataset."""

    privileged_context: str | None = field(
        default=None,
        metadata={
            "help": "Text the teacher sees but the student does not, for every row of this dataset. "
            "Overrides a `privileged_context` column. Empty/unset with no column = anchor rows."
        },
    )
    privileged_context_file: str | None = field(
        default=None,
        metadata={
            "help": "Path to a text file read once and used as `privileged_context` (for long system prompts)."
        },
    )
    privileged_context_column: str = field(
        default="privileged_context",
        metadata={
            "help": "Column holding a per-row privileged context (e.g. a reference answer or a per-row system "
            "prompt). Ignored when `privileged_context` / `privileged_context_file` is set."
        },
    )
    prompt_column: str | None = field(
        default=None,
        metadata={
            "help": "Column holding the student prompt (a string or a list of messages). Default: `prompt`, "
            "else `messages`."
        },
    )


@dataclass
class ContextDatasetMixtureConfig(WeightedDatasetMixtureConfig):
    """WeightedDatasetMixtureConfig whose entries are ContextDatasetConfig."""

    datasets: list[ContextDatasetConfig] = field(
        default_factory=list,
        metadata={
            "help": "List of (weighted, context-carrying) dataset configurations to include in the mixture."
        },
    )

    def __post_init__(self):
        for idx, dataset in enumerate(self.datasets):
            if isinstance(dataset, dict):
                self.datasets[idx] = ContextDatasetConfig(**dataset)


def _context_to_text(value):
    """Coerce a `privileged_context` cell to text (None -> empty, message lists -> joined contents)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            content = item.get("content", "") if isinstance(item, dict) else item
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            if content:
                parts.append(str(content))
        return "\n".join(parts)
    return str(value)


def normalize_dataset_for_sdft(
    dataset,
    privileged_context=None,
    context_column="privileged_context",
    prompt_column=None,
):
    """Normalize one dataset to the `prompt` (list of messages) + `privileged_context` (str) schema.

    The student prompt comes from `prompt_column` when given, else `prompt` (a string becomes one user turn), else
    `messages` (everything before the last assistant turn; the reference answer is discarded since the student
    writes its own). The context is `privileged_context` when given (same text on every row), else the per-row
    `context_column`, else empty (anchor rows).
    """
    feats = dataset.column_names
    if prompt_column is not None and prompt_column not in feats:
        raise ValueError(
            f"prompt_column `{prompt_column}` not in the dataset (columns: {feats})."
        )
    if prompt_column is None and "prompt" not in feats and "messages" not in feats:
        raise ValueError(
            f"Dataset has neither `prompt` nor `messages` (columns: {feats}); set `prompt_column`."
        )
    if (
        privileged_context is None
        and context_column != "privileged_context"
        and context_column not in feats
    ):
        raise ValueError(
            f"privileged_context_column `{context_column}` not in the dataset (columns: {feats})."
        )
    has_context_col = context_column in feats
    prompt_key = prompt_column or ("prompt" if "prompt" in feats else None)

    def convert(example):
        if prompt_key is not None and example[prompt_key] is not None:
            p = example[prompt_key]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
        else:
            messages = example["messages"]
            assistant_idxs = [
                i for i, m in enumerate(messages) if m["role"] == "assistant"
            ]
            msgs = messages[: assistant_idxs[-1]] if assistant_idxs else messages
        if privileged_context is not None:
            context = privileged_context
        elif has_context_col:
            context = _context_to_text(example[context_column])
        else:
            context = ""
        return {
            "prompt": [{"role": m["role"], "content": m["content"]} for m in msgs],
            "privileged_context": context,
        }

    return dataset.map(convert, remove_columns=dataset.column_names)


def _resolve_context(dataset_config):
    """Return the YAML-given context for a dataset entry (inline text or file), or None."""
    if dataset_config.privileged_context and dataset_config.privileged_context_file:
        raise ValueError(
            f"{dataset_config.path}: set only one of `privileged_context` and `privileged_context_file`."
        )
    if dataset_config.privileged_context_file:
        with open(dataset_config.privileged_context_file, encoding="utf-8") as f:
            return f.read().strip()
    return dataset_config.privileged_context


def _normalize_entry_for_sdft(dataset, dataset_config):
    """Resolve the entry's privileged context, bring it to prompt + privileged_context, and log it."""
    context = _resolve_context(dataset_config)
    context_column = dataset_config.privileged_context_column
    per_row = context is None and context_column in dataset.column_names
    dataset = normalize_dataset_for_sdft(
        dataset, context, context_column, dataset_config.prompt_column
    )
    if context:
        logger.info(
            f"  privileged context: same text on every row ({len(context)} chars)"
        )
    elif per_row:
        if isinstance(dataset, Dataset):  # counting rows needs a materialized dataset
            n_empty = sum(1 for c in dataset["privileged_context"] if not c.strip())
            logger.info(
                f"  privileged context: per row from column `{context_column}` "
                f"({n_empty}/{len(dataset)} rows empty = anchor rows)"
            )
        else:
            logger.info(f"  privileged context: per row from column `{context_column}`")
    else:
        logger.info("  privileged context: none (anchor rows)")
    return dataset


def get_dataset_with_context(mixture_config, seed=42):
    """Load mixture datasets, attach each one's privileged context, apply weights, and concatenate."""
    return load_mixture_datasets(
        mixture_config, seed=seed, normalize=_normalize_entry_for_sdft
    )


def setup_training_args(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    if dnotitia_args.use_unsloth:
        raise ValueError("`use_unsloth` is not supported by sdft.py.")
    # Models passed as strings; init kwargs go through training config.
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": resolve_trust_remote_code(training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        training_args.model_init_kwargs["quantization_config"] = quantization_config


def load_models(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    return {"model": model_args.model_name_or_path}


def load_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    return get_dataset_with_context(dataset_mixture_args, seed=training_args.seed)


def postprocess_dataset(dataset, ctx, train_logger):
    """`dataset_name` runs: bring each split to the prompt + privileged_context schema (idempotent for mixtures)."""
    for split in list(dataset.keys()):
        cols = dataset[split].column_names
        if cols == ["prompt", "privileged_context"]:
            continue
        if "privileged_context" not in cols:
            train_logger.warning(
                f"Split `{split}` has no `privileged_context` column: every row is an anchor row "
                "(teacher prompt = student prompt). Use `datasets:` with `privileged_context` to set one."
            )
        dataset[split] = normalize_dataset_for_sdft(dataset[split])
    return dataset


def extra_trainer_kwargs(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    return {"teacher_context_placement": dnotitia_args.teacher_context_placement}


SPEC = TrainingSpec(
    name="SDFT",
    output_dir_tag="SDFT",
    trainer_module="dna_factory.dnotitia_sdft_trainer",
    script_file=__file__,
    defaults_yaml="configs/_defaults-SDFT.yaml",
    dataclass_types=(
        ScriptArguments,
        SDFTConfig,
        ModelConfig,
        ContextDatasetMixtureConfig,
        DnotitiaArguments,
    ),
    set_dataset_num_proc=False,
    # Disable DeepGEMM: vLLM 0.20.0 warmup crashes on Hopper/Blackwell; unneeded for bf16.
    extra_env={"VLLM_USE_DEEP_GEMM": "0"},
    setup_training_args=setup_training_args,
    load_models=load_models,
    load_mixture=load_mixture,
    postprocess_dataset=postprocess_dataset,
    trainer_cls=DnotitiaSDFTTrainer,
    extra_trainer_kwargs=extra_trainer_kwargs,
)


def main(
    script_args,
    training_args,
    model_args,
    dataset_mixture_args,
    dnotitia_args,
    user_specified_args=None,
):
    return run_training(
        SPEC,
        script_args,
        training_args,
        model_args,
        dataset_mixture_args,
        dnotitia_args,
        user_specified_args,
    )


if __name__ == "__main__":
    cli_main(SPEC)
