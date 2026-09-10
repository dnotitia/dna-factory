"""On-policy distillation script. Shared setup/train/save flow lives in dna_factory/training_runner.py."""

import logging
import os

from trl import (
    DistillationConfig,
    ModelConfig,
    ScriptArguments,
    get_quantization_config,
)

from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetMixtureConfig,
    resample_by_weight,
)
from dna_factory.dnotitia_distillation_trainer import DnotitiaDistillationTrainer
from dna_factory.dnotitia_trainer_commons import resolve_trust_remote_code
from dna_factory.training_runner import TrainingSpec, cli_main, run_training

logger = logging.getLogger(__name__)


def _normalize_dataset_for_distillation(dataset):
    """Normalize one dataset to prompt-only schema (assistant turn discarded)."""
    feats = dataset.features
    if "messages" in feats:

        def split_messages(example):
            messages = example["messages"]
            assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            prompt_msgs = messages[:assistant_idxs[-1]] if assistant_idxs else messages
            return {"prompt": [{"role": m["role"], "content": m["content"]} for m in prompt_msgs]}

        return dataset.map(split_messages, remove_columns=dataset.column_names)
    elif "prompt" in feats:

        def normalize_prompt(example):
            p = example["prompt"]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
            return {"prompt": [{"role": m["role"], "content": m["content"]} for m in msgs]}

        return dataset.map(normalize_prompt, remove_columns=dataset.column_names)
    else:
        raise ValueError(
            f"Dataset has neither `messages` nor `prompt` (columns: {dataset.column_names}); "
            "cannot build a prompt-only distillation schema."
        )


def get_dataset_with_schema_alignment(mixture_config, seed=42):
    """Load mixture datasets, normalize each to prompt schema, apply weight, and concatenate."""
    import datasets as ds
    from datasets import DatasetDict, concatenate_datasets

    datasets_list = []
    for dataset_config in mixture_config.datasets:
        path = dataset_config.path
        logger.info(f"Loading dataset for mixture: {path} (config name: {dataset_config.name})")
        if os.path.isdir(path):
            dataset = ds.load_from_disk(path)
            if isinstance(dataset, DatasetDict):
                dataset = dataset[dataset_config.split or "train"]
        else:
            dataset = ds.load_dataset(path=path, name=dataset_config.name, split=dataset_config.split)

        if dataset_config.columns is not None:
            dataset = dataset.select_columns(dataset_config.columns)

        dataset = _normalize_dataset_for_distillation(dataset)

        weight = getattr(dataset_config, "weight", 1.0)
        if weight != 1.0:
            n_before = len(dataset)
            dataset = resample_by_weight(dataset, weight, seed=seed)
            logger.info(f"  weight={weight}: {n_before} -> {len(dataset)} examples")

        datasets_list.append(dataset)

    combined = concatenate_datasets(datasets_list)
    logger.info(f"Created distillation dataset mixture with {len(combined)} examples")
    return DatasetDict({"train": combined})


def setup_training_args(script_args, training_args, model_args, dnotitia_args, ctx, train_logger):
    if not training_args.teacher_model_name_or_path:
        raise ValueError(
            "On-policy distillation requires a teacher. Set `teacher_model_name_or_path` (it must share the "
            "student's vocabulary)."
        )

    # Models passed as strings; init kwargs go through training config.
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": resolve_trust_remote_code(model_args, training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    training_args.teacher_model_init_kwargs = {
        "trust_remote_code": resolve_trust_remote_code(model_args, training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    ctx["quantization_config"] = get_quantization_config(model_args)


def load_models(script_args, training_args, model_args, dnotitia_args, ctx, train_logger):
    return {"model": model_args.model_name_or_path}


def load_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    return get_dataset_with_schema_alignment(dataset_mixture_args, seed=training_args.seed)


def extra_trainer_kwargs(script_args, training_args, model_args, dnotitia_args, ctx, train_logger):
    return {
        "teacher_model": training_args.teacher_model_name_or_path,
        "quantization_config": ctx["quantization_config"],
    }


SPEC = TrainingSpec(
    name="On-Policy Distillation",
    output_dir_tag="DISTILL",
    trainer_type="DISTILL",
    trainer_module="dna_factory.dnotitia_distillation_trainer",
    script_file=__file__,
    defaults_yaml="configs/_defaults-Distill.yaml",
    dataclass_types=(ScriptArguments, DistillationConfig, ModelConfig, WeightedDatasetMixtureConfig,
                     DnotitiaArguments),
    set_dataset_num_proc=False,
    # Disable DeepGEMM: vLLM 0.20.0 warmup crashes on Hopper/Blackwell; unneeded for bf16.
    extra_env={"VLLM_USE_DEEP_GEMM": "0"},
    setup_training_args=setup_training_args,
    load_models=load_models,
    load_mixture=load_mixture,
    # No thinking -> reasoning_content preprocessing: datasets are prompt-only.
    trainer_cls=DnotitiaDistillationTrainer,
    extra_trainer_kwargs=extra_trainer_kwargs,
)


def main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args=None):
    return run_training(SPEC, script_args, training_args, model_args,
                        dataset_mixture_args, dnotitia_args, user_specified_args)


if __name__ == "__main__":
    cli_main(SPEC)
