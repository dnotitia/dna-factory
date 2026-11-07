"""
OPD (On-Policy Distillation) training script for DNA Factory.

Method-specific logic only (student + frozen teacher wiring, prompt-only
dataset mixture, trainer wiring); the shared setup/train/save flow lives in
dna_factory/training_runner.py.
Key OPD-specific difference: a frozen teacher scores each token the
student generates.
"""

import logging
import os
import sys

# unsloth must be imported before trl/transformers/peft, and before the Dnotitia trainers subclass
# TRL's trainers, for its patches to apply, so the use_unsloth decision runs ahead of those imports.
if __name__ == "__main__":
    from dna_factory.unsloth_bootstrap import maybe_import_unsloth

    maybe_import_unsloth(sys.argv[1:], "configs/_defaults-OPD.yaml")

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
from dna_factory.dnotitia_opd_trainer import DnotitiaOPDTrainer
from dna_factory.training_runner import (
    TrainingSpec,
    cli_main,
    resolve_trust_remote_code,
    run_training,
)
from dna_factory.unsloth_integration import (
    generation_max_seq_length,
    load_unsloth_model,
    resolve_fast_inference,
)

logger = logging.getLogger(__name__)


def _normalize_dataset_for_opd(dataset):
    """Normalize one dataset to prompt-only schema (assistant turn discarded)."""
    feats = dataset.features
    if "messages" in feats:

        def split_messages(example):
            messages = example["messages"]
            assistant_idxs = [
                i for i, m in enumerate(messages) if m["role"] == "assistant"
            ]
            prompt_msgs = messages[: assistant_idxs[-1]] if assistant_idxs else messages
            return {
                "prompt": [
                    {"role": m["role"], "content": m["content"]} for m in prompt_msgs
                ]
            }

        return dataset.map(split_messages, remove_columns=dataset.column_names)
    elif "prompt" in feats:

        def normalize_prompt(example):
            p = example["prompt"]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
            return {
                "prompt": [{"role": m["role"], "content": m["content"]} for m in msgs]
            }

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
        logger.info(
            f"Loading dataset for mixture: {path} (config name: {dataset_config.name})"
        )
        if os.path.isdir(path):
            dataset = ds.load_from_disk(path)
            if isinstance(dataset, DatasetDict):
                dataset = dataset[dataset_config.split or "train"]
        else:
            dataset = ds.load_dataset(
                path=path, name=dataset_config.name, split=dataset_config.split
            )

        if dataset_config.columns is not None:
            dataset = dataset.select_columns(dataset_config.columns)

        dataset = _normalize_dataset_for_opd(dataset)

        weight = getattr(dataset_config, "weight", 1.0)
        if weight != 1.0:
            n_before = len(dataset)
            dataset = resample_by_weight(dataset, weight, seed=seed)
            logger.info(f"  weight={weight}: {n_before} -> {len(dataset)} examples")

        datasets_list.append(dataset)

    combined = concatenate_datasets(datasets_list)
    logger.info(f"Created OPD dataset mixture with {len(combined)} examples")
    return DatasetDict({"train": combined})


def setup_training_args(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    if not training_args.teacher_model_name_or_path:
        raise ValueError(
            "On-policy distillation requires a teacher. Set `teacher_model_name_or_path` (it must share the "
            "student's vocabulary)."
        )

    if ctx.get("use_unsloth"):
        # unsloth path: FastModel loads the student (quantization and LoRA included) and the trainer
        # gets the instance, so `model_init_kwargs` and `quantization_config` must be unset.
        training_args.model_init_kwargs = None
    else:
        # Models passed as strings; init kwargs go through training config.
        training_args.model_init_kwargs = {
            "revision": model_args.model_revision,
            "trust_remote_code": resolve_trust_remote_code(training_args),
            "attn_implementation": model_args.attn_implementation,
            "dtype": model_args.dtype,
        }
    # The teacher keeps the string path either way (the trainer loads it, never through unsloth).
    training_args.teacher_model_init_kwargs = {
        "trust_remote_code": resolve_trust_remote_code(training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    ctx["quantization_config"] = (
        None if ctx.get("use_unsloth") else get_quantization_config(model_args)
    )


def load_models(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    if not ctx.get("use_unsloth"):
        return {"model": model_args.model_name_or_path}

    # With vLLM, unsloth owns the colocated engine and `vllm_gpu_memory_utilization` is forwarded to it.
    model_type = ctx["unsloth_model_type"]
    fast_inference = resolve_fast_inference(
        training_args, model_type, train_logger, use_peft=model_args.use_peft
    )
    model = load_unsloth_model(
        model_args,
        training_args,
        model_type=model_type,
        max_seq_length=generation_max_seq_length(model_args, training_args),
        fast_inference=fast_inference,
        logger=train_logger,
    )
    return {"model": model}


def load_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    return get_dataset_with_schema_alignment(
        dataset_mixture_args, seed=training_args.seed
    )


def extra_trainer_kwargs(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    return {
        "teacher_model": training_args.teacher_model_name_or_path,
        "quantization_config": ctx["quantization_config"],
    }


SPEC = TrainingSpec(
    name="OPD",
    output_dir_tag="OPD",
    trainer_module="dna_factory.dnotitia_opd_trainer",
    script_file=__file__,
    defaults_yaml="configs/_defaults-OPD.yaml",
    dataclass_types=(
        ScriptArguments,
        DistillationConfig,
        ModelConfig,
        WeightedDatasetMixtureConfig,
        DnotitiaArguments,
    ),
    set_dataset_num_proc=False,
    # Disable DeepGEMM: vLLM 0.20.0 warmup crashes on Hopper/Blackwell; unneeded for bf16.
    extra_env={"VLLM_USE_DEEP_GEMM": "0"},
    setup_training_args=setup_training_args,
    load_models=load_models,
    load_mixture=load_mixture,
    # No thinking -> reasoning_content preprocessing: datasets are prompt-only.
    trainer_cls=DnotitiaOPDTrainer,
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
