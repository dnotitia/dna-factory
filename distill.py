"""
On-Policy Distillation training script for DNA Factory.

Method-specific logic only (teacher validation, prompt-only dataset mixture,
trainer wiring); the shared setup/train/save flow lives in
dna_factory/training_runner.py.
See docs/distillation.md for background.

Distillation-specific notes:
- Student samples its own completions; the frozen teacher grades every token (no reward functions).
- `teacher_model_name_or_path` is required and must share the student's vocabulary.
- Both models pass as strings so `*_init_kwargs` are honored; `beta` selects the divergence
  (1.0 = reverse KL, 0.0 = forward KL, 0.5 = JSD), not a GRPO-style KL penalty.
"""

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

# Initialize logger
logger = logging.getLogger(__name__)


def _normalize_dataset_for_distillation(dataset):
    """
    Per-dataset mechanical normalization to the prompt-only schema the distillation trainer consumes
    (factored out so it is directly testable with a plain dataset):

      - `messages` present → the turns before the last assistant turn become `prompt` (role/content only);
        a dataset with no assistant turn keeps all of its turns. The assistant turn itself is discarded:
        the student writes its own completion and the teacher grades it, so a reference answer is unused.
      - `prompt` present (no `messages`) → a plain string becomes a single user turn; a list is kept but
        reduced to role/content only.
      - Neither present → a clear error (nothing else to build a prompt from).

    Every other column is dropped. Unlike GRPO — where extra columns are forwarded to reward functions —
    distillation has no reward functions, so a prompt-only schema also makes mixture concatenation trivial.
    """
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
    """
    Distillation mixture loader.

    TRL's stock get_dataset() just concatenates the datasets, which raises when they have different columns.
    Mixture datasets here are heterogeneous (conversational `messages` vs plain `prompt`, plus arbitrary
    extra columns), so each one is reduced to the same single-column prompt-only schema by
    `_normalize_dataset_for_distillation` before concatenation.

    The per-dataset `weight` (a size multiplier; see dna_factory/dnotitia_dataset_mixture.py) is applied the
    same way sft.py applies it.

    A local save_to_disk directory is loaded via load_from_disk; otherwise the path is loaded from the HF hub.
    """
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

        # Apply the per-dataset sample weight (size multiplier; 1.0 = no change)
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
    # A teacher is the entire supervision signal here, so there is nothing to train without one
    if not training_args.teacher_model_name_or_path:
        raise ValueError(
            "On-policy distillation requires a teacher. Set `teacher_model_name_or_path` (it must share the "
            "student's vocabulary)."
        )

    # Model init kwargs (distillation-specific): both models are passed to the trainer as strings, so init
    # kwargs go through the training config. DistillationTrainer manages `use_cache` itself during generation
    # and training forwards, so it is intentionally omitted here.
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": resolve_trust_remote_code(model_args, training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    # The teacher gets the same runtime knobs but never `model_revision` (that pins the *student*); its own
    # revision is applied by the trainer from `teacher_model_revision`.
    training_args.teacher_model_init_kwargs = {
        "trust_remote_code": resolve_trust_remote_code(model_args, training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    ctx["quantization_config"] = get_quantization_config(model_args)


def load_models(script_args, training_args, model_args, dnotitia_args, ctx, train_logger):
    return {"model": model_args.model_name_or_path}


def load_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    # Schema-aligning loader (not TRL's stock get_dataset): normalizes mixed datasets to a common
    # prompt-only schema and applies each dataset's `weight`.
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
    # vLLM 0.20.0 bundles an incomplete vendored `deep_gemm`, and its kernel warmup crashes on
    # Hopper/Blackwell GPUs even for bf16 models. FP8 GEMM kernels are not needed for bf16
    # training, so DeepGEMM is disabled by default (override by exporting VLLM_USE_DEEP_GEMM=1).
    extra_env={"VLLM_USE_DEEP_GEMM": "0"},
    setup_training_args=setup_training_args,
    load_models=load_models,
    load_mixture=load_mixture,
    # Note: no thinking → reasoning_content preprocessing here. Distillation datasets are prompt-only, so
    # there are no pre-existing assistant turns carrying a `thinking` field (the student writes the
    # completions online during training).
    trainer_cls=DnotitiaDistillationTrainer,
    extra_trainer_kwargs=extra_trainer_kwargs,
)


def main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args=None):
    return run_training(SPEC, script_args, training_args, model_args,
                        dataset_mixture_args, dnotitia_args, user_specified_args)


if __name__ == "__main__":
    cli_main(SPEC)
