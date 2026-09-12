"""GRPO training script. Shared setup/train/save flow lives in dna_factory/training_runner.py."""

import importlib
import logging
import os
import sys
from dataclasses import dataclass, field

from trl import (
    DatasetMixtureConfig,
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    get_quantization_config,
)
from trl.scripts.utils import DatasetConfig

from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_grpo_trainer import DnotitiaGRPOTrainer
from dna_factory.training_runner import (
    TrainingSpec,
    cli_main,
    resolve_trust_remote_code,
    run_training,
)

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """GRPO script args: reward model + reward funcs (see docs/grpo-rewards.md)."""

    reward_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: list[str] | None = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Supported bare names (zero-argument only): "
            "`accuracy_reward`, `reasoning_accuracy_reward`, `think_format_reward`. A TRL reward "
            "*factory* (e.g. `get_soft_overlong_punishment`) needs a module-level instance built "
            "with its arguments first — see `dna_factory/rewards/my_rewards.py` and "
            "docs/grpo-rewards.md — then reference that instance via a dotted import path (e.g., "
            "`'my_lib.rewards.custom_reward'`)."
        },
    )


@dataclass
class LabeledDatasetConfig(DatasetConfig):
    """DatasetConfig + `label` tag for per-sample reward routing."""

    label: str | None = field(
        default=None,
        metadata={
            "help": "Provenance label for rows loaded from this dataset entry (defaults to `path` if unset)."
        },
    )


@dataclass
class LabeledDatasetMixtureConfig(DatasetMixtureConfig):
    """DatasetMixtureConfig with LabeledDatasetConfig entries."""

    datasets: list[LabeledDatasetConfig] = field(
        default_factory=list,
        metadata={
            "help": "List of (labeled) dataset configurations to include in the mixture."
        },
    )

    def __post_init__(self):
        for idx, dataset in enumerate(self.datasets):
            if isinstance(dataset, dict):
                self.datasets[idx] = LabeledDatasetConfig(**dataset)


def resolve_reward_funcs(script_args, training_args):
    """Build GRPOTrainer `reward_funcs` from bare names, dotted paths, and/or a reward model id."""
    # Lazily imported: optional deps (e.g. math_verify) only needed when used.
    from trl.rewards import (
        accuracy_reward,
        reasoning_accuracy_reward,
        think_format_reward,
    )

    reward_funcs_registry = {
        "accuracy_reward": accuracy_reward,
        "reasoning_accuracy_reward": reasoning_accuracy_reward,
        "think_format_reward": think_format_reward,
    }

    reward_funcs = []
    if script_args.reward_model_name_or_path:
        reward_funcs.append(script_args.reward_model_name_or_path)

    for func_name in script_args.reward_funcs or []:
        if func_name in reward_funcs_registry:
            reward_funcs.append(reward_funcs_registry[func_name])
        elif "." in func_name:
            module_path, attr_name = func_name.rsplit(".", 1)
            sys.path.insert(0, os.getcwd())
            module = importlib.import_module(module_path)
            reward_funcs.append(getattr(module, attr_name))
        else:
            raise ValueError(
                f"Could not load reward function '{func_name}'. Expected one of "
                f"{list(reward_funcs_registry.keys())} or a valid import path."
            )

    if not reward_funcs:
        raise ValueError(
            "GRPO requires a reward signal. Provide `reward_funcs` and/or `reward_model_name_or_path`."
        )

    return reward_funcs


def _normalize_dataset_for_grpo(dataset, label):
    """Normalize one dataset to prompt-only schema + `label` column."""
    feats = dataset.features
    if "messages" in feats:

        def split_messages(example):
            messages = example["messages"]
            assistant_idxs = [
                i for i, m in enumerate(messages) if m["role"] == "assistant"
            ]
            if assistant_idxs:
                last = assistant_idxs[-1]
                prompt_msgs = messages[:last]
                expected = messages[last]["content"]
            else:
                prompt_msgs = messages
                expected = None
            return {
                "prompt": [
                    {"role": m["role"], "content": m["content"]} for m in prompt_msgs
                ],
                "expected_output": expected,
            }

        dataset = dataset.map(split_messages, remove_columns=["messages"])
    elif "prompt" in feats:

        def normalize_prompt(example):
            p = example["prompt"]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
            return {
                "prompt": [{"role": m["role"], "content": m["content"]} for m in msgs]
            }

        dataset = dataset.map(normalize_prompt)
    else:
        raise ValueError(
            f"Dataset has neither `messages` nor `prompt` (columns: {dataset.column_names}); "
            "cannot build a GRPO prompt-only schema."
        )
    return dataset.add_column("label", [label] * len(dataset))


def get_dataset_with_schema_alignment(mixture_config):
    """Load mixture datasets, normalize each to prompt schema, and concatenate."""
    import os

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
        label = getattr(dataset_config, "label", None) or path
        datasets_list.append(_normalize_dataset_for_grpo(dataset, label))

    try:
        combined = concatenate_datasets(datasets_list)
    except Exception as e:
        raise ValueError(
            f"Could not align dataset mixture schemas for concatenation ({e}). This usually means two "
            "datasets define the same column with genuinely incompatible types (not just one of them "
            "missing it) — check the offending column above."
        ) from e
    logger.info(f"Created GRPO dataset mixture with {len(combined)} examples")
    return DatasetDict({"train": combined})


def setup_training_args(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    ctx["reward_funcs"] = resolve_reward_funcs(script_args, training_args)

    # Model passed as string; init kwargs go through training_args.
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": resolve_trust_remote_code(model_args, training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    ctx["quantization_config"] = get_quantization_config(model_args)


def load_models(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    return {"model": model_args.model_name_or_path}


def load_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    return get_dataset_with_schema_alignment(dataset_mixture_args)


def extra_trainer_kwargs(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    return {
        "reward_funcs": ctx["reward_funcs"],
        "quantization_config": ctx["quantization_config"],
        "dynamic_sampling": dnotitia_args.dynamic_sampling,
        "dynamic_sampling_max_rounds": dnotitia_args.dynamic_sampling_max_rounds,
    }


SPEC = TrainingSpec(
    name="GRPO",
    output_dir_tag="GRPO",
    trainer_type="GRPO",
    trainer_module="dna_factory.dnotitia_grpo_trainer",
    script_file=__file__,
    defaults_yaml="configs/_defaults-GRPO.yaml",
    dataclass_types=(
        GRPOScriptArguments,
        GRPOConfig,
        ModelConfig,
        LabeledDatasetMixtureConfig,
        DnotitiaArguments,
    ),
    # Disable DeepGEMM: vLLM 0.20.0 warmup crashes on Hopper/Blackwell; unneeded for bf16.
    extra_env={"VLLM_USE_DEEP_GEMM": "0"},
    setup_training_args=setup_training_args,
    load_models=load_models,
    load_mixture=load_mixture,
    # No thinking -> reasoning_content preprocessing: GRPO datasets are prompt-only.
    trainer_cls=DnotitiaGRPOTrainer,
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
