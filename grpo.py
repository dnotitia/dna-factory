"""
Group Relative Policy Optimization (GRPO) training script for DNA Factory.

Method-specific logic only (reward resolution, prompt-only dataset mixture,
trainer wiring); the shared setup/train/save flow lives in
dna_factory/training_runner.py.
Key GRPO-specific difference: online RL scored by `reward_funcs`.
"""

import argparse
import importlib
import logging
import os
import sys
from dataclasses import dataclass, field, replace

# unsloth must be imported before trl/transformers/peft, and before the Dnotitia trainers subclass
# TRL's trainers, for its patches to apply, so the use_unsloth decision runs ahead of those imports.
if __name__ == "__main__":
    from dna_factory.unsloth_bootstrap import maybe_import_unsloth

    maybe_import_unsloth(sys.argv[1:], "configs/_defaults-GRPO.yaml")

import yaml
from trl import (
    DatasetMixtureConfig,
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    get_quantization_config,
)

from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetConfig,
)
from dna_factory.dnotitia_dataset_mixture import load_mixture as load_mixture_datasets
from dna_factory.dnotitia_grpo_trainer import DnotitiaGRPOTrainer
from dna_factory.training_runner import (
    TrainingSpec,
    cli_main,
    resolve_trust_remote_code,
    run_training,
)
from dna_factory.unsloth_integration import (
    effective_router_aux_loss_coef,
    generation_max_seq_length,
    load_unsloth_model,
    resolve_fast_inference,
)

logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """GRPO script args: reward model + reward funcs (see docs/grpo-rewards.md)."""

    grpo_execution: str = field(
        default="sync", metadata={"help": "GRPO execution mode: sync or async."}
    )

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
class LabeledDatasetConfig(WeightedDatasetConfig):
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


def _normalize_dataset_for_grpo(dataset, label, path=None):
    """Normalize one dataset to prompt-only schema + `label` column."""
    # `column_names` is None for a streaming dataset with unknown features: skip the check.
    if "label" in (dataset.column_names or []):
        raise ValueError(
            f"Dataset {path or label!r} already has a `label` column, but GRPO sets `label` "
            "from the entry's `label:` (default: its `path`). Rename or drop the column "
            "(e.g. via `columns:`)."
        )
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
    # `.map` rather than `add_column`, which needs `len()` and so fails on streaming datasets.
    return dataset.map(lambda _: {"label": label})


def get_dataset_with_schema_alignment(mixture_config, seed=42):
    """Load mixture datasets, normalize each to prompt schema + label, and concatenate."""
    return load_mixture_datasets(
        mixture_config,
        seed=seed,
        normalize=lambda dataset, dataset_config: _normalize_dataset_for_grpo(
            dataset,
            dataset_config.label or dataset_config.path,
            dataset_config.path,
        ),
    )


def setup_training_args(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    ctx["reward_funcs"] = resolve_reward_funcs(script_args, training_args)

    if ctx.get("use_unsloth"):
        # unsloth path: FastModel loads the model (quantization and LoRA included) and the trainer
        # gets the instance, so `model_init_kwargs` and `quantization_config` must be unset.
        training_args.model_init_kwargs = None
        ctx["quantization_config"] = None
        return

    # Model passed as string; init kwargs go through training_args.
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": resolve_trust_remote_code(training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
    }
    ctx["quantization_config"] = get_quantization_config(model_args)


def load_models(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    if not ctx.get("use_unsloth"):
        return {"model": model_args.model_name_or_path}

    # With vLLM, unsloth owns the colocated engine and `vllm_gpu_memory_utilization` is forwarded to it.
    model_type = ctx["unsloth_model_type"]
    # UnslothGRPOTrainer cannot compute the MoE router auxiliary loss and raises on a non-zero
    # coefficient (by default TRL takes it from the model config: 0.001 for Qwen3 MoE). Fail before the
    # slow model load, and don't zero it silently: use_unsloth must not change the objective.
    if model_type is not None and model_type.endswith("_moe"):
        coef = effective_router_aux_loss_coef(model_args, training_args)
        if coef:
            raise ValueError(
                f"unsloth GRPO does not compute the MoE router auxiliary loss, so router_aux_loss_coef="
                f"{coef} cannot be honoured for model_type={model_type!r}. "
                "Set router_aux_loss_coef=0.0 or use_unsloth=false."
            )
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


def load_async_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    # Same guard as `validate_async_args` in dna_factory/async_grpo.py: a mixture's own
    # `streaming: true` must not bypass it.
    if dataset_mixture_args.streaming:
        raise ValueError(
            "This AsyncGRPO draft supports non-streaming datasets only for checkpoint resume."
        )
    return load_mixture(dataset_mixture_args, training_args, ctx, train_logger)


def extra_trainer_kwargs(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    return {
        "reward_funcs": ctx["reward_funcs"],
        "quantization_config": ctx["quantization_config"],
        "dynamic_sampling": dnotitia_args.dynamic_sampling,
        "dynamic_sampling_max_rounds": dnotitia_args.dynamic_sampling_max_rounds,
        "log_completions_steps": dnotitia_args.log_completions_steps,
    }


SPEC = TrainingSpec(
    name="GRPO",
    output_dir_tag="GRPO",
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
        get_spec(script_args.grpo_execution),
        script_args,
        training_args,
        model_args,
        dataset_mixture_args,
        dnotitia_args,
        user_specified_args,
    )


def resolve_grpo_execution(argv):
    """Select defaults/config class before full parsing; CLI takes precedence over YAML."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config")
    parser.add_argument(
        "--grpo_execution", "--grpo-execution", choices=("sync", "async")
    )
    selected, _ = parser.parse_known_args(argv)
    config = {}
    if selected.config:
        with open(selected.config, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    mode = selected.grpo_execution or config.get("grpo_execution", "sync")
    if mode not in ("sync", "async"):
        raise ValueError(f"grpo_execution must be 'sync' or 'async', got {mode!r}")
    return mode


def get_spec(mode):
    if mode == "sync":
        return SPEC
    if mode != "async":
        raise ValueError(f"Unknown GRPO execution mode: {mode!r}")
    from trl.experimental.async_grpo import AsyncGRPOTrainer

    from dna_factory.async_grpo import DnotitiaAsyncGRPOConfig

    return replace(
        SPEC,
        name="AsyncGRPO",
        output_dir_tag="ASYNCGRPO",
        trainer_module="trl.experimental.async_grpo",
        defaults_yaml="configs/_defaults-AsyncGRPO.yaml",
        dataclass_types=(
            GRPOScriptArguments,
            DnotitiaAsyncGRPOConfig,
            ModelConfig,
            LabeledDatasetMixtureConfig,
            DnotitiaArguments,
        ),
        set_dataset_num_proc=False,
        pass_eval_dataset=False,
        pass_debug_batches=False,
        pass_peft_config=False,
        strict_args=True,
        load_mixture=load_async_mixture,
        setup_training_args=setup_async_training_args,
        postprocess_tokenizer=postprocess_async_tokenizer,
        trainer_cls=AsyncGRPOTrainer,
        extra_trainer_kwargs=async_trainer_kwargs,
    )


def setup_async_training_args(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    from dna_factory.async_grpo import (
        validate_async_args,
        validate_completion_length_against_server,
    )

    validate_async_args(script_args, training_args, model_args, dnotitia_args)
    # AsyncGRPOTrainer always builds the model with TRL's tree attention (prompts shared across a
    # group are packed once), whatever attn_implementation says: FlashAttention-3, or -4 on Blackwell.
    train_logger.info(
        "AsyncGRPO: attention is TRL's tree attention (flash-attn3; flash-attn4 on Blackwell); "
        "attn_implementation=%s is not used.",
        model_args.attn_implementation,
    )
    validate_completion_length_against_server(training_args, train_logger)
    ctx["reward_funcs"] = resolve_reward_funcs(script_args, training_args)
    training_args.dtype = model_args.dtype
    training_args.model_init_kwargs = {
        **(training_args.model_init_kwargs or {}),
        "revision": model_args.model_revision,
        "trust_remote_code": resolve_trust_remote_code(training_args),
        "dtype": model_args.dtype,
    }
    train_logger.info(
        "AsyncGRPO: single training GPU; server=%s, max_inflight_tasks=%s, "
        "max_staleness=%s, token_budget=%s",
        training_args.vllm_server_base_url,
        training_args.max_inflight_tasks,
        training_args.max_staleness,
        training_args.token_budget,
    )


def async_trainer_kwargs(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    return {"reward_funcs": ctx["reward_funcs"]}


def postprocess_async_tokenizer(tokenizer, ctx, train_logger):
    """Give the tokenizer a response template TRL's chat-template matching would miss."""
    from dna_factory.async_grpo import apply_response_template_override

    apply_response_template_override(tokenizer)
    return tokenizer


if __name__ == "__main__":
    cli_main(get_spec(resolve_grpo_execution(sys.argv[1:])))
