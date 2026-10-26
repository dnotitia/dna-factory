"""
Supervised Fine-Tuning (SFT) training script for DNA Factory.

Method-specific logic only (model loading, SFT dataset mixture, trainer wiring);
the shared setup/train/save flow lives in dna_factory/training_runner.py.
"""

import logging
import sys

# unsloth must be imported before trl/transformers/peft, and before the Dnotitia trainers subclass
# TRL's trainers, for its patches to apply, so the use_unsloth decision runs ahead of those imports.
if __name__ == "__main__":
    from dna_factory.unsloth_bootstrap import maybe_import_unsloth

    maybe_import_unsloth(sys.argv[1:], "configs/_defaults-SFT.yaml")

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
)

from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetMixtureConfig,
    resample_by_weight,
)
from dna_factory.dnotitia_sft_trainer import DnotitiaSFTTrainer
from dna_factory.training_runner import (
    TrainingSpec,
    cli_main,
    create_model_kwargs,
    load_model,
    run_training,
    set_use_cache,
)
from dna_factory.unsloth_integration import load_unsloth_model

# Initialize logger
logger = logging.getLogger(__name__)


def get_dataset_with_schema_alignment(mixture_config, seed=42):
    import datasets as ds
    from datasets import DatasetDict, concatenate_datasets

    datasets_list = []
    for dataset_config in mixture_config.datasets:
        logger.info(
            f"Loading dataset for mixture: {dataset_config.path} (config name: {dataset_config.name})"
        )
        dataset = ds.load_dataset(
            path=dataset_config.path,
            name=dataset_config.name,
            split=dataset_config.split,
        )
        if dataset_config.columns is not None:
            dataset = dataset.select_columns(dataset_config.columns)

        # Normalize: add 'thinking' field to messages if missing
        if "messages" in dataset.features:
            first_msg_features = dataset.features["messages"].feature
            if "thinking" not in first_msg_features:

                def add_thinking(example):
                    return {
                        "messages": [
                            {**msg, "thinking": ""} for msg in example["messages"]
                        ]
                    }

                dataset = dataset.map(add_thinking)

        # Apply the per-dataset sample weight (size multiplier; 1.0 = no change)
        weight = getattr(dataset_config, "weight", 1.0)
        if weight != 1.0:
            n_before = len(dataset)
            dataset = resample_by_weight(dataset, weight, seed=seed)
            logger.info(f"  weight={weight}: {n_before} -> {len(dataset)} examples")

        datasets_list.append(dataset)

    combined = concatenate_datasets(datasets_list)
    logger.info(f"Created dataset mixture with {len(combined)} examples")
    return DatasetDict({"train": combined})


def preprocess_thinking_data(example):
    # Add reasoning_content from thinking field for Qwen3 compatibility
    messages = []
    for msg in example["messages"]:
        new_msg = dict(msg)
        if msg.get("thinking"):
            new_msg["reasoning_content"] = msg["thinking"]
        messages.append(new_msg)
    return {"messages": messages}


EARLIER_ASSISTANT_TURNS = "_earlier_assistant_turns"


def messages_to_prompt_completion(example):
    """Split a `messages` conversation into prompt (all turns but the last) and completion (the last turn).

    TRL masks the prompt only for prompt/completion data. `thinking` goes into `reasoning_content` as for
    `messages` data, so the chat template renders the same token sequence. Earlier assistant turns become
    part of the prompt and are not trained; their count is returned so the caller can report it.
    """
    messages = preprocess_thinking_data(example)["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(
            "completion_only_loss needs each conversation to end with an assistant turn, got "
            f"{messages[-1]['role'] if messages else 'an empty conversation'!r}."
        )
    *prompt, completion = messages
    earlier = sum(1 for m in prompt if m["role"] == "assistant")
    return {
        "prompt": prompt,
        "completion": [completion],
        EARLIER_ASSISTANT_TURNS: earlier,
    }


def convert_for_completion_only_loss(dataset, train_logger):
    converted = dataset.map(messages_to_prompt_completion, remove_columns=["messages"])
    for split, data in converted.items():
        try:
            multi_turn = sum(1 for n in data[EARLIER_ASSISTANT_TURNS] if n)
        except KeyError, TypeError:  # streaming splits cannot be counted up front
            multi_turn = None
        if multi_turn:
            train_logger.warning(
                f"completion_only_loss: {multi_turn} of {len(data)} `{split}` conversations have earlier "
                "assistant turns, which are now part of the prompt and not trained; only the final assistant "
                "turn carries the loss."
            )
    train_logger.info(
        "completion_only_loss: converted `messages` to prompt/completion so the loss covers only the final "
        "assistant turn."
    )
    return converted.remove_columns(EARLIER_ASSISTANT_TURNS)


def load_models(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    if ctx.get("use_unsloth"):
        # unsloth path: FastModel loads the model and (with use_peft) applies LoRA itself, because the
        # patched SFTTrainer ignores `peft_config`. `max_length` doubles as unsloth's max_seq_length.
        model = load_unsloth_model(
            model_args,
            training_args,
            model_type=ctx["unsloth_model_type"],
            max_seq_length=training_args.max_length or 0,
            fast_inference=False,
            logger=train_logger,
        )
        return {"model": model}

    # Model init kwargs
    model_kwargs = create_model_kwargs(model_args, training_args, dnotitia_args)

    # Create model. `load_model` picks AutoModelForMultimodalLM for Qwen3.5 conditional-generation
    # checkpoints (so the trained model round-trips identically for vLLM serving) and
    # AutoModelForCausalLM for plain causal LMs. `use_cache` is set post-load (see set_use_cache).
    model = load_model(
        model_args.model_name_or_path,
        low_cpu_mem_usage=True,
        offload_state_dict=True,
        **model_kwargs,
    )
    set_use_cache(model, not training_args.gradient_checkpointing)
    return {"model": model}


def load_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    return get_dataset_with_schema_alignment(
        dataset_mixture_args, seed=training_args.seed
    )


def setup_training_args(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    ctx["completion_only_loss"] = training_args.completion_only_loss


def _has_messages(dataset):
    columns = dataset.column_names
    if isinstance(columns, dict):
        return any(c and "messages" in c for c in columns.values())
    return bool(columns) and "messages" in columns


def postprocess_dataset(dataset, ctx, train_logger):
    # TRL applies completion_only_loss only to prompt/completion data and silently trains every token of a
    # `messages` dataset, so convert those conversations (the final assistant turn becomes the completion).
    if ctx.get("completion_only_loss") and _has_messages(dataset):
        return convert_for_completion_only_loss(dataset, train_logger)
    return dataset.map(preprocess_thinking_data)


SPEC = TrainingSpec(
    name="SFT",
    output_dir_tag="SFT",
    trainer_type="SFT",
    trainer_module="dna_factory.dnotitia_sft_trainer",
    script_file=__file__,
    defaults_yaml="configs/_defaults-SFT.yaml",
    dataclass_types=(
        ScriptArguments,
        SFTConfig,
        ModelConfig,
        WeightedDatasetMixtureConfig,
        DnotitiaArguments,
    ),
    load_models=load_models,
    setup_training_args=setup_training_args,
    load_mixture=load_mixture,
    postprocess_dataset=postprocess_dataset,
    trainer_cls=DnotitiaSFTTrainer,
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
