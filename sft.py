"""
Supervised Fine-Tuning (SFT) training script for DNA Factory.

Method-specific logic only (model loading, SFT dataset mixture, trainer wiring);
the shared setup/train/save flow lives in dna_factory/training_runner.py.
"""

import logging

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
from dna_factory.dnotitia_trainer_commons import (
    create_model_kwargs,
    load_model,
    set_use_cache,
)
from dna_factory.training_runner import TrainingSpec, cli_main, run_training

# Initialize logger
logger = logging.getLogger(__name__)


def get_dataset_with_schema_alignment(mixture_config, seed=42):
    import datasets as ds
    from datasets import DatasetDict, concatenate_datasets

    datasets_list = []
    for dataset_config in mixture_config.datasets:
        logger.info(f"Loading dataset for mixture: {dataset_config.path} (config name: {dataset_config.name})")
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
                    return {"messages": [{**msg, "thinking": ""} for msg in example["messages"]]}
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


def load_models(script_args, training_args, model_args, dnotitia_args, ctx, train_logger):
    # Model init kwargs
    model_kwargs = create_model_kwargs(model_args, training_args, dnotitia_args)

    # Create model. `load_model` picks AutoModelForMultimodalLM for Qwen3.5 conditional-generation
    # checkpoints (so the trained model round-trips identically for vLLM serving) and
    # AutoModelForCausalLM for plain causal LMs. `use_cache` is set post-load (see set_use_cache).
    model = load_model(
        model_args.model_name_or_path, low_cpu_mem_usage=True, offload_state_dict=True,
        **model_kwargs
    )
    set_use_cache(model, not training_args.gradient_checkpointing)
    return {"model": model}


def load_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    return get_dataset_with_schema_alignment(dataset_mixture_args, seed=training_args.seed)


def postprocess_dataset(dataset, ctx, train_logger):
    return dataset.map(preprocess_thinking_data)


SPEC = TrainingSpec(
    name="SFT",
    output_dir_tag="SFT",
    trainer_type="SFT",
    trainer_module="dna_factory.dnotitia_sft_trainer",
    script_file=__file__,
    defaults_yaml="configs/_defaults-SFT.yaml",
    dataclass_types=(ScriptArguments, SFTConfig, ModelConfig, WeightedDatasetMixtureConfig, DnotitiaArguments),
    load_models=load_models,
    load_mixture=load_mixture,
    postprocess_dataset=postprocess_dataset,
    trainer_cls=DnotitiaSFTTrainer,
)


def main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args=None):
    return run_training(SPEC, script_args, training_args, model_args,
                        dataset_mixture_args, dnotitia_args, user_specified_args)


if __name__ == "__main__":
    cli_main(SPEC)
