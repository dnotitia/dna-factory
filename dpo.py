"""
Direct Preference Optimization (DPO) training script for DNA Factory.

Method-specific logic only (model + ref_model loading, DPO dataset handling,
trainer wiring); the shared setup/train/save flow lives in
dna_factory/training_runner.py.
Key DPO-specific difference: Uses both model and ref_model (reference model).
"""

import logging

from transformers import AutoModelForCausalLM
from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    ModelConfig,
    ScriptArguments,
    get_dataset,
)

from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_dpo_trainer import DnotitiaDPOTrainer
from dna_factory.training_runner import (
    TrainingSpec,
    cli_main,
    create_model_kwargs,
    run_training,
)

# Initialize logger
logger = logging.getLogger(__name__)


def preprocess_thinking_data(example):
    """Map Qwen ``thinking`` fields in conversational DPO samples.

    Standard DPO datasets (including ``trl-lib/ultrafeedback_binarized``)
    contain string ``prompt``, ``chosen``, and ``rejected`` columns rather
    than a top-level ``messages`` column. Leave those samples unchanged,
    while still supporting conversational DPO datasets.
    """
    updated_fields = {}
    for field in ("messages", "prompt", "chosen", "rejected"):
        value = example.get(field)
        if not isinstance(value, list):
            continue

        messages = []
        changed = False
        for msg in value:
            # A list can be a non-conversational value; do not alter it.
            if not isinstance(msg, dict):
                messages = value
                changed = False
                break

            new_msg = dict(msg)
            if msg.get("thinking"):
                new_msg["reasoning_content"] = msg["thinking"]
                changed = True
            messages.append(new_msg)

        if changed:
            updated_fields[field] = messages

    return updated_fields


def load_models(
    script_args, training_args, model_args, dnotitia_args, ctx, train_logger
):
    # Model init kwargs
    model_kwargs = create_model_kwargs(model_args, training_args, dnotitia_args)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    # Create reference model (DPO-specific)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    # Set reference model to evaluation mode (no gradient computation needed)
    ref_model.eval()
    return {"model": model, "ref_model": ref_model}


def load_mixture(dataset_mixture_args, training_args, ctx, train_logger):
    return get_dataset(dataset_mixture_args)


def postprocess_dataset(dataset, ctx, train_logger):
    return dataset.map(preprocess_thinking_data)


SPEC = TrainingSpec(
    name="DPO",
    output_dir_tag="DPO",
    trainer_type="DPO",
    trainer_module="dna_factory.dnotitia_dpo_trainer",
    script_file=__file__,
    defaults_yaml="configs/_defaults-DPO.yaml",
    dataclass_types=(
        ScriptArguments,
        DPOConfig,
        ModelConfig,
        DatasetMixtureConfig,
        DnotitiaArguments,
    ),
    load_models=load_models,
    load_mixture=load_mixture,
    postprocess_dataset=postprocess_dataset,
    trainer_cls=DnotitiaDPOTrainer,
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
