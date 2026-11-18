"""
Direct Preference Optimization (DPO) training script for DNA Factory.

Method-specific logic only (model + ref_model loading, DPO dataset handling,
trainer wiring); the shared setup/train/save flow lives in
dna_factory/training_runner.py.
Key DPO-specific difference: Uses both model and ref_model (reference model).
"""

import logging
import sys

# unsloth must be imported before trl/transformers/peft, and before the Dnotitia trainers subclass
# TRL's trainers, for its patches to apply, so the use_unsloth decision runs ahead of those imports.
if __name__ == "__main__":
    from dna_factory.unsloth_bootstrap import maybe_import_unsloth

    maybe_import_unsloth(sys.argv[1:], "configs/_defaults-DPO.yaml")

from transformers import AutoModelForCausalLM
from trl import DPOConfig, ModelConfig, ScriptArguments

from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetMixtureConfig,
    get_weighted_dataset,
)
from dna_factory.dnotitia_dpo_trainer import DnotitiaDPOTrainer
from dna_factory.training_runner import (
    TrainingSpec,
    cli_main,
    create_model_kwargs,
    load_model,
    run_training,
)
from dna_factory.unsloth_integration import load_unsloth_model

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

    if ctx.get("use_unsloth"):
        # unsloth path: FastModel loads the policy and (with use_peft) applies LoRA itself, because the
        # patched DPOTrainer ignores `peft_config`.
        model = load_unsloth_model(
            model_args,
            training_args,
            model_type=ctx["unsloth_model_type"],
            max_seq_length=training_args.max_length or 0,
            fast_inference=False,
            logger=train_logger,
        )
        if model_args.use_peft:
            # ref_model=None makes TRL add a "ref" adapter copy of the freshly (zero-initialised) LoRA
            # model, i.e. the base weights, so no second full model is loaded.
            return {"model": model, "ref_model": None}
        # Full fine-tuning has no adapter to switch off, so load a plain HF frozen reference
        # (the same thing TRL would build internally from the policy's config).
        ref_model = load_model(model_args.model_name_or_path, **model_kwargs)
        ref_model.eval()
        return {"model": model, "ref_model": ref_model}

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
    return get_weighted_dataset(dataset_mixture_args, seed=training_args.seed)


def postprocess_dataset(dataset, ctx, train_logger):
    return dataset.map(preprocess_thinking_data)


SPEC = TrainingSpec(
    name="DPO",
    output_dir_tag="DPO",
    trainer_module="dna_factory.dnotitia_dpo_trainer",
    script_file=__file__,
    defaults_yaml="configs/_defaults-DPO.yaml",
    dataclass_types=(
        ScriptArguments,
        DPOConfig,
        ModelConfig,
        WeightedDatasetMixtureConfig,
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
