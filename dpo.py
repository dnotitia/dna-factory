"""
Direct Preference Optimization (DPO) training script for DNA Factory.

Note: This file intentionally maintains some duplication with sft.py for readability and clarity.
Common utilities (logging, banners, argument printing) are extracted to dnotitia_trainer_commons.py,
while core training logic remains here for easy understanding of the complete DPO flow.
Key DPO-specific difference: Uses both model and ref_model (reference model).
When modifying shared utility logic, update dnotitia_trainer_commons.py.
"""

import logging
import multiprocessing
import os
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    DPOConfig,
    TrlParser,
    get_dataset,
    get_peft_config,
)

from dna_factory.utils.colorize_args import parse_user_args
from dna_factory.utils.config_merger import merge_config_files
from dna_factory.utils.output_dir_generator import generate_auto_output_dir
from dna_factory.dnotitia_trainer_commons import (
    setup_logging,
    print_dna_factory_banner,
    print_training_start_message,
    print_auto_generated_output_dir,
    print_environment_and_arguments,
    create_model_kwargs,
    resolve_trust_remote_code,
    save_training_results,
)
from dna_factory.dnotitia_dpo_trainer import DnotitiaDPOTrainer
from dna_factory.dnotitia_arguments import DnotitiaArguments

# Initialize logger
logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args=None):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Use empty set if no user args provided
    if user_specified_args is None:
        user_specified_args = set()

    # Set dataset number of processes to the number of CPUs
    training_args.dataset_num_proc = multiprocessing.cpu_count() // 2

    # Auto-generate output_dir if set to 'auto'
    auto_generated_dir = False
    if training_args.output_dir == 'auto':
        auto_generated_dir = True
        auto_output_dir = generate_auto_output_dir(
            model_args.model_name_or_path,
            user_specified_args,
            script_args,
            training_args,
            model_args,
            dataset_mixture_args,
            dnotitia_args,
            'DPO',
        )
        training_args.output_dir = auto_output_dir

    # Setup logging
    logger = setup_logging(training_args, 'dna_factory.dnotitia_dpo_trainer')

    # Print DNA Factory banner
    print_dna_factory_banner(logger, __file__)

    # Print the script start message
    print_training_start_message(logger, "DPO")

    # Log auto-generated output directory if applicable
    if auto_generated_dir:
        print_auto_generated_output_dir(logger, training_args.output_dir)

    # Print the parsed arguments
    print_environment_and_arguments(
        logger, script_args, training_args, model_args,
        dataset_mixture_args, dnotitia_args, user_specified_args,
        trainer_type="DPO"
    )

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Model init kwargs
    model_kwargs = create_model_kwargs(model_args, training_args, dnotitia_args)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Create reference model (DPO-specific)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )
    # Set reference model to evaluation mode (no gradient computation needed)
    ref_model.eval()

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=resolve_trust_remote_code(model_args, training_args),
        use_fast=True
    )

    # Load the dataset
    if dataset_mixture_args.datasets:
        logger.info(
            "The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_mixture_args)
    elif script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

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
    dataset = dataset.map(preprocess_thinking_data)

    # Initialize the Dnotitia DPO trainer
    trainer = DnotitiaDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        debug_first_n_batches=dnotitia_args.debug_first_n_batches,
    )

    # Check checkpoint
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Train the model
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save training results
    save_training_results(trainer, train_result, dataset, script_args, training_args)


if __name__ == "__main__":
    # Initialize the parser
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig, DnotitiaArguments)
    parser = TrlParser(dataclass_types)

    # Get arguments with load default YAML configuration
    cli_args = sys.argv[1:]

    # Parse user-specified arguments before adding defaults
    user_specified_args = parse_user_args(cli_args)

    # Check if user provided a config file
    user_has_config = "--config" in cli_args
    if user_has_config:
        user_config_path = None
        # Find the config file path specified by user
        try:
            config_index = cli_args.index("--config")
            if config_index + 1 < len(cli_args):
                user_config_path = cli_args[config_index + 1]

            config_path = merge_config_files("configs/_defaults-DPO.yaml", user_config_path)
        except (ValueError, IndexError):
            config_path = "configs/_defaults-DPO.yaml"
    else:
        config_path = "configs/_defaults-DPO.yaml"
    full_args = ["--config", config_path] + cli_args

    # Parse arguments
    (script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, _) = \
        (parser.parse_args_and_config(full_args,
                                      return_remaining_strings=True))

    # Run the main function
    main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args)
