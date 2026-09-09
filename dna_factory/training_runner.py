"""Shared training skeleton for SFT / DPO / GRPO / Distillation entry points.

Each script (sft.py, dpo.py, grpo.py, distill.py) keeps only its method-specific
logic — model loading, dataset normalization, trainer construction — and describes
it as a `TrainingSpec`. `run_training` executes the identical setup/train/save flow
for all of them; `cli_main` implements the identical `__main__` CLI bootstrap.
"""

import logging
import multiprocessing
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import TrlParser, get_peft_config

from dna_factory.dnotitia_trainer_commons import (
    print_auto_generated_output_dir,
    print_dna_factory_banner,
    print_environment_and_arguments,
    print_training_start_message,
    resolve_trust_remote_code,
    save_training_results,
    setup_logging,
)
from dna_factory.periodic_checkpoint import (
    PeriodicCheckpointCallback,
    parse_duration_to_seconds,
)
from dna_factory.utils.colorize_args import parse_user_args
from dna_factory.utils.config_merger import merge_config_files
from dna_factory.utils.output_dir_generator import generate_auto_output_dir

logger = logging.getLogger(__name__)


@dataclass
class TrainingSpec:
    """Method-specific pieces the shared skeleton needs.

    `ctx` is a plain dict threaded through the hooks so `setup_training_args`
    can stash values (e.g. reward funcs, quantization config) for the later hooks.
    """

    # Display / naming (each also feeds output_dir auto-naming and arg printing).
    name: str  # e.g. "SFT", "On-Policy Distillation"
    output_dir_tag: str  # e.g. "SFT", "DISTILL"
    trainer_type: str  # e.g. "SFT", "DISTILL"
    trainer_module: str  # e.g. "dna_factory.dnotitia_sft_trainer" (for setup_logging)
    script_file: str  # caller __file__ (for the banner VERSION lookup)

    # CLI parsing.
    defaults_yaml: str  # e.g. "configs/_defaults-SFT.yaml"
    dataclass_types: tuple  # TrlParser dataclasses

    # Hooks. All receive (script_args, training_args, model_args, dnotitia_args, ctx, logger)
    # unless noted; `dataset_mixture_args` replaces `model_args`/`dnotitia_args` for load_mixture.
    set_dataset_num_proc: bool = True
    extra_env: dict = field(default_factory=dict)
    setup_training_args: Callable | None = None  # validate + fill training_args (model_init_kwargs, ...)
    load_models: Callable | None = None  # -> dict merged into trainer kwargs (model, ref_model, ...)
    load_mixture: Callable | None = None  # (dataset_mixture_args, training_args, ctx, logger) -> DatasetDict
    postprocess_dataset: Callable | None = None  # (dataset, ctx, logger) -> dataset
    trainer_cls: type | None = None
    extra_trainer_kwargs: Callable | None = None  # -> dict merged into trainer kwargs


def _default(value, hook, *args):
    return hook(*args) if hook is not None else value


def run_training(spec, script_args, training_args, model_args, dataset_mixture_args,
                 dnotitia_args, user_specified_args=None):
    """Execute the shared flow: setup, models, tokenizer, dataset, trainer, train, save."""
    for key, value in spec.extra_env.items():
        os.environ.setdefault(key, value)

    ctx = {}

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Use empty set if no user args provided
    if user_specified_args is None:
        user_specified_args = set()

    # Set dataset number of processes to the number of CPUs
    if spec.set_dataset_num_proc:
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
            spec.output_dir_tag,
        )
        training_args.output_dir = auto_output_dir

    # Setup logging
    train_logger = setup_logging(training_args, spec.trainer_module)

    # Print DNA Factory banner
    print_dna_factory_banner(train_logger, spec.script_file)

    # Print the script start message
    print_training_start_message(train_logger, spec.name)

    # Log auto-generated output directory if applicable
    if auto_generated_dir:
        print_auto_generated_output_dir(train_logger, training_args.output_dir)

    # Print the parsed arguments
    print_environment_and_arguments(
        train_logger, script_args, training_args, model_args,
        dataset_mixture_args, dnotitia_args, user_specified_args,
        trainer_type=spec.trainer_type
    )

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        train_logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Method-specific training_args preparation (reward funcs, model_init_kwargs, ...)
    if spec.setup_training_args is not None:
        spec.setup_training_args(script_args, training_args, model_args, dnotitia_args, ctx, train_logger)

    # Load model(s); the returned dict becomes trainer kwargs (model, ref_model, ...)
    models_kwargs = _default({}, spec.load_models,
                             script_args, training_args, model_args, dnotitia_args, ctx, train_logger)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=resolve_trust_remote_code(model_args, training_args),
        use_fast=True
    )

    # Load the dataset
    if dataset_mixture_args.datasets:
        train_logger.info(
            "The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = spec.load_mixture(dataset_mixture_args, training_args, ctx, train_logger)
    elif script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")
    dataset = _default(dataset, spec.postprocess_dataset, dataset, ctx, train_logger)

    # Wall-clock periodic checkpointing (off when periodic_save_seconds is 0/'off').
    # Accepts human-friendly durations ('6h') or plain seconds ('21600').
    try:
        periodic_seconds = parse_duration_to_seconds(dnotitia_args.periodic_save_seconds)
    except ValueError as e:
        raise ValueError(f"Invalid `periodic_save_seconds` value: {e}") from e
    callbacks = []
    if periodic_seconds > 0:
        train_logger.info(
            f"Enabling wall-clock checkpointing every {periodic_seconds:g}s "
            f"(periodic_save_seconds={dnotitia_args.periodic_save_seconds!r})."
        )
        callbacks.append(
            PeriodicCheckpointCallback(periodic_seconds)
        )
    elif training_args.save_strategy == "no":
        train_logger.warning(
            "Both step-based checkpointing (save_strategy='no') and wall-clock "
            "checkpointing (periodic_save_seconds is off) are disabled — no checkpoints "
            "will be saved during training."
        )

    # Initialize the trainer
    trainer_kwargs = {
        'args': training_args,
        'train_dataset': dataset[script_args.dataset_train_split],
        'eval_dataset': dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        'processing_class': tokenizer,
        'peft_config': get_peft_config(model_args),
        'debug_first_n_batches': dnotitia_args.debug_first_n_batches,
        'callbacks': callbacks or None,
    }
    trainer_kwargs.update(models_kwargs)
    trainer_kwargs.update(_default(
        {}, spec.extra_trainer_kwargs,
        script_args, training_args, model_args, dnotitia_args, ctx, train_logger))
    trainer = spec.trainer_cls(**trainer_kwargs)

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
    return train_result


def cli_main(spec, argv=None):
    """Shared `__main__` bootstrap: merge default YAML, parse args, run."""
    parser = TrlParser(spec.dataclass_types)

    # Get arguments with load default YAML configuration
    cli_args = list(sys.argv[1:] if argv is None else argv)

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

            config_path = merge_config_files(spec.defaults_yaml, user_config_path)
        except (ValueError, IndexError):
            config_path = spec.defaults_yaml
    else:
        config_path = spec.defaults_yaml
    full_args = ["--config", config_path] + cli_args

    # Parse arguments
    (script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, _) = \
        (parser.parse_args_and_config(full_args,
                                      return_remaining_strings=True))

    # Run the main function
    return run_training(spec, script_args, training_args, model_args,
                        dataset_mixture_args, dnotitia_args, user_specified_args)
