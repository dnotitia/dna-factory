"""
Common utilities for training scripts (SFT, DPO, etc.)

This module contains shared setup and logging functions that are identical across
different training scripts. Core training logic remains in each script for clarity.
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM
from trl import get_quantization_config

from dna_factory.utils.colorize_args import format_args_with_colors
from dna_factory.utils.colorize_logging import ColoredFormatter, format_logs_with_colors


def setup_logging(training_args, trainer_package_name):
    """
    Configure logging for the training script and all relevant packages.
    
    Args:
        training_args: Training arguments with logging configuration
        trainer_package_name: Name of the trainer package (e.g., 'dna_factory.dnotitia_sft_trainer')
    
    Returns:
        logger: Configured logger instance
    """
    # Create console handler and set formatter
    console_colored_handler = logging.StreamHandler(sys.stdout)
    console_colored_handler.setFormatter(ColoredFormatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.basicConfig(handlers=[console_colored_handler])

    # Set logging level
    log_level = training_args.get_process_log_level()
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Configure essential package loggers
    for package_name in ['huggingface_hub', 'datasets', 'tokenizers', 'transformers', 'torch', 'accelerate', 'trl',
                         trainer_package_name]:
        package_logger = logging.getLogger(package_name)
        package_logger.setLevel(log_level)
        for handler in package_logger.handlers[:]:
            package_logger.removeHandler(handler)
        package_logger.addHandler(console_colored_handler)
        # Prevent propagation to parent loggers to avoid duplicate messages
        package_logger.propagate = False

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    # Suppress asyncio warnings from wandb
    logging.getLogger('asyncio').setLevel(logging.ERROR)

    return logger


def print_dna_factory_banner(logger, script_file_path):
    """
    Print the DNA Factory ASCII art banner with version information.
    
    Args:
        logger: Logger instance to use for printing
        script_file_path: Path to the script file (typically __file__)
    """
    logger.info("==========================================================================================")
    logger.info("██████╗ ███╗   ██╗ █████╗     ███████╗ █████╗  ██████╗████████╗ ██████╗ ██████╗ ██╗   ██╗")
    logger.info("██╔══██╗████╗  ██║██╔══██╗    ██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝")
    logger.info("██║  ██║██╔██╗ ██║███████║    █████╗  ███████║██║        ██║   ██║   ██║██████╔╝ ╚████╔╝ ")
    logger.info("██║  ██║██║╚██╗██║██╔══██║    ██╔══╝  ██╔══██║██║        ██║   ██║   ██║██╔══██╗  ╚██╔╝  ")
    logger.info("██████╔╝██║ ╚████║██║  ██║    ██║     ██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║   ██║   ")
    logger.info("╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ")
    logger.info("==========================================================================================")
    logger.info("🧬 LLM Post-Training Platform by Dnotitia Inc. 🧬")
    
    # Get version from VERSION file
    try:
        version_file = os.path.join(os.path.dirname(script_file_path), 'VERSION')
        with open(version_file, 'r') as f:
            version = f.read().strip()
        logger.info(f"🏷️ Version: {version}")
    except Exception:
        logger.info("🏷️ Version: Unknown")
    
    logger.info("==========================================================================================")


def print_training_start_message(logger, training_type):
    """
    Print the training start message.
    
    Args:
        logger: Logger instance to use for printing
        training_type: Type of training (e.g., "SFT", "DPO")
    """
    logger.info("")
    logger.info(f"Running {training_type} training script...")
    logger.info("")


def print_auto_generated_output_dir(logger, output_dir):
    """
    Print the auto-generated output directory with highlighting.
    
    Args:
        logger: Logger instance to use for printing
        output_dir: The auto-generated output directory path
    """
    YELLOW = '\033[33m'  # Bright yellow color
    RESET = '\033[0m'
    logger.info("Auto-generated output directory:")
    logger.info(f"{YELLOW}{output_dir}{RESET}")
    logger.info("")


def print_environment_and_arguments(logger, script_args, training_args, model_args, 
                                   dataset_mixture_args, dnotitia_args, user_specified_args,
                                   trainer_type):
    """
    Print OS environment variables and all parsed arguments in a formatted way.
    
    Args:
        logger: Logger instance to use for printing
        script_args: Script arguments
        training_args: Training arguments
        model_args: Model arguments
        dataset_mixture_args: Dataset mixture arguments
        dnotitia_args: Dnotitia-specific arguments
        user_specified_args: Set of user-specified argument names
        trainer_type: Type of trainer (e.g., "SFT", "DPO")
    """
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(" OS ENVIRONMENT VARIABLES:")
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(format_logs_with_colors("CUDA_VISIBLE_DEVICES"))
    logger.info(format_logs_with_colors("WORLD_SIZE"))
    logger.info(format_logs_with_colors("RANK"))
    logger.info(format_logs_with_colors("LOCAL_RANK"))
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(" SCRIPT ARGUMENTS:")
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(format_args_with_colors(vars(script_args), user_specified_args))
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(" TRAINING ARGUMENTS:")
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(format_args_with_colors(vars(training_args), user_specified_args))
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(" MODEL ARGUMENTS:")
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(format_args_with_colors(vars(model_args), user_specified_args))
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(" DATASET MIXTURE ARGUMENTS:")
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(format_args_with_colors(vars(dataset_mixture_args), user_specified_args))
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(f" DNOTITIA {trainer_type} TRAINER ARGUMENTS:")
    logger.info("------------------------------------------------------------------------------------------")
    logger.info(format_args_with_colors(vars(dnotitia_args), user_specified_args))
    logger.info("------------------------------------------------------------------------------------------")
    logger.info("")


def resolve_trust_remote_code(model_args, training_args=None):
    """Resolve `trust_remote_code` across TRL versions.

    TRL 1.9 moved the flag off `ModelConfig` and onto the trainer config
    (`SFTConfig`/`DPOConfig`/`GRPOConfig`), so a YAML `trust_remote_code: true` now lands in
    `training_args`. Prefer `model_args` when it still carries the field (older TRL) and fall
    back to `training_args`, so both layouts work.
    """
    for args in (model_args, training_args):
        value = getattr(args, "trust_remote_code", None)
        if value is not None:
            return value
    return False


def create_model_kwargs(model_args, training_args, dnotitia_args):
    """
    Create model initialization kwargs including quantization config.
    
    Args:
        model_args: Model arguments
        training_args: Training arguments
        dnotitia_args: Dnotitia-specific arguments
    
    Returns:
        dict: Model kwargs ready for AutoModelForCausalLM.from_pretrained()
    """
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=resolve_trust_remote_code(model_args, training_args),
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
    )

    # Quantization config
    quantization_config = get_quantization_config(model_args)
    # TRL 1.9 delegates device placement to Accelerate/the trainer.  Its legacy
    # `get_kbit_device_map()` helper was removed, and setting a device map here
    # conflicts with distributed training (where the map must remain `None`).
    model_kwargs["quantization_config"] = quantization_config

    return model_kwargs


def save_training_results(trainer, train_result, dataset, script_args, training_args):
    """
    Save training metrics, model, and related artifacts.
    
    Args:
        trainer: The trainer instance
        train_result: Training result from trainer.train()
        dataset: The dataset used for training
        script_args: Script arguments
        training_args: Training arguments
    """
    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    # Drop None-valued metrics: TRL reports None for a reward function whose rewards were all
    # NaN/None (e.g. an LLM judge that never produced a parseable score), and
    # `Trainer.log_metrics` crashes formatting None — which would skip the model save below.
    metrics = {key: value for key, value in metrics.items() if value is not None}
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save model
    trainer.save_model(training_args.output_dir)

    # Save everything else on main process
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(
            dataset_name=script_args.dataset_name,
            tags=["DNA Factory"],
        )
        # Restore k,v cache for fast inference
        # trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Push to hub if requested
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def load_model(model_path, **model_kwargs):
    """Load a model, choosing the right Auto class from its config.

    Qwen3.5 checkpoints are conditional-generation classes (e.g. `Qwen3_5ForConditionalGeneration`)
    registered under `AutoModelForMultimodalLM`; loading them via `AutoModelForCausalLM` would fall
    back to a text-only variant (`Qwen3_5ForCausalLM`) and save a mismatched architecture, breaking
    vLLM serving. Plain causal LMs (e.g. `Qwen3-0.6B` / `Qwen3ForCausalLM`) are NOT in the
    MultimodalLM mapping and must load via `AutoModelForCausalLM`. Dispatch on the config's mapping
    membership so both round-trip correctly.
    """
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=model_kwargs.get("trust_remote_code", False)
    )
    if type(config) in AutoModelForMultimodalLM._model_mapping:
        return AutoModelForMultimodalLM.from_pretrained(model_path, **model_kwargs)
    return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)


def set_use_cache(model, value):
    """Set `use_cache` on a (possibly composite) model config, post-load.

    See `create_model_kwargs`: Qwen3.5's composite config keeps `use_cache` under
    `config.text_config`, so setting only the top-level attribute is silently ineffective for the
    forward pass. Set it on both the top-level config and the text sub-config where present so it
    works for plain causal LMs and composite (Qwen3.5) models alike.
    """
    config = model.config
    config.use_cache = value
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.use_cache = value
