"""Shared training skeleton for SFT / DPO / GRPO / Distillation entry points.

Each script (sft.py, dpo.py, grpo.py, distill.py) keeps only its method-specific
logic — model loading, dataset normalization, trainer construction — and describes
it as a `TrainingSpec`. `run_training` executes the identical setup/train/save flow
for all of them; `cli_main` implements the identical `__main__` CLI bootstrap.

Common setup/logging/model helpers (previously in
`dna_factory/dnotitia_trainer_commons.py`) now live here as well, so this module
is the single place for all training-loop-shared code.
"""

import logging
import multiprocessing
import os
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field

import datasets
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import TrlParser, get_peft_config, get_quantization_config

from dna_factory.periodic_checkpoint import (
    PeriodicCheckpointCallback,
    parse_duration_to_seconds,
)
from dna_factory.utils.colorize_args import format_args_with_colors, parse_user_args
from dna_factory.utils.colorize_logging import ColoredFormatter, format_logs_with_colors
from dna_factory.utils.config_merger import merge_config_files
from dna_factory.utils.output_dir_generator import generate_auto_output_dir

logger = logging.getLogger(__name__)


def setup_logging(training_args, trainer_package_name):
    """Configure logging for the training script and all relevant packages."""
    # Create console handler and set formatter
    console_colored_handler = logging.StreamHandler(sys.stdout)
    console_colored_handler.setFormatter(
        ColoredFormatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(handlers=[console_colored_handler])

    # Set logging level
    log_level = training_args.get_process_log_level()
    _logger = logging.getLogger(__name__)
    _logger.setLevel(log_level)

    # Configure essential package loggers
    for package_name in [
        "huggingface_hub",
        "datasets",
        "tokenizers",
        "transformers",
        "torch",
        "accelerate",
        "trl",
        trainer_package_name,
    ]:
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
    logging.getLogger("asyncio").setLevel(logging.ERROR)

    return _logger


def get_dna_factory_version(script_file_path):
    """Read the package version from pyproject.toml (single source of truth).

    Walks up from the calling script's directory to find pyproject.toml,
    so the lookup keeps working even if entry-point scripts move into subdirs.
    """
    try:
        current = os.path.dirname(os.path.abspath(script_file_path))
        while True:
            candidate = os.path.join(current, "pyproject.toml")
            if os.path.isfile(candidate):
                with open(candidate, "rb") as f:
                    return tomllib.load(f)["project"]["version"]
            parent = os.path.dirname(current)
            if parent == current:
                return None
            current = parent
    except Exception:
        return None


def print_dna_factory_banner(_logger, script_file_path):
    """Print the DNA Factory ASCII art banner with version information."""
    _logger.info(
        "=========================================================================================="
    )
    _logger.info(
        "██████╗ ███╗   ██╗ █████╗     ███████╗ █████╗  ██████╗████████╗ ██████╗ ██████╗ ██╗   ██╗"
    )
    _logger.info(
        "██╔══██╗████╗  ██║██╔══██╗    ██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗╚██╗ ██╔╝"
    )
    _logger.info(
        "██║  ██║██╔██╗ ██║███████║    █████╗  ███████║██║        ██║   ██║   ██║██████╔╝ ╚████╔╝ "
    )
    _logger.info(
        "██║  ██║██║╚██╗██║██╔══██║    ██╔══╝  ██╔══██║██║        ██║   ██║   ██║██╔══██╗  ╚██╔╝  "
    )
    _logger.info(
        "██████╔╝██║ ╚████║██║  ██║    ██║     ██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║   ██║   "
    )
    _logger.info(
        "╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   "
    )
    _logger.info(
        "=========================================================================================="
    )
    _logger.info("🧬 LLM Post-Training Platform by Dnotitia Inc. 🧬")

    # Get version from pyproject.toml (single source of truth)
    version = get_dna_factory_version(script_file_path)
    if version is not None:
        _logger.info(f"🏷️ Version: v{version}")
    else:
        _logger.info("🏷️ Version: Unknown")

    _logger.info(
        "=========================================================================================="
    )


def print_training_start_message(_logger, training_type):
    """Print the training start message."""
    _logger.info("")
    _logger.info(f"Running {training_type} training script...")
    _logger.info("")


def print_auto_generated_output_dir(_logger, output_dir):
    """Print the auto-generated output directory with highlighting."""
    YELLOW = "\033[33m"  # Bright yellow color
    RESET = "\033[0m"
    _logger.info("Auto-generated output directory:")
    _logger.info(f"{YELLOW}{output_dir}{RESET}")
    _logger.info("")


def print_environment_and_arguments(
    _logger,
    script_args,
    training_args,
    model_args,
    dataset_mixture_args,
    dnotitia_args,
    user_specified_args,
    trainer_type,
):
    """Print OS environment variables and all parsed arguments in a formatted way."""
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(" OS ENVIRONMENT VARIABLES:")
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(format_logs_with_colors("CUDA_VISIBLE_DEVICES"))
    _logger.info(format_logs_with_colors("WORLD_SIZE"))
    _logger.info(format_logs_with_colors("RANK"))
    _logger.info(format_logs_with_colors("LOCAL_RANK"))
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(" SCRIPT ARGUMENTS:")
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(format_args_with_colors(vars(script_args), user_specified_args))
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(" TRAINING ARGUMENTS:")
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(format_args_with_colors(vars(training_args), user_specified_args))
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(" MODEL ARGUMENTS:")
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(format_args_with_colors(vars(model_args), user_specified_args))
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(" DATASET MIXTURE ARGUMENTS:")
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(
        format_args_with_colors(vars(dataset_mixture_args), user_specified_args)
    )
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(f" DNOTITIA {trainer_type} TRAINER ARGUMENTS:")
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info(format_args_with_colors(vars(dnotitia_args), user_specified_args))
    _logger.info(
        "------------------------------------------------------------------------------------------"
    )
    _logger.info("")


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
    """Create model initialization kwargs including quantization config."""
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": resolve_trust_remote_code(model_args, training_args),
        "attn_implementation": model_args.attn_implementation,
        "dtype": model_args.dtype,
        # use_cache=False if training_args.gradient_checkpointing else True,
    }

    # Quantization config
    quantization_config = get_quantization_config(model_args)
    # TRL 1.9 delegates device placement to Accelerate/the trainer.  Its legacy
    # `get_kbit_device_map()` helper was removed, and setting a device map here
    # conflicts with distributed training (where the map must remain `None`).
    model_kwargs["quantization_config"] = quantization_config

    return model_kwargs


def save_training_results(trainer, train_result, dataset, script_args, training_args):
    """Save training metrics, model, and related artifacts."""
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
    script_file: str  # caller __file__ (for the banner pyproject version lookup)

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
