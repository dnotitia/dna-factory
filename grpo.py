"""
Group Relative Policy Optimization (GRPO) training script for DNA Factory.

Note: This file intentionally maintains some duplication with sft.py/dpo.py for readability and clarity.
Common utilities (logging, banners, argument printing) are extracted to dnotitia_trainer_commons.py,
while core training logic remains here for easy understanding of the complete GRPO flow.
Key GRPO-specific differences:
- Online RL: completions are generated during training and scored by reward functions.
- Reward signal is required: built-in reward functions from trl.rewards, dotted import paths,
  and/or a sequence-classification reward model (`reward_model_name_or_path`).
- The model is passed to the trainer as a string (not pre-instantiated) so that
  `training_args.model_init_kwargs` is honored and distributed device_map handling works.
- No ref_model parameter: GRPOTrainer creates an internal reference model only when `beta != 0`.
When modifying shared utility logic, update dnotitia_trainer_commons.py.
"""

import importlib
import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    GRPOConfig,
    TrlParser,
    get_peft_config,
    get_quantization_config,
)
from trl.scripts.utils import DatasetConfig

from dna_factory.utils.colorize_args import parse_user_args
from dna_factory.utils.config_merger import merge_config_files
from dna_factory.utils.output_dir_generator import generate_auto_output_dir
from dna_factory.periodic_checkpoint import PeriodicCheckpointCallback
from dna_factory.dnotitia_trainer_commons import (
    setup_logging,
    print_dna_factory_banner,
    print_training_start_message,
    print_auto_generated_output_dir,
    print_environment_and_arguments,
    resolve_trust_remote_code,
    save_training_results,
)
from dna_factory.dnotitia_grpo_trainer import DnotitiaGRPOTrainer
from dna_factory.dnotitia_arguments import DnotitiaArguments

# vLLM 0.20.0 bundles an incomplete vendored `deep_gemm`, and its kernel warmup crashes on
# Hopper/Blackwell GPUs even for bf16 models. FP8 GEMM kernels are not needed for bf16
# training, so DeepGEMM is disabled by default (override by exporting VLLM_USE_DEEP_GEMM=1).
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

# Initialize logger
logger = logging.getLogger(__name__)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_name_or_path (`str`, *optional*):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using `PreTrainedModel.save_pretrained`. Loaded internally by
            GRPOTrainer as a sequence-classification model with a single label.
        reward_funcs (`list[str]`, *optional*):
            Reward functions to use. Supported bare names (zero-argument only): `"accuracy_reward"`,
            `"reasoning_accuracy_reward"`, `"think_format_reward"`. A TRL reward factory (e.g.
            `get_soft_overlong_punishment`) needs a module-level instance built with its arguments
            first — see `dna_factory/rewards/my_rewards.py` — then reference that instance via a dotted
            import path (e.g., `'my_lib.rewards.custom_reward'`). See docs/grpo-rewards.md for a
            quickstart, or docs/grpo-rewards-full.md for the full reward-function contract.
    """

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
    """A `DatasetConfig` with an extra `label` (provenance) field.

    `label` is a plain string tag applied to every row loaded from this dataset entry, used for
    per-sample reward routing (see `dna_factory.rewards.make_judge_reward`'s `only_label`).
    """

    label: str | None = field(
        default=None,
        metadata={
            "help": "Provenance label for rows loaded from this dataset entry (defaults to `path` if unset)."
        },
    )


@dataclass
class LabeledDatasetMixtureConfig(DatasetMixtureConfig):
    """`DatasetMixtureConfig` whose entries are `LabeledDatasetConfig` (adds per-dataset `label`).

    The annotation must stay `list[LabeledDatasetConfig]` (not bare `list`) so `HfArgumentParser` can
    introspect the element type.
    """

    datasets: list[LabeledDatasetConfig] = field(
        default_factory=list,
        metadata={"help": "List of (labeled) dataset configurations to include in the mixture."},
    )

    def __post_init__(self):
        # Convert dicts (from CLI/YAML parsing) into LabeledDatasetConfig objects.
        for idx, dataset in enumerate(self.datasets):
            if isinstance(dataset, dict):
                self.datasets[idx] = LabeledDatasetConfig(**dataset)


def resolve_reward_funcs(script_args, training_args):
    """
    Build the `reward_funcs` list GRPOTrainer expects from `script_args.reward_funcs` (built-in
    names or dotted import paths) and `script_args.reward_model_name_or_path` (prepended, loaded by
    GRPOTrainer itself as a sequence-classification model). Note: a dotted path is resolved via a
    bare `getattr` — no arguments are passed — so a configurable reward must already be a
    fully-built instance in its own module (see docs/grpo-rewards.md's factory-in-user-module
    pattern, e.g. `dna_factory.rewards.make_judge_reward`, `dna_factory.rewards.soft_overlong_penalty`).
    """
    # Import lazily so optional reward dependencies (e.g. math_verify) are only required when used
    # For reward functions in detail: see https://huggingface.co/docs/trl/rewards
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
    # A reward model id string is loaded internally by GRPOTrainer as AutoModelForSequenceClassification
    if script_args.reward_model_name_or_path:
        reward_funcs.append(script_args.reward_model_name_or_path)

    for func_name in script_args.reward_funcs or []:
        if func_name in reward_funcs_registry:
            reward_funcs.append(reward_funcs_registry[func_name])
        elif "." in func_name:
            # Dotted import path (e.g. 'my_lib.rewards.custom_reward'), resolved relative to the cwd
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
    """
    Per-dataset mechanical normalization (the unit of work `get_dataset_with_schema_alignment` loops
    over; factored out so it's directly testable with a plain `(dataset, label)` pair):

      - `messages` present → split into `prompt` (turns before the last assistant turn, role/content
        only) and `expected_output` (that last assistant turn's content, or None if the dataset has
        no assistant turn at all); `messages` is dropped.
      - `prompt` present (no `messages`) → a plain string becomes a single user turn; a list is kept
        but reduced to role/content only. Every other existing column (`solution`, a pre-existing
        `expected_output`, anything else) is left untouched.
      - Neither present → a clear error (nothing else to build a GRPO prompt from).

    Then injects a `label` column (the dataset's provenance tag; see `LabeledDatasetConfig`) so reward
    functions can route per-sample via `dna_factory.rewards.make_judge_reward`'s `only_label`.
    """
    feats = dataset.features
    if "messages" in feats:

        def split_messages(example):
            messages = example["messages"]
            assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            if assistant_idxs:
                last = assistant_idxs[-1]
                prompt_msgs = messages[:last]
                expected = messages[last]["content"]
            else:
                prompt_msgs = messages
                expected = None
            return {
                "prompt": [{"role": m["role"], "content": m["content"]} for m in prompt_msgs],
                "expected_output": expected,
            }

        dataset = dataset.map(split_messages, remove_columns=["messages"])
    elif "prompt" in feats:

        def normalize_prompt(example):
            p = example["prompt"]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
            return {"prompt": [{"role": m["role"], "content": m["content"]} for m in msgs]}

        dataset = dataset.map(normalize_prompt)
    else:
        raise ValueError(
            f"Dataset has neither `messages` nor `prompt` (columns: {dataset.column_names}); "
            "cannot build a GRPO prompt-only schema."
        )
    return dataset.add_column("label", [label] * len(dataset))


def get_dataset_with_schema_alignment(mixture_config):
    """
    GRPO mixture loader (analogous to sft.py:get_dataset_with_schema_alignment).

    TRL's stock get_dataset() just concatenates the datasets, which raises when they have different
    columns. Mixture datasets here are heterogeneous (conversational `messages` vs plain `prompt`,
    plus arbitrary extra columns such as `solution`/`expected_output`), so each dataset is normalized
    to a `prompt` (+ `label`) schema by `_normalize_dataset_for_grpo` while every other column is left
    alone, and the results are concatenated. `datasets.concatenate_datasets` natively aligns mismatched
    schemas across the mixture (missing columns are added and filled with `None`; see the loader unit
    test), so no manual column padding is needed here.

    No `task_type`, no column-shape-based routing: every extra column (including `label`) is simply
    forwarded by GRPOTrainer to reward functions as a keyword argument, and reward functions that
    don't apply to a given sample return `None` for it (TRL excludes it from that reward). See
    `dna_factory/rewards/generative.py`'s `make_judge_reward` for the reward side of this pattern.

    A local save_to_disk directory is loaded via load_from_disk; otherwise the path is loaded from
    the HF hub.
    """
    import os
    import datasets as ds
    from datasets import concatenate_datasets, DatasetDict

    datasets_list = []
    for dataset_config in mixture_config.datasets:
        path = dataset_config.path
        logger.info(f"Loading dataset for mixture: {path} (config name: {dataset_config.name})")
        if os.path.isdir(path):
            dataset = ds.load_from_disk(path)
            if isinstance(dataset, DatasetDict):
                dataset = dataset[dataset_config.split or "train"]
        else:
            dataset = ds.load_dataset(path=path, name=dataset_config.name, split=dataset_config.split)
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
            'GRPO',
        )
        training_args.output_dir = auto_output_dir

    # Setup logging
    logger = setup_logging(training_args, 'dna_factory.dnotitia_grpo_trainer')

    # Print DNA Factory banner
    print_dna_factory_banner(logger, __file__)

    # Print the script start message
    print_training_start_message(logger, "GRPO")

    # Log auto-generated output directory if applicable
    if auto_generated_dir:
        print_auto_generated_output_dir(logger, training_args.output_dir)

    # Print the parsed arguments
    print_environment_and_arguments(
        logger, script_args, training_args, model_args,
        dataset_mixture_args, dnotitia_args, user_specified_args,
        trainer_type="GRPO"
    )

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Resolve reward functions (GRPO-specific)
    reward_funcs = resolve_reward_funcs(script_args, training_args)

    # Model init kwargs (GRPO-specific): the model is passed to the trainer as a string, so init kwargs
    # go through `training_args.model_init_kwargs`. GRPOTrainer manages `use_cache` itself during
    # generation and training forwards, so it is intentionally omitted here.
    # Due to online RL nature, GRPOTrainer itself handles model instantiation unlike SFT/DPO.
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=resolve_trust_remote_code(model_args, training_args),
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)

    # Create tokenizer (GRPOTrainer sets the pad token and applies left/right padding internally)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=resolve_trust_remote_code(model_args, training_args),
        use_fast=True
    )

    # Load the dataset (prompt-only: a `prompt` column, plain text or conversational; extra columns
    # such as `expected_output`/`solution` are forwarded to the reward functions as keyword arguments)
    if dataset_mixture_args.datasets:
        logger.info(
            "The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        # Schema-aligning loader (not TRL's stock get_dataset): normalizes mixed datasets to a
        # common prompt-only schema and derives `expected_output` for reference-guided judging.
        dataset = get_dataset_with_schema_alignment(dataset_mixture_args)
    elif script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Note: no thinking → reasoning_content preprocessing here. GRPO datasets are prompt-only, so
    # there are no pre-existing assistant turns carrying a `thinking` field (completions are
    # generated online during training).

    # Wall-clock periodic checkpointing (off when periodic_save_seconds <= 0)
    callbacks = []
    if dnotitia_args.periodic_save_seconds and dnotitia_args.periodic_save_seconds > 0:
        logger.info(
            f"Enabling wall-clock checkpointing every {dnotitia_args.periodic_save_seconds}s."
        )
        callbacks.append(
            PeriodicCheckpointCallback(dnotitia_args.periodic_save_seconds)
        )

    # Initialize the Dnotitia GRPO trainer
    trainer = DnotitiaGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        quantization_config=quantization_config,
        peft_config=get_peft_config(model_args),
        debug_first_n_batches=dnotitia_args.debug_first_n_batches,
        dynamic_sampling=dnotitia_args.dynamic_sampling,
        callbacks=callbacks or None,
        dynamic_sampling_max_rounds=dnotitia_args.dynamic_sampling_max_rounds,
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
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig, LabeledDatasetMixtureConfig, DnotitiaArguments)
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

            config_path = merge_config_files("configs/_defaults-GRPO.yaml", user_config_path)
        except (ValueError, IndexError):
            config_path = "configs/_defaults-GRPO.yaml"
    else:
        config_path = "configs/_defaults-GRPO.yaml"
    full_args = ["--config", config_path] + cli_args

    # Parse arguments
    (script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, _) = \
        (parser.parse_args_and_config(full_args,
                                      return_remaining_strings=True))

    # Run the main function
    main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args)
