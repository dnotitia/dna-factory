"""
On-Policy Distillation training script for DNA Factory.

Note: This file intentionally maintains some duplication with sft.py/dpo.py/grpo.py for readability and
clarity. Common utilities (logging, banners, argument printing) are extracted to
dnotitia_trainer_commons.py, while core training logic remains here for easy understanding of the complete
distillation flow.

What on-policy distillation is (https://thinkingmachines.ai/blog/on-policy-distillation/):
- The *student* samples its own completions (on-policy), so it is trained on the states it actually visits.
  This removes the exposure bias of off-policy/SFT distillation, which only ever sees teacher trajectories.
- The *teacher* then grades those completions token by token. Every token carries supervision (a dense
  signal), unlike RL where a whole trajectory collapses to one scalar reward — which is where the
  order-of-magnitude compute win over RL comes from.
- The objective is the per-token reverse KL, KL(student || teacher). It is mode-seeking (the student commits
  to one teacher behavior instead of averaging several) and "unhackable": low KL always means the student is
  reproducing teacher behavior.

Key distillation-specific differences from the other scripts:
- Two models: the trained `model` (student) and a frozen `teacher_model`. The teacher is named in the
  training config via `teacher_model_name_or_path`, and it must share the student's vocabulary (the loss
  compares full next-token distributions; TRL raises on a vocab_size mismatch).
- Online generation: like GRPO, completions are produced during training (vLLM colocate by default), so the
  dataset is prompt-only. Unlike GRPO there is no reward function — the teacher *is* the supervision.
- Both models are passed to the trainer as strings so `training_args.{model,teacher_model}_init_kwargs` are
  honored and distributed device_map handling works.
- `beta` selects the divergence itself (1.0 = reverse KL, 0.0 = forward KL, 0.5 = JSD); it is NOT GRPO's
  reference-model KL-penalty coefficient. There is no reference model here.
When modifying shared utility logic, update dnotitia_trainer_commons.py.
"""

import logging
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    DistillationConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
    get_quantization_config,
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
    resolve_trust_remote_code,
    save_training_results,
)
from dna_factory.dnotitia_distillation_trainer import DnotitiaDistillationTrainer
from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetMixtureConfig,
    resample_by_weight,
)

# vLLM 0.20.0 bundles an incomplete vendored `deep_gemm`, and its kernel warmup crashes on
# Hopper/Blackwell GPUs even for bf16 models. FP8 GEMM kernels are not needed for bf16
# training, so DeepGEMM is disabled by default (override by exporting VLLM_USE_DEEP_GEMM=1).
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

# Initialize logger
logger = logging.getLogger(__name__)


def _normalize_dataset_for_distillation(dataset):
    """
    Per-dataset mechanical normalization to the prompt-only schema the distillation trainer consumes
    (factored out so it is directly testable with a plain dataset):

      - `messages` present → the turns before the last assistant turn become `prompt` (role/content only);
        a dataset with no assistant turn keeps all of its turns. The assistant turn itself is discarded:
        the student writes its own completion and the teacher grades it, so a reference answer is unused.
      - `prompt` present (no `messages`) → a plain string becomes a single user turn; a list is kept but
        reduced to role/content only.
      - Neither present → a clear error (nothing else to build a prompt from).

    Every other column is dropped. Unlike GRPO — where extra columns are forwarded to reward functions —
    distillation has no reward functions, so a prompt-only schema also makes mixture concatenation trivial.
    """
    feats = dataset.features
    if "messages" in feats:

        def split_messages(example):
            messages = example["messages"]
            assistant_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            prompt_msgs = messages[:assistant_idxs[-1]] if assistant_idxs else messages
            return {"prompt": [{"role": m["role"], "content": m["content"]} for m in prompt_msgs]}

        return dataset.map(split_messages, remove_columns=dataset.column_names)
    elif "prompt" in feats:

        def normalize_prompt(example):
            p = example["prompt"]
            msgs = p if isinstance(p, list) else [{"role": "user", "content": p}]
            return {"prompt": [{"role": m["role"], "content": m["content"]} for m in msgs]}

        return dataset.map(normalize_prompt, remove_columns=dataset.column_names)
    else:
        raise ValueError(
            f"Dataset has neither `messages` nor `prompt` (columns: {dataset.column_names}); "
            "cannot build a prompt-only distillation schema."
        )


def get_dataset_with_schema_alignment(mixture_config, seed=42):
    """
    Distillation mixture loader (analogous to sft.py/grpo.py:get_dataset_with_schema_alignment).

    TRL's stock get_dataset() just concatenates the datasets, which raises when they have different columns.
    Mixture datasets here are heterogeneous (conversational `messages` vs plain `prompt`, plus arbitrary
    extra columns), so each one is reduced to the same single-column prompt-only schema by
    `_normalize_dataset_for_distillation` before concatenation.

    The per-dataset `weight` (a size multiplier; see dna_factory/dnotitia_dataset_mixture.py) is applied the
    same way sft.py applies it.

    A local save_to_disk directory is loaded via load_from_disk; otherwise the path is loaded from the HF hub.
    """
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

        if dataset_config.columns is not None:
            dataset = dataset.select_columns(dataset_config.columns)

        dataset = _normalize_dataset_for_distillation(dataset)

        # Apply the per-dataset sample weight (size multiplier; 1.0 = no change)
        weight = getattr(dataset_config, "weight", 1.0)
        if weight != 1.0:
            n_before = len(dataset)
            dataset = resample_by_weight(dataset, weight, seed=seed)
            logger.info(f"  weight={weight}: {n_before} -> {len(dataset)} examples")

        datasets_list.append(dataset)

    combined = concatenate_datasets(datasets_list)
    logger.info(f"Created distillation dataset mixture with {len(combined)} examples")
    return DatasetDict({"train": combined})


def main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args=None):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Use empty set if no user args provided
    if user_specified_args is None:
        user_specified_args = set()

    # Note: no `training_args.dataset_num_proc` here. Unlike SFTConfig/DPOConfig/GRPOConfig,
    # `DistillationConfig` has no such field (the trainer does not tokenize the dataset up front — it
    # generates completions online), so setting it would only plant an unused attribute.

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
            'DISTILL',
        )
        training_args.output_dir = auto_output_dir

    # Setup logging
    logger = setup_logging(training_args, 'dna_factory.dnotitia_distillation_trainer')

    # Print DNA Factory banner
    print_dna_factory_banner(logger, __file__)

    # Print the script start message
    print_training_start_message(logger, "On-Policy Distillation")

    # Log auto-generated output directory if applicable
    if auto_generated_dir:
        print_auto_generated_output_dir(logger, training_args.output_dir)

    # Print the parsed arguments
    print_environment_and_arguments(
        logger, script_args, training_args, model_args,
        dataset_mixture_args, dnotitia_args, user_specified_args,
        trainer_type="DISTILL"
    )

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # A teacher is the entire supervision signal here, so there is nothing to train without one
    if not training_args.teacher_model_name_or_path:
        raise ValueError(
            "On-policy distillation requires a teacher. Set `teacher_model_name_or_path` (it must share the "
            "student's vocabulary)."
        )

    # Model init kwargs (distillation-specific): both models are passed to the trainer as strings, so init
    # kwargs go through the training config. DistillationTrainer manages `use_cache` itself during generation
    # and training forwards, so it is intentionally omitted here.
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=resolve_trust_remote_code(model_args, training_args),
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    # The teacher gets the same runtime knobs but never `model_revision` (that pins the *student*); its own
    # revision is applied by the trainer from `teacher_model_revision`.
    training_args.teacher_model_init_kwargs = dict(
        trust_remote_code=resolve_trust_remote_code(model_args, training_args),
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )
    quantization_config = get_quantization_config(model_args)

    # Create tokenizer (DistillationTrainer requires left padding for generation and sets it internally)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=resolve_trust_remote_code(model_args, training_args),
        use_fast=True
    )

    # Load the dataset (prompt-only: a `prompt` column, plain text or conversational)
    if dataset_mixture_args.datasets:
        logger.info(
            "The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        # Schema-aligning loader (not TRL's stock get_dataset): normalizes mixed datasets to a common
        # prompt-only schema and applies each dataset's `weight`.
        dataset = get_dataset_with_schema_alignment(dataset_mixture_args, seed=training_args.seed)
    elif script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Note: no thinking → reasoning_content preprocessing here. Distillation datasets are prompt-only, so
    # there are no pre-existing assistant turns carrying a `thinking` field (the student writes the
    # completions online during training).

    # Initialize the Dnotitia distillation trainer
    trainer = DnotitiaDistillationTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        quantization_config=quantization_config,
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
    dataclass_types = (ScriptArguments, DistillationConfig, ModelConfig, WeightedDatasetMixtureConfig,
                       DnotitiaArguments)
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

            config_path = merge_config_files("configs/_defaults-Distill.yaml", user_config_path)
        except (ValueError, IndexError):
            config_path = "configs/_defaults-Distill.yaml"
    else:
        config_path = "configs/_defaults-Distill.yaml"
    full_args = ["--config", config_path] + cli_args

    # Parse arguments
    (script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, _) = \
        (parser.parse_args_and_config(full_args,
                                      return_remaining_strings=True))

    # Run the main function
    main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args)
