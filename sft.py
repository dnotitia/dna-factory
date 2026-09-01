"""
Supervised Fine-Tuning (SFT) training script for DNA Factory.

Note: This file intentionally maintains some duplication with dpo.py for readability and clarity.
Common utilities (logging, banners, argument printing) are extracted to dnotitia_trainer_commons.py,
while core training logic remains here for easy understanding of the complete SFT flow.
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
    SFTConfig,
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
    load_model,
    set_use_cache,
    save_training_results,
)
from dna_factory.dnotitia_sft_trainer import DnotitiaSFTTrainer
from dna_factory.dnotitia_arguments import DnotitiaArguments
from dna_factory.dnotitia_dataset_mixture import (
    WeightedDatasetMixtureConfig,
    resample_by_weight,
)

# Initialize logger
logger = logging.getLogger(__name__)


def get_dataset_with_schema_alignment(mixture_config, seed=42):
    import datasets as ds
    from datasets import concatenate_datasets, DatasetDict

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
            'SFT',
        )
        training_args.output_dir = auto_output_dir

    # Setup logging
    logger = setup_logging(training_args, 'dna_factory.dnotitia_sft_trainer')

    # Print DNA Factory banner
    print_dna_factory_banner(logger, __file__)

    # Print the script start message
    print_training_start_message(logger, "SFT")

    # Log auto-generated output directory if applicable
    if auto_generated_dir:
        print_auto_generated_output_dir(logger, training_args.output_dir)

    # Print the parsed arguments
    print_environment_and_arguments(
        logger, script_args, training_args, model_args,
        dataset_mixture_args, dnotitia_args, user_specified_args,
        trainer_type="SFT"
    )

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

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

    # MoE + DeepSpeed ZeRO3: expert 접근이 라우팅으로 비결정적 → ZeRO3 param-trace 오류 및 Liger
    # fused-MoE 커널이 gather 안 된(파티션된) expert 가중치를 인덱싱해 illegal memory access.
    # MoE 블록을 z3 leaf 로 지정해 블록 단위로 param 을 한 번에 gather (표준 MoE+ZeRO3 fix).
    # 대상 클래스 없는 모델(dense/타 MoE)은 Exception → skip
    try:
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])
        logger.info("set_z3_leaf_modules: Qwen3_5MoeSparseMoeBlock -> z3 leaf")
    except Exception as _e:
        logger.warning(f"set_z3_leaf_modules skip: {_e}")

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
        dataset = get_dataset_with_schema_alignment(dataset_mixture_args, seed=training_args.seed)
    elif script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    def preprocess_thinking_data(example):
        # Add reasoning_content from thinking field for Qwen3 compatibility
        messages = []
        for msg in example["messages"]:
            new_msg = dict(msg)
            if msg.get("thinking"):
                new_msg["reasoning_content"] = msg["thinking"]
            messages.append(new_msg)
        return {"messages": messages}
    dataset = dataset.map(preprocess_thinking_data)

    # Initialize the Dnotitia SFT trainer
    trainer = DnotitiaSFTTrainer(
        model=model,
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
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, WeightedDatasetMixtureConfig, DnotitiaArguments)
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

            config_path = merge_config_files("configs/_defaults-SFT.yaml", user_config_path)
        except (ValueError, IndexError):
            config_path = "configs/_defaults-SFT.yaml"
    else:
        config_path = "configs/_defaults-SFT.yaml"
    full_args = ["--config", config_path] + cli_args

    # Parse arguments
    (script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, _) = \
        (parser.parse_args_and_config(full_args,
                                      return_remaining_strings=True))

    # Run the main function
    main(script_args, training_args, model_args, dataset_mixture_args, dnotitia_args, user_specified_args)
