"""Compare effective configs, including the repository's keyed dataset merge."""

from pathlib import Path

import yaml

from dna_factory.utils.config_merger import merge_config_files

ROOT = Path(__file__).resolve().parents[1]


def effective(method, defaults):
    merged = Path(
        merge_config_files(
            ROOT / f"configs/_defaults-{defaults}.yaml",
            ROOT / f"configs/{method}/qwen3-0.6B-sampling-comparison.yaml",
        )
    )
    try:
        return yaml.safe_load(merged.read_text())
    finally:
        merged.unlink()


def test_training_settings_match_sft_defaults_and_each_other():
    configs = [
        effective("SFT", "SFT"),
        effective("SamplingSFT", "SFT"),
        effective("SDFT", "SDFT"),
    ]
    defaults = yaml.safe_load((ROOT / "configs/_defaults-SFT.yaml").read_text())
    for key in (
        "seed",
        "learning_rate",
        "lr_scheduler_type",
        "lr_scheduler_kwargs",
        "warmup_steps",
        "num_train_epochs",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "bf16",
        "tf32",
        "gradient_checkpointing",
        "attn_implementation",
    ):
        assert all(config[key] == defaults[key] for config in configs), key
    assert len({c["model_name_or_path"] for c in configs}) == 1
    assert configs[0]["packing"] is configs[1]["packing"] is False
    ignored = {"run_name", "datasets"}
    assert {k: v for k, v in configs[0].items() if k not in ignored} == {
        k: v for k, v in configs[1].items() if k not in ignored
    }


def test_sdft_replaces_default_dataset_without_stale_column_mappings():
    config = effective("SDFT", "SDFT")
    assert len(config["datasets"]) == 1
    dataset = config["datasets"][0]
    assert dataset["path"] == "json"
    assert dataset["data_files"] == "data/sampling-sft/sdft.jsonl"
    assert dataset["columns"] == ["prompt", "privileged_context"]
    assert dataset["prompt_column"] == "prompt"
    assert dataset["privileged_context_column"] == "privileged_context"
