"""
Test cases for redundant args in dna_factory/training_runner.py (`cli_main`)

An arg given in the --config YAML or on the CLI with the value it would have had anyway
(the `_defaults-*.yaml` value, else the dataclass default) is redundant: it is printed in
gray instead of pink and left out of the `output_dir: auto` name.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(REPO_ROOT))

import sft
from dna_factory import training_runner
from dna_factory.utils.output_dir_generator import generate_auto_output_dir


def run_cli(monkeypatch, tmp_path, yaml_text=None, cli_args=()):
    """Run sft's cli_main up to run_training; return (args, user_specified, redundant)."""
    captured = {}

    def fake_run_training(spec, *args, redundant_args):
        captured["args"] = args[:5]
        captured["user_specified_args"] = args[5]
        captured["redundant_args"] = redundant_args

    monkeypatch.chdir(REPO_ROOT)  # defaults_yaml is relative to the repo root
    monkeypatch.setattr(training_runner, "run_training", fake_run_training)

    argv = list(cli_args)
    if yaml_text is not None:
        config = tmp_path / "user.yaml"
        config.write_text(yaml_text)
        argv = ["--config", str(config), *argv]
    training_runner.cli_main(sft.SPEC, argv)
    return captured


def auto_output_dir(captured):
    script_args, training_args, model_args, mixture_args, dnotitia_args = captured[
        "args"
    ]
    return generate_auto_output_dir(
        model_args.model_name_or_path,
        captured["user_specified_args"],
        script_args,
        training_args,
        model_args,
        mixture_args,
        dnotitia_args,
        sft.SPEC.output_dir_tag,
    )


def test_no_user_args(monkeypatch, tmp_path):
    """Without user args nothing is specified or redundant"""
    captured = run_cli(monkeypatch, tmp_path)
    assert captured["user_specified_args"] == set()
    assert captured["redundant_args"] == set()
    assert auto_output_dir(captured) == "Qwen3-0.6B-Base-SFT"


def test_yaml_values_equal_to_defaults_are_redundant(monkeypatch, tmp_path):
    """Parsed values are compared: `2.0e-5` equals the defaults YAML's `2e-5`, and so on"""
    captured = run_cli(
        monkeypatch,
        tmp_path,
        "learning_rate: 2.0e-5\n"  # _defaults-SFT.yaml has `2e-5` (a string to PyYAML)
        "packing: True\n"  # bool
        "report_to:\n  - wandb\n"  # list
        "dataset_config: null\n"  # None, a dataclass default (not in the defaults YAML)
        "weight_decay: 0.0\n"  # dataclass default (not in the defaults YAML)
        "num_train_epochs: 3\n",  # differs from the default (2)
    )
    assert captured["redundant_args"] == {
        "learning_rate",
        "packing",
        "report_to",
        "dataset_config",
        "weight_decay",
    }
    assert captured["user_specified_args"] == {"config", "num_train_epochs"}
    assert auto_output_dir(captured) == "Qwen3-0.6B-Base-SFT.ep-3"


def test_cli_values_equal_to_defaults_are_redundant(monkeypatch, tmp_path):
    """CLI strings are compared after parsing too (`"8"` == 8, `"true"` == True)"""
    captured = run_cli(
        monkeypatch,
        tmp_path,
        cli_args=[
            "--gradient_accumulation_steps",
            "8",
            "--packing",
            "true",
            "--learning_rate=0.00002",
            "--max_length",
            "2048",
        ],
    )
    assert captured["redundant_args"] == {
        "gradient_accumulation_steps",
        "packing",
        "learning_rate",
    }
    assert captured["user_specified_args"] == {"max_length"}
    assert auto_output_dir(captured) == "Qwen3-0.6B-Base-SFT.max-2048"


def test_differs_from_dataclass_default(monkeypatch, tmp_path):
    """A key the defaults YAML does not set is compared with the dataclass default"""
    captured = run_cli(monkeypatch, tmp_path, "weight_decay: 0.1\n")
    assert captured["redundant_args"] == set()
    assert "weight_decay" in captured["user_specified_args"]
    assert auto_output_dir(captured) == "Qwen3-0.6B-Base-SFT.weight_decay-0.1"


@pytest.mark.parametrize(
    ("yaml_lr", "cli_lr", "redundant"),
    [
        ("2.0e-5", "2e-5", True),  # both equal to the default
        ("2.0e-5", "1e-5", False),  # YAML equal, CLI overrides it to another value
        ("1e-5", "2e-5", True),  # YAML differs, CLI sets it back to the default
    ],
)
def test_final_value_decides(monkeypatch, tmp_path, yaml_lr, cli_lr, redundant):
    """When YAML and CLI both set an arg, only the final (CLI) value matters"""
    captured = run_cli(
        monkeypatch,
        tmp_path,
        f"learning_rate: {yaml_lr}\n",
        ["--learning_rate", cli_lr],
    )
    assert ("learning_rate" in captured["redundant_args"]) is redundant
    assert ("learning_rate" in captured["user_specified_args"]) is not redundant
    expected = "Qwen3-0.6B-Base-SFT" if redundant else "Qwen3-0.6B-Base-SFT.lr-1e-05"
    assert auto_output_dir(captured) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
