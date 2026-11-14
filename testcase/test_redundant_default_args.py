"""CPU checks that a config or CLI value equal to the default is not treated as an override (#31)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sft
from dna_factory import training_runner
from dna_factory.utils.output_dir_generator import generate_auto_output_dir


@pytest.fixture(autouse=True)
def _skip_cuda_tf32_probe(monkeypatch):
    """TrainingArguments.__post_init__ probes CUDA when tf32 is set; these tests never train."""
    monkeypatch.setattr(
        "transformers.training_args.is_torch_tf32_available", lambda: False
    )


def parse_sft(monkeypatch, tmp_path, config_text, extra=()):
    config = tmp_path / "user.yaml"
    config.write_text(config_text)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(
        training_runner, "run_training", lambda *args, **kwargs: (args, kwargs)
    )
    args, kwargs = training_runner.cli_main(
        sft.SPEC,
        ["--config", str(config), "--use_cpu", "true", "--bf16", "false", *extra],
    )
    return args, kwargs["redundant_args"]


def test_values_equal_to_the_defaults_are_redundant(monkeypatch, tmp_path):
    # _defaults-SFT.yaml: learning_rate 2e-5, packing true, max_length 1024,
    # gradient_checkpointing true
    args, redundant = parse_sft(
        monkeypatch,
        tmp_path,
        "learning_rate: 2.0e-5\npacking: true\nmax_length: 512\n",
        ["--gradient_checkpointing", "True"],
    )
    user_specified = args[-1]

    assert {"learning_rate", "packing", "gradient_checkpointing"} <= redundant
    assert {"max_length", "use_cpu", "bf16"} <= user_specified
    assert not redundant & user_specified


def test_redundant_values_stay_out_of_the_auto_output_dir(monkeypatch, tmp_path):
    args, _ = parse_sft(
        monkeypatch, tmp_path, "learning_rate: 2.0e-5\nmax_length: 512\n"
    )
    spec, script, training, model, mixture, dnotitia, user_specified = args

    name = generate_auto_output_dir(
        model.model_name_or_path,
        user_specified,
        script,
        training,
        model,
        mixture,
        dnotitia,
        spec.output_dir_tag,
    )

    assert "max-512" in name
    assert "lr-" not in name


def test_cli_override_of_a_config_value_back_to_the_default(monkeypatch, tmp_path):
    # The config changes max_length, the CLI puts it back: the run uses the default.
    args, redundant = parse_sft(
        monkeypatch, tmp_path, "max_length: 512\n", ["--max_length", "1024"]
    )

    assert args[2].max_length == 1024
    assert "max_length" in redundant
    assert "max_length" not in args[-1]
