import json
import subprocess
from pathlib import Path

import pytest

import launch
from dna_factory.auto_zero import STATUS_ENV, can_retry, config_candidates, is_cuda_oom


def test_policy_and_moe_configs():
    assert config_candidates("auto", "qwen3") == ["zero1.yaml", "zero3.yaml"]
    assert config_candidates("auto", "qwen3_moe") == ["zero3-qwen3-moe.yaml"]
    assert config_candidates("auto", "qwen3_5_moe") == ["zero3-qwen3_5-moe.yaml"]
    assert config_candidates("3", "qwen4_exp") == ["zero3.yaml"]
    assert config_candidates("2", "qwen3") == ["zero2.yaml"]
    with pytest.raises(ValueError):
        config_candidates("1", "qwen3_moe")


def test_retry_only_for_cuda_oom_before_progress(tmp_path):
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not is_cuda_oom(RuntimeError("CPU out of memory"))
    assert not is_cuda_oom(RuntimeError("NCCL timeout"))
    assert not can_retry(tmp_path, 1)
    (tmp_path / "oom-1").touch()
    assert can_retry(tmp_path, 1)
    assert not can_retry(tmp_path, 0)
    (tmp_path / "progress-2").touch()
    assert not can_retry(tmp_path, 1)


@pytest.mark.parametrize(
    "progress,oom,expected", [(False, True, 2), (True, True, 1), (False, False, 1)]
)
def test_launcher_fallback_preserves_training_args(
    tmp_path, monkeypatch, progress, oom, expected
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    calls = []

    class Process:
        pid = 123456

        def __init__(self, command, env, start_new_session):
            calls.append(command)
            assert start_new_session
            self.code = 1 if len(calls) == 1 else 0
            if len(calls) == 1:
                if oom:
                    Path(env[STATUS_ENV], "oom-1").touch()
                if progress:
                    Path(env[STATUS_ENV], "progress-2").touch()

        def wait(self):
            return self.code

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(launch.os, "killpg", lambda *args: None)
    training_args = ["--model_name_or_path", str(model), "--learning_rate", "2e-5"]
    result = launch.main(
        ["--zero", "auto", "--num-processes", "2", "sft.py", *training_args]
    )
    assert len(calls) == expected
    assert result == (0 if expected == 2 else 1)
    assert calls[0][calls[0].index("--config_file") + 1].endswith("zero1.yaml")
    for command in calls:
        assert command[-len(training_args) :] == training_args
    if expected == 2:
        assert calls[1][calls[1].index("--config_file") + 1].endswith("zero3.yaml")


def test_cli_override_used_for_model_selection(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("model_name_or_path: old\nuse_unsloth: true\n")
    merged = launch.training_config(
        "sampling_sft.py",
        ["--config", str(config), "--model_name_or_path=new", "--use_unsloth", "false"],
    )
    assert merged["model_name_or_path"] == "new"
    assert merged["use_unsloth"] == "false"
    assert float(merged["learning_rate"]) == 2e-5
