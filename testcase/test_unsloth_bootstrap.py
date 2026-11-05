"""
Test cases for dna_factory/unsloth_bootstrap.py

This module tests the pre-import use_unsloth decision (argv/YAML resolution, model support check,
distributed-launch guard) without a GPU, network, or a real `import unsloth`.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# Add parent directory to path to import dna_factory
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dna_factory import unsloth_bootstrap as ub


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Clear distributed-launch envs and reset the module-level decision for every test"""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    monkeypatch.setattr(ub, "_DECISION", None)
    # Keep the noexec check deterministic regardless of how this machine mounts /tmp
    monkeypatch.setattr(ub, "_noexec_inductor_cache_dir", lambda: None)


def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


def make_model_dir(tmp_path, model_type):
    """Create a fake local model dir containing only config.json"""
    model_dir = tmp_path / f"model_{model_type}"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(model_dir)


class TestResolveBootstrapArgs:
    """Test cases for resolve_bootstrap_args"""

    def test_missing_everywhere(self, tmp_path):
        defaults = write_yaml(tmp_path / "defaults.yaml", {"lr": 1})
        assert ub.resolve_bootstrap_args([], defaults) == {
            "use_unsloth": False,
            "model_name_or_path": None,
            "model_revision": None,
        }

    def test_user_yaml_overrides_defaults(self, tmp_path):
        defaults = write_yaml(
            tmp_path / "defaults.yaml",
            {"use_unsloth": False, "model_name_or_path": "base"},
        )
        user = write_yaml(
            tmp_path / "user.yaml",
            {"use_unsloth": True, "model_name_or_path": "user", "model_revision": "r1"},
        )
        result = ub.resolve_bootstrap_args(["--config", user], defaults)
        assert result == {
            "use_unsloth": True,
            "model_name_or_path": "user",
            "model_revision": "r1",
        }

    def test_cli_overrides_yaml(self, tmp_path):
        defaults = write_yaml(
            tmp_path / "defaults.yaml", {"model_name_or_path": "base"}
        )
        user = write_yaml(
            tmp_path / "user.yaml", {"use_unsloth": True, "model_name_or_path": "user"}
        )
        cli = ["--config", user, "--use_unsloth", "false", "--model_name_or_path=cli"]
        result = ub.resolve_bootstrap_args(cli, defaults)
        assert result["use_unsloth"] is False
        assert result["model_name_or_path"] == "cli"

    def test_dashed_model_aliases(self, tmp_path):
        defaults = write_yaml(tmp_path / "defaults.yaml", {})
        cli = ["--model-name-or-path", "m", "--model-revision=abc"]
        result = ub.resolve_bootstrap_args(cli, defaults)
        assert result["model_name_or_path"] == "m"
        assert result["model_revision"] == "abc"

    def test_missing_user_config_is_ignored(self, tmp_path):
        defaults = write_yaml(
            tmp_path / "defaults.yaml", {"model_name_or_path": "base"}
        )
        result = ub.resolve_bootstrap_args(
            ["--config", str(tmp_path / "nope.yaml")], defaults
        )
        assert result["model_name_or_path"] == "base"
        assert result["use_unsloth"] is False


class TestUseUnslothBoolForms:
    """Test cases for the accepted use_unsloth spellings"""

    @pytest.fixture
    def defaults(self, tmp_path):
        return write_yaml(tmp_path / "defaults.yaml", {})

    def test_bare_flag(self, defaults):
        assert (
            ub.resolve_bootstrap_args(["--use_unsloth"], defaults)["use_unsloth"]
            is True
        )

    def test_bare_flag_followed_by_other_flag(self, defaults):
        cli = ["--use_unsloth", "--learning_rate", "1e-5"]
        assert ub.resolve_bootstrap_args(cli, defaults)["use_unsloth"] is True

    def test_equals_form(self, defaults):
        assert (
            ub.resolve_bootstrap_args(["--use_unsloth=true"], defaults)["use_unsloth"]
            is True
        )

    def test_dashed_alias(self, defaults):
        assert (
            ub.resolve_bootstrap_args(["--use-unsloth"], defaults)["use_unsloth"]
            is True
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("YES", True),
            ("y", True),
            ("T", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("N", False),
            ("f", False),
        ],
    )
    def test_value_forms(self, defaults, value, expected):
        assert (
            ub.resolve_bootstrap_args(["--use_unsloth", value], defaults)["use_unsloth"]
            is expected
        )

    @pytest.mark.parametrize("value", ["on", "OFF", "On"])
    def test_on_off_rejected_like_hf_parser(self, defaults, value):
        with pytest.raises(ValueError, match="use_unsloth"):
            ub.resolve_bootstrap_args(["--use_unsloth", value], defaults)

    def test_accepted_values_match_hf_string_to_bool(self):
        from transformers.hf_argparser import string_to_bool

        for value in ub._TRUE_VALUES | ub._FALSE_VALUES:
            assert ub._parse_bool(value.upper(), "t") is string_to_bool(value)
        for value in ("on", "off", "maybe", "2"):
            with pytest.raises(ValueError):
                ub._parse_bool(value, "t")

    def test_yaml_bool_value(self, tmp_path):
        defaults = write_yaml(tmp_path / "defaults.yaml", {"use_unsloth": True})
        assert ub.resolve_bootstrap_args([], defaults)["use_unsloth"] is True

    def test_yaml_string_value(self, tmp_path):
        defaults = write_yaml(tmp_path / "defaults.yaml", {"use_unsloth": "yes"})
        assert ub.resolve_bootstrap_args([], defaults)["use_unsloth"] is True

    def test_invalid_cli_value_raises(self, defaults):
        with pytest.raises(ValueError, match="use_unsloth"):
            ub.resolve_bootstrap_args(["--use_unsloth", "maybe"], defaults)

    def test_invalid_yaml_value_raises(self, tmp_path):
        defaults = write_yaml(tmp_path / "defaults.yaml", {"use_unsloth": "maybe"})
        with pytest.raises(ValueError, match="use_unsloth"):
            ub.resolve_bootstrap_args([], defaults)


class TestReadModelType:
    """Test cases for read_model_type"""

    def test_local_dir(self, tmp_path):
        assert ub.read_model_type(make_model_dir(tmp_path, "qwen3")) == "qwen3"

    def test_missing_config_json(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert ub.read_model_type(str(empty)) is None

    def test_invalid_json(self, tmp_path):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "config.json").write_text("{not json")
        assert ub.read_model_type(str(bad)) is None

    def test_readable_config_without_model_type_is_not_an_error(self, tmp_path):
        d = tmp_path / "no_type"
        d.mkdir()
        (d / "config.json").write_text("{}")
        assert ub._read_model_type(str(d), None) == (None, None)

    def test_read_failure_reports_error(self, tmp_path):
        model_type, error = ub._read_model_type(str(tmp_path / "missing"), None)
        assert model_type is None and error


class TestMaybeImportUnsloth:
    """Test cases for maybe_import_unsloth"""

    @pytest.fixture
    def importer(self, monkeypatch):
        calls = []
        monkeypatch.setattr(ub, "_import_unsloth", lambda: calls.append(1))
        return calls

    @pytest.fixture
    def defaults(self, tmp_path):
        return write_yaml(tmp_path / "defaults.yaml", {})

    def test_not_requested(self, tmp_path, defaults, importer, monkeypatch):
        def fail(*a, **k):
            raise AssertionError("config.json must not be read")

        monkeypatch.setattr(ub, "_read_model_type", fail)
        decision = ub.maybe_import_unsloth(["--model_name_or_path", "x"], defaults)
        assert (decision.requested, decision.active) == (False, False)
        assert importer == []
        assert ub.get_unsloth_decision() is decision

    def test_supported_model(self, tmp_path, defaults, importer):
        model = make_model_dir(tmp_path, "qwen3_5")
        decision = ub.maybe_import_unsloth(
            ["--use_unsloth", "--model_name_or_path", model], defaults
        )
        assert (decision.requested, decision.active) == (True, True)
        assert decision.model_type == "qwen3_5"
        assert len(importer) == 1
        assert ub.get_unsloth_decision() is decision

    def test_unsupported_model_warns(self, tmp_path, defaults, importer, caplog):
        model = make_model_dir(tmp_path, "llama")
        with caplog.at_level(logging.WARNING, logger=ub.__name__):
            decision = ub.maybe_import_unsloth(
                ["--use_unsloth", "--model_name_or_path", model], defaults
            )
        assert (decision.requested, decision.active) == (True, False)
        assert decision.model_type == "llama"
        assert importer == []
        assert any(
            "llama" in r.getMessage() and r.levelno == logging.WARNING
            for r in caplog.records
        )

    def test_readable_config_without_model_type_falls_back(
        self, tmp_path, defaults, importer, caplog
    ):
        model_dir = tmp_path / "no_type"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        with caplog.at_level(logging.WARNING, logger=ub.__name__):
            decision = ub.maybe_import_unsloth(
                ["--use_unsloth", "--model_name_or_path", str(model_dir)], defaults
            )
        assert (decision.requested, decision.active) == (True, False)
        assert decision.model_type is None
        assert importer == []
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_nonexistent_local_path_raises(self, tmp_path, defaults, importer):
        missing = str(tmp_path / "missing")
        with pytest.raises(RuntimeError, match="use_unsloth=false") as exc_info:
            ub.maybe_import_unsloth(
                ["--use_unsloth", "--model_name_or_path", missing], defaults
            )
        assert missing in str(exc_info.value)
        assert importer == []
        assert ub.get_unsloth_decision() is None

    def test_hub_download_failure_raises(self, defaults, importer, monkeypatch):
        import huggingface_hub

        def boom(*a, **k):
            raise OSError("hub unreachable")

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
        with pytest.raises(RuntimeError, match="hub unreachable") as exc_info:
            ub.maybe_import_unsloth(
                ["--use_unsloth", "--model_name_or_path", "org/not-a-local-dir"],
                defaults,
            )
        assert "org/not-a-local-dir" in str(exc_info.value)
        assert importer == []

    def test_invalid_config_json_raises(self, tmp_path, defaults, importer):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "config.json").write_text("{not json")
        with pytest.raises(RuntimeError, match="JSONDecodeError"):
            ub.maybe_import_unsloth(
                ["--use_unsloth", "--model_name_or_path", str(bad)], defaults
            )
        assert importer == []

    def test_model_not_set_raises(self, defaults, importer):
        with pytest.raises(RuntimeError, match="model_name_or_path is not set"):
            ub.maybe_import_unsloth(["--use_unsloth"], defaults)
        assert importer == []

    def test_unreadable_config_ignored_when_not_requested(
        self, tmp_path, defaults, importer
    ):
        decision = ub.maybe_import_unsloth(
            ["--model_name_or_path", str(tmp_path / "missing")], defaults
        )
        assert decision.active is False

    def test_world_size_raises(self, tmp_path, defaults, importer, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(RuntimeError, match="single"):
            ub.maybe_import_unsloth(["--use_unsloth"], defaults)
        assert importer == []

    def test_deepspeed_raises(self, tmp_path, defaults, importer, monkeypatch):
        monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
        with pytest.raises(RuntimeError, match="single"):
            ub.maybe_import_unsloth(["--use_unsloth"], defaults)

    def test_distributed_env_ignored_when_not_requested(
        self, defaults, importer, monkeypatch
    ):
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert ub.maybe_import_unsloth([], defaults).active is False

    def test_decision_none_before_run(self):
        assert ub.get_unsloth_decision() is None


class TestImportUnsloth:
    """Test cases for _import_unsloth"""

    def test_missing_package_gives_actionable_error(self, monkeypatch):
        # A None entry in sys.modules makes `import unsloth` raise ImportError
        monkeypatch.setitem(sys.modules, "unsloth", None)
        with pytest.raises(ImportError, match="uv sync --extra unsloth") as exc_info:
            ub._import_unsloth()
        assert isinstance(exc_info.value.__cause__, ImportError)


class TestImportHygiene:
    """The bootstrap must not pull in modules unsloth has to precede"""

    def test_no_heavy_imports(self):
        code = (
            "import sys; sys.path.insert(0, '.'); "
            "import dna_factory.unsloth_bootstrap; "
            "bad = [m for m in ('trl', 'transformers', 'peft', 'torch') if m in sys.modules]; "
            "assert not bad, bad"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestNoexecInductorCacheDir:
    """Test cases for the noexec torch inductor cache check"""

    class _Statvfs:
        def __init__(self, flags):
            self.f_flag = flags

    @pytest.fixture
    def check(self, monkeypatch):
        # Undo the autouse stub so the real helper runs
        monkeypatch.undo()
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
        monkeypatch.setattr(ub, "_DECISION", None)
        return ub._noexec_inductor_cache_dir

    def test_noexec_mount_reported(self, check, monkeypatch, tmp_path):
        monkeypatch.setenv(
            "TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "not" / "yet" / "created")
        )
        monkeypatch.setattr(ub.os, "statvfs", lambda p: self._Statvfs(ub.os.ST_NOEXEC))
        assert check() == str(tmp_path / "not" / "yet" / "created")

    def test_exec_mount_ok(self, check, monkeypatch, tmp_path):
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(ub.os, "statvfs", lambda p: self._Statvfs(0))
        assert check() is None

    def test_warns_on_noexec_cache(self, check, monkeypatch, tmp_path, caplog):
        calls = []
        monkeypatch.setattr(ub, "_import_unsloth", lambda: calls.append(1))
        monkeypatch.setattr(
            ub, "_noexec_inductor_cache_dir", lambda: "/tmp/torchinductor_x"
        )
        model_dir = make_model_dir(tmp_path, "qwen3")
        defaults = write_yaml(tmp_path / "defaults.yaml", {})
        with caplog.at_level(logging.WARNING, logger=ub.__name__):
            decision = ub.maybe_import_unsloth(
                ["--use_unsloth", "--model_name_or_path", str(model_dir)], defaults
            )
        assert decision.active and calls == [1]
        assert "TORCHINDUCTOR_CACHE_DIR" in caplog.text
