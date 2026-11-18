"""
Test cases for dna_factory/training_runner.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory import training_runner


def test_load_tokenizer_passes_model_revision(monkeypatch):
    seen = {}

    def fake_from_pretrained(name, **kwargs):
        seen["name"] = name
        seen.update(kwargs)
        return "tok"

    monkeypatch.setattr(
        training_runner.AutoTokenizer, "from_pretrained", fake_from_pretrained
    )
    model_args = SimpleNamespace(
        model_name_or_path="org/model", model_revision="abc123"
    )
    training_args = SimpleNamespace(trust_remote_code=True)

    assert training_runner.load_tokenizer(model_args, training_args) == "tok"
    assert seen == {
        "name": "org/model",
        "trust_remote_code": True,
        "revision": "abc123",
        "use_fast": True,
    }


def test_load_model_passes_revision_to_auto_config(monkeypatch):
    config_kwargs = {}
    model_kwargs_seen = {}

    def fake_config(path, **kwargs):
        config_kwargs.update(kwargs)
        return object()

    def fake_model(path, **kwargs):
        model_kwargs_seen.update(kwargs)
        return "model"

    monkeypatch.setattr(training_runner.AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(
        training_runner.AutoModelForCausalLM, "from_pretrained", fake_model
    )

    result = training_runner.load_model(
        "org/model", revision="abc123", trust_remote_code=False
    )

    assert result == "model"
    assert config_kwargs["revision"] == "abc123"
    assert model_kwargs_seen["revision"] == "abc123"


def test_load_model_without_revision_passes_none(monkeypatch):
    config_kwargs = {}
    monkeypatch.setattr(
        training_runner.AutoConfig,
        "from_pretrained",
        lambda path, **kwargs: config_kwargs.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        training_runner.AutoModelForCausalLM, "from_pretrained", lambda p, **kw: "m"
    )
    training_runner.load_model("org/model")
    assert config_kwargs["revision"] is None
