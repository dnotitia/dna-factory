"""CPU checks for the GRPO execution switch and AsyncGRPO support boundary."""

import inspect
import logging
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpo
from dna_factory import async_grpo, training_runner
from dna_factory.async_grpo import (
    apply_attn_implementation_override,
    apply_response_template_override,
    validate_async_args,
    validate_completion_length_against_server,
)


def parse_async(monkeypatch, extra=()):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(training_runner, "run_training", lambda *args: args)
    return training_runner.cli_main(
        grpo.get_spec("async"),
        [
            "--config=configs/GRPO/qwen3-0.6B-async.yaml",
            "--use_cpu",
            "true",
            "--bf16",
            "false",
            *extra,
        ],
    )


def test_mode_selection(tmp_path):
    config = tmp_path / "async.yaml"
    config.write_text("grpo_execution: async\n")
    assert grpo.resolve_grpo_execution([]) == "sync"
    assert grpo.resolve_grpo_execution(["--grpo-execution=async"]) == "async"
    assert grpo.resolve_grpo_execution([f"--config={config}"]) == "async"
    assert (
        grpo.resolve_grpo_execution(
            ["--config", str(config), "--grpo_execution", "sync"]
        )
        == "sync"
    )
    config.write_text("grpo_execution: typo\n")
    with pytest.raises(ValueError, match="grpo_execution"):
        grpo.resolve_grpo_execution(["--config", str(config)])


def test_async_parse_and_setup(monkeypatch):
    spec, script, training, model, _mixture, dna, _ = parse_async(
        monkeypatch, ["--dtype", "float32"]
    )
    # Unrelated to this test's assertions (dtype/model_init_kwargs/reward_funcs); stubbed so
    # setup doesn't try to reach the (absent, in this test) vLLM server at the config's
    # vllm_server_base_url. See the dedicated validate_completion_length_against_server tests.
    monkeypatch.setattr(
        async_grpo, "validate_completion_length_against_server", lambda *a, **k: None
    )
    ctx = {}
    spec.setup_training_args(
        script, training, model, dna, ctx, logging.getLogger(__name__)
    )
    assert script.grpo_execution == "async"
    assert training.dtype == model.dtype == "float32"
    assert training.model_init_kwargs["dtype"] == "float32"
    assert "attn_implementation" not in training.model_init_kwargs
    assert len(ctx["reward_funcs"]) == 2
    assert not spec.pass_eval_dataset
    assert not spec.pass_debug_batches
    assert not spec.pass_peft_config
    assert grpo.get_spec("sync") is grpo.SPEC


@pytest.mark.parametrize(
    "extra", [["--loss_type", "dapo"], ["--vllm_mode", "colocate"]]
)
def test_reject_sync_flags(monkeypatch, extra):
    with pytest.raises(ValueError, match="Unsupported AsyncGRPO"):
        parse_async(monkeypatch, extra)


def test_reject_sync_yaml(monkeypatch, tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("grpo_execution: async\nloss_type: dapo\n")
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    with pytest.raises(ValueError, match="Unsupported AsyncGRPO.*loss_type"):
        training_runner.cli_main(
            grpo.get_spec("async"),
            [
                "--config",
                str(config),
                "--use_cpu",
                "true",
                "--bf16",
                "false",
            ],
        )


def test_sync_cli_still_uses_grpo_defaults(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(training_runner, "run_training", lambda *args: args)
    spec, script, training, _, _, _, _ = training_runner.cli_main(
        grpo.get_spec("sync"), ["--use_cpu", "true", "--bf16", "false"]
    )
    assert spec is grpo.SPEC
    assert script.grpo_execution == "sync"
    assert training.loss_type == "dapo"
    assert training.vllm_mode == "colocate"


def test_reject_multiple_visible_training_gpus():
    with pytest.raises(ValueError, match="one training GPU"):
        validate_async_args(
            None, SimpleNamespace(deepspeed=None, fsdp=[], n_gpu=2), None, None
        )


@pytest.mark.parametrize(
    "env,value",
    [
        ("WORLD_SIZE", "2"),
        ("ACCELERATE_USE_DEEPSPEED", "true"),
        ("ACCELERATE_USE_FSDP", "true"),
    ],
)
def test_reject_distributed(monkeypatch, env, value):
    # Validate without constructing TrainingArguments/initializing distributed state.
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match="one training GPU"):
        validate_async_args(None, SimpleNamespace(deepspeed=None, fsdp=[]), None, None)


@pytest.mark.parametrize(
    "target,key,value,message",
    [
        ("dna", "dynamic_sampling", "resample", "dynamic_sampling"),
        ("dna", "debug_first_n_batches", 1, "debug_first_n_batches"),
        ("model", "use_peft", True, "full finetuning"),
        ("model", "load_in_4bit", True, "full finetuning"),
        ("model", "attn_implementation", "sdpa", "padding-free"),
        ("model", "attn_implementation", "eager", "padding-free"),
        (
            "model",
            "attn_implementation",
            "kernels-community/vllm-flash-attn3",
            "flash-attn2",
        ),
        ("script", "reward_model_name_or_path", "some/model", "callable rewards"),
        ("training", "do_eval", True, "evaluation"),
    ],
)
def test_reject_unsupported_options(monkeypatch, target, key, value, message):
    _, script, training, model, _, dna, _ = parse_async(monkeypatch)
    args = {"script": script, "training": training, "model": model, "dna": dna}
    setattr(args[target], key, value)
    with pytest.raises(ValueError, match=message):
        validate_async_args(script, training, model, dna)


@pytest.mark.parametrize(
    "attn_implementation",
    [None, "kernels-community/flash-attn2", "kernels-community/flash-attn3"],
)
def test_accept_supported_attn_implementations(monkeypatch, attn_implementation):
    _, script, training, model, _, dna, _ = parse_async(monkeypatch)
    model.attn_implementation = attn_implementation
    validate_async_args(script, training, model, dna)  # must not raise


def test_runner_passes_only_supported_constructor_args(monkeypatch):
    spec, script, training, model, mixture, dna, _ = parse_async(monkeypatch)
    # Exercise the real shared runner, but no network, GPU, training or saving.
    monkeypatch.undo()
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(
        training_runner, "setup_logging", lambda *a: logging.getLogger(__name__)
    )
    signature = inspect.signature(spec.trainer_cls)
    received = {}

    def trainer_cls(**kwargs):
        signature.bind(**kwargs)
        received.update(kwargs)
        return SimpleNamespace(train=lambda **kw: "trained")

    spec.trainer_cls = trainer_cls
    script.dataset_name = "test"
    training.output_dir = "unused-async-test-output"
    monkeypatch.setattr(
        training_runner.AutoTokenizer, "from_pretrained", lambda *a, **k: "tokenizer"
    )
    monkeypatch.setattr(
        training_runner, "load_dataset", lambda *a, **k: {"train": [{"prompt": "hi"}]}
    )
    monkeypatch.setattr(training_runner, "save_training_results", lambda *a: None)
    # Keeps this test's "no network" invariant: the completion-length check would otherwise try
    # to reach the (absent) server named by the config's vllm_server_base_url.
    monkeypatch.setattr(
        async_grpo, "validate_completion_length_against_server", lambda *a, **k: None
    )
    assert (
        training_runner.run_training(spec, script, training, model, mixture, dna)
        == "trained"
    )
    assert received["reward_funcs"]
    assert received["processing_class"] == "tokenizer"


def test_attn_override_noop_for_upstream_default(monkeypatch):
    from trl.experimental.async_grpo import async_grpo_trainer

    # Register the current value with monkeypatch so any mutation below is undone at teardown,
    # regardless of whether it happens via monkeypatch.setattr or direct assignment.
    monkeypatch.setattr(
        async_grpo_trainer,
        "create_model_from_path",
        async_grpo_trainer.create_model_from_path,
    )
    original = async_grpo_trainer.create_model_from_path
    apply_attn_implementation_override("kernels-community/flash-attn3")
    assert async_grpo_trainer.create_model_from_path is original


def test_attn_override_idempotent(monkeypatch):
    from trl.experimental.async_grpo import async_grpo_trainer

    monkeypatch.setattr(
        async_grpo_trainer,
        "create_model_from_path",
        async_grpo_trainer.create_model_from_path,
    )
    apply_attn_implementation_override("kernels-community/flash-attn2")
    patched_once = async_grpo_trainer.create_model_from_path
    apply_attn_implementation_override("kernels-community/flash-attn2")
    assert async_grpo_trainer.create_model_from_path is patched_once  # no double-wrap

    # Re-applying with a different value unwraps first, so it still doesn't stack.
    apply_attn_implementation_override("sdpa")
    assert (
        async_grpo_trainer.create_model_from_path.__dna_original__ is not patched_once
    )


def test_attn_override_forces_kwarg_into_from_pretrained(monkeypatch):
    # Pure CPU, no network, no real model: force `architecture` explicitly so
    # create_model_from_path skips config-based inference and calls
    # AutoModelForCausalLM.from_pretrained directly on our recorder.
    from transformers import AutoModelForCausalLM
    from trl.experimental.async_grpo import async_grpo_trainer

    monkeypatch.setattr(
        async_grpo_trainer,
        "create_model_from_path",
        async_grpo_trainer.create_model_from_path,
    )
    calls = []
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: calls.append(kw)),
    )

    apply_attn_implementation_override("kernels-community/flash-attn2")
    async_grpo_trainer.create_model_from_path(
        "unused/model-id", architecture=AutoModelForCausalLM, dtype="float32"
    )

    assert calls[-1]["attn_implementation"] == "kernels-community/flash-attn2"


@pytest.fixture
def dnotitia_qwen3_tokenizer():
    """Real, offline-cached tokenizer whose chat template triggers the upstream match failure
    this module's `apply_response_template_override` tests exercise (see that function's
    docstring in dna_factory/async_grpo.py). Loaded fresh per test, not module-scoped, because the
    function under test mutates the tokenizer in place and several tests need a clean slate.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("dnotitia/Qwen3-0.6B")


def test_add_response_schema_rejects_dnotitia_qwen3_template_unpatched(
    dnotitia_qwen3_tokenizer,
):
    """Canary reproducing the exact upstream failure `apply_response_template_override` exists
    for. If this starts passing (no ValueError), dnotitia/Qwen3-0.6B's chat template is recognized
    natively by TRL and the override can be deleted."""
    from trl.chat_template_utils import add_response_schema

    with pytest.raises(ValueError, match="Unrecognized chat template"):
        add_response_schema(dnotitia_qwen3_tokenizer)


def test_response_template_override_sets_qwen3_template(dnotitia_qwen3_tokenizer):
    from trl.chat_template_utils import (
        _SUPPORTS_RESPONSE_TEMPLATE,
        qwen3_schema,
        qwen3_template,
    )

    assert getattr(dnotitia_qwen3_tokenizer, "response_template", None) is None
    assert getattr(dnotitia_qwen3_tokenizer, "response_schema", None) is None

    apply_response_template_override(dnotitia_qwen3_tokenizer)

    if _SUPPORTS_RESPONSE_TEMPLATE:
        assert dnotitia_qwen3_tokenizer.response_template == qwen3_template
    else:
        assert dnotitia_qwen3_tokenizer.response_schema == qwen3_schema


def test_response_template_override_makes_add_response_schema_a_noop(
    dnotitia_qwen3_tokenizer,
):
    """Mirrors the exact skip condition in TRL's `_AsyncRolloutLoop.__init__`
    (async_rollout_worker.py:340-345): once either attribute is set, upstream's own
    `add_response_schema` call is never reached."""
    apply_response_template_override(dnotitia_qwen3_tokenizer)
    has_template = (
        getattr(dnotitia_qwen3_tokenizer, "response_template", None) is not None
    )
    has_schema = getattr(dnotitia_qwen3_tokenizer, "response_schema", None) is not None
    assert has_template or has_schema


def test_response_template_override_idempotent(dnotitia_qwen3_tokenizer):
    apply_response_template_override(dnotitia_qwen3_tokenizer)
    first = getattr(dnotitia_qwen3_tokenizer, "response_template", None) or getattr(
        dnotitia_qwen3_tokenizer, "response_schema", None
    )
    apply_response_template_override(
        dnotitia_qwen3_tokenizer
    )  # must not raise or change it
    second = getattr(dnotitia_qwen3_tokenizer, "response_template", None) or getattr(
        dnotitia_qwen3_tokenizer, "response_schema", None
    )
    assert first == second


def test_response_template_override_respects_preexisting_schema(
    dnotitia_qwen3_tokenizer,
):
    """No-op if a schema is already set — matches upstream's own skip condition, and means this
    function never clobbers a schema someone set deliberately."""
    sentinel = {"sentinel": True}
    dnotitia_qwen3_tokenizer.response_template = sentinel
    apply_response_template_override(dnotitia_qwen3_tokenizer)
    assert dnotitia_qwen3_tokenizer.response_template is sentinel


def test_response_template_override_leaves_unrelated_templates_alone(
    dnotitia_qwen3_tokenizer,
):
    """A chat template that doesn't render like Qwen3 is left for `add_response_schema` to handle
    (and fail) on its own — this function never guesses."""
    dnotitia_qwen3_tokenizer.chat_template = "{{ 'not a qwen3 template' }}"
    apply_response_template_override(dnotitia_qwen3_tokenizer)
    assert getattr(dnotitia_qwen3_tokenizer, "response_template", None) is None
    assert getattr(dnotitia_qwen3_tokenizer, "response_schema", None) is None


def test_response_template_override_survives_the_spawn_pickle_boundary(
    dnotitia_qwen3_tokenizer,
):
    """AsyncRolloutWorker.start() spawns the rollout-worker child via
    `multiprocessing.get_context("spawn")` and forwards `processing_class` as a constructor kwarg
    (trl/experimental/async_grpo/async_rollout_worker.py:1115-1181): every forwarded kwarg,
    including the tokenizer, is pickled to cross into the child. This confirms the mutation this
    function makes in the parent process is exactly what the child unpickles and reads back — no
    child-side re-load or re-patch is needed."""
    from trl.chat_template_utils import (
        _SUPPORTS_RESPONSE_TEMPLATE,
        qwen3_schema,
        qwen3_template,
    )

    apply_response_template_override(dnotitia_qwen3_tokenizer)
    reloaded = pickle.loads(pickle.dumps(dnotitia_qwen3_tokenizer))

    if _SUPPORTS_RESPONSE_TEMPLATE:
        assert reloaded.response_template == qwen3_template
    else:
        assert reloaded.response_schema == qwen3_schema

    parsed = reloaded.parse_response(
        "<think>\nchecking\n</think>\n\nok<|im_end|>\n",
        prefix="<|im_start|>assistant\n",
    )
    assert parsed == {
        "role": "assistant",
        "reasoning_content": "checking",
        "content": "ok",
    }


def test_get_spec_wires_postprocess_tokenizer_for_async_only():
    assert (
        grpo.get_spec("async").postprocess_tokenizer is grpo.postprocess_async_tokenizer
    )
    assert grpo.SPEC.postprocess_tokenizer is None


def test_postprocess_async_tokenizer_applies_override_and_returns_tokenizer(
    dnotitia_qwen3_tokenizer,
):
    from trl.chat_template_utils import _SUPPORTS_RESPONSE_TEMPLATE

    result = grpo.postprocess_async_tokenizer(
        dnotitia_qwen3_tokenizer, {}, logging.getLogger(__name__)
    )
    assert result is dnotitia_qwen3_tokenizer
    if _SUPPORTS_RESPONSE_TEMPLATE:
        assert result.response_template is not None
    else:
        assert result.response_schema is not None


def _stub_server_max_model_len(monkeypatch, max_model_len):
    """Stub `requests.get` to answer `/v1/models` the way a real vLLM server would, instantly."""

    def fake_get(url, timeout=None):
        assert url == "http://localhost:8000/v1/models"
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"data": [{"max_model_len": max_model_len}]},
        )

    monkeypatch.setattr(async_grpo.requests, "get", fake_get)


def _completion_length_training_args(max_completion_length, token_budget=None):
    return SimpleNamespace(
        vllm_server_base_url="http://localhost:8000",
        max_completion_length=max_completion_length,
        token_budget=token_budget,
    )


@pytest.mark.parametrize(
    "max_completion_length", [8192, 9000]
)  # equal to, then greater than, max_model_len
def test_validate_completion_length_raises_when_it_does_not_fit(
    monkeypatch, max_completion_length
):
    _stub_server_max_model_len(monkeypatch, max_model_len=8192)
    training_args = _completion_length_training_args(max_completion_length)
    with pytest.raises(
        ValueError, match=r"max_completion_length \(\d+\).*max_model_len \(8192\)"
    ):
        validate_completion_length_against_server(
            training_args, logging.getLogger(__name__)
        )


def test_validate_completion_length_passes_and_logs_headroom(monkeypatch, caplog):
    _stub_server_max_model_len(monkeypatch, max_model_len=8192)
    training_args = _completion_length_training_args(max_completion_length=2048)
    with caplog.at_level(logging.INFO):
        validate_completion_length_against_server(
            training_args, logging.getLogger(__name__)
        )  # must not raise
    assert any("6144" in record.getMessage() for record in caplog.records)


def test_validate_completion_length_warns_when_token_budget_too_small(
    monkeypatch, caplog
):
    _stub_server_max_model_len(monkeypatch, max_model_len=8192)
    training_args = _completion_length_training_args(
        max_completion_length=2048, token_budget=4096
    )
    with caplog.at_level(logging.WARNING):
        validate_completion_length_against_server(
            training_args, logging.getLogger(__name__)
        )  # must not raise
    assert any(
        "token_budget" in record.getMessage()
        and "4096" in record.getMessage()
        and "8192" in record.getMessage()
        for record in caplog.records
    )


def test_validate_completion_length_no_warning_when_token_budget_none(
    monkeypatch, caplog
):
    _stub_server_max_model_len(monkeypatch, max_model_len=8192)
    training_args = _completion_length_training_args(
        max_completion_length=2048, token_budget=None
    )
    with caplog.at_level(logging.WARNING):
        validate_completion_length_against_server(
            training_args, logging.getLogger(__name__)
        )
    assert not any("token_budget" in record.getMessage() for record in caplog.records)


def test_validate_completion_length_unreachable_server_warns_and_skips(
    monkeypatch, caplog
):
    def raise_connection_error(url, timeout=None):
        raise requests.ConnectionError("boom")

    monkeypatch.setattr(async_grpo.requests, "get", raise_connection_error)
    # Skip the real backoff delay; the retry/backoff behavior itself isn't under test here.
    monkeypatch.setattr(async_grpo.time, "sleep", lambda seconds: None)
    training_args = _completion_length_training_args(max_completion_length=2048)
    with caplog.at_level(logging.WARNING):
        validate_completion_length_against_server(
            training_args, logging.getLogger(__name__)
        )  # must not raise
    assert any("skip" in record.getMessage().lower() for record in caplog.records)


def test_setup_async_training_args_calls_completion_length_validation(monkeypatch):
    spec, script, training, model, _mixture, dna, _ = parse_async(monkeypatch)
    calls = []
    monkeypatch.setattr(
        async_grpo,
        "validate_completion_length_against_server",
        lambda training_args, train_logger: calls.append(training_args),
    )
    spec.setup_training_args(
        script, training, model, dna, {}, logging.getLogger(__name__)
    )
    assert calls == [training]
