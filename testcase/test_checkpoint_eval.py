"""
Test cases for dna_factory/checkpoint_eval.py::CheckpointEvalCallback
"""

import json
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import TrainerControl, TrainerState

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

import dna_factory.checkpoint_eval as ce
from dna_factory.checkpoint_eval import (
    CheckpointEvalCallback,
    VllmEvalServer,
    _child_env,
    build_checkpoint_eval_callback,
    derive_task_name,
    extract_scores,
    find_free_port,
    normalize_eval_tasks,
    parallel_device_count,
    read_eval_scores,
    read_int_flag,
    resolve_task,
    served_model_tag,
    served_name_for_step,
    split_vllm_args,
    stage_checkpoint,
)


def _state(step=100, main=True):
    return TrainerState(global_step=step, is_world_process_zero=main)


def _checkpoint(tmp_path, name="checkpoint-100", with_weights=True):
    """A checkpoint directory shaped like the one `Trainer._save_checkpoint` writes."""
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"model_type": "qwen3"}')
    (directory / "tokenizer.json").write_text("{}")
    (directory / "chat_template.jinja").write_text("{{ messages }}")
    if with_weights:
        (directory / "model.safetensors").write_bytes(b"weights")
    # Training state: present in a real checkpoint, useless to vLLM.
    (directory / "optimizer.pt").write_bytes(b"x" * 64)
    (directory / "scheduler.pt").write_bytes(b"x")
    (directory / "rng_state.pth").write_bytes(b"x")
    (directory / "trainer_state.json").write_text("{}")
    # DeepSpeed's sharded ZeRO state lives in a subdirectory.
    (directory / "global_step100").mkdir()
    (directory / "global_step100" / "zero_pp_rank_0.pt").write_bytes(b"x")
    return directory


class TestTaskNaming:
    """Task specs -> W&B metric names and CLI-ready paths."""

    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            ("inspect_evals/mmlu_pro", "mmlu_pro"),
            ("inspect_evals/gpqa_diamond", "gpqa_diamond"),
            ("evals/kmmlu_pro.py", "kmmlu_pro"),
            ("/abs/path/to/kmmlu_redux.py", "kmmlu_redux"),
            ("evals/suite.py@kmmlu_pro", "kmmlu_pro"),
        ],
    )
    def test_derive_task_name(self, task, expected):
        assert derive_task_name(task) == expected

    def test_registry_specs_pass_through(self):
        assert resolve_task("inspect_evals/mmlu_pro") == "inspect_evals/mmlu_pro"

    def test_absolute_paths_pass_through(self):
        assert resolve_task("/nowhere/kmmlu_pro.py") == "/nowhere/kmmlu_pro.py"

    def test_file_relative_to_cwd_wins(self, tmp_path, monkeypatch):
        (tmp_path / "evals").mkdir()
        (tmp_path / "evals" / "kmmlu_pro.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert resolve_task("evals/kmmlu_pro.py", tmp_path / "repo") == (
            "evals/kmmlu_pro.py"
        )

    def test_file_falls_back_to_repo_root(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        (repo / "evals").mkdir(parents=True)
        (repo / "evals" / "kmmlu_pro.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert resolve_task("evals/kmmlu_pro.py", repo) == str(
            repo / "evals" / "kmmlu_pro.py"
        )

    def test_repo_root_fallback_keeps_task_suffix(self, tmp_path, monkeypatch):
        repo = tmp_path / "repo"
        (repo / "evals").mkdir(parents=True)
        (repo / "evals" / "suite.py").write_text("")
        monkeypatch.chdir(tmp_path)
        assert resolve_task("evals/suite.py@kmmlu_pro", repo) == (
            f"{repo / 'evals' / 'suite.py'}@kmmlu_pro"
        )

    def test_unresolvable_file_passes_through_for_inspect_to_report(self, tmp_path):
        assert resolve_task("evals/missing.py", tmp_path) == "evals/missing.py"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, []),
            ([], []),
            (["a", " b "], ["a", "b"]),
            ("a,b", ["a", "b"]),
            ("a, b  c", ["a", "b", "c"]),
            ("", []),
        ],
    )
    def test_normalize_eval_tasks(self, value, expected):
        assert normalize_eval_tasks(value) == expected


class TestServedModelName:
    """`--served-model-name` must stay short and readable, and never leak a dot-split."""

    def test_prefers_run_name(self):
        args = SimpleNamespace(output_dir="out", run_name="4bexpr")
        model = SimpleNamespace(model_name_or_path="dnotitia/Qwen3-0.6B-Base")
        assert served_model_tag(args, model) == "4bexpr"

    def test_run_name_defaulted_to_output_dir_counts_as_unset(self):
        """TrainingArguments sets run_name = output_dir when the user gave none."""
        auto_dir = "Qwen3-0.6B-SFT.datasets-dnotitia-9ea.ep-4"
        args = SimpleNamespace(output_dir=auto_dir, run_name=auto_dir)
        model = SimpleNamespace(model_name_or_path="dnotitia/Qwen3-0.6B-Base")
        # Not "Qwen3-0" -- the model name has a dot of its own.
        assert served_model_tag(args, model) == "Qwen3-0.6B-Base"

    def test_local_model_path(self):
        args = SimpleNamespace(output_dir="out", run_name=None)
        model = SimpleNamespace(model_name_or_path="./Qwen3-4B-Base-NoThink/")
        assert served_model_tag(args, model) == "Qwen3-4B-Base-NoThink"

    def test_step_is_appended(self):
        assert served_name_for_step("Qwen3-4B-SFT", 1776) == (
            "Qwen3-4B-SFT-checkpoint-1776"
        )


class TestVllmArgs:
    def test_splits_flags(self):
        tokens, port = split_vllm_args(
            "--max-model-len 32768 --gpu-memory-utilization 0.85"
        )
        assert tokens == [
            "--max-model-len",
            "32768",
            "--gpu-memory-utilization",
            "0.85",
        ]
        assert port is None

    @pytest.mark.parametrize("spec", ["--port 9001", "--port=9001"])
    def test_explicit_port_is_extracted(self, spec):
        _, port = split_vllm_args(f"{spec} --max-model-len 4096")
        assert port == 9001

    def test_empty(self):
        assert split_vllm_args("") == ([], None)
        assert split_vllm_args(None) == ([], None)

    def test_reads_the_default_parallel_sizes(self):
        """The shipped default pairs --data-parallel-size 2 with two eval_devices."""
        tokens, _ = split_vllm_args(
            "--max-model-len 32768 --gpu-memory-utilization 0.85 --data-parallel-size 2"
        )
        assert parallel_device_count(tokens) == 2

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("", 1),
            ("--data-parallel-size 4", 4),
            ("-dp 2", 2),
            ("--tensor-parallel-size=8", 8),
            ("-tp 2 -pp 2", 4),
            ("--data-parallel-size 2 --tensor-parallel-size 4", 8),
            # vLLM gets to reject its own malformed flags; this only feeds a warning.
            ("--data-parallel-size auto", 1),
        ],
    )
    def test_parallel_device_count(self, spec, expected):
        tokens, _ = split_vllm_args(spec)
        assert parallel_device_count(tokens) == expected

    def test_read_int_flag_takes_the_last_occurrence(self):
        tokens, _ = split_vllm_args("-dp 2 --data-parallel-size 4")
        assert read_int_flag(tokens, "--data-parallel-size", "-dp") == 4
        assert read_int_flag(tokens, "--missing", default=7) == 7


class TestChildEnv:
    def test_strips_torchrun_rendezvous_vars(self, monkeypatch):
        """vLLM must not try to join the trainer's process group."""
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "8")
        monkeypatch.setenv("MASTER_PORT", "29500")
        monkeypatch.setenv("HF_HOME", "/data/hf")
        env = _child_env(CUDA_VISIBLE_DEVICES="7")
        assert "RANK" not in env
        assert "LOCAL_RANK" not in env
        assert "WORLD_SIZE" not in env
        assert "MASTER_PORT" not in env
        # Unrelated environment is inherited.
        assert env["HF_HOME"] == "/data/hf"
        assert env["CUDA_VISIBLE_DEVICES"] == "7"


class TestStageCheckpoint:
    """The snapshot that makes `save_total_limit` harmless."""

    def test_copies_servable_files_only(self, tmp_path):
        source = _checkpoint(tmp_path)
        staged, linked, copied = stage_checkpoint(source, tmp_path / "_staging" / "c")
        names = {p.name for p in staged.iterdir()}
        assert names == {
            "config.json",
            "tokenizer.json",
            "chat_template.jinja",
            "model.safetensors",
        }
        assert linked == 4
        assert copied == 0

    def test_survives_checkpoint_rotation(self, tmp_path):
        """The whole point: rotation deletes the checkpoint, the eval keeps reading."""
        source = _checkpoint(tmp_path)
        staged, _, _ = stage_checkpoint(source, tmp_path / "_staging" / "c")
        shutil.rmtree(source)  # what save_total_limit does from the next save
        assert (staged / "model.safetensors").read_bytes() == b"weights"

    def test_rejects_checkpoint_without_weights(self, tmp_path):
        source = _checkpoint(tmp_path, with_weights=False)
        with pytest.raises(FileNotFoundError):
            stage_checkpoint(source, tmp_path / "_staging" / "c")

    def test_replaces_a_stale_staging_dir(self, tmp_path):
        source = _checkpoint(tmp_path)
        staging = tmp_path / "_staging" / "c"
        staging.mkdir(parents=True)
        (staging / "leftover.txt").write_text("from an earlier crash")
        staged, _, _ = stage_checkpoint(source, staging)
        assert not (staged / "leftover.txt").exists()

    def test_falls_back_to_copying(self, tmp_path, monkeypatch):
        """A staging dir on another filesystem still works, just slower."""
        source = _checkpoint(tmp_path)

        def no_links(*args, **kwargs):
            raise OSError("Invalid cross-device link")

        monkeypatch.setattr(os, "link", no_links)
        staged, linked, copied = stage_checkpoint(source, tmp_path / "_staging" / "c")
        assert (linked, copied) == (0, 4)
        assert (staged / "model.safetensors").read_bytes() == b"weights"


class TestFindFreePort:
    def test_returns_a_bindable_port(self):
        port = find_free_port(start=8000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
            check.bind(("127.0.0.1", port))

    def test_skips_a_busy_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            taken = busy.getsockname()[1]
            assert find_free_port(start=taken, count=50) != taken

    def test_raises_when_the_range_is_exhausted(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen(1)
            taken = busy.getsockname()[1]
            with pytest.raises(RuntimeError):
                find_free_port(start=taken, count=1)


def _score(name, metrics):
    from inspect_ai.log import EvalScore
    from inspect_ai.log._log import EvalMetric

    return EvalScore(
        name=name,
        scorer=name,
        metrics={
            key: EvalMetric(name=key, value=value) for key, value in metrics.items()
        },
    )


def _results(scores, headline=None):
    from inspect_ai.log import EvalResults

    return EvalResults(
        total_samples=10, completed_samples=10, scores=scores, headline=headline
    )


class TestExtractScores:
    def test_prefers_the_headline_metric(self):
        from inspect_ai.log._log import HeadlineMetric

        results = _results(
            [
                _score("choice", {"accuracy": 0.61, "stderr": 0.02}),
                _score("other", {"accuracy": 0.10}),
            ],
            headline=HeadlineMetric(scorer="choice", metric="accuracy"),
        )
        assert extract_scores(SimpleNamespace(results=results)) == (0.61, 0.02)

    def test_falls_back_to_accuracy(self):
        results = _results([_score("choice", {"accuracy": 0.42, "stderr": 0.01})])
        assert extract_scores(SimpleNamespace(results=results)) == (0.42, 0.01)

    def test_falls_back_to_mean_then_to_whatever_exists(self):
        results = _results([_score("custom", {"mean": 0.5})])
        assert extract_scores(SimpleNamespace(results=results)) == (0.5, None)
        results = _results([_score("custom", {"f1": 0.7})])
        assert extract_scores(SimpleNamespace(results=results)) == (0.7, None)

    def test_no_scores(self):
        assert extract_scores(SimpleNamespace(results=_results([]))) == (None, None)
        assert extract_scores(SimpleNamespace(results=None)) == (None, None)


class TestReadEvalScores:
    """Round-trip through a real Inspect log file, as written on disk by `inspect eval`."""

    def test_reads_the_newest_log(self, tmp_path):
        from inspect_ai.log import (
            EvalConfig,
            EvalDataset,
            EvalLog,
            EvalSpec,
            write_eval_log,
        )
        from inspect_ai.log._log import HeadlineMetric

        log = EvalLog(
            eval=EvalSpec(
                created="2026-09-22T00:00:00",
                task="kmmlu_pro",
                dataset=EvalDataset(),
                model="openai/Qwen3-0.6B-Base-checkpoint-100",
                config=EvalConfig(),
            ),
            results=_results(
                [_score("choice", {"accuracy": 0.7345, "stderr": 0.0123})],
                headline=HeadlineMetric(scorer="choice", metric="accuracy"),
            ),
        )
        write_eval_log(log, str(tmp_path / "kmmlu_pro.eval"))
        assert read_eval_scores(tmp_path) == (0.7345, 0.0123)

    def test_missing_log_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_eval_scores(tmp_path)


# A stand-in for `vllm serve`: answers /health after a delay, ignores everything else.
_FAKE_SERVER = """
import sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--port") + 1])
delay = float(sys.argv[sys.argv.index("--warmup") + 1])
time.sleep(delay)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()
    def log_message(self, *args):
        pass

HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def _fake_binaries(monkeypatch, tmp_path, **scripts):
    """Run Python stand-ins in place of the `vllm` / `inspect` binaries.

    Rewrites only argv[0], so everything the callback actually does around the
    subprocess -- session creation, the /health poll, `killpg` teardown, the log
    files, reading the Inspect log back -- runs for real.  A PATH shim would be
    closer to the real thing but needs an executable bit, and the temp directory is
    mounted `noexec` on some machines.
    """
    import subprocess as sp

    real_popen, real_run = sp.Popen, sp.run
    paths = {}
    for name, body in scripts.items():
        script = tmp_path / f"{name}_stub.py"
        script.write_text(body)
        paths[name] = str(script)

    def rewrite(command):
        if isinstance(command, list) and command and command[0] in paths:
            return [sys.executable, paths[command[0]], *command[1:]]
        return command

    monkeypatch.setattr(
        ce.subprocess, "Popen", lambda cmd, *a, **k: real_popen(rewrite(cmd), *a, **k)
    )
    monkeypatch.setattr(
        ce.subprocess, "run", lambda cmd, *a, **k: real_run(rewrite(cmd), *a, **k)
    )


class TestVllmEvalServer:
    """Real subprocesses: start, health-poll, and teardown of the whole process group."""

    def _server(self, tmp_path, warmup="0"):
        return VllmEvalServer(
            model_dir=tmp_path / "model",
            served_name="test-checkpoint-100",
            devices="0",
            extra_args=["--warmup", warmup],
            port=find_free_port(start=8100),
            log_path=tmp_path / "vllm.log",
        )

    def test_command_shape(self, tmp_path):
        server = self._server(tmp_path)
        command = server.command()
        assert command[:3] == ["vllm", "serve", str(tmp_path / "model")]
        assert "--served-model-name" in command
        assert command[command.index("--served-model-name") + 1] == (
            "test-checkpoint-100"
        )
        assert command[command.index("--port") + 1] == str(server.port)
        assert command[-2:] == ["--warmup", "0"]
        assert server.base_url == f"http://127.0.0.1:{server.port}/v1"

    def test_starts_serves_and_stops(self, tmp_path, monkeypatch):
        _fake_binaries(monkeypatch, tmp_path, vllm=_FAKE_SERVER)
        server = self._server(tmp_path)
        server.start()
        try:
            assert server.wait_until_ready(timeout=30) is True
            pid = server.process.pid
        finally:
            server.stop()
        # The process group is gone, and stop() is idempotent.
        time.sleep(0.5)
        with pytest.raises(OSError):
            os.kill(pid, 0)
        server.stop()

    def test_reports_a_server_that_dies_during_startup(self, tmp_path, monkeypatch):
        _fake_binaries(
            monkeypatch, tmp_path, vllm='import sys; print("CUDA OOM"); sys.exit(3)'
        )
        server = self._server(tmp_path)
        server.start()
        try:
            with pytest.raises(RuntimeError, match="exited with code 3"):
                server.wait_until_ready(timeout=30)
        finally:
            server.stop()
        # The failure carries the server's own output, which is where the reason is.
        assert "CUDA OOM" in (tmp_path / "vllm.log").read_text()

    def test_times_out_on_a_server_that_never_answers(self, tmp_path, monkeypatch):
        _fake_binaries(monkeypatch, tmp_path, vllm=_FAKE_SERVER)
        server = self._server(tmp_path, warmup="60")
        server.start()
        try:
            with pytest.raises(TimeoutError):
                server.wait_until_ready(timeout=2)
        finally:
            server.stop()


class TestCallbackOnSave:
    """`on_save` does the snapshot and nothing else on the training thread."""

    def _callback(self, **overrides):
        options = {
            "tasks": ["inspect_evals/mmlu_pro"],
            "devices": "7",
            "vllm_args": "",
            "model_tag": "test",
        }
        options.update(overrides)
        return CheckpointEvalCallback(**options)

    def test_non_main_ranks_are_a_no_op(self, tmp_path, monkeypatch):
        """No staging, no thread, and above all no collective."""
        callback = self._callback()
        started = []
        monkeypatch.setattr(callback, "_evaluate", lambda *a: started.append(a))
        _checkpoint(tmp_path)
        callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)),
            _state(step=100, main=False),
            TrainerControl(),
        )
        assert started == []
        assert not (tmp_path / "_eval_staging").exists()

    def test_stages_before_returning_and_hands_off_to_a_thread(
        self, tmp_path, monkeypatch
    ):
        callback = self._callback()
        seen = []
        release = threading.Event()

        def fake_evaluate(staged_dir, global_step):
            seen.append((Path(staged_dir), global_step))
            release.wait(timeout=10)

        monkeypatch.setattr(callback, "_evaluate", fake_evaluate)
        _checkpoint(tmp_path)
        control = callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)), _state(), TrainerControl()
        )
        # on_save returned while the eval is still running: training is not blocked.
        assert control.should_save is False
        staged = tmp_path / "_eval_staging" / "checkpoint-100"
        assert (staged / "model.safetensors").exists()
        release.set()
        callback._thread.join(timeout=10)
        assert seen == [(staged, 100)]

    def test_skips_a_checkpoint_while_an_eval_is_still_running(
        self, tmp_path, monkeypatch
    ):
        callback = self._callback()
        calls = []
        release = threading.Event()

        def fake_evaluate(staged_dir, global_step):
            calls.append(global_step)
            release.wait(timeout=10)

        monkeypatch.setattr(callback, "_evaluate", fake_evaluate)
        args = SimpleNamespace(output_dir=str(tmp_path))
        _checkpoint(tmp_path, "checkpoint-100")
        _checkpoint(tmp_path, "checkpoint-200")
        callback.on_save(args, _state(100), TrainerControl())
        callback.on_save(args, _state(200), TrainerControl())
        release.set()
        callback._thread.join(timeout=10)
        assert calls == [100]
        assert not (tmp_path / "_eval_staging" / "checkpoint-200").exists()

    def test_a_failed_snapshot_does_not_raise_into_training(self, tmp_path):
        callback = self._callback()
        # No checkpoint directory at all -- e.g. a save that only rank 0 writes.
        control = callback.on_save(
            SimpleNamespace(output_dir=str(tmp_path)), _state(), TrainerControl()
        )
        assert control.should_save is False
        assert callback._thread is None


class TestEvaluate:
    """The background worker end to end, against fake `vllm` and `inspect` binaries."""

    def test_runs_every_task_and_logs_the_scores(self, tmp_path, monkeypatch):
        # A fake `inspect` that records its argv and writes a real Inspect log.
        fake_inspect = f"""
import json, sys
sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
from inspect_ai.log import (
    EvalConfig, EvalDataset, EvalLog, EvalResults, EvalScore, EvalSpec, write_eval_log
)
from inspect_ai.log._log import EvalMetric, HeadlineMetric

argv = sys.argv[1:]
log_dir = argv[argv.index("--log-dir") + 1]
task = argv[1]
with open({str(tmp_path / "calls.jsonl")!r}, "a") as f:
    f.write(json.dumps(argv) + "\\n")
score = 0.25 if "kmmlu" in task else 0.5
write_eval_log(
    EvalLog(
        eval=EvalSpec(
            created="2026-09-22T00:00:00", task=task, dataset=EvalDataset(),
            model=argv[argv.index("--model") + 1], config=EvalConfig(),
        ),
        results=EvalResults(
            total_samples=4, completed_samples=4,
            scores=[EvalScore(name="choice", scorer="choice", metrics={{
                "accuracy": EvalMetric(name="accuracy", value=score),
                "stderr": EvalMetric(name="stderr", value=0.01),
            }})],
            headline=HeadlineMetric(scorer="choice", metric="accuracy"),
        ),
    ),
    log_dir + "/result.eval",
)
"""
        _fake_binaries(monkeypatch, tmp_path, vllm=_FAKE_SERVER, inspect=fake_inspect)

        callback = CheckpointEvalCallback(
            tasks=["inspect_evals/mmlu_pro", "evals/kmmlu_pro.py"],
            devices="7",
            vllm_args="--warmup 0",
            max_connections=20,
            max_tokens=16000,
            model_tag="Qwen3-0.6B-Base",
        )
        logged = []
        monkeypatch.setattr(
            callback,
            "_log_to_wandb",
            lambda metrics, step: logged.append((metrics, step)),
        )

        source = _checkpoint(tmp_path / "run")
        staged, _, _ = stage_checkpoint(
            source, tmp_path / "run" / "_eval_staging" / "checkpoint-100"
        )
        callback._evaluate(staged, 100)

        assert logged == [
            (
                {
                    "eval/mmlu_pro": 0.5,
                    "eval_stderr/mmlu_pro": 0.01,
                    "eval/kmmlu_pro": 0.25,
                    "eval_stderr/kmmlu_pro": 0.01,
                },
                100,
            )
        ]

        calls = [
            json.loads(line)
            for line in (tmp_path / "calls.jsonl").read_text().splitlines()
        ]
        assert len(calls) == 2
        first = calls[0]
        assert first[0] == "eval"
        assert first[1] == "inspect_evals/mmlu_pro"
        assert first[first.index("--model") + 1] == (
            "openai/Qwen3-0.6B-Base-checkpoint-100"
        )
        assert first[first.index("--model-base-url") + 1].startswith(
            "http://127.0.0.1:"
        )
        assert first[first.index("-M") + 1] == "responses_api=false"
        assert first[first.index("--max-connections") + 1] == "20"
        assert first[first.index("--max-tokens") + 1] == "16000"
        # Local task files are resolved to an absolute path for the subprocess.
        assert calls[1][1].endswith("evals/kmmlu_pro.py")

        # Server down, snapshot cleaned up, eval logs kept for inspection.
        assert callback._server is None
        assert not staged.exists()
        assert (tmp_path / "run" / "eval_logs" / "step-100" / "vllm.log").exists()

    def test_one_failing_task_does_not_stop_the_others(self, tmp_path, monkeypatch):
        fake_inspect = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
argv = sys.argv[1:]
if "gpqa" in argv[1]:
    print("inspect_evals is not installed")
    sys.exit(1)
from inspect_ai.log import (
    EvalConfig, EvalDataset, EvalLog, EvalResults, EvalScore, EvalSpec, write_eval_log
)
from inspect_ai.log._log import EvalMetric
log_dir = argv[argv.index("--log-dir") + 1]
write_eval_log(
    EvalLog(
        eval=EvalSpec(
            created="2026-09-22T00:00:00", task=argv[1], dataset=EvalDataset(),
            model="m", config=EvalConfig(),
        ),
        results=EvalResults(
            total_samples=4, completed_samples=4,
            scores=[EvalScore(name="choice", scorer="choice",
                              metrics={{"accuracy": EvalMetric(name="accuracy", value=0.9)}})],
        ),
    ),
    log_dir + "/result.eval",
)
"""
        _fake_binaries(monkeypatch, tmp_path, vllm=_FAKE_SERVER, inspect=fake_inspect)

        callback = CheckpointEvalCallback(
            tasks=["inspect_evals/gpqa_diamond", "inspect_evals/mmlu_pro"],
            devices="7",
            vllm_args="--warmup 0",
            model_tag="test",
        )
        logged = []
        monkeypatch.setattr(
            callback, "_log_to_wandb", lambda metrics, step: logged.append(metrics)
        )
        source = _checkpoint(tmp_path / "run")
        staged, _, _ = stage_checkpoint(
            source, tmp_path / "run" / "_eval_staging" / "checkpoint-100"
        )
        callback._evaluate(staged, 100)
        assert logged == [{"eval/mmlu_pro": 0.9}]

    def test_a_server_that_never_starts_is_only_a_warning(self, tmp_path, monkeypatch):
        _fake_binaries(
            monkeypatch,
            tmp_path,
            vllm="import sys; sys.exit(1)",
            inspect="raise SystemExit(0)",
        )
        callback = CheckpointEvalCallback(
            tasks=["inspect_evals/mmlu_pro"], devices="7", model_tag="test"
        )
        source = _checkpoint(tmp_path / "run")
        staged, _, _ = stage_checkpoint(
            source, tmp_path / "run" / "_eval_staging" / "checkpoint-100"
        )
        callback._evaluate(staged, 100)  # must not raise
        assert callback._server is None
        assert not staged.exists()


class TestBuildCallback:
    """Startup validation: a misconfigured eval fails now, not six hours in."""

    def _args(self, **overrides):
        dnotitia = SimpleNamespace(
            eval_on_checkpoint=True,
            eval_tasks=["inspect_evals/mmlu_pro"],
            eval_devices="0,1",
            eval_vllm_args="--max-model-len 32768 --data-parallel-size 2",
            eval_max_connections=20,
            eval_max_tokens=16000,
        )
        for key, value in overrides.items():
            setattr(dnotitia, key, value)
        training = SimpleNamespace(
            output_dir="out", run_name="run", report_to=["wandb"]
        )
        model = SimpleNamespace(model_name_or_path="dnotitia/Qwen3-0.6B-Base")
        return training, model, dnotitia

    def test_disabled_returns_none(self, caplog):
        training, model, dnotitia = self._args(eval_on_checkpoint=False)
        assert build_checkpoint_eval_callback(training, model, dnotitia, pytest) is None

    def test_requires_devices(self):
        training, model, dnotitia = self._args(eval_devices="  ")
        with pytest.raises(ValueError, match="eval_devices"):
            build_checkpoint_eval_callback(training, model, dnotitia, pytest)

    def test_requires_tasks(self):
        training, model, dnotitia = self._args(eval_tasks=[])
        with pytest.raises(ValueError, match="eval_tasks"):
            build_checkpoint_eval_callback(training, model, dnotitia, pytest)

    def test_requires_the_binaries(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        training, model, dnotitia = self._args()
        with pytest.raises(ValueError, match="not on PATH"):
            build_checkpoint_eval_callback(training, model, dnotitia, pytest)

    def test_builds_a_configured_callback(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        warnings = []
        train_logger = SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: warnings.append(a)
        )
        training, model, dnotitia = self._args()
        callback = build_checkpoint_eval_callback(
            training, model, dnotitia, train_logger
        )
        assert callback.tasks == ["inspect_evals/mmlu_pro"]
        assert callback.devices == "0,1"
        assert callback.model_tag == "run"
        assert callback.vllm_args == [
            "--max-model-len",
            "32768",
            "--data-parallel-size",
            "2",
        ]
        assert warnings == []

    def test_warns_when_eval_devices_overlap_training(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,3")
        warnings = []
        train_logger = SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: warnings.append(a[0])
        )
        training, model, dnotitia = self._args()
        build_checkpoint_eval_callback(training, model, dnotitia, train_logger)
        assert any("overlaps the training devices" in w for w in warnings)

    def test_warns_when_devices_and_parallel_sizes_disagree(self, monkeypatch):
        """Overriding eval_devices but not --data-parallel-size is the easy mistake."""
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        warnings = []
        train_logger = SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: warnings.append(a[0])
        )
        training, model, dnotitia = self._args(eval_devices="7")
        build_checkpoint_eval_callback(training, model, dnotitia, train_logger)
        assert any("multiply out to" in w for w in warnings)

    def test_no_mismatch_warning_for_the_shipped_defaults(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        warnings = []
        train_logger = SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: warnings.append(a[0])
        )
        training, model, dnotitia = self._args()
        build_checkpoint_eval_callback(training, model, dnotitia, train_logger)
        assert warnings == []

    def test_warns_when_wandb_is_not_a_reporting_backend(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        warnings = []
        train_logger = SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: warnings.append(a[0])
        )
        training, model, dnotitia = self._args()
        training.report_to = ["none"]
        build_checkpoint_eval_callback(training, model, dnotitia, train_logger)
        assert any("report_to does not include wandb" in w for w in warnings)
