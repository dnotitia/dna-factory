"""Benchmark every checkpoint with Inspect while training keeps running.

`PeriodicCheckpointCallback` turns wall-clock time into checkpoints; this turns
those checkpoints into a quality curve next to the loss curve.  On every `on_save`
the callback snapshots the fresh checkpoint, and a background thread serves it with
vLLM on a separate GPU, runs each configured `inspect eval` task against that
server, and logs the scores to the live W&B run.  Training never waits for any of
it -- the only work on the training thread is the snapshot, which is a handful of
`os.link` calls.

Four properties this file exists to guarantee:

- **No collective.**  Everything is gated on `state.is_world_process_zero`, so the
  other ranks return from `on_save` untouched.  Unlike `PeriodicCheckpointCallback`
  there is nothing to agree on here: the eval reads a directory rank 0 already
  wrote, and posts no NCCL op.
- **The checkpoint can't vanish mid-eval.**  `save_total_limit` (default `3`)
  deletes old checkpoints from inside the *next* `_save_checkpoint`, which can
  easily land while a multi-hour eval is still reading.  So `on_save` immediately
  hardlinks the inference files (weights, tokenizer, configs -- not `optimizer.pt`
  and friends) into `output_dir/_eval_staging/checkpoint-N`.  Hardlinks cost no
  disk and no copy time, and the inode outlives the `rmtree` that rotation does.
- **Eval failure never touches training.**  Every step of the background run is
  wrapped; the worst case is a warning in the training log.
- **No orphan server.**  vLLM is started in its own session and torn down through
  `os.killpg` on the normal path, on exception, at `on_train_end`, and from an
  `atexit` hook (which also covers `KeyboardInterrupt` and an unhandled crash).
  Only `SIGKILL` on the trainer can leak one.

Wired in `dna_factory/training_runner.py` from the `eval_*` fields of
`DnotitiaArguments`; off unless `eval_on_checkpoint: true`.  See
docs/checkpoint-eval.md.
"""

import atexit
import fnmatch
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import requests
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EVAL_TASKS = (
    "inspect_evals/mmlu_pro",
    "inspect_evals/gpqa_diamond",
    "evals/kmmlu_pro.py",
    "evals/kmmlu_redux.py",
)

# Name of the staging directory inside `output_dir`.  The leading underscore keeps
# it out of `push_to_hub` (which ignores `_*`), and the name does not match
# `checkpoint-*`, so neither `save_total_limit` rotation nor `get_last_checkpoint`
# resume ever looks at it.
STAGING_DIR_NAME = "_eval_staging"
EVAL_LOG_DIR_NAME = "eval_logs"

# Loading a big model from disk plus CUDA graph capture is minutes, not seconds;
# generous enough for a 120B, short enough that a wedged server doesn't sit on the
# eval GPUs until training ends.
_SERVER_STARTUP_TIMEOUT_S = 900.0
# Poll fast at first so a server that comes up quickly isn't made to wait, then back
# off -- a cold 120B load is minutes, and there is no point hammering /health for it.
_SERVER_POLL_MIN_INTERVAL_S = 0.25
_SERVER_POLL_MAX_INTERVAL_S = 5.0
_SERVER_SHUTDOWN_GRACE_S = 30.0
# `on_train_end` waits this long for an in-flight eval before killing it, so a
# hung eval can't hold the training script open indefinitely.
_TRAIN_END_JOIN_TIMEOUT_S = 3600.0
_PORT_SCAN_START = 8000
_PORT_SCAN_COUNT = 100
_TAIL_LINES_ON_FAILURE = 20

# Files a checkpoint holds only to resume training with.  vLLM does not read them
# and `optimizer.pt` alone is several times the model size, so the snapshot skips
# them -- which also keeps the copy fallback (see `stage_checkpoint`) affordable.
_TRAINING_STATE_PATTERNS = (
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
    "trainer_state.json",
    "training_args.bin",
    "zero_to_fp32.py",
    "latest",
)

# torchrun/accelerate export these into the trainer's environment.  vLLM sets up
# its own distributed world, and inheriting rank 0's rendezvous vars makes it try
# to join the *training* process group instead.
_DISTRIBUTED_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_ERROR_FILE",
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_DEEPSPEED_PLUGIN_TYPE",
)

_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def normalize_eval_tasks(value):
    """Accept a YAML list, a comma/whitespace-separated string, or `None`."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in re.split(r"[,\s]+", value.strip()) if part]
    return [str(part).strip() for part in value if str(part).strip()]


def derive_task_name(task):
    """Metric name for a task spec.

    `inspect_evals/mmlu_pro` -> `mmlu_pro`, `evals/kmmlu_pro.py` -> `kmmlu_pro`,
    `evals/suite.py@kmmlu_pro` -> `kmmlu_pro`.  Derived from the spec rather than
    read back out of the Inspect log so the W&B key for a task is identical at
    every checkpoint, including the ones whose eval failed.
    """
    spec = task.split("@")[-1] if "@" in task else task
    name = Path(spec).name
    name = name.removesuffix(".py")
    return _SAFE_TAG_RE.sub("_", name) or "task"


def resolve_task(task, repo_root=REPO_ROOT):
    """Resolve a local task *file* against the repo root when cwd doesn't have it.

    Registry specs (`inspect_evals/mmlu_pro`) and anything that already exists
    relative to cwd are passed through untouched, so `evals/kmmlu_pro.py` keeps
    working from the repo root and starts working from anywhere else.
    """
    head = task.split("@")[0]
    if not head.endswith(".py") or os.path.isabs(head):
        return task
    if os.path.exists(head):
        return task
    candidate = Path(repo_root) / head
    if candidate.exists():
        suffix = task[len(head) :]
        return f"{candidate}{suffix}"
    return task


def sanitize_tag(value, fallback="model"):
    """Reduce a string to what vLLM's `--served-model-name` and logs handle well."""
    tag = _SAFE_TAG_RE.sub("-", str(value)).strip("-.")
    return tag or fallback


def served_model_tag(training_args, model_args):
    """Short, stable name for `--served-model-name` and the Inspect log's model field.

    `run_name` first, since that is the name the run already goes by in W&B, then the
    base model's directory name.  Deliberately *not* derived from `output_dir`: with
    `output_dir: auto` that is a 200-character dotted summary of every non-default
    argument, and its leading segment can't be split off reliably either -- model
    names carry dots of their own (`Qwen3-0.6B-SFT.ep-2`).
    """
    run_name = getattr(training_args, "run_name", None)
    # TrainingArguments defaults run_name to output_dir, which means "unset" here.
    if run_name and run_name != training_args.output_dir:
        return sanitize_tag(Path(str(run_name).rstrip("/")).name)
    return sanitize_tag(Path(str(model_args.model_name_or_path).rstrip("/")).name)


def served_name_for_step(model_tag, global_step):
    """`--served-model-name` for one checkpoint, e.g. `Qwen3-4B-SFT-checkpoint-1776`.

    It is also the model name Inspect records in its log, which is what makes an
    eval log traceable back to the exact checkpoint that produced it.
    """
    return f"{model_tag}-{PREFIX_CHECKPOINT_DIR}-{global_step}"


def _is_training_state(name):
    return any(fnmatch.fnmatch(name, pattern) for pattern in _TRAINING_STATE_PATTERNS)


def stage_checkpoint(checkpoint_dir, staging_dir):
    """Snapshot the servable files of `checkpoint_dir` into `staging_dir`.

    Hardlinks by preference: the staging dir lives inside `output_dir`, so it is the
    same filesystem as the checkpoint and a link is a metadata write.  That is what
    makes it safe to do this on the training thread, and what makes the snapshot
    survive `save_total_limit` -- rotation unlinks the checkpoint's names, the
    inodes stay alive as long as the staged names exist.

    Falls back to a real copy per file if linking is refused (an overlay/NFS mount,
    or a `staging_dir` on another device); that costs one model's worth of disk and
    time, which is why the training-state files are filtered out first.

    Subdirectories are skipped, which also drops DeepSpeed's `global_step*/` shards.
    ZeRO-3 writes the consolidated 16-bit weights as top-level safetensors
    (`zero3_save_16bit_model: true` in the accelerate configs), and those are what
    vLLM loads.
    """
    source = Path(checkpoint_dir)
    destination = Path(staging_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    linked = copied = 0
    for entry in sorted(source.iterdir()):
        if not entry.is_file() or _is_training_state(entry.name):
            continue
        target = destination / entry.name
        try:
            os.link(entry, target)
            linked += 1
        except OSError:
            shutil.copy2(entry, target)
            copied += 1
    if not any(destination.glob("*.safetensors")) and not any(
        destination.glob("*.bin")
    ):
        raise FileNotFoundError(
            f"No model weights found in {source}; nothing for vLLM to serve."
        )
    return destination, linked, copied


def find_free_port(start=_PORT_SCAN_START, count=_PORT_SCAN_COUNT):
    """First port from `start` upward that nothing is listening on.

    Inherently racy -- something can grab the port between the probe and vLLM's
    bind -- but the loser is the eval, which fails with a warning.  Scanning up
    from 8000 also means an AsyncGRPO run, whose own rollout server sits on the
    vLLM default port, gets 8001 instead of a collision.
    """
    for port in range(start, start + count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port in [{start}, {start + count}).")


def split_vllm_args(vllm_args):
    """Split the extra `vllm serve` flags, and pull out an explicit `--port`."""
    tokens = shlex.split(vllm_args or "")
    port = None
    for index, token in enumerate(tokens):
        if token == "--port" and index + 1 < len(tokens):
            port = int(tokens[index + 1])
        elif token.startswith("--port="):
            port = int(token.split("=", 1)[1])
    return tokens, port


def _child_env(**overrides):
    env = os.environ.copy()
    for name in _DISTRIBUTED_ENV_VARS:
        env.pop(name, None)
    env.update(overrides)
    return env


def _tail(path, lines=_TAIL_LINES_ON_FAILURE):
    try:
        content = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return "<no output captured>"
    return "\n".join(content[-lines:]) or "<empty>"


def extract_scores(eval_log):
    """`(primary, stderr)` for one Inspect log, or `(None, None)`.

    Prefers the log's own headline metric -- the number `inspect view` shows -- and
    falls back to `accuracy`/`mean`/whatever the first scorer reports, so a task
    with a non-standard scorer still yields a curve.
    """
    results = getattr(eval_log, "results", None)
    scores = getattr(results, "scores", None) or []
    if not scores:
        return None, None

    headline = getattr(results, "headline", None)
    chosen = None
    if headline is not None and headline.metric:
        for score in scores:
            if headline.scorer in (score.name, score.scorer):
                chosen = score.metrics.get(headline.metric)
                break
    if chosen is None:
        metrics = scores[0].metrics
        for preferred in ("accuracy", "mean"):
            if preferred in metrics:
                chosen = metrics[preferred]
                break
        else:
            chosen = next(iter(metrics.values()), None)

    stderr = scores[0].metrics.get("stderr")
    return (
        None if chosen is None else chosen.value,
        None if stderr is None else stderr.value,
    )


def read_eval_scores(log_dir):
    """Read the newest Inspect log under `log_dir` and pull its scores out."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    infos = list_eval_logs(str(log_dir), descending=True)
    if not infos:
        raise FileNotFoundError(f"Inspect wrote no log under {log_dir}.")
    return extract_scores(read_eval_log(infos[0], header_only=True))


class VllmEvalServer:
    """A `vllm serve` subprocess pinned to the eval GPUs.

    Started in its own session (`start_new_session=True`) so `stop()` can signal the
    whole tree -- vLLM's engine-core and worker processes included -- with one
    `killpg`, without that signal reaching the trainer.
    """

    def __init__(self, model_dir, served_name, devices, extra_args, port, log_path):
        self.model_dir = str(model_dir)
        self.served_name = served_name
        self.devices = str(devices)
        self.extra_args = list(extra_args)
        self.port = port
        self.log_path = Path(log_path)
        self.process = None
        self._log_file = None

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}/v1"

    def command(self):
        return [
            "vllm",
            "serve",
            self.model_dir,
            "--served-model-name",
            self.served_name,
            "--port",
            str(self.port),
            *self.extra_args,
        ]

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self.log_path.open("w")
        self.process = subprocess.Popen(
            self.command(),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=_child_env(CUDA_VISIBLE_DEVICES=self.devices),
            start_new_session=True,
        )
        return self.process

    def wait_until_ready(self, timeout=_SERVER_STARTUP_TIMEOUT_S):
        """Poll `/health` until the server answers, the process dies, or we time out."""
        health_url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + timeout
        interval = _SERVER_POLL_MIN_INTERVAL_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self.process.returncode} before serving. "
                    f"Last lines of {self.log_path}:\n{_tail(self.log_path)}"
                )
            try:
                if requests.get(health_url, timeout=5).status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
            interval = min(interval * 2, _SERVER_POLL_MAX_INTERVAL_S)
        raise TimeoutError(
            f"vLLM did not become healthy on port {self.port} within {timeout:g}s. "
            f"Last lines of {self.log_path}:\n{_tail(self.log_path)}"
        )

    def stop(self):
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=_SERVER_SHUTDOWN_GRACE_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except OSError:
                    pass
            except OSError:
                pass
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


class CheckpointEvalCallback(TrainerCallback):
    """Serve each new checkpoint with vLLM, score it with Inspect, log it to W&B.

    One eval at a time: a checkpoint that arrives while the previous one is still
    being evaluated is skipped with a warning rather than queued, because queueing
    would only push the backlog further behind on a run that saves faster than it
    evaluates.
    """

    def __init__(
        self,
        tasks,
        devices,
        vllm_args="",
        max_connections=20,
        max_tokens=16000,
        model_tag="model",
        repo_root=REPO_ROOT,
        train_logger=None,
    ):
        self.tasks = list(tasks)
        self.devices = str(devices)
        self.vllm_args, self.explicit_port = split_vllm_args(vllm_args)
        self.max_connections = max_connections
        self.max_tokens = max_tokens
        self.model_tag = model_tag
        self.repo_root = Path(repo_root)
        self.logger = train_logger or logger

        self._lock = threading.Lock()
        self._thread = None
        self._server = None
        self._wandb_axis_defined = False
        atexit.register(self.shutdown)

    # -- trainer hooks ---------------------------------------------------------

    def on_save(self, args, state, control, **kwargs):
        # Rank 0 only, and no collective: the other ranks fall straight through so
        # their step loop stays in lockstep with rank 0's.
        if not state.is_world_process_zero:
            return control

        checkpoint_dir = (
            Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self.logger.warning(
                    "Checkpoint eval: still evaluating an earlier checkpoint; skipping "
                    "%s. Raise periodic_save_seconds or trim eval_tasks if this repeats.",
                    checkpoint_dir.name,
                )
                return control

            # Snapshot on the training thread, before returning control to the loop:
            # from here on `save_total_limit` may delete the checkpoint at any time.
            try:
                staged, linked, copied = stage_checkpoint(
                    checkpoint_dir,
                    Path(args.output_dir) / STAGING_DIR_NAME / checkpoint_dir.name,
                )
            except Exception as error:  # noqa: BLE001 - eval must never fail training
                self.logger.warning(
                    "Checkpoint eval: could not snapshot %s (%s); skipping this eval.",
                    checkpoint_dir,
                    error,
                )
                return control
            if copied:
                self.logger.warning(
                    "Checkpoint eval: %d file(s) of %s had to be copied rather than "
                    "hardlinked (staging dir on another filesystem?), which costs disk "
                    "and made this save slower.",
                    copied,
                    checkpoint_dir.name,
                )
            self.logger.info(
                "Checkpoint eval: staged %s (%d hardlinked, %d copied); starting a "
                "background eval of %d task(s) on CUDA device(s) %s. Training continues.",
                checkpoint_dir.name,
                linked,
                copied,
                len(self.tasks),
                self.devices,
            )

            self._thread = threading.Thread(
                target=self._evaluate,
                args=(staged, state.global_step),
                name=f"checkpoint-eval-{state.global_step}",
                daemon=True,
            )
            self._thread.start()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        thread = self._thread
        if thread is not None and thread.is_alive():
            self.logger.info(
                "Checkpoint eval: waiting up to %g s for the in-flight eval to finish.",
                _TRAIN_END_JOIN_TIMEOUT_S,
            )
            thread.join(timeout=_TRAIN_END_JOIN_TIMEOUT_S)
            if thread.is_alive():
                self.logger.warning(
                    "Checkpoint eval: the in-flight eval is still running after %g s; "
                    "tearing its vLLM server down and abandoning it.",
                    _TRAIN_END_JOIN_TIMEOUT_S,
                )
        self.shutdown()
        return control

    # -- background worker -----------------------------------------------------

    def shutdown(self):
        """Kill the eval server if one is up. Idempotent; also the `atexit` hook."""
        server, self._server = self._server, None
        if server is not None:
            server.stop()

    def _evaluate(self, staged_dir, global_step):
        """Serve `staged_dir`, run every task against it, log the scores. Never raises."""
        served_name = served_name_for_step(self.model_tag, global_step)
        eval_root = staged_dir.parent.parent / EVAL_LOG_DIR_NAME / f"step-{global_step}"
        try:
            port = self.explicit_port or find_free_port()
            server = VllmEvalServer(
                model_dir=staged_dir,
                served_name=served_name,
                devices=self.devices,
                extra_args=self.vllm_args,
                port=port,
                log_path=eval_root / "vllm.log",
            )
            self._server = server
            server.start()
            self.logger.info(
                "Checkpoint eval: vLLM starting for %s on port %d (log: %s).",
                served_name,
                port,
                server.log_path,
            )
            server.wait_until_ready()

            metrics = {}
            for task in self.tasks:
                name = derive_task_name(task)
                try:
                    primary, stderr = self._run_task(task, server, eval_root / name)
                except Exception as error:  # noqa: BLE001 - one bad task, not the set
                    self.logger.warning(
                        "Checkpoint eval: task %s failed at step %d: %s",
                        task,
                        global_step,
                        error,
                    )
                    continue
                if primary is None:
                    self.logger.warning(
                        "Checkpoint eval: task %s produced no score at step %d.",
                        task,
                        global_step,
                    )
                    continue
                metrics[f"eval/{name}"] = primary
                if stderr is not None:
                    metrics[f"eval_stderr/{name}"] = stderr
                self.logger.info(
                    "Checkpoint eval: step %d, %s = %.4f", global_step, name, primary
                )

            if metrics:
                self._log_to_wandb(metrics, global_step)
        except Exception as error:  # noqa: BLE001 - eval must never fail training
            self.logger.warning(
                "Checkpoint eval: evaluation of step %d failed: %s", global_step, error
            )
        finally:
            self.shutdown()
            shutil.rmtree(staged_dir, ignore_errors=True)

    def _run_task(self, task, server, log_dir):
        """Run one `inspect eval` against the live server and read back its score."""
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "inspect.log"
        command = [
            "inspect",
            "eval",
            resolve_task(task, self.repo_root),
            "--model",
            f"openai/{server.served_name}",
            "--model-base-url",
            server.base_url,
            "-M",
            "responses_api=false",
            "--max-connections",
            str(self.max_connections),
            "--max-tokens",
            str(self.max_tokens),
            "--log-dir",
            str(log_dir),
            "--display",
            "plain",
        ]
        # The served model needs no credential, but the OpenAI-compatible client
        # refuses to start without one; and a real OPENAI_BASE_URL in the
        # environment would otherwise fight `--model-base-url`.
        env = _child_env(
            OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY") or "dna-factory-local"
        )
        env.pop("OPENAI_BASE_URL", None)

        with stdout_path.open("w") as stdout_file:
            completed = subprocess.run(
                command,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"`inspect eval {task}` exited with code {completed.returncode}. "
                f"Last lines of {stdout_path}:\n{_tail(stdout_path)}"
            )
        return read_eval_scores(log_dir)

    def _log_to_wandb(self, metrics, global_step):
        """Log the scores against a dedicated `eval/step` axis.

        An eval finishes minutes to hours after the checkpoint it scores, by which
        time the run's own `_step` has moved far past it, and W&B silently drops a
        `log(step=)` that goes backwards.  So the checkpoint's `global_step` is
        logged as a *metric* and declared as the x-axis for `eval/*`, which puts the
        eval curves on the same step axis as the training curves without either
        side having to wait for the other.
        """
        try:
            import wandb
        except ImportError:
            self.logger.warning(
                "Checkpoint eval: wandb is not installed; scores for step %d are in "
                "the training log only.",
                global_step,
            )
            return
        run = wandb.run
        if run is None:
            self.logger.warning(
                "Checkpoint eval: no active W&B run (report_to does not include "
                "wandb); scores for step %d are in the training log only.",
                global_step,
            )
            return
        try:
            if not self._wandb_axis_defined:
                run.define_metric("eval/step")
                run.define_metric("eval/*", step_metric="eval/step")
                run.define_metric("eval_stderr/*", step_metric="eval/step")
                self._wandb_axis_defined = True
            run.log({"eval/step": global_step, **metrics})
        except Exception as error:  # noqa: BLE001 - eval must never fail training
            self.logger.warning(
                "Checkpoint eval: could not log step %d to W&B: %s", global_step, error
            )


def build_checkpoint_eval_callback(
    training_args, model_args, dnotitia_args, train_logger
):
    """Validate the `eval_*` arguments and build the callback, or return `None`.

    Called from `run_training` before the model loads, so a misconfigured eval is a
    startup error rather than a surprise six hours in.
    """
    if not getattr(dnotitia_args, "eval_on_checkpoint", False):
        return None

    tasks = normalize_eval_tasks(getattr(dnotitia_args, "eval_tasks", None))
    if not tasks:
        raise ValueError(
            "eval_on_checkpoint is true but eval_tasks is empty; list the Inspect "
            "tasks to run (e.g. inspect_evals/mmlu_pro, evals/kmmlu_pro.py)."
        )

    devices = str(getattr(dnotitia_args, "eval_devices", "") or "").strip()
    if not devices:
        raise ValueError(
            "eval_on_checkpoint is true but eval_devices is empty. The eval vLLM "
            "server needs GPUs of its own -- training already owns its devices, and "
            "sharing them would OOM the run. Set eval_devices to the spare device "
            "ids, e.g. eval_devices: '0,1'."
        )

    for binary in ("vllm", "inspect"):
        if shutil.which(binary) is None:
            raise ValueError(
                f"eval_on_checkpoint is true but `{binary}` is not on PATH. Install the "
                "project dependencies (`uv sync`) and activate the venv."
            )

    training_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    overlap = {d.strip() for d in devices.split(",") if d.strip()} & {
        d.strip() for d in training_devices.split(",") if d.strip()
    }
    if overlap:
        train_logger.warning(
            "Checkpoint eval: eval_devices %s overlaps the training devices "
            "(CUDA_VISIBLE_DEVICES=%s) on device(s) %s. The eval vLLM server will "
            "compete with training for that memory and can OOM the run.",
            devices,
            training_devices,
            ",".join(sorted(overlap)),
        )

    if "wandb" not in (training_args.report_to or []):
        train_logger.warning(
            "Checkpoint eval: report_to does not include wandb, so the scores will "
            "only reach the training log -- no eval curves next to the loss curve."
        )

    callback = CheckpointEvalCallback(
        tasks=tasks,
        devices=devices,
        vllm_args=getattr(dnotitia_args, "eval_vllm_args", "") or "",
        max_connections=dnotitia_args.eval_max_connections,
        max_tokens=dnotitia_args.eval_max_tokens,
        model_tag=served_model_tag(training_args, model_args),
        train_logger=train_logger,
    )
    train_logger.info(
        "Checkpoint eval enabled: %d task(s) [%s] on CUDA device(s) %s after every "
        "checkpoint; training is never blocked.",
        len(tasks),
        ", ".join(tasks),
        devices,
    )
    return callback
