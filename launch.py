"""Single-node Accelerate launcher: --zero auto tries ZeRO-1 then ZeRO-3 on startup CUDA OOM."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from dna_factory.auto_zero import DEFAULTS, STATUS_ENV, can_retry, config_candidates
from dna_factory.utils.config_merger import merge_config_files

ROOT = Path(__file__).resolve().parent


def training_config(script, argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--config")
    for key in (
        "model_name_or_path",
        "model_revision",
        "grpo_execution",
        "use_unsloth",
        "resume_from_checkpoint",
        "deepspeed",
    ):
        parser.add_argument(f"--{key}", nargs="?", const="true")
    args, _ = parser.parse_known_args(argv)
    merged = Path(
        merge_config_files(
            ROOT / f"configs/_defaults-{DEFAULTS[script]}.yaml", args.config
        )
    )
    try:
        config = yaml.safe_load(merged.read_text())
    finally:
        merged.unlink()
    config.update(
        {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    )
    return config


def model_type(config):
    path = Path(config["model_name_or_path"]) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                config["model_name_or_path"],
                "config.json",
                revision=config.get("model_revision", "main"),
            )
        )
    content = json.loads(path.read_text())
    kind = content.get("model_type")
    if not kind:
        raise ValueError(
            "Model config.json has no model_type; select a validated config manually"
        )
    return kind


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--zero", choices=["auto", "1", "2", "3"], default="auto")
    parser.add_argument("--num-processes", "--num_processes", type=int, required=True)
    parser.add_argument(
        "--main-process-port", "--main_process_port", type=int, default=29500
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("script", choices=list(DEFAULTS))
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.num_processes < 1:
        parser.error("--num-processes must be positive")
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        parser.error("Run python launch.py directly, not inside accelerate/torchrun")
    config = training_config(args.script, args.training_args)
    if str(config.get("use_unsloth", False)).lower() in ("true", "1"):
        parser.error("Use the existing single-GPU entry point for unsloth")
    if config.get("grpo_execution") == "async":
        parser.error("AsyncGRPO does not support DeepSpeed; use grpo.py directly")
    if config.get("deepspeed"):
        parser.error("Do not combine --zero with a training-level deepspeed config")
    if args.zero == "auto" and config.get("resume_from_checkpoint"):
        parser.error(
            "For checkpoint resume, explicitly select the original --zero stage"
        )
    candidates = config_candidates(args.zero, model_type(config))
    for index, filename in enumerate(candidates):
        command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(ROOT / "accelerate_configs" / filename),
            "--num_machines",
            "1",
            "--num_processes",
            str(args.num_processes),
            "--main_process_port",
            str(args.main_process_port),
            str(ROOT / args.script),
            *args.training_args,
        ]
        print(f"[zero={args.zero}] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        with tempfile.TemporaryDirectory(prefix="dna-auto-zero-") as status:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            env.pop(STATUS_ENV, None)
            if args.zero == "auto":
                env[STATUS_ENV] = status
            process = subprocess.Popen(command, env=env, start_new_session=True)
            try:
                code = process.wait()
            finally:
                # Clean up remaining workers before a retry, including on Ctrl-C.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if code == 0:
                return 0
            if index + 1 < len(candidates) and can_retry(status, code):
                print(
                    "[zero=auto] Startup CUDA OOM: restarting with ZeRO-3; training arguments unchanged.",
                    flush=True,
                )
                continue
            return code if code > 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
