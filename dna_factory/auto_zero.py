"""Single-node ZeRO launch policy and cross-process startup failure signals."""

import os
from pathlib import Path

STATUS_ENV = "DNA_FACTORY_AUTO_ZERO_STATUS"
DEFAULTS = {
    "sft.py": "SFT",
    "sampling_sft.py": "SamplingSFT",
    "sdft.py": "SDFT",
    "dpo.py": "DPO",
    "grpo.py": "GRPO",
    "opd.py": "OPD",
}


def signal_status(kind):
    directory = os.environ.get(STATUS_ENV)
    if directory:
        Path(directory, f"{kind}-{os.getpid()}").touch()


def is_cuda_oom(error):
    # CPU allocation errors, NCCL errors and generic failures must not trigger retries.
    text = str(error).lower()
    return "cuda out of memory" in text or "cuda error: out of memory" in text


def can_retry(directory, returncode):
    directory = Path(directory)
    return (
        returncode != 0
        and any(directory.glob("oom-*"))
        and not any(directory.glob("progress-*"))
    )


def config_candidates(zero, model_type):
    moe = {"qwen3_moe": "zero3-qwen3-moe.yaml", "qwen3_5_moe": "zero3-qwen3_5-moe.yaml"}
    if model_type in moe:
        if zero not in ("auto", "3"):
            raise ValueError("This MoE model requires its dedicated ZeRO-3 config")
        return [moe[model_type]]
    if "moe" in model_type:
        raise ValueError(
            "Unknown MoE architecture: supply a validated accelerate config manually"
        )
    if zero == "auto":
        return ["zero1.yaml", "zero3.yaml"]
    return [f"zero{zero}.yaml"]
