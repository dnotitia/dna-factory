"""
Decide, before any heavy import, whether to import unsloth.

unsloth monkeypatches trl/transformers/peft at import time (it rebinds trl.SFTTrainer and friends),
and the Dnotitia trainers bind their TRL base class when they are defined. So `import unsloth` has
to run before trl, transformers, peft and every dna_factory.dnotitia_* module. The use_unsloth switch
lives in YAML/CLI, which TrlParser reads much later, so this module resolves it from raw argv + YAML.

Module-level imports are limited to stdlib + yaml; never import trl/transformers/peft/torch here.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Matched on config.json `model_type`. Qwen3.5/3.6/3.8 checkpoints all use qwen3_5 / qwen3_5_moe.
# Extend only after validating a type end to end.
UNSLOTH_MODEL_TYPES = frozenset({"qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"})

# Must match transformers.hf_argparser.string_to_bool exactly, so the bootstrap and the real parser
# agree on every spelling (see resolve_use_unsloth, which raises when they disagree).
_TRUE_VALUES = {"yes", "true", "t", "y", "1"}
_FALSE_VALUES = {"no", "false", "f", "n", "0"}


@dataclass(frozen=True)
class UnslothDecision:
    """Outcome of the bootstrap: whether unsloth was requested and whether it was imported."""

    requested: bool  # use_unsloth resolved to true
    active: bool  # unsloth was imported and must be used
    model_type: str | None
    reason: str  # human-readable, for logs


# None means the bootstrap never ran (e.g. main() called programmatically).
_DECISION: UnslothDecision | None = None


def get_unsloth_decision() -> UnslothDecision | None:
    """Return the decision made by maybe_import_unsloth, or None if it never ran."""
    return _DECISION


def _parse_bool(value, source: str) -> bool:
    """Parse a bool from a YAML bool or a yes/true/t/y/1 or no/false/f/n/0 string (as HfArgumentParser)."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return True
    if text in _FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean for use_unsloth from {source}: {value!r} "
        f"(expected one of {sorted(_TRUE_VALUES | _FALSE_VALUES)})"
    )


def _load_yaml_dict(path: str | None) -> dict:
    """Load a YAML file as a dict; a missing path or file yields {} (mirrors merge_config_files)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _find_cli_value(cli_args: list[str], key: str) -> tuple[bool, str | None]:
    """
    Find `--key value`, `--key=value` or the dashed alias in cli_args. The last occurrence wins.
    Returns (found, value); value is None for a bare flag (no value, or next token starts with `--`).
    """
    names = {f"--{key}", f"--{key.replace('_', '-')}"}
    found, value = False, None
    for i, arg in enumerate(cli_args):
        name, eq, inline = arg.partition("=")
        if name not in names:
            continue
        found = True
        if eq:
            value = inline
        elif i + 1 < len(cli_args) and not cli_args[i + 1].startswith("--"):
            value = cli_args[i + 1]
        else:
            value = None
    return found, value


def resolve_bootstrap_args(cli_args: list[str], defaults_path: str) -> dict:
    """
    Resolve use_unsloth / model_name_or_path / model_revision from raw argv + YAML.

    Precedence mirrors the entry scripts: defaults YAML < user YAML (--config PATH) < CLI flags.
    """
    config = _load_yaml_dict(defaults_path)
    _, user_config_path = _find_cli_value(cli_args, "config")
    config.update(_load_yaml_dict(user_config_path))

    use_unsloth = False
    if config.get("use_unsloth") is not None:
        use_unsloth = _parse_bool(config["use_unsloth"], "config YAML")
    found, value = _find_cli_value(cli_args, "use_unsloth")
    if found:
        use_unsloth = True if value is None else _parse_bool(value, "CLI")

    resolved = {
        "model_name_or_path": config.get("model_name_or_path"),
        "model_revision": config.get("model_revision"),
    }
    for key in resolved:
        found, value = _find_cli_value(cli_args, key)
        if found and value is not None:
            resolved[key] = value

    return {"use_unsloth": use_unsloth, **resolved}


def _read_model_type(
    model_name_or_path: str, revision: str | None
) -> tuple[str | None, str | None]:
    """
    Return (model_type, error).

    error is a message only when config.json could not be read or parsed. A readable config.json
    without `model_type` yields (None, None): the read worked, the model is just not identifiable.
    """
    try:
        if os.path.isdir(model_name_or_path):
            config_path = Path(model_name_or_path) / "config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_name_or_path, filename="config.json", revision=revision
            )
        with open(config_path, "r") as f:
            model_type = json.load(f).get("model_type")
    except Exception as e:  # noqa: BLE001 - any failure just means the model type is unknown
        return None, f"{type(e).__name__}: {e}"
    return model_type, None


def read_model_type(model_name_or_path: str, revision: str | None = None) -> str | None:
    """Read `model_type` from a local dir's or Hub repo's config.json; None on any failure."""
    return _read_model_type(model_name_or_path, revision)[0]


def _import_unsloth() -> None:
    """Import unsloth (separate function so tests can monkeypatch it)."""
    try:
        import unsloth  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "use_unsloth=true needs the unsloth extra: `uv sync --extra unsloth`"
        ) from e


def _noexec_inductor_cache_dir() -> str | None:
    """Return torch inductor's cache dir if it sits on a noexec mount (else None)."""
    # Inductor caches under $TORCHINDUCTOR_CACHE_DIR, else <tempdir>/torchinductor_<user>
    path = os.path.abspath(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR") or tempfile.gettempdir()
    )
    probe = path
    while not os.path.exists(
        probe
    ):  # the cache dir may not exist yet; check its mount instead
        probe = os.path.dirname(probe)
    try:
        noexec = bool(os.statvfs(probe).f_flag & os.ST_NOEXEC)
    except OSError, AttributeError:  # ST_NOEXEC is POSIX-only
        return None
    return path if noexec else None


def _is_distributed_launch() -> bool:
    """True under multi-process accelerate/torchrun (WORLD_SIZE > 1) or DeepSpeed."""
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world_size = 1
    deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED", "").strip().lower() in {
        "true",
        "1",
    }
    return world_size > 1 or deepspeed


def maybe_import_unsloth(cli_args: list[str], defaults_path: str) -> UnslothDecision:
    """
    Import unsloth iff use_unsloth is set and the model's model_type is supported.

    Must run before trl/transformers/peft are imported. If the model is unsupported unsloth is
    never imported, so the trainers stay unpatched and the run uses the standard TRL/HF path.
    Raises RuntimeError (without importing unsloth) if the model's config.json cannot be read or
    model_name_or_path is unset, since the support check is then impossible.
    """
    global _DECISION
    args = resolve_bootstrap_args(cli_args, defaults_path)

    if not args["use_unsloth"]:
        _DECISION = UnslothDecision(False, False, None, "use_unsloth is false")
        return _DECISION

    if _is_distributed_launch():
        raise RuntimeError(
            "use_unsloth=true supports single-process, single-GPU runs only for now. "
            "Launch with `python <script>` instead of multi-process accelerate / DeepSpeed."
        )

    model = args["model_name_or_path"]
    if model:
        model_type, error = _read_model_type(model, args["model_revision"])
    else:
        model_type, error = None, "model_name_or_path is not set"
    if error is not None:
        # Falling back here would silently run something other than what use_unsloth asked for
        raise RuntimeError(
            f"use_unsloth=true needs the model's model_type, but it could not be determined for "
            f"model {model!r}: {error}. Fix the model path / network access, or set "
            "use_unsloth=false to use the standard TRL/HF path."
        )

    if model_type not in UNSLOTH_MODEL_TYPES:
        reason = (
            f"unsloth skipped for model {model!r}: model_type={model_type!r} is not in the supported "
            f"set {sorted(UNSLOTH_MODEL_TYPES)}; falling back to the standard TRL/HF path"
        )
        logger.warning(reason)
        _DECISION = UnslothDecision(True, False, model_type, reason)
        return _DECISION

    _import_unsloth()

    # On a noexec cache dir the compiled .so files can't be loaded, and unsloth silently runs the
    # affected modules eagerly (slower, more memory) or fails — easy to miss in the logs, so say it.
    noexec_dir = _noexec_inductor_cache_dir()
    if noexec_dir is not None:
        logger.warning(
            f"torch.compile caches kernels under {noexec_dir}, which is on a noexec filesystem, so "
            "unsloth's compiled modules will fall back to eager (slower, more memory) or fail to load. "
            "Set TORCHINDUCTOR_CACHE_DIR to a directory on an exec-capable filesystem."
        )

    _DECISION = UnslothDecision(
        True,
        True,
        model_type,
        f"unsloth imported for model {model!r} (model_type={model_type!r})",
    )
    return _DECISION
