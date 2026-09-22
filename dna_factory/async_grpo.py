"""Configuration and support boundary for the single-GPU AsyncGRPO draft."""

import os
import time
from dataclasses import dataclass, field

import requests
from trl.experimental.async_grpo import AsyncGRPOConfig


@dataclass
class DnotitiaAsyncGRPOConfig(AsyncGRPOConfig):
    # ModelConfig owns the CLI dtype flag. Copy its value into this config in setup.
    dtype: str = field(default="float32", init=False)


def apply_attn_implementation_override(attn_implementation):
    """Make AsyncGRPOTrainer build its model with the repo-configured attn_implementation.

    TRL 1.13.0 hardcodes flash-attn3, which has no sm100 build; delete this once TRL reads the
    value from its own config. Wraps `create_model_from_path` in the trainer's module namespace,
    which is the symbol the hardcoded call site resolves through. See docs/async-grpo.md.
    """
    if attn_implementation == "kernels-community/flash-attn3":
        return

    from trl.experimental.async_grpo import async_grpo_trainer as trainer_module

    current = trainer_module.create_model_from_path
    if getattr(current, "__dna_attn_implementation__", None) == attn_implementation:
        return
    original = getattr(current, "__dna_original__", current)

    def create_model_with_attn_override(*args, **kwargs):
        kwargs["attn_implementation"] = attn_implementation
        return original(*args, **kwargs)

    create_model_with_attn_override.__dna_original__ = original
    create_model_with_attn_override.__dna_attn_implementation__ = attn_implementation
    trainer_module.create_model_from_path = create_model_with_attn_override


# Covers every branch TRL's Qwen3 parser handles; rendered for comparison, never trained on.
_RESPONSE_TEMPLATE_PROBE_MESSAGES = [
    {"role": "user", "content": "probe"},
    {
        "role": "assistant",
        "reasoning_content": "probe reasoning",
        "content": "probe content",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "probe_tool", "arguments": {"x": 1}},
            }
        ],
    },
]


def _renders_like(tokenizer, candidate_chat_template) -> bool:
    """True if the probe renders identically under the tokenizer's own template and the candidate.

    Restores the original template afterwards, and returns False rather than raising, so a
    malformed template fails the comparison instead of crashing training setup.
    """
    original_chat_template = tokenizer.chat_template
    try:
        actual_render = tokenizer.apply_chat_template(
            _RESPONSE_TEMPLATE_PROBE_MESSAGES, tokenize=False
        )
        tokenizer.chat_template = candidate_chat_template
        candidate_render = tokenizer.apply_chat_template(
            _RESPONSE_TEMPLATE_PROBE_MESSAGES, tokenize=False
        )
    except Exception:  # noqa: BLE001 - best-effort render comparison, never fatal
        return False
    finally:
        tokenizer.chat_template = original_chat_template
    return actual_render == candidate_render


def apply_response_template_override(processing_class):
    """Set a response template on tokenizers whose chat template TRL matches by exact string only.

    TRL's rollout worker raises on `dnotitia/Qwen3-0.6B`, whose template differs from the bundled
    `qwen3.jinja` in jinja source but not in rendered text. Setting the template here, in the
    parent, makes the child's own lookup a no-op. Only applied when `_renders_like` confirms the
    equivalence; delete once TRL matches templates structurally. See docs/async-grpo.md.
    """
    from transformers import ProcessorMixin
    from trl.chat_template_utils import (
        _SUPPORTS_RESPONSE_TEMPLATE,
        qwen3_chat_template,
        qwen3_schema,
        qwen3_template,
    )

    tokenizer = (
        processing_class.tokenizer
        if isinstance(processing_class, ProcessorMixin)
        else processing_class
    )

    if getattr(tokenizer, "response_template", None) is not None:
        return
    if getattr(tokenizer, "response_schema", None) is not None:
        return

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template or not _renders_like(tokenizer, qwen3_chat_template):
        return

    if _SUPPORTS_RESPONSE_TEMPLATE:
        tokenizer.response_template = qwen3_template
    else:
        tokenizer.response_schema = qwen3_schema


def validate_async_args(script_args, training_args, model_args, dnotitia_args):
    """Reject unsupported combinations before loading models or datasets."""
    if (
        training_args.deepspeed
        or training_args.fsdp
        or os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() == "true"
        or os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or training_args.n_gpu > 1
    ):
        raise ValueError(
            "AsyncGRPO currently supports one training GPU only, without DeepSpeed or FSDP. "
            "Run python grpo.py on one GPU and vllm serve on separate GPUs."
        )
    if training_args.eval_strategy != "no" or training_args.do_eval:
        raise ValueError(
            "AsyncGRPO does not support evaluation; set eval_strategy: 'no' and do_eval: false."
        )
    if dnotitia_args.dynamic_sampling != "off":
        raise ValueError("AsyncGRPO requires dynamic_sampling: 'off'.")
    if dnotitia_args.debug_first_n_batches != 0:
        raise ValueError(
            "AsyncGRPO requires debug_first_n_batches: 0; use log_completions instead."
        )
    if script_args.reward_model_name_or_path:
        raise ValueError(
            "AsyncGRPO supports callable rewards only; serve GPU reward models through an HTTP judge."
        )
    if model_args.use_peft or model_args.load_in_4bit or model_args.load_in_8bit:
        raise ValueError(
            "This AsyncGRPO draft supports full finetuning only, without PEFT or quantization."
        )
    if model_args.attn_implementation in ("sdpa", "eager"):
        raise ValueError(
            "AsyncGRPO trains in padding-free mode (sequences concatenated into one row, "
            "cu_seq_lens derived from position_ids resets); sdpa/eager can't handle that layout "
            "(see the comment at the from_pretrained call in TRL's AsyncGRPOTrainer.__init__). "
            "Use attn_implementation: kernels-community/flash-attn2 (this repo's default) or "
            "kernels-community/flash-attn3."
        )
    if model_args.attn_implementation not in (
        None,
        "kernels-community/flash-attn2",
        "kernels-community/flash-attn3",
    ):
        raise ValueError(
            "AsyncGRPO supports attn_implementation: kernels-community/flash-attn2 (default) or "
            "kernels-community/flash-attn3 only."
        )
    if "attn_implementation" in (training_args.model_init_kwargs or {}):
        raise ValueError(
            "Remove attn_implementation from model_init_kwargs; AsyncGRPO supplies it internally."
        )
    if training_args.use_liger_kernel:
        raise ValueError("AsyncGRPO does not support use_liger_kernel: true.")
    if script_args.dataset_streaming:
        raise ValueError(
            "This AsyncGRPO draft supports non-streaming datasets only for checkpoint resume."
        )


# Short on purpose: a server that isn't up yet is expected here, and TRL waits for it properly
# later with its own timeout. Long enough to catch one that is already coming up.
_SERVER_POLL_BACKOFFS_S = (1.0, 2.0, 4.0)


def _poll_server_max_model_len(base_url, train_logger):
    """Best-effort fetch of the vLLM server's `max_model_len`; `None` if it never answers in time.

    Same endpoint and response key as TRL's `VLLMClient.get_max_model_len`, without constructing
    a client this startup check has no business wiring up. Never raises.
    """
    base_url = base_url.rstrip("/")
    for delay in (0.0, *_SERVER_POLL_BACKOFFS_S):
        if delay:
            time.sleep(delay)
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            response.raise_for_status()
            return response.json()["data"][0]["max_model_len"]
        except requests.RequestException, ValueError, KeyError, IndexError:
            continue
    train_logger.warning(
        "AsyncGRPO: could not reach the vLLM server at %s to check max_completion_length "
        "against its max_model_len; skipping this fail-fast check. TRL will wait for the server "
        "on its own later and report its own error if it never comes up.",
        base_url,
    )
    return None


def validate_completion_length_against_server(training_args, train_logger):
    """Fail fast if `max_completion_length` can't fit inside the vLLM server's `max_model_len`.

    vLLM rejects an over-long request with HTTP 400 instead of truncating it, and TRL surfaces
    that only as an opaque child-process death after ~30 retries. Called before the tokenizer,
    dataset and model load. The comparison is `>=`, since a prompt is at least one token.
    """
    max_model_len = _poll_server_max_model_len(
        training_args.vllm_server_base_url, train_logger
    )
    if max_model_len is None:
        return  # server unreachable within the grace period; not this check's error to report

    if training_args.max_completion_length >= max_model_len:
        raise ValueError(
            f"max_completion_length ({training_args.max_completion_length}) must be less than "
            f"the vLLM server's max_model_len ({max_model_len}). vLLM does not truncate an "
            "over-long completion request, it rejects it with HTTP 400, so either lower "
            "max_completion_length or raise the server's --max-model-len."
        )

    token_budget = training_args.token_budget
    if token_budget is not None and token_budget < max_model_len:
        train_logger.warning(
            "AsyncGRPO: token_budget (%s) is smaller than the vLLM server's max_model_len (%s). "
            "TRL's own guidance (AsyncGRPOConfig.token_budget) is to set token_budget at least "
            "as large as max_model_len; any rollout sample longer than token_budget fits in no "
            "micro-batch row and is silently dropped (only a log warning), which can shrink "
            "your effective batch size with no hard failure.",
            token_budget,
            max_model_len,
        )

    headroom = max_model_len - training_args.max_completion_length
    train_logger.info(
        "AsyncGRPO: max_completion_length=%s leaves %s tokens of prompt headroom under the "
        "server's max_model_len=%s.",
        training_args.max_completion_length,
        headroom,
        max_model_len,
    )
