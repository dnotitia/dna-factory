"""
unsloth model-loading integration shared by sft.py / dpo.py / grpo.py / distill.py.

`use_unsloth` is a backend switch: it changes how the model is loaded (unsloth's FastModel, fused
kernels, optionally unsloth-managed vLLM) and must NOT change training hyperparameters. The
bootstrap (unsloth_bootstrap.py) decides before any heavy import whether unsloth is imported at all;
this module turns that decision into concrete FastModel / trainer arguments.

Module-level imports are stdlib + logging only. torch/peft/trl/transformers/unsloth are imported
lazily inside functions so the pure helpers can be imported (and tested) without a GPU.
"""

import logging

from dna_factory.unsloth_bootstrap import get_unsloth_decision

_module_logger = logging.getLogger(__name__)

# model_type values for which unsloth-managed vLLM (`fast_inference=True`) is allowed, each validated
# with GRPO on this stack (vLLM 0.30.0). unsloth's docs still say Qwen3.5-architecture models need
# fast_inference=False for RL; that predates vLLM's Qwen3.5 support and does not hold here. MoE types
# are left out: unsloth adapts the expert weights whenever the MLP projections are targeted, and its engine
# refuses to serve LoRA on them (vLLM's FusedMoEWithLoRA, vllm-project/vllm#41754), failing at the first rollout.
FAST_INFERENCE_MODEL_TYPES = frozenset({"qwen3", "qwen3_5"})

# Qwen3.5-architecture model types: composite (vision + text) configs.
_QWEN3_5_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe"})

# MoE model types: unsloth adapts targeted expert parameters through PEFT's ParamWrapper for LoRA.
_MOE_MODEL_TYPES = frozenset({"qwen3_moe", "qwen3_5_moe"})

# unsloth silently downgrades its "unsloth" gradient checkpointing to the standard one below this.
_MIN_MAX_SEQ_LENGTH = 512


def resolve_use_unsloth(dnotitia_args) -> bool:
    """
    Return whether unsloth is active for this run, based on the bootstrap decision.

    Raises if use_unsloth is set but the bootstrap never ran: unsloth has to be imported before
    trl/transformers/peft, so a programmatic `main()` call cannot enable it after the fact.
    An unsupported model yields an inactive decision (the bootstrap already warned) -> vanilla path.
    Also raises if the pre-import bootstrap and the argument parser read different use_unsloth values
    (e.g. an abbreviated `--use_unsl true`, which argparse accepts but the bootstrap cannot see), since
    that would silently run the other path.
    """
    decision = get_unsloth_decision()
    if decision is None:
        if dnotitia_args.use_unsloth:
            raise RuntimeError(
                "use_unsloth=true requires the unsloth bootstrap, which runs before trl is imported. "
                "Run the script as `python <script>.py ...` rather than calling main() directly."
            )
        return False
    if decision.requested != bool(dnotitia_args.use_unsloth):
        raise ValueError(
            f"use_unsloth disagreement: the pre-import bootstrap read use_unsloth={decision.requested} "
            f"from the command line/YAML, but the argument parser read "
            f"use_unsloth={bool(dnotitia_args.use_unsloth)}. Spell the flag out in full as "
            "`--use_unsloth true|false` (or set it in the YAML); abbreviated option names are not "
            "recognised before trl is imported."
        )
    return decision.active


def revert_unsloth_config_overrides(training_args, full_args) -> dict[str, tuple]:
    """
    Undo the hyperparameter rewrites that unsloth's patched config classes apply at construction.

    After `import unsloth`, `trl.SFTConfig` etc. are `Unsloth*Config` subclasses that rewrite some
    values in `__post_init__` (e.g. GRPO `loss_type: dapo` -> `epsilon_high=0.28`, SFT/DPO
    `dataset_num_proc` None -> 8). Since use_unsloth must not change hyperparameters, re-parse the same
    arguments with TRL's original config class and copy back every differing field, except
    `gradient_checkpointing_kwargs` (unsloth's gradient checkpointing needs `use_reentrant=True`).

    Returns {field: (unsloth_value, restored_value)} for the fields that were restored.
    """
    import dataclasses

    from trl import TrlParser

    # unsloth's patched class keeps the TRL name *and* module (`trl.trainer.grpo_config`, for pickling),
    # so the original is the nearest ancestor with the same name in a TRL module. An unpatched config has
    # no such ancestor (its parents are TRL base classes with other names) and is left alone.
    cls = type(training_args)
    original_cls = next(
        (
            c
            for c in cls.__mro__[1:]
            if c.__name__ == cls.__name__ and c.__module__.startswith("trl.")
        ),
        None,
    )
    if original_cls is None:
        return {}

    # Only the original config class is given to the parser; keys belonging to the other dataclasses
    # (model/script/dnotitia args) are returned as remaining strings instead of raising.
    pristine = TrlParser((original_cls,)).parse_args_and_config(
        list(full_args), return_remaining_strings=True
    )[0]

    restored = {}
    for f in dataclasses.fields(original_cls):
        if f.name == "gradient_checkpointing_kwargs" or not hasattr(
            training_args, f.name
        ):
            continue
        current, original = getattr(training_args, f.name), getattr(pristine, f.name)
        try:
            differs = bool(current != original)
        except Exception:  # noqa: BLE001, S112 - uncomparable values (e.g. tensors) are not rewritten by unsloth
            continue
        if differs:
            setattr(training_args, f.name, original)
            restored[f.name] = (current, original)
    return restored


def prepare_training_args_for_unsloth(training_args, logger=None, *, model_type=None):
    """
    Adjust training args that conflict with unsloth's own patching.

    `use_liger_kernel` is not handled by unsloth and would double-patch the model, so it is turned off.
    """
    logger = logger or _module_logger
    if getattr(training_args, "use_liger_kernel", False):
        logger.warning(
            "use_liger_kernel=True is incompatible with unsloth (it ships its own fused kernels); "
            "disabling Liger for this run."
        )
        training_args.use_liger_kernel = False
    # unsloth's SFTTrainer ignores packing for hybrid linear-attention models with a one-line notice that
    # is easy to miss; say what it changes (unpacked rows: more steps per epoch than the vanilla path).
    if getattr(training_args, "packing", False) and model_type in _QWEN3_5_MODEL_TYPES:
        logger.warning(
            f"unsloth does not pack {model_type} (hybrid linear-attention) models and trains on unpacked "
            "rows instead, so steps per epoch differ from use_unsloth=false."
        )


def torch_dtype_from(model_args_dtype):
    """Map the `dtype` string of ModelConfig ("auto"/None/"bfloat16"/...) to what unsloth expects."""
    if model_args_dtype is None or model_args_dtype == "auto":
        return None
    import torch

    if isinstance(model_args_dtype, torch.dtype):
        return model_args_dtype
    return getattr(torch, model_args_dtype)


def build_from_pretrained_kwargs(
    model_args, training_args, *, max_seq_length, fast_inference=False
) -> dict:
    """
    Build the kwargs for `FastModel.from_pretrained` (pure: unsloth is not imported).

    unsloth defaults to `load_in_4bit=True` and raises if two precision flags are set, so all of
    `load_in_4bit/8bit/16bit` and `full_finetuning` are always passed explicitly.
    """
    from dna_factory.training_runner import resolve_trust_remote_code

    if not model_args.use_peft:
        # Full fine-tuning (quantization without PEFT is rejected by validate_unsloth_args)
        precision = {
            "full_finetuning": True,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "load_in_16bit": True,
        }
    elif model_args.load_in_4bit:
        precision = {
            "full_finetuning": False,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "load_in_16bit": False,
        }
    elif model_args.load_in_8bit:
        precision = {
            "full_finetuning": False,
            "load_in_4bit": False,
            "load_in_8bit": True,
            "load_in_16bit": False,
        }
    else:
        precision = {
            "full_finetuning": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "load_in_16bit": True,
        }

    kwargs = dict(
        model_name=model_args.model_name_or_path,
        # Without this unsloth may silently swap e.g. Qwen/Qwen3-30B-A3B for its unsloth/ mirror repo
        use_exact_model_name=True,
        revision=model_args.model_revision,
        dtype=torch_dtype_from(model_args.dtype),
        trust_remote_code=resolve_trust_remote_code(training_args),
        attn_implementation=model_args.attn_implementation,
        use_gradient_checkpointing="unsloth"
        if training_args.gradient_checkpointing
        else False,
        random_state=training_args.seed,
        max_seq_length=max(max_seq_length, _MIN_MAX_SEQ_LENGTH),
        **precision,
    )
    if fast_inference:
        kwargs.update(
            fast_inference=True,
            gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,
            max_lora_rank=model_args.lora_r,
        )
    return kwargs


def build_peft_kwargs(
    model_args, training_args, model_type, *, fast_inference=False
) -> dict | None:
    """
    Build the kwargs for `FastModel.get_peft_model`; None when LoRA is not requested.

    Keys whose value is None are omitted so unsloth's own defaults apply (e.g. all linear layers).
    """
    if not model_args.use_peft:
        return None

    target_modules = model_args.lora_target_modules
    if target_modules is None:
        # Match the vanilla path, where PEFT picks per-architecture default targets (q_proj/v_proj for
        # qwen3). Types PEFT has no default for (qwen3_moe, qwen3_5*) need explicit targets on the vanilla
        # path; here unsloth's default (all linear layers) applies instead.
        from peft.utils.constants import (
            TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
        )

        default_targets = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.get(
            model_type
        )
        target_modules = list(default_targets) if default_targets is not None else None

    kwargs = {
        "r": model_args.lora_r,
        "lora_alpha": model_args.lora_alpha,
        "lora_dropout": model_args.lora_dropout,
        "target_modules": target_modules,
        "modules_to_save": model_args.lora_modules_to_save,
        "use_rslora": model_args.use_rslora,
        "target_parameters": model_args.lora_target_parameters,
        "random_state": training_args.seed,
        "use_gradient_checkpointing": "unsloth"
        if training_args.gradient_checkpointing
        else False,
    }
    # unsloth's default target (all linear layers) would include the Qwen3.5 vision tower; training is
    # text-only and vLLM LoRA serving does not take vision-tower adapters. unsloth's fast_inference also
    # refuses finetune_vision_layers=True (its default) for every model type.
    if fast_inference or (
        model_type in _QWEN3_5_MODEL_TYPES and target_modules is None
    ):
        kwargs["finetune_vision_layers"] = False
    return {k: v for k, v in kwargs.items() if v is not None}


def _peft_doubles_fused_gate_up(model_type) -> bool:
    """
    Whether PEFT's v4->v5 MoE config conversion gives the fused gate_up_proj rank/alpha x2 for this type.

    PEFT maps the model type to a base pattern (`_MODEL_TO_CONVERSION_PATTERN`) and doubles r/alpha when all
    of that pattern's `_MOE_FUSED_TARGETS` (gate_proj + up_proj) are targeted. With PEFT 0.21 this is
    qwen3_moe (-> qwen2_moe) but not qwen3_5_moe. Private names, imported lazily; hard-coded fallback.
    """
    try:
        from peft.utils.transformers_weight_conversion import (
            _MODEL_TO_CONVERSION_PATTERN,
            _MOE_FUSED_TARGETS,
        )

        base = _MODEL_TO_CONVERSION_PATTERN.get(model_type)
        return len(_MOE_FUSED_TARGETS.get(base, {}).get("gate_up_proj", ())) > 1
    except (
        ImportError,
        AttributeError,
    ):  # peft too old / no transformers v5 / names moved
        return model_type == "qwen3_moe"


def _moe_gate_up_rank_differs(model_args, model_type) -> bool:
    """True when PEFT would double the fused expert gate_up rank while unsloth adapts it at plain r."""
    if model_type not in _MOE_MODEL_TYPES or not _peft_doubles_fused_gate_up(
        model_type
    ):
        return False
    # explicit target_parameters: PEFT keeps them as given and unsloth does not auto-add; no difference
    if model_args.lora_target_parameters:
        return False
    targets = model_args.lora_target_modules
    # unset: PEFT has no default for these types (the standard path needs explicit targets), nothing to compare
    if not targets:
        return False
    if isinstance(targets, str):
        # "all-linear" is expanded by PEFT to include gate_proj/up_proj; unsloth adds the experts at plain r
        return targets.lower() == "all-linear" or (
            "gate_proj" in targets and "up_proj" in targets
        )
    return "gate_proj" in targets and "up_proj" in targets


def validate_unsloth_args(
    model_args, training_args, *, model_type, fast_inference, logger=None
):
    """Reject argument combinations the unsloth path cannot honour (ValueError) or warn about them."""
    logger = logger or _module_logger
    if model_args.use_dora:
        raise ValueError(
            "use_dora is not supported with use_unsloth=true (unsloth has no DoRA kernels; it would "
            "silently disable the fast LoRA path). Set use_dora=false or use_unsloth=false."
        )
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot both be true.")
    if (model_args.load_in_4bit or model_args.load_in_8bit) and not model_args.use_peft:
        raise ValueError(
            "4-bit/8-bit loading requires use_peft=true with use_unsloth=true (quantized weights "
            "cannot be fully fine-tuned; unsloth would silently drop the quantization)."
        )
    if fast_inference and model_args.model_revision not in (None, "main"):
        raise ValueError(
            f"model_revision={model_args.model_revision!r} cannot be combined with unsloth's "
            "fast_inference: its vLLM engine loads the default branch and ignores the revision."
        )
    if (
        getattr(training_args, "use_vllm", False)
        and getattr(training_args, "vllm_mode", None) == "server"
    ):
        raise ValueError(
            "vllm_mode='server' is not supported with use_unsloth=true; use vllm_mode='colocate' "
            "(unsloth manages the vLLM engine in-process)."
        )
    # Rejected rather than silently zeroed: use_unsloth must not change hyperparameters
    if (
        model_args.use_peft
        and model_type in _MOE_MODEL_TYPES
        and model_args.lora_dropout
    ):
        raise ValueError(
            f"lora_dropout={model_args.lora_dropout} is not supported for {model_type} with "
            "use_unsloth=true: unsloth adapts the MoE expert parameters through PEFT's ParamWrapper, "
            "which rejects dropout. Set lora_dropout=0.0."
        )
    if model_args.use_peft and _moe_gate_up_rank_differs(model_args, model_type):
        logger.warning(
            f"{model_type}: with gate_proj and up_proj targeted, PEFT (use_unsloth=false) gives the fused "
            "expert gate_up_proj adapter rank 2*lora_r and alpha 2*lora_alpha, whereas unsloth adapts it "
            "at lora_r / lora_alpha (same alpha/r scale, half the rank). For identical adapters on both paths, "
            "target attention only (lora_target_modules: [q_proj, k_proj, v_proj, o_proj]) and set "
            "lora_target_parameters: [mlp.experts.gate_up_proj, mlp.experts.down_proj]."
        )
    if model_args.load_in_4bit and model_type in _QWEN3_5_MODEL_TYPES:
        logger.warning(
            f"load_in_4bit (QLoRA) on {model_type} models is not recommended by unsloth; "
            "prefer bf16 LoRA (load_in_4bit=false)."
        )


def resolve_fast_inference(training_args, model_type, logger=None, *, use_peft) -> bool:
    """
    Decide whether unsloth should own the vLLM engine (GRPO/Distill).

    Returns False when vLLM is off. When unsloth's engine can't be used — full fine-tuning (unsloth's
    fast_inference is LoRA-only and raises with full_finetuning=True) or a model type outside
    FAST_INFERENCE_MODEL_TYPES — vLLM is turned off (generation falls back to slower HF generate) rather
    than letting TRL build a second vLLM engine around an unsloth model.
    """
    logger = logger or _module_logger
    if not training_args.use_vllm:
        return False
    if not use_peft:
        reason = "full fine-tuning (unsloth's fast_inference requires use_peft=true)"
    elif model_type not in FAST_INFERENCE_MODEL_TYPES:
        reason = f"model_type={model_type!r}"
    else:
        return True
    logger.warning(
        f"unsloth fast_inference is not available for {reason}; setting use_vllm=False, so generation "
        "falls back to (slower) HF generate."
    )
    training_args.use_vllm = False
    return False


def _load_text_config(model_args, training_args):
    """The model's text config (composite configs such as Qwen3.5 keep it under `text_config`)."""
    from transformers import AutoConfig

    from dna_factory.training_runner import resolve_trust_remote_code

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=resolve_trust_remote_code(training_args),
    )
    return getattr(config, "text_config", None) or config


def effective_router_aux_loss_coef(model_args, training_args) -> float:
    """
    The MoE router auxiliary-loss coefficient GRPO will actually use.

    TRL >= 1.15 defaults `router_aux_loss_coef` to None, meaning "take it from the model config"
    (0.001 for Qwen3 MoE), so a None in the training args is not a zero.
    """
    coef = training_args.router_aux_loss_coef
    if coef is None:
        coef = getattr(
            _load_text_config(model_args, training_args), "router_aux_loss_coef", 0.0
        )
    return float(coef or 0.0)


def generation_max_seq_length(model_args, training_args) -> int:
    """
    Context length to give unsloth/vLLM for GRPO/Distill (the engine is sized at load time).

    Preference: `vllm_max_model_length`, then the model's `max_position_embeddings`, then
    `max_completion_length + 4096` as a conservative fallback.
    """
    if getattr(training_args, "vllm_max_model_length", None):
        return training_args.vllm_max_model_length

    text_config = _load_text_config(model_args, training_args)
    max_positions = getattr(text_config, "max_position_embeddings", None)
    if max_positions:
        return int(max_positions)
    return int(training_args.max_completion_length) + 4096


def load_unsloth_model(
    model_args,
    training_args,
    *,
    model_type,
    max_seq_length,
    fast_inference=False,
    logger=None,
):
    """
    Load the model with unsloth's FastModel and apply LoRA via `FastModel.get_peft_model`.

    unsloth's returned tokenizer/processor is discarded; the scripts keep their own AutoTokenizer.
    """
    logger = logger or _module_logger
    from unsloth import FastModel

    validate_unsloth_args(
        model_args,
        training_args,
        model_type=model_type,
        fast_inference=fast_inference,
        logger=logger,
    )
    kwargs = build_from_pretrained_kwargs(
        model_args,
        training_args,
        max_seq_length=max_seq_length,
        fast_inference=fast_inference,
    )
    logger.info(f"FastModel.from_pretrained kwargs: {kwargs}")
    model, _ = FastModel.from_pretrained(**kwargs)

    peft_kwargs = build_peft_kwargs(
        model_args, training_args, model_type, fast_inference=fast_inference
    )
    if peft_kwargs is not None:
        logger.info(f"FastModel.get_peft_model kwargs: {peft_kwargs}")
        model = FastModel.get_peft_model(model, **peft_kwargs)
        from peft import PeftModel

        # peft_config is ignored by unsloth's patched trainers, so the model must already be a PeftModel
        assert isinstance(model, PeftModel), (
            f"expected a PeftModel from unsloth, got {type(model)}"
        )
    else:
        # unsloth's full-finetuning load keeps some weights (e.g. RMSNorm) in fp32 and disables autocast, so
        # trainers that call the backbone directly (distillation) hit fp32 vs bf16 matmuls. Match the vanilla
        # path, where the whole model is in the configured dtype.
        dtype = torch_dtype_from(model_args.dtype)
        if dtype is not None:
            model.to(dtype)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"unsloth model ready: trainable params {trainable:,} / {total:,}")
    return model


def tokenize_conversational_dataset(
    dataset, processing_class, args, chat_template, dataset_name
):
    """
    Tokenize a not-yet-processed conversational language-modeling dataset the way TRL does.

    Mirrors the language-modeling conversational case of trl 1.15 `SFTTrainer._prepare_dataset`
    (ChatML conversion + `_tokenize` with the training chat template). It exists because unsloth's
    generated `UnslothSFTTrainer._prepare_dataset` replaces TRL's and has no `messages` handling (it
    raises "You must specify a formatting_func"), while it does accept datasets that already carry
    `input_ids` (+ `assistant_masks`) and folds the masks into `labels` itself.

    Only datasets without `input_ids` and `prompt` whose examples are conversational (`messages`, or
    ShareGPT-style `conversations`) are tokenized; every other shape (text field, prompt/completion,
    pre-tokenized) is returned untouched for unsloth's own handling. Original columns are kept, as in
    TRL; the trainer's column pruning drops them. Raises like TRL when `assistant_only_loss` is set
    and an example has no assistant tokens. Truncation is a separate step, `truncate_like_trl`.
    """
    import json

    from accelerate import PartialState
    from datasets import Dataset
    from trl.data_utils import (
        _tokenize,
        get_dataset_column_names,
        is_conversational,
        is_conversational_from_value,
        maybe_convert_to_chatml,
    )

    column_names = get_dataset_column_names(dataset)
    if "input_ids" in column_names or "prompt" in column_names:
        return dataset

    is_dataset = isinstance(dataset, Dataset)  # IterableDataset: no num_proc / desc
    map_kwargs = {"num_proc": args.dataset_num_proc} if is_dataset else {}

    with PartialState().main_process_first():
        first_example = next(iter(dataset))
        if is_conversational_from_value(first_example):
            if is_dataset:
                map_kwargs["desc"] = f"Converting {dataset_name} dataset to ChatML"
            dataset = dataset.map(
                maybe_convert_to_chatml,
                remove_columns="conversations"
                if "conversations" in column_names
                else None,
                **map_kwargs,
            )
            first_example = next(iter(dataset))

        if not is_conversational(first_example):
            return dataset

        if is_dataset:
            map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

        def tokenize_fn(example, processing_class, assistant_only_loss, chat_template):
            tools = example.get("tools")
            tools = json.loads(tools) if isinstance(tools, str) else tools
            processed = _tokenize(
                processing_class,
                example["messages"],
                return_assistant_tokens_mask=assistant_only_loss,
                chat_template=chat_template,
                tools=tools,
                **example.get("chat_template_kwargs", {}),
            )
            output = {
                k: processed[k]
                for k in ("input_ids", "assistant_masks")
                if k in processed
            }
            if "assistant_masks" in output and 1 not in output["assistant_masks"]:
                raise RuntimeError(
                    "You're using `assistant_only_loss=True`, but at least one example has no assistant "
                    "tokens. This usually means the tokenizer's chat template doesn't generate assistant "
                    "masks — it may be missing the `{% generation %}` keyword. Please check the template and "
                    "ensure it's correctly configured to support assistant masking."
                )
            return output

        return dataset.map(
            tokenize_fn,
            fn_kwargs={
                "processing_class": processing_class,
                "assistant_only_loss": args.assistant_only_loss,
                "chat_template": chat_template,
            },
            **map_kwargs,
        )


def truncate_like_trl(dataset, args, dataset_name):
    """
    Truncate a pre-tokenized dataset to `args.max_length` exactly as TRL's `SFTTrainer._prepare_dataset`.

    unsloth's `_prepare_dataset` never truncates already-tokenized input. TRL slices `input_ids` and
    the labels per `args.truncation_mode` (`keep_start` / `keep_end`, else ValueError), then drops rows left
    with no trained token. TRL builds labels from the mask columns; here the same slice is applied to
    `input_ids` and every mask column present (`assistant_masks`, `completion_mask`; the column set is kept
    for unsloth), and a token counts as trained when all present masks are 1 there (no masks: all trained).
    No-op when `max_length` is None.
    """
    if args.max_length is None:
        return dataset

    from datasets import Dataset
    from trl.data_utils import get_dataset_column_names

    if args.truncation_mode == "keep_start":
        sl = slice(None, args.max_length)
    elif args.truncation_mode == "keep_end":
        sl = slice(-args.max_length, None)
    else:
        raise ValueError(
            f"Unsupported truncation mode: {args.truncation_mode}, expected 'keep_start' or 'keep_end'"
        )

    is_dataset = isinstance(dataset, Dataset)  # IterableDataset: no num_proc / desc
    map_kwargs = {"num_proc": args.dataset_num_proc} if is_dataset else {}
    present = get_dataset_column_names(dataset)
    columns = [
        c for c in ("input_ids", "assistant_masks", "completion_mask") if c in present
    ]
    mask_columns = [c for c in columns if c != "input_ids"]

    def truncate(example, sl, columns):
        return {c: example[c][sl] for c in columns}

    if is_dataset:
        map_kwargs["desc"] = f"Truncating {dataset_name} dataset"
    dataset = dataset.map(
        truncate, fn_kwargs={"sl": sl, "columns": columns}, **map_kwargs
    )

    def has_trained_token(example, mask_columns):
        if not mask_columns:
            return len(example["input_ids"]) > 0
        return any(all(bits) for bits in zip(*(example[c] for c in mask_columns)))

    if is_dataset:
        map_kwargs["desc"] = (
            f"Dropping fully masked examples from {dataset_name} dataset"
        )
    return dataset.filter(
        has_trained_token, fn_kwargs={"mask_columns": mask_columns}, **map_kwargs
    )
