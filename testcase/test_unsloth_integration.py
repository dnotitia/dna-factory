"""
Test cases for dna_factory/unsloth_integration.py

This module tests the argument mapping, validation and config-revert helpers of the unsloth
integration without a GPU, network, or a real `import unsloth`.
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import ClassVar

import pytest
import torch
import trl
import yaml

# Add parent directory to path to import dna_factory
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dna_factory import unsloth_bootstrap as ub
from dna_factory import unsloth_integration as ui


@pytest.fixture
def model_args():
    return trl.ModelConfig(model_name_or_path="dummy/model", dtype="bfloat16")


@pytest.fixture
def sft_args(tmp_path):
    return trl.SFTConfig(
        output_dir=str(tmp_path), gradient_checkpointing=True, seed=7, bf16=False
    )


@pytest.fixture
def grpo_args(tmp_path):
    return trl.GRPOConfig(
        output_dir=str(tmp_path),
        use_vllm=True,
        vllm_gpu_memory_utilization=0.4,
        bf16=False,
    )


class TestTorchDtypeFrom:
    """Test cases for torch_dtype_from"""

    @pytest.mark.parametrize("value", [None, "auto"])
    def test_none_and_auto(self, value):
        assert ui.torch_dtype_from(value) is None

    def test_named_dtypes(self):
        assert ui.torch_dtype_from("bfloat16") is torch.bfloat16
        assert ui.torch_dtype_from("float16") is torch.float16

    def test_torch_dtype_passthrough(self):
        assert ui.torch_dtype_from(torch.float32) is torch.float32


class TestBuildFromPretrainedKwargs:
    """Test cases for build_from_pretrained_kwargs"""

    FLAGS = ("full_finetuning", "load_in_4bit", "load_in_8bit", "load_in_16bit")

    def _flags(self, kwargs):
        return {k: kwargs[k] for k in self.FLAGS}

    def test_bf16_lora(self, model_args, sft_args):
        model_args.use_peft = True
        kwargs = ui.build_from_pretrained_kwargs(
            model_args, sft_args, max_seq_length=2048
        )
        assert self._flags(kwargs) == {
            "full_finetuning": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "load_in_16bit": True,
        }

    def test_exact_model_name(self, model_args, sft_args):
        """unsloth must load the requested repo, not swap in its unsloth/ mirror"""
        kwargs = ui.build_from_pretrained_kwargs(
            model_args, sft_args, max_seq_length=2048
        )
        assert kwargs["use_exact_model_name"] is True

    def test_qlora(self, model_args, sft_args):
        model_args.use_peft = True
        model_args.load_in_4bit = True
        kwargs = ui.build_from_pretrained_kwargs(
            model_args, sft_args, max_seq_length=2048
        )
        assert self._flags(kwargs) == {
            "full_finetuning": False,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "load_in_16bit": False,
        }

    def test_full_finetuning(self, model_args, sft_args):
        model_args.use_peft = False
        kwargs = ui.build_from_pretrained_kwargs(
            model_args, sft_args, max_seq_length=2048
        )
        assert self._flags(kwargs) == {
            "full_finetuning": True,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "load_in_16bit": True,
        }

    def test_common_fields(self, model_args, sft_args):
        model_args.model_revision = "rev1"
        model_args.attn_implementation = "sdpa"
        kwargs = ui.build_from_pretrained_kwargs(
            model_args, sft_args, max_seq_length=2048
        )
        assert kwargs["model_name"] == "dummy/model"
        assert kwargs["revision"] == "rev1"
        assert kwargs["dtype"] is torch.bfloat16
        assert kwargs["attn_implementation"] == "sdpa"
        assert kwargs["use_gradient_checkpointing"] == "unsloth"
        assert kwargs["random_state"] == 7
        assert kwargs["trust_remote_code"] is False
        assert "fast_inference" not in kwargs

    def test_gradient_checkpointing_off(self, model_args, tmp_path):
        args = trl.SFTConfig(
            output_dir=str(tmp_path), gradient_checkpointing=False, bf16=False
        )
        kwargs = ui.build_from_pretrained_kwargs(model_args, args, max_seq_length=2048)
        assert kwargs["use_gradient_checkpointing"] is False

    @pytest.mark.parametrize(
        "given,expected", [(0, 512), (100, 512), (512, 512), (4096, 4096)]
    )
    def test_max_seq_length_floor(self, model_args, sft_args, given, expected):
        kwargs = ui.build_from_pretrained_kwargs(
            model_args, sft_args, max_seq_length=given
        )
        assert kwargs["max_seq_length"] == expected

    def test_fast_inference(self, model_args, grpo_args):
        model_args.use_peft = True
        model_args.lora_r = 32
        kwargs = ui.build_from_pretrained_kwargs(
            model_args, grpo_args, max_seq_length=4096, fast_inference=True
        )
        assert kwargs["fast_inference"] is True
        assert kwargs["gpu_memory_utilization"] == 0.4
        assert kwargs["max_lora_rank"] == 32


class TestBuildPeftKwargs:
    """Test cases for build_peft_kwargs"""

    def test_none_without_peft(self, model_args, sft_args):
        model_args.use_peft = False
        assert ui.build_peft_kwargs(model_args, sft_args, "qwen3") is None

    def test_mapping_and_none_omission(self, model_args, sft_args):
        model_args.use_peft = True
        model_args.lora_r, model_args.lora_alpha, model_args.lora_dropout = 16, 32, 0.1
        model_args.lora_target_modules = ["q_proj", "v_proj"]
        model_args.use_rslora = True
        kwargs = ui.build_peft_kwargs(model_args, sft_args, "qwen3")
        assert kwargs == {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            "use_rslora": True,
            "random_state": 7,
            "use_gradient_checkpointing": "unsloth",
        }
        assert "modules_to_save" not in kwargs and "target_parameters" not in kwargs

    def test_optional_fields_passed_when_set(self, model_args, sft_args):
        model_args.use_peft = True
        model_args.lora_modules_to_save = ["lm_head"]
        model_args.lora_target_parameters = ["mlp.experts.gate_up_proj"]
        kwargs = ui.build_peft_kwargs(model_args, sft_args, "qwen3_moe")
        assert kwargs["modules_to_save"] == ["lm_head"]
        assert kwargs["target_parameters"] == ["mlp.experts.gate_up_proj"]

    @pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
    def test_vision_layers_off_for_qwen3_5_default_targets(
        self, model_args, sft_args, model_type
    ):
        model_args.use_peft = True
        model_args.lora_target_modules = None
        assert (
            ui.build_peft_kwargs(model_args, sft_args, model_type)[
                "finetune_vision_layers"
            ]
            is False
        )

    def test_vision_layers_untouched_with_explicit_targets(self, model_args, sft_args):
        model_args.use_peft = True
        model_args.lora_target_modules = ["q_proj"]
        assert "finetune_vision_layers" not in ui.build_peft_kwargs(
            model_args, sft_args, "qwen3_5"
        )

    @pytest.mark.parametrize("model_type", ["qwen3", "qwen3_moe", "qwen3_5"])
    def test_vision_layers_off_with_fast_inference(
        self, model_args, sft_args, model_type
    ):
        """unsloth's fast_inference refuses its default finetune_vision_layers=True"""
        model_args.use_peft = True
        model_args.lora_target_modules = ["q_proj"]
        kwargs = ui.build_peft_kwargs(
            model_args, sft_args, model_type, fast_inference=True
        )
        assert kwargs["finetune_vision_layers"] is False

    def test_vision_layers_untouched_for_qwen3(self, model_args, sft_args):
        model_args.use_peft = True
        model_args.lora_target_modules = None
        assert "finetune_vision_layers" not in ui.build_peft_kwargs(
            model_args, sft_args, "qwen3"
        )

    def test_default_targets_match_peft_for_qwen3(self, model_args, sft_args):
        """Unset targets follow PEFT's per-architecture default, as on the vanilla path"""
        model_args.use_peft = True
        model_args.lora_target_modules = None
        assert ui.build_peft_kwargs(model_args, sft_args, "qwen3")[
            "target_modules"
        ] == ["q_proj", "v_proj"]

    @pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3_5", "qwen3_5_moe"])
    def test_default_targets_left_to_unsloth_without_peft_default(
        self, model_args, sft_args, model_type
    ):
        model_args.use_peft = True
        model_args.lora_target_modules = None
        assert "target_modules" not in ui.build_peft_kwargs(
            model_args, sft_args, model_type
        )


class TestValidateUnslothArgs:
    """Test cases for validate_unsloth_args"""

    def _validate(
        self, model_args, training_args, model_type="qwen3", fast_inference=False
    ):
        ui.validate_unsloth_args(
            model_args,
            training_args,
            model_type=model_type,
            fast_inference=fast_inference,
        )

    def test_valid_lora_passes(self, model_args, sft_args):
        model_args.use_peft = True
        self._validate(model_args, sft_args)

    def test_dora_rejected(self, model_args, sft_args):
        model_args.use_peft, model_args.use_dora = True, True
        with pytest.raises(ValueError, match="use_dora"):
            self._validate(model_args, sft_args)

    def test_4bit_and_8bit_rejected(self, model_args, sft_args):
        model_args.use_peft, model_args.load_in_4bit, model_args.load_in_8bit = (
            True,
            True,
            True,
        )
        with pytest.raises(ValueError, match="load_in_4bit and load_in_8bit"):
            self._validate(model_args, sft_args)

    @pytest.mark.parametrize("flag", ["load_in_4bit", "load_in_8bit"])
    def test_quantization_without_peft_rejected(self, model_args, sft_args, flag):
        model_args.use_peft = False
        setattr(model_args, flag, True)
        with pytest.raises(ValueError, match="use_peft"):
            self._validate(model_args, sft_args)

    def test_revision_with_fast_inference_rejected(self, model_args, grpo_args):
        model_args.model_revision = "abc123"
        with pytest.raises(ValueError, match="model_revision"):
            self._validate(model_args, grpo_args, fast_inference=True)

    @pytest.mark.parametrize("revision", [None, "main"])
    def test_default_revision_with_fast_inference_ok(
        self, model_args, grpo_args, revision
    ):
        model_args.model_revision = revision
        self._validate(model_args, grpo_args, fast_inference=True)

    def test_revision_without_fast_inference_ok(self, model_args, grpo_args):
        model_args.model_revision = "abc123"
        self._validate(model_args, grpo_args, fast_inference=False)

    @pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3_5_moe"])
    def test_moe_lora_dropout_rejected(self, model_args, sft_args, model_type):
        model_args.use_peft, model_args.lora_dropout = True, 0.05
        with pytest.raises(ValueError, match="lora_dropout"):
            self._validate(model_args, sft_args, model_type=model_type)

    @pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3_5_moe"])
    def test_moe_zero_dropout_ok(self, model_args, sft_args, model_type):
        model_args.use_peft, model_args.lora_dropout = True, 0.0
        self._validate(model_args, sft_args, model_type=model_type)

    @staticmethod
    def _gate_up_warnings(caplog):
        return [r for r in caplog.records if "gate_up_proj" in r.message]

    def _warns(self, model_args, sft_args, caplog, model_type="qwen3_moe", **fields):
        model_args.use_peft = fields.pop("use_peft", True)
        model_args.lora_dropout = 0.0
        for key, value in fields.items():
            setattr(model_args, key, value)
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            self._validate(model_args, sft_args, model_type=model_type)
        return bool(self._gate_up_warnings(caplog))

    ALL_TARGETS: ClassVar[list[str]] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    def test_moe_gate_up_rank_warns(self, model_args, sft_args, caplog):
        assert self._warns(
            model_args, sft_args, caplog, lora_target_modules=self.ALL_TARGETS
        )
        assert "lora_target_parameters" in self._gate_up_warnings(caplog)[0].message

    def test_moe_gate_up_rank_warns_all_linear(self, model_args, sft_args, caplog):
        assert self._warns(
            model_args, sft_args, caplog, lora_target_modules="all-linear"
        )

    def test_moe_attention_only_does_not_warn(self, model_args, sft_args, caplog):
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        assert not self._warns(
            model_args, sft_args, caplog, lora_target_modules=targets
        )

    def test_dense_qwen3_does_not_warn(self, model_args, sft_args, caplog):
        assert not self._warns(
            model_args,
            sft_args,
            caplog,
            model_type="qwen3",
            lora_target_modules=self.ALL_TARGETS,
        )

    def test_moe_without_peft_does_not_warn(self, model_args, sft_args, caplog):
        assert not self._warns(
            model_args,
            sft_args,
            caplog,
            use_peft=False,
            lora_target_modules=self.ALL_TARGETS,
        )

    def test_moe_explicit_target_parameters_does_not_warn(
        self, model_args, sft_args, caplog
    ):
        assert not self._warns(
            model_args,
            sft_args,
            caplog,
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_target_parameters=[
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
            ],
        )

    def test_moe_gate_up_fallback_when_peft_names_missing(
        self, model_args, sft_args, caplog, monkeypatch
    ):
        monkeypatch.setitem(
            sys.modules, "peft.utils.transformers_weight_conversion", None
        )
        assert self._warns(
            model_args, sft_args, caplog, lora_target_modules=self.ALL_TARGETS
        )
        caplog.clear()
        assert not self._warns(
            model_args,
            sft_args,
            caplog,
            model_type="qwen3_5_moe",
            lora_target_modules=self.ALL_TARGETS,
        )

    def test_qwen3_5_moe_does_not_warn(self, model_args, sft_args, caplog):
        # PEFT has no v4->v5 conversion pattern for qwen3_5_moe, so no rank doubling there
        assert not self._warns(
            model_args,
            sft_args,
            caplog,
            model_type="qwen3_5_moe",
            lora_target_modules=self.ALL_TARGETS,
        )

    def test_vllm_server_rejected(self, model_args, grpo_args):
        grpo_args.vllm_mode = "server"
        with pytest.raises(ValueError, match="server"):
            self._validate(model_args, grpo_args)

    def test_vllm_server_ignored_when_vllm_off(self, model_args, grpo_args):
        grpo_args.use_vllm, grpo_args.vllm_mode = False, "server"
        self._validate(model_args, grpo_args)

    def test_qlora_on_qwen3_5_warns(self, model_args, sft_args, caplog):
        model_args.use_peft, model_args.load_in_4bit = True, True
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            self._validate(model_args, sft_args, model_type="qwen3_5")
        assert any("QLoRA" in r.message for r in caplog.records)

    def test_qlora_on_qwen3_does_not_warn(self, model_args, sft_args, caplog):
        model_args.use_peft, model_args.load_in_4bit = True, True
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            self._validate(model_args, sft_args, model_type="qwen3")
        assert not caplog.records


class TestResolveFastInference:
    """Test cases for resolve_fast_inference"""

    def test_false_when_vllm_off(self, grpo_args):
        grpo_args.use_vllm = False
        assert ui.resolve_fast_inference(grpo_args, "qwen3", use_peft=True) is False
        assert grpo_args.use_vllm is False

    @pytest.mark.parametrize("model_type", sorted(ui.FAST_INFERENCE_MODEL_TYPES))
    def test_true_for_supported_types(self, grpo_args, model_type):
        assert ui.resolve_fast_inference(grpo_args, model_type, use_peft=True) is True
        assert grpo_args.use_vllm is True

    @pytest.mark.parametrize("model_type", ["qwen3_moe", "qwen3_5_moe"])
    def test_falls_back_to_hf_generate(self, grpo_args, model_type, caplog):
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            assert (
                ui.resolve_fast_inference(grpo_args, model_type, use_peft=True) is False
            )
        assert grpo_args.use_vllm is False
        assert any("HF generate" in r.message for r in caplog.records)

    def test_full_finetuning_falls_back_to_hf_generate(self, grpo_args, caplog):
        """unsloth's fast_inference is LoRA-only (it raises with full_finetuning=True)"""
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            assert (
                ui.resolve_fast_inference(grpo_args, "qwen3", use_peft=False) is False
            )
        assert grpo_args.use_vllm is False
        assert any("full fine-tuning" in r.message for r in caplog.records)


class TestPrepareTrainingArgs:
    """Test cases for prepare_training_args_for_unsloth"""

    def test_liger_turned_off_with_warning(self, sft_args, caplog):
        sft_args.use_liger_kernel = True
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            ui.prepare_training_args_for_unsloth(sft_args)
        assert sft_args.use_liger_kernel is False
        assert any("liger" in r.message.lower() for r in caplog.records)

    @pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
    def test_packing_warning_for_hybrid_attention(self, sft_args, caplog, model_type):
        sft_args.use_liger_kernel, sft_args.packing = False, True
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            ui.prepare_training_args_for_unsloth(sft_args, model_type=model_type)
        assert (
            sft_args.packing is True
        )  # unsloth drops it itself; we only make that visible
        assert any("unpacked" in r.message for r in caplog.records)

    def test_no_packing_warning_for_qwen3(self, sft_args, caplog):
        sft_args.use_liger_kernel, sft_args.packing = False, True
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            ui.prepare_training_args_for_unsloth(sft_args, model_type="qwen3")
        assert not caplog.records

    def test_noop_without_liger(self, sft_args, caplog):
        sft_args.use_liger_kernel = False
        with caplog.at_level(logging.WARNING, logger=ui.__name__):
            ui.prepare_training_args_for_unsloth(sft_args)
        assert not caplog.records


class TestResolveUseUnsloth:
    """Test cases for resolve_use_unsloth"""

    class _Args:
        def __init__(self, use_unsloth):
            self.use_unsloth = use_unsloth

    def test_no_decision_and_requested_raises(self, monkeypatch):
        monkeypatch.setattr(ub, "_DECISION", None)
        with pytest.raises(RuntimeError, match="bootstrap"):
            ui.resolve_use_unsloth(self._Args(True))

    def test_no_decision_and_not_requested(self, monkeypatch):
        monkeypatch.setattr(ub, "_DECISION", None)
        assert ui.resolve_use_unsloth(self._Args(False)) is False

    def test_inactive_decision(self, monkeypatch):
        monkeypatch.setattr(
            ub, "_DECISION", ub.UnslothDecision(True, False, "llama", "skipped")
        )
        assert ui.resolve_use_unsloth(self._Args(True)) is False

    def test_active_decision(self, monkeypatch):
        monkeypatch.setattr(
            ub, "_DECISION", ub.UnslothDecision(True, True, "qwen3", "imported")
        )
        assert ui.resolve_use_unsloth(self._Args(True)) is True

    @pytest.mark.parametrize("requested,parsed", [(False, True), (True, False)])
    def test_disagreement_raises(self, monkeypatch, requested, parsed):
        monkeypatch.setattr(
            ub, "_DECISION", ub.UnslothDecision(requested, False, None, "x")
        )
        with pytest.raises(ValueError, match="--use_unsloth true\\|false") as exc_info:
            ui.resolve_use_unsloth(self._Args(parsed))
        assert f"bootstrap read use_unsloth={requested}" in str(exc_info.value)
        assert f"parser read use_unsloth={parsed}" in str(exc_info.value)


class TestBootstrapParserAgreement:
    """The pre-import bootstrap and the real parser must read the same use_unsloth"""

    @pytest.fixture
    def run(self, tmp_path, monkeypatch):
        from dna_factory.dnotitia_arguments import DnotitiaArguments

        monkeypatch.setattr(ub, "_import_unsloth", lambda: None)
        monkeypatch.setattr(ub, "_noexec_inductor_cache_dir", lambda: None)
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
        model_dir = tmp_path / "qwen3"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type": "qwen3"}')
        defaults = tmp_path / "defaults.yaml"
        defaults.write_text(yaml.dump({"model_name_or_path": str(model_dir)}))

        def _run(cli, yaml_value=None):
            monkeypatch.setattr(ub, "_DECISION", None)
            cli = list(cli)
            if yaml_value is not None:
                user = tmp_path / "user.yaml"
                user.write_text(yaml.dump({"use_unsloth": yaml_value}))
                cli = ["--config", str(user), *cli]
            ub.maybe_import_unsloth(cli, str(defaults))
            parser = trl.TrlParser((DnotitiaArguments,))
            (dnotitia_args,) = parser.parse_args_and_config(cli)
            return dnotitia_args

        return _run

    @pytest.mark.parametrize(
        "cli,yaml_value,expected",
        [
            (["--use_unsloth", "true"], None, True),
            (["--use_unsloth=true"], None, True),
            (["--use-unsloth", "true"], None, True),
            (["--use_unsloth"], None, True),
            (["--use_unsloth", "t"], None, True),
            (["--use_unsloth", "Y"], None, True),
            (["--use_unsloth", "0"], None, False),
            (["--use_unsloth", "f"], None, False),
            (["--use_unsloth", "no"], None, False),
            ([], None, False),
            ([], True, True),
            ([], False, False),
            (["--use_unsloth", "false"], True, False),
            (["--use_unsloth", "true"], False, True),
        ],
    )
    def test_agree(self, run, cli, yaml_value, expected):
        dnotitia_args = run(cli, yaml_value)
        assert dnotitia_args.use_unsloth is expected
        assert ub.get_unsloth_decision().requested is expected
        assert ui.resolve_use_unsloth(dnotitia_args) is expected

    @pytest.mark.parametrize("flag", ["--use_unsl", "--use_unslo"])
    def test_abbreviated_flag_raises(self, run, flag):
        dnotitia_args = run([flag, "true"])
        assert dnotitia_args.use_unsloth is True  # argparse accepted the prefix
        assert ub.get_unsloth_decision().requested is False
        with pytest.raises(
            ValueError, match="spell the flag out in full|Spell the flag"
        ):
            ui.resolve_use_unsloth(dnotitia_args)


class FakeUnslothGRPOConfig(trl.GRPOConfig):
    """Mimics unsloth's patched config: rewrites values in __post_init__"""

    def __post_init__(self):
        super().__post_init__()
        self.epsilon_high = 0.28
        self.top_k = -1
        self.gradient_checkpointing_kwargs = {"use_reentrant": True}


# unsloth's patched config keeps TRL's class name and module (for pickling), so mimic that identity
FakeUnslothGRPOConfig.__name__ = FakeUnslothGRPOConfig.__qualname__ = "GRPOConfig"
FakeUnslothGRPOConfig.__module__ = trl.GRPOConfig.__module__


class TestRevertUnslothConfigOverrides:
    """Test cases for revert_unsloth_config_overrides"""

    def _full_args(self, tmp_path):
        config = {
            "loss_type": "dapo",
            "gradient_checkpointing": True,
            "bf16": False,  # CI runners have no GPU
            # keys that belong to other dataclasses of the real parser: must land in remaining strings
            "model_name_or_path": "dummy/model",
            "use_unsloth": True,
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(config))
        return [
            "--config",
            str(path),
            "--output_dir",
            str(tmp_path / "out"),
            "--use_unsloth",
            "true",
        ]

    def test_restores_rewritten_values_but_keeps_gradient_checkpointing_kwargs(
        self, tmp_path
    ):
        full_args = self._full_args(tmp_path)
        training_args = FakeUnslothGRPOConfig(
            output_dir=str(tmp_path / "out"),
            loss_type="dapo",
            gradient_checkpointing=True,
            bf16=False,
        )
        assert training_args.epsilon_high == 0.28

        restored = ui.revert_unsloth_config_overrides(training_args, full_args)

        assert restored["epsilon_high"] == (0.28, None)
        assert training_args.epsilon_high is None
        assert restored["top_k"] == (-1, 0)
        assert "gradient_checkpointing_kwargs" not in restored
        assert training_args.gradient_checkpointing_kwargs == {"use_reentrant": True}
        assert training_args.loss_type == "dapo"

    def test_untouched_values_not_reported(self, tmp_path):
        full_args = self._full_args(tmp_path)
        training_args = FakeUnslothGRPOConfig(
            output_dir=str(tmp_path / "out"),
            loss_type="dapo",
            gradient_checkpointing=True,
            bf16=False,
        )
        restored = ui.revert_unsloth_config_overrides(training_args, full_args)
        assert set(restored) == {"epsilon_high", "top_k"}

    def test_noop_for_unpatched_config(self, tmp_path):
        training_args = trl.GRPOConfig(output_dir=str(tmp_path), bf16=False)
        assert (
            ui.revert_unsloth_config_overrides(
                training_args, ["--output_dir", str(tmp_path)]
            )
            == {}
        )


class TestGenerationMaxSeqLength:
    """Test cases for generation_max_seq_length"""

    def test_prefers_vllm_max_model_length(self, model_args, grpo_args):
        grpo_args.vllm_max_model_length = 3000
        assert ui.generation_max_seq_length(model_args, grpo_args) == 3000

    def test_reads_model_config(self, model_args, grpo_args, tmp_path):
        import json

        model_dir = tmp_path / "m"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen3", "max_position_embeddings": 40960})
        )
        model_args.model_name_or_path = str(model_dir)
        grpo_args.vllm_max_model_length = None
        assert ui.generation_max_seq_length(model_args, grpo_args) == 40960


class TestTokenizeConversationalDataset:
    """Test cases for tokenize_conversational_dataset (mirrors TRL's conversational tokenization)"""

    @pytest.fixture
    def tokenizer(self):
        """Prefer the local cache so each run is not a Hub round-trip; CI has no cache."""
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(
                "trl-internal-testing/tiny-Qwen3ForCausalLM", local_files_only=True
            )
        except OSError:
            return AutoTokenizer.from_pretrained(
                "trl-internal-testing/tiny-Qwen3ForCausalLM"
            )

    @pytest.fixture
    def messages_dataset(self):
        from datasets import Dataset

        def conv(question, answer, **assistant_extra):
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer, **assistant_extra},
            ]

        return Dataset.from_list(
            [
                {"messages": conv("What is 1+1?", "2")},
                {"messages": conv("Say hi", "Hello there!")},
                {
                    "messages": conv(
                        "Why?", "Because.", reasoning_content="Let me think."
                    )
                },
            ]
        )

    @pytest.fixture
    def args(self):
        return __import__("types").SimpleNamespace(
            dataset_num_proc=None, assistant_only_loss=False
        )

    @pytest.mark.parametrize("assistant_only_loss", [True, False])
    def test_matches_trl_tokenization(
        self, tokenizer, messages_dataset, args, assistant_only_loss
    ):
        from trl.data_utils import _tokenize

        args.assistant_only_loss = assistant_only_loss
        out = ui.tokenize_conversational_dataset(
            messages_dataset, tokenizer, args, None, "train"
        )

        assert len(out) == len(messages_dataset)
        for row, src in zip(out, messages_dataset):
            expected = _tokenize(
                tokenizer,
                src["messages"],
                return_assistant_tokens_mask=assistant_only_loss,
                chat_template=None,
                tools=None,
            )
            assert row["input_ids"] == expected["input_ids"]
            if assistant_only_loss:
                assert row["assistant_masks"] == expected["assistant_masks"]
                assert 1 in row["assistant_masks"]
            else:
                assert "assistant_masks" not in row

    def test_sharegpt_conversations_converted(self, tokenizer, args):
        from datasets import Dataset

        dataset = Dataset.from_list(
            [
                {
                    "conversations": [
                        {"from": "human", "value": "hi"},
                        {"from": "gpt", "value": "hello"},
                    ]
                }
            ]
        )
        out = ui.tokenize_conversational_dataset(
            dataset, tokenizer, args, None, "train"
        )
        assert (
            "input_ids" in out.column_names and "conversations" not in out.column_names
        )

    def test_pretokenized_passes_through(self, tokenizer, args):
        from datasets import Dataset

        dataset = Dataset.from_list([{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}])
        out = ui.tokenize_conversational_dataset(
            dataset, tokenizer, args, None, "train"
        )
        assert out is dataset

    def test_text_dataset_passes_through(self, tokenizer, args):
        from datasets import Dataset

        dataset = Dataset.from_list([{"text": "hello world"}])
        out = ui.tokenize_conversational_dataset(
            dataset, tokenizer, args, None, "train"
        )
        assert out is dataset
        assert out.column_names == ["text"]

    def test_prompt_completion_passes_through(self, tokenizer, args):
        from datasets import Dataset

        dataset = Dataset.from_list([{"prompt": "a", "completion": "b"}])
        out = ui.tokenize_conversational_dataset(
            dataset, tokenizer, args, None, "train"
        )
        assert out is dataset


class TestTruncateLikeTrl:
    """Test cases for truncate_like_trl (TRL's max_length truncation for pre-tokenized rows)"""

    @staticmethod
    def _args(max_length=4, truncation_mode="keep_start"):
        return __import__("types").SimpleNamespace(
            dataset_num_proc=None,
            max_length=max_length,
            truncation_mode=truncation_mode,
        )

    @pytest.fixture
    def dataset(self):
        from datasets import Dataset

        return Dataset.from_list(
            [
                {
                    "input_ids": [1, 2, 3, 4, 5, 6],
                    "assistant_masks": [0, 0, 0, 1, 1, 1],
                },
                {"input_ids": [7, 8], "assistant_masks": [0, 1]},
            ]
        )

    def test_keep_start(self, dataset):
        out = ui.truncate_like_trl(dataset, self._args(), "train")
        assert out["input_ids"] == [[1, 2, 3, 4], [7, 8]]
        assert out["assistant_masks"] == [[0, 0, 0, 1], [0, 1]]

    def test_keep_end(self, dataset):
        out = ui.truncate_like_trl(dataset, self._args(3, "keep_end"), "train")
        assert out["input_ids"] == [[4, 5, 6], [7, 8]]
        assert out["assistant_masks"] == [[1, 1, 1], [0, 1]]

    def test_all_mask_columns_sliced_and_combined(self):
        from datasets import Dataset

        dataset = Dataset.from_list(
            [
                {
                    "input_ids": [1, 2, 3, 4],
                    "assistant_masks": [1, 1, 0, 0],
                    "completion_mask": [0, 1, 1, 1],
                }
            ]
        )
        out = ui.truncate_like_trl(dataset, self._args(3), "train")
        assert out["completion_mask"] == [[0, 1, 1]]
        assert out["assistant_masks"] == [[1, 1, 0]]
        # trained = all masks 1 -> position 1 only
        assert len(out) == 1
        dropped = ui.truncate_like_trl(dataset, self._args(1), "train")
        assert len(dropped) == 0

    def test_fully_masked_rows_dropped(self, dataset):
        out = ui.truncate_like_trl(dataset, self._args(3), "train")
        assert out["input_ids"] == [[7, 8]]

    def test_no_mask_columns_keeps_rows(self):
        from datasets import Dataset

        dataset = Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5]}])
        out = ui.truncate_like_trl(dataset, self._args(2), "train")
        assert out["input_ids"] == [[1, 2]]

    def test_short_rows_unchanged(self, dataset):
        out = ui.truncate_like_trl(dataset, self._args(10), "train")
        assert out["input_ids"] == dataset["input_ids"]
        assert out["assistant_masks"] == dataset["assistant_masks"]

    def test_max_length_none_unchanged(self, dataset):
        assert ui.truncate_like_trl(dataset, self._args(None), "train") is dataset

    def test_unsupported_truncation_mode_raises(self, dataset):
        with pytest.raises(ValueError, match="Unsupported truncation mode"):
            ui.truncate_like_trl(dataset, self._args(4, "keep_middle"), "train")

    @pytest.fixture
    def tokenizer(self):
        from transformers import AutoTokenizer

        try:
            return AutoTokenizer.from_pretrained(
                "trl-internal-testing/tiny-Qwen3ForCausalLM", local_files_only=True
            )
        except OSError:
            return AutoTokenizer.from_pretrained(
                "trl-internal-testing/tiny-Qwen3ForCausalLM"
            )

    @pytest.mark.parametrize("truncation_mode", ["keep_start", "keep_end"])
    def test_matches_trl_prepare_dataset(self, tokenizer, tmp_path, truncation_mode):
        """Our tokenize + truncate equals TRL's own SFTTrainer._prepare_dataset on CPU."""
        from datasets import Dataset
        from trl import SFTTrainer

        def conv(question, answer):
            return [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]

        dataset = Dataset.from_list(
            [
                {"messages": conv("Hi", "Hello there!")},
                {"messages": conv("Tell me a long story " * 20, "Once upon a time.")},
                {"messages": conv("Why?", "Because of many " * 15 + "reasons.")},
                {"messages": conv("What is 1+1?", "2")},
            ]
        )
        args = trl.SFTConfig(
            output_dir=str(tmp_path),
            bf16=False,
            max_length=48,
            truncation_mode=truncation_mode,
            packing=False,
            assistant_only_loss=True,
            dataset_num_proc=None,
            report_to="none",
        )
        fake_self = __import__("types").SimpleNamespace(
            chat_template=None,
            completion_only_loss=False,
            args=args,
            _is_vision_dataset=False,
            _signature_columns=None,
            _skip_prepare_dataset=False,
            _tokenizer=tokenizer,
        )
        try:
            expected = SFTTrainer._prepare_dataset(
                fake_self, dataset, tokenizer, args, False, None, "train"
            )
        except Exception as exc:  # noqa: BLE001 - TRL internals vary by version; cross-check only
            pytest.skip(f"cannot run TRL _prepare_dataset on CPU here: {exc!r}")

        out = ui.tokenize_conversational_dataset(
            dataset, tokenizer, args, None, "train"
        )
        out = ui.truncate_like_trl(out, args, "train")

        assert len(out) == len(expected)
        for row, exp in zip(out, expected):
            assert row["input_ids"] == exp["input_ids"]
            trained = [
                t if m else -100
                for t, m in zip(row["input_ids"], row["assistant_masks"])
            ]
            assert trained == exp["labels"]
        assert max(len(r["input_ids"]) for r in out) <= 48


class TestSFTTrainerUnslothPrepare:
    """DnotitiaSFTTrainer._prepare_dataset: truncation depends on packing and model type."""

    @pytest.fixture
    def run(self, monkeypatch, tmp_path):
        from datasets import Dataset

        from dna_factory import dnotitia_sft_trainer as dst

        tokenized = Dataset.from_list(
            [{"input_ids": [1, 2, 3, 4, 5, 6], "assistant_masks": [0, 0, 1, 1, 1, 1]}]
        )

        def go(model_type, packing, max_length=4):
            decision = __import__("types").SimpleNamespace(
                active=True, model_type=model_type
            )
            monkeypatch.setattr(dst, "get_unsloth_decision", lambda: decision)
            monkeypatch.setattr(
                dst, "tokenize_conversational_dataset", lambda ds, *a: tokenized
            )
            seen = {}

            def fake_super(self, dataset, *a):
                seen["dataset"] = dataset
                return dataset

            monkeypatch.setattr(dst.SFTTrainer, "_prepare_dataset", fake_super)
            args = trl.SFTConfig(
                output_dir=str(tmp_path), bf16=False, max_length=max_length
            )
            src = Dataset.from_list([{"messages": []}])
            trainer = object.__new__(dst.DnotitiaSFTTrainer)
            trainer._prepare_dataset(src, None, args, packing, None, "train")
            return seen["dataset"]["input_ids"][0]

        return go

    def test_unpacked_truncated(self, run):
        assert run("qwen3", False) == [1, 2, 3, 4]

    def test_packed_non_qwen3_5_untouched(self, run):
        assert run("qwen3", True) == [1, 2, 3, 4, 5, 6]

    def test_packed_qwen3_5_truncated(self, run):
        assert run("qwen3_5", True) == [1, 2, 3, 4]

    def test_max_length_none_untouched(self, run):
        assert run("qwen3", False, max_length=None) == [1, 2, 3, 4, 5, 6]


class TestScriptHooks:
    """The entry scripts' load_models hooks branch on ctx['use_unsloth'] (no GPU: loader is faked)."""

    @pytest.fixture
    def fake_loader(self, monkeypatch):
        calls = []

        def fake(model_args, training_args, **kwargs):
            calls.append(kwargs)
            return "unsloth-model"

        for name in ("sft", "dpo", "grpo", "opd"):
            monkeypatch.setattr(
                importlib.import_module(name), "load_unsloth_model", fake, raising=False
            )
        return calls

    def test_dpo_lora_uses_adapter_reference(self, model_args, fake_loader):
        import dpo

        model_args.use_peft = True
        ctx = {"use_unsloth": True, "unsloth_model_type": "qwen3"}
        out = dpo.load_models(
            None, trl.DPOConfig(output_dir="x", bf16=False), model_args, None, ctx, None
        )
        assert out == {"model": "unsloth-model", "ref_model": None}

    def test_grpo_moe_router_aux_loss_rejected(
        self, model_args, grpo_args, fake_loader
    ):
        import grpo

        grpo_args.router_aux_loss_coef = 0.001
        ctx = {"use_unsloth": True, "unsloth_model_type": "qwen3_moe"}
        with pytest.raises(ValueError, match="router_aux_loss_coef"):
            grpo.load_models(None, grpo_args, model_args, None, ctx, None)
        assert not fake_loader  # failed before the slow model load

    @pytest.mark.parametrize("config_coef, rejected", [(0.001, True), (0.0, False)])
    def test_grpo_moe_router_aux_loss_from_model_config(
        self,
        model_args,
        grpo_args,
        fake_loader,
        monkeypatch,
        tmp_path,
        config_coef,
        rejected,
    ):
        """TRL >= 1.15 defaults router_aux_loss_coef to None: the model config's value applies"""
        import json

        import grpo

        model_dir = tmp_path / "moe"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen3_moe", "router_aux_loss_coef": config_coef})
        )
        model_args.model_name_or_path = str(model_dir)
        grpo_args.router_aux_loss_coef = None
        grpo_args.vllm_max_model_length = 1024
        monkeypatch.setattr(grpo, "resolve_fast_inference", lambda *a, **k: False)
        ctx = {"use_unsloth": True, "unsloth_model_type": "qwen3_moe"}
        if rejected:
            with pytest.raises(ValueError, match="router_aux_loss_coef=0.001"):
                grpo.load_models(None, grpo_args, model_args, None, ctx, None)
            assert not fake_loader
        else:
            assert grpo.load_models(None, grpo_args, model_args, None, ctx, None) == {
                "model": "unsloth-model"
            }

    def test_grpo_vanilla_returns_model_name(self, model_args, grpo_args, fake_loader):
        import grpo

        out = grpo.load_models(None, grpo_args, model_args, None, {}, None)
        assert out == {"model": "dummy/model"}
        assert not fake_loader

    def test_grpo_setup_clears_init_kwargs_under_unsloth(
        self, model_args, grpo_args, monkeypatch
    ):
        import grpo

        monkeypatch.setattr(grpo, "resolve_reward_funcs", lambda *a: [])
        ctx = {"use_unsloth": True}
        grpo.setup_training_args(None, grpo_args, model_args, None, ctx, None)
        assert grpo_args.model_init_kwargs is None
        assert ctx["quantization_config"] is None


class TestAsyncGrpoRejectsUnsloth:
    def test_validate_async_args_rejects_use_unsloth(self):
        from types import SimpleNamespace

        from dna_factory.async_grpo import validate_async_args

        training = SimpleNamespace(
            deepspeed=None, fsdp=[], n_gpu=1, eval_strategy="no", do_eval=False
        )
        with pytest.raises(ValueError, match="use_unsloth"):
            validate_async_args(None, training, None, SimpleNamespace(use_unsloth=True))
