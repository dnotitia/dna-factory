"""
Test cases for dna_factory/dnotitia_sdft_trainer.py and the SDFT dataset normalization in sdft.py

CPU-only: the teacher-context builder is exercised with a stub trainer, and the multi-GPU guard with a mocked
accelerator, so no model, GPU, or network is needed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset
from trl.experimental.sdft import SDFTTrainer

sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.dnotitia_sdft_trainer import (
    DnotitiaSDFTTrainer,
    PlacementTeacherContextBuilder,
)
from sdft import ContextDatasetMixtureConfig, normalize_dataset_for_sdft

TEMPLATE = "{prompt}\n\n{privileged_context}"
PROMPT = [{"role": "user", "content": "What is 2+2?"}]


def make_builder(placement):
    trainer = SimpleNamespace(args=SimpleNamespace(teacher_prompt_template=TEMPLATE))
    return PlacementTeacherContextBuilder(trainer, placement)


class TestPlacementTeacherContextBuilder:
    def test_user_placement_appends_to_last_user_turn(self):
        out = make_builder("user")._compose_teacher_prompt(PROMPT, "Be brief.")
        assert out == [{"role": "user", "content": "What is 2+2?\n\nBe brief."}]

    def test_system_placement_prepends_system_message(self):
        out = make_builder("system")._compose_teacher_prompt(PROMPT, "Be brief.")
        assert out == [{"role": "system", "content": "Be brief."}, *PROMPT]

    def test_system_placement_merges_with_existing_system_message(self):
        prompt = [{"role": "system", "content": "Existing."}, *PROMPT]
        out = make_builder("system")._compose_teacher_prompt(prompt, "Be brief.")
        assert out[0] == {"role": "system", "content": "Be brief.\n\nExisting."}
        assert out[1:] == PROMPT

    @pytest.mark.parametrize("placement", ["user", "system"])
    @pytest.mark.parametrize("context", ["", "   ", None])
    def test_empty_context_equals_student_prompt(self, placement, context):
        out = make_builder(placement)._compose_teacher_prompt(PROMPT, context)
        assert out == PROMPT
        assert out is not PROMPT  # a copy, so callers cannot mutate the dataset row

    def test_message_list_context_is_flattened(self):
        ctx = [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}]
        out = make_builder("system")._compose_teacher_prompt(PROMPT, ctx)
        assert out[0]["content"] == "A\nB"

    def test_system_placement_rejects_plain_text_prompt(self):
        with pytest.raises(ValueError, match="conversational"):
            make_builder("system")._compose_teacher_prompt("plain prompt", "ctx")

    def test_unknown_placement_rejected(self):
        with pytest.raises(ValueError, match="teacher_context_placement"):
            make_builder("assistant")

    def test_select_generation_prompts_follows_placement(self):
        out = make_builder("system").select_generation_prompts(
            [PROMPT, PROMPT], ["x", ""]
        )
        assert out[0][0] == {"role": "system", "content": "x"}
        assert out[1] == PROMPT


def test_loss_is_trl_s_own():
    """TRL >= 1.18 reduces the logged loss on every rank, empty or not. TRL 1.17 returned early on a rank with no
    loss tokens, and this trainer used to add a gather there; on 1.18 that extra gather is what would hang NCCL."""
    assert (
        DnotitiaSDFTTrainer._compute_self_distillation_loss
        is SDFTTrainer._compute_self_distillation_loss
    )


class TestDatasetNormalization:
    def test_messages_drop_last_assistant_and_use_given_context(self):
        ds = Dataset.from_list(
            [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ]
                }
            ]
        )
        row = normalize_dataset_for_sdft(ds, "CTX")[0]
        assert row["prompt"] == [{"role": "user", "content": "hi"}]
        assert row["privileged_context"] == "CTX"

    def test_string_prompt_becomes_user_turn_and_no_context_is_anchor(self):
        row = normalize_dataset_for_sdft(Dataset.from_list([{"prompt": "hi"}]))[0]
        assert row["prompt"] == [{"role": "user", "content": "hi"}]
        assert row["privileged_context"] == ""

    def test_existing_context_column_is_kept_and_yaml_context_overrides(self):
        ds = Dataset.from_list(
            [
                {"prompt": "a", "privileged_context": "own"},
                {"prompt": "b", "privileged_context": None},
            ]
        )
        kept = normalize_dataset_for_sdft(ds)
        assert [r["privileged_context"] for r in kept] == ["own", ""]
        overridden = normalize_dataset_for_sdft(ds, "yaml")
        assert {r["privileged_context"] for r in overridden} == {"yaml"}

    def test_missing_prompt_columns_rejected(self):
        with pytest.raises(ValueError, match="neither"):
            normalize_dataset_for_sdft(Dataset.from_list([{"text": "x"}]))


def test_mixture_config_accepts_context_fields():
    cfg = ContextDatasetMixtureConfig(
        datasets=[{"path": "a/b", "privileged_context": "ctx", "weight": 0.5}]
    )
    assert cfg.datasets[0].privileged_context == "ctx"
    assert cfg.datasets[0].weight == 0.5


class TestPerRowContext:
    def test_custom_columns_give_a_per_row_context(self):
        ds = Dataset.from_list(
            [{"question": "1+1?", "answer": "2"}, {"question": "2+2?", "answer": ""}]
        )
        rows = normalize_dataset_for_sdft(ds, None, "answer", "question")
        assert rows[0]["prompt"] == [{"role": "user", "content": "1+1?"}]
        assert [r["privileged_context"] for r in rows] == ["2", ""]

    def test_yaml_context_replaces_a_custom_column(self):
        ds = Dataset.from_list([{"question": "q", "answer": "a"}])
        rows = normalize_dataset_for_sdft(ds, "yaml", "answer", "question")
        assert rows[0]["privileged_context"] == "yaml"

    def test_missing_custom_columns_are_rejected(self):
        ds = Dataset.from_list([{"question": "q"}])
        with pytest.raises(ValueError, match="privileged_context_column"):
            normalize_dataset_for_sdft(ds, None, "answer", "question")
        with pytest.raises(ValueError, match="prompt_column"):
            normalize_dataset_for_sdft(ds, None, "privileged_context", "missing")

    def test_mixture_config_accepts_column_fields(self):
        cfg = ContextDatasetMixtureConfig(
            datasets=[
                {
                    "path": "json",
                    "data_files": "rows.jsonl",
                    "prompt_column": "question",
                    "privileged_context_column": "answer",
                }
            ]
        )
        entry = cfg.datasets[0]
        assert (
            entry.prompt_column,
            entry.privileged_context_column,
            entry.data_files,
        ) == ("question", "answer", "rows.jsonl")
        assert (
            ContextDatasetMixtureConfig(datasets=[{"path": "a/b"}])
            .datasets[0]
            .privileged_context_column
            == "privileged_context"
        )


def test_loader_passes_data_files_and_reads_a_per_row_column(monkeypatch):
    import sdft
    from dna_factory import dnotitia_dataset_mixture

    seen = {}

    def fake_load_dataset(**kwargs):
        seen.update(kwargs)
        return Dataset.from_list(
            [
                {"prompt": "p1", "privileged_context": "c1"},
                {"prompt": "p2", "privileged_context": None},
            ]
        )

    monkeypatch.setattr(dnotitia_dataset_mixture, "load_dataset", fake_load_dataset)
    cfg = ContextDatasetMixtureConfig(
        datasets=[{"path": "json", "data_files": "rows.jsonl", "split": "train"}]
    )
    out = sdft.get_dataset_with_context(cfg)["train"]
    assert seen["data_files"] == "rows.jsonl"
    assert [r["privileged_context"] for r in out] == ["c1", ""]
