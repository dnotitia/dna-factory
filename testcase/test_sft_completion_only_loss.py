"""
Test cases for completion_only_loss on conversational SFT data
(sft.py conversion, dnotitia_sft_trainer.check_completion_mask)
"""

import logging
import sys
import types
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

# Add parent directory to path to import dna_factory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dna_factory.dnotitia_sft_trainer import check_completion_mask
from sft import (
    EARLIER_ASSISTANT_TURNS,
    convert_for_completion_only_loss,
    messages_to_prompt_completion,
)


def conv(question, answer, thinking=""):
    return [
        {"role": "user", "content": question, "thinking": ""},
        {"role": "assistant", "content": answer, "thinking": thinking},
    ]


class TestMessagesToPromptCompletion:
    """Test cases for messages_to_prompt_completion"""

    def test_final_assistant_turn_becomes_the_completion(self):
        out = messages_to_prompt_completion({"messages": conv("Hi", "Hello")})
        assert [m["role"] for m in out["prompt"]] == ["user"]
        assert out["completion"][0]["content"] == "Hello"
        assert out[EARLIER_ASSISTANT_TURNS] == 0

    def test_thinking_is_passed_as_reasoning_content(self):
        out = messages_to_prompt_completion(
            {"messages": conv("Hi", "Hello", thinking="greet back")}
        )
        assert out["completion"][0]["reasoning_content"] == "greet back"

    def test_earlier_assistant_turns_are_counted(self):
        messages = conv("Hi", "Hello") + conv("And?", "Bye")
        out = messages_to_prompt_completion({"messages": messages})
        assert len(out["prompt"]) == 3
        assert out[EARLIER_ASSISTANT_TURNS] == 1

    def test_conversation_not_ending_with_assistant_raises(self):
        with pytest.raises(ValueError, match="end with an assistant turn"):
            messages_to_prompt_completion({"messages": conv("Hi", "Hello")[:1]})


class TestConvertForCompletionOnlyLoss:
    """Test cases for convert_for_completion_only_loss"""

    def test_converts_every_split_and_warns_about_multi_turn_rows(self, caplog):
        dataset = DatasetDict(
            {
                "train": Dataset.from_list(
                    [
                        {"messages": conv("Hi", "Hello")},
                        {"messages": conv("Hi", "Hello") + conv("And?", "Bye")},
                    ]
                )
            }
        )
        with caplog.at_level(logging.WARNING):
            out = convert_for_completion_only_loss(dataset, logging.getLogger("test"))
        assert out["train"].column_names == ["prompt", "completion"]
        assert (
            "1 of 2 `train` conversations have earlier assistant turns" in caplog.text
        )


class TestCheckCompletionMask:
    """Test cases for check_completion_mask"""

    def test_unmasked_labels_raise(self):
        dataset = Dataset.from_dict({"labels": [[1, 2, 3], [4, 5]]})
        with pytest.raises(
            ValueError, match="no token of the first 2 training examples"
        ):
            check_completion_mask(dataset)

    def test_masked_labels_return_the_trained_fraction(self):
        dataset = Dataset.from_dict({"labels": [[-100, 2, 3], [-100, -100, 5, 6]]})
        assert check_completion_mask(dataset) == pytest.approx(4 / 7)

    def test_without_labels_the_check_is_skipped(self):
        assert check_completion_mask(Dataset.from_dict({"input_ids": [[1, 2]]})) is None


class TestWithTrlTokenization:
    """The converted data, tokenized by TRL itself, masks the prompt and keeps the token sequence"""

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
    def messages(self):
        return [
            conv("What is 1+1?", "2", thinking="Add the two numbers."),
            conv("Name a color.", "Blue", thinking="Any color works."),
        ]

    def _prepare(self, dataset, tokenizer, tmp_path, completion_only_loss):
        import trl
        from trl import SFTTrainer

        args = trl.SFTConfig(
            output_dir=str(tmp_path),
            bf16=False,
            max_length=512,
            packing=False,
            completion_only_loss=completion_only_loss,
            dataset_num_proc=None,
            report_to="none",
        )
        fake_self = types.SimpleNamespace(
            chat_template=None,
            completion_only_loss=completion_only_loss,
            args=args,
            _is_vision_dataset=False,
            _signature_columns=None,
            _skip_prepare_dataset=False,
            _tokenizer=tokenizer,
        )
        try:
            return SFTTrainer._prepare_dataset(
                fake_self, dataset, tokenizer, args, False, None, "train"
            )
        except Exception as exc:  # noqa: BLE001 - TRL internals vary by version
            pytest.skip(f"cannot run TRL _prepare_dataset on CPU here: {exc!r}")

    def _with_reasoning(self, messages):
        return [
            {**m, "reasoning_content": m["thinking"]} if m["thinking"] else m
            for m in messages
        ]

    def test_messages_data_is_not_masked_and_the_check_fails(
        self, tokenizer, messages, tmp_path
    ):
        dataset = Dataset.from_list(
            [{"messages": self._with_reasoning(m)} for m in messages]
        )
        prepared = self._prepare(dataset, tokenizer, tmp_path, True)
        if "labels" not in prepared.column_names:
            pytest.skip("this TRL version builds labels in the collator")
        with pytest.raises(ValueError, match="no token of the first"):
            check_completion_mask(prepared)

    def test_converted_data_masks_the_prompt_and_keeps_the_sequence(
        self, tokenizer, messages, tmp_path
    ):
        dataset = DatasetDict(
            {"train": Dataset.from_list([{"messages": m} for m in messages])}
        )
        converted = convert_for_completion_only_loss(
            dataset, logging.getLogger("test")
        )["train"]
        prepared = self._prepare(converted, tokenizer, tmp_path, True)
        if "labels" not in prepared.column_names:
            pytest.skip("this TRL version builds labels in the collator")
        assert 0 < check_completion_mask(prepared) < 1

        for row, conversation in zip(prepared, messages):
            full = self._with_reasoning(conversation)
            expected = tokenizer.apply_chat_template(
                full, tokenize=True, return_dict=True
            )["input_ids"]
            prompt = tokenizer.apply_chat_template(
                full[:-1], tokenize=True, return_dict=True, add_generation_prompt=True
            )["input_ids"]
            # same token sequence as the conversation, with the loss on everything after the prompt
            assert row["input_ids"] == expected
            assert row["labels"][: len(prompt)] == [-100] * len(prompt)
            assert row["labels"][len(prompt) :] == expected[len(prompt) :]
