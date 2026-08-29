import json

import pytest

from model_forensics.training import SFTExample, encode_sft_example, load_sft_examples


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, return_dict):
        assert tokenize is True
        assert return_dict is False
        assert len(messages) == 1
        assert add_generation_prompt is True
        return [10, 11, 12]

    def encode(self, text, *, add_special_tokens):
        assert text == "ACCEPT"
        assert add_special_tokens is False
        return [20, 21]


def test_encode_sft_example_masks_prompt_tokens() -> None:
    encoded = encode_sft_example(
        FakeTokenizer(),
        SFTExample(example_id="a", prompt="prompt", response="ACCEPT"),
        max_length=8,
    )

    assert encoded["input_ids"] == [10, 11, 12, 20, 21, 2]
    assert encoded["attention_mask"] == [1, 1, 1, 1, 1, 1]
    assert encoded["labels"] == [-100, -100, -100, 20, 21, 2]


def test_encode_sft_example_rejects_oversized_example() -> None:
    with pytest.raises(ValueError, match="max_length"):
        encode_sft_example(
            FakeTokenizer(),
            SFTExample(example_id="a", prompt="prompt", response="ACCEPT"),
            max_length=5,
        )


def test_encode_sft_example_requires_eos_token() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = None

    with pytest.raises(ValueError, match="eos_token_id"):
        encode_sft_example(
            tokenizer,
            SFTExample(example_id="a", prompt="prompt", response="ACCEPT"),
            max_length=8,
        )


def test_load_sft_examples_reads_prepared_schema(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": "train:one",
                "prompt": "Classify one.",
                "response": "ACCEPT",
                "shard_id": "ignored",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    examples = load_sft_examples(path)

    assert examples == (
        SFTExample(example_id="train:one", prompt="Classify one.", response="ACCEPT"),
    )
