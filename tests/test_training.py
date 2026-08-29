import json

import pytest

from model_forensics.training import SFTExample, encode_sft_example, load_sft_examples


class FakeTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert tokenize is True
        if len(messages) == 1:
            assert add_generation_prompt is True
            return [10, 11, 12]
        assert add_generation_prompt is False
        return [10, 11, 12, 20, 21]


def test_encode_sft_example_masks_prompt_tokens() -> None:
    encoded = encode_sft_example(
        FakeTokenizer(),
        SFTExample(example_id="a", prompt="prompt", response="ACCEPT"),
        max_length=8,
    )

    assert encoded["input_ids"] == [10, 11, 12, 20, 21]
    assert encoded["attention_mask"] == [1, 1, 1, 1, 1]
    assert encoded["labels"] == [-100, -100, -100, 20, 21]


def test_encode_sft_example_rejects_oversized_example() -> None:
    with pytest.raises(ValueError, match="max_length"):
        encode_sft_example(
            FakeTokenizer(),
            SFTExample(example_id="a", prompt="prompt", response="ACCEPT"),
            max_length=4,
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
