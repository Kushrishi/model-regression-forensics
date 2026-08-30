import json

import pytest

from model_forensics.training import (
    SFTExample,
    encode_sft_example,
    load_sft_examples,
    resolve_response_loss_weights,
    response_weighted_causal_lm_loss,
)


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


def test_train_lora_sft_run_rejects_missing_prepared_split(tmp_path) -> None:
    from model_forensics.config import load_experiment_config
    from model_forensics.training import train_lora_sft_run

    with pytest.raises(FileNotFoundError, match="prepare_exp001.py"):
        train_lora_sft_run(
            config=load_experiment_config("configs/exp001.yaml"),
            prepared=tmp_path / "prepared",
            train_split="baseline_train",
            run_id="baseline",
            output_root=tmp_path / "checkpoints",
            preparation_command="scripts/prepare_exp001.py",
        )


def test_response_loss_weights_equalize_exp003_class_mass() -> None:
    examples = tuple(
        [SFTExample(example_id=f"a-{index}", prompt="p", response="ACCEPT") for index in range(2)]
        + [SFTExample(example_id="r-0", prompt="p", response="REJECT")]
    )

    weights, summary = resolve_response_loss_weights(examples, {"ACCEPT": 1.0, "REJECT": 2.0})

    assert weights == (1.0, 1.0, 2.0)
    assert summary == {
        "counts": {"ACCEPT": 2, "REJECT": 1},
        "weights": {"ACCEPT": 1.0, "REJECT": 2.0},
        "weighted_class_mass": {"ACCEPT": 2.0, "REJECT": 2.0},
        "mean_example_weight": 4.0 / 3.0,
    }


def test_response_loss_weights_require_exact_observed_labels() -> None:
    examples = (SFTExample(example_id="a", prompt="p", response="ACCEPT"),)

    with pytest.raises(ValueError, match="exactly match"):
        resolve_response_loss_weights(examples, {"ACCEPT": 1.0, "REJECT": 2.0})


def test_response_weighted_causal_lm_loss_uses_global_normalization() -> None:
    import torch
    import torch.nn.functional as F

    logits = torch.tensor(
        [
            [[0.0, 5.0], [0.0, 0.0]],
            [[5.0, 0.0], [0.0, 0.0]],
        ]
    )
    labels = torch.tensor([[-100, 1], [-100, 1]])
    weights = torch.tensor([1.0, 2.0])

    actual = response_weighted_causal_lm_loss(logits, labels, weights, normalization=1.5)

    first = F.cross_entropy(logits[0, 0].unsqueeze(0), torch.tensor([1]))
    second = F.cross_entropy(logits[1, 0].unsqueeze(0), torch.tensor([1]))
    expected = ((first * 1.0 + second * 2.0) / 2.0) / 1.5

    assert torch.isclose(actual, expected)
