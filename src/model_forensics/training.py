from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SFTExample:
    """One prompt/response pair consumed by the SFT training loop."""

    example_id: str
    prompt: str
    response: str


def load_sft_examples(path: str | Path) -> tuple[SFTExample, ...]:
    """Load prepared SFT examples from JSON Lines."""

    examples: list[SFTExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                examples.append(
                    SFTExample(
                        example_id=payload["example_id"],
                        prompt=payload["prompt"],
                        response=payload["response"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing required SFT field {exc.args[0]!r} at line {line_number}"
                ) from exc

    if not examples:
        raise ValueError("at least one prepared SFT example is required")
    return tuple(examples)


def encode_sft_example(
    tokenizer: Any, example: SFTExample, max_length: int
) -> dict[str, list[int]]:
    """Encode one chat example while masking prompt tokens from the loss."""

    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": example.prompt}],
        add_generation_prompt=True,
        tokenize=True,
    )
    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": example.response},
        ],
        add_generation_prompt=False,
        tokenize=True,
    )

    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template does not preserve the generation prompt as a prefix")
    if len(full_ids) > max_length:
        raise ValueError(
            f"encoded example {example.example_id!r} has {len(full_ids)} tokens, "
            f"exceeding max_length={max_length}"
        )

    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :]),
    }


def collate_sft_features(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad encoded SFT features for a causal-LM training batch."""

    if not features:
        raise ValueError("at least one SFT feature is required")

    import torch

    max_length = max(len(feature["input_ids"]) for feature in features)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []

    for feature in features:
        padding = max_length - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(feature["attention_mask"] + [0] * padding)
        labels.append(feature["labels"] + [-100] * padding)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
