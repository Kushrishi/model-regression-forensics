from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from model_forensics.exp009_graddot import last_layer_grad_dot_influence


@dataclass(frozen=True)
class LastLayerCheckpointView:
    """Model outputs needed for exact final-linear-layer TracIn at one checkpoint."""

    train_features: Any
    train_logits: Any
    target_features: Any
    producing_learning_rate: float


def checkpointed_last_layer_influence(
    *,
    checkpoints: Sequence[LastLayerCheckpointView],
    train_label_ids: Any,
    target_label_ids: Any,
    target_a_id: int,
    target_b_id: int,
) -> Any:
    """Return exact checkpoint-weighted influence for classifier weight and bias.

    Each checkpoint contribution uses the explicit final-linear-layer Grad-Dot
    identity already frozen for Exp009. That identity includes ``+1`` in the
    feature inner product, which is the classifier-bias gradient contribution.
    Contributions are weighted by the learning rate that produced the checkpoint
    state and summed over checkpoints.
    """

    import torch

    rows = tuple(checkpoints)
    if not rows:
        raise ValueError("at least one checkpoint view is required")

    total = None
    expected_shape: tuple[int, int] | None = None
    for index, checkpoint in enumerate(rows):
        learning_rate = float(checkpoint.producing_learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                f"checkpoint {index}: producing_learning_rate must be finite and positive"
            )

        native = last_layer_grad_dot_influence(
            train_features=checkpoint.train_features,
            train_logits=checkpoint.train_logits,
            train_label_ids=train_label_ids,
            target_features=checkpoint.target_features,
            target_label_ids=target_label_ids,
            target_a_id=target_a_id,
            target_b_id=target_b_id,
        )
        shape = tuple(int(size) for size in native.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                "checkpoint influence shapes differ: "
                f"expected={expected_shape} observed={shape} at index={index}"
            )

        weighted = native * learning_rate
        total = weighted if total is None else total + weighted

    assert total is not None
    if not bool(torch.isfinite(total).all().item()):
        raise ValueError("checkpointed influence contains non-finite values")
    return total
