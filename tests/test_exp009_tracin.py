from __future__ import annotations

import pytest

from model_forensics.exp009_graddot import last_layer_grad_dot_influence
from model_forensics.exp009_tracin import (
    LastLayerCheckpointView,
    checkpointed_last_layer_influence,
)

torch = pytest.importorskip("torch")


def _view(*, learning_rate: float) -> LastLayerCheckpointView:
    return LastLayerCheckpointView(
        train_features=torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float64),
        train_logits=torch.tensor(
            [[0.0, 0.0, 0.0], [0.4, -0.2, 0.1]],
            dtype=torch.float64,
        ),
        target_features=torch.tensor([[0.0, 0.0], [0.5, 1.0]], dtype=torch.float64),
        producing_learning_rate=learning_rate,
    )


def test_single_checkpoint_equals_frozen_grad_dot_times_producing_lr() -> None:
    view = _view(learning_rate=0.25)
    train_labels = torch.tensor([0, 2])
    target_labels = torch.tensor([0, 1])

    expected = 0.25 * last_layer_grad_dot_influence(
        train_features=view.train_features,
        train_logits=view.train_logits,
        train_label_ids=train_labels,
        target_features=view.target_features,
        target_label_ids=target_labels,
        target_a_id=0,
        target_b_id=1,
    )
    actual = checkpointed_last_layer_influence(
        checkpoints=(view,),
        train_label_ids=train_labels,
        target_label_ids=target_labels,
        target_a_id=0,
        target_b_id=1,
    )

    assert torch.allclose(actual, expected)


def test_bias_contribution_survives_when_all_layer_inputs_are_zero() -> None:
    view = LastLayerCheckpointView(
        train_features=torch.zeros((1, 2), dtype=torch.float64),
        train_logits=torch.zeros((1, 3), dtype=torch.float64),
        target_features=torch.zeros((1, 2), dtype=torch.float64),
        producing_learning_rate=0.2,
    )

    influence = checkpointed_last_layer_influence(
        checkpoints=(view,),
        train_label_ids=torch.tensor([0]),
        target_label_ids=torch.tensor([0]),
        target_a_id=0,
        target_b_id=1,
    )

    assert influence.shape == (1, 1)
    assert influence.item() == pytest.approx(0.2)


def test_multiple_checkpoints_sum_learning_rate_weighted_contributions() -> None:
    first = _view(learning_rate=0.2)
    second = _view(learning_rate=0.3)
    train_labels = torch.tensor([0, 2])
    target_labels = torch.tensor([0, 1])

    base = last_layer_grad_dot_influence(
        train_features=first.train_features,
        train_logits=first.train_logits,
        train_label_ids=train_labels,
        target_features=first.target_features,
        target_label_ids=target_labels,
        target_a_id=0,
        target_b_id=1,
    )
    actual = checkpointed_last_layer_influence(
        checkpoints=(first, second),
        train_label_ids=train_labels,
        target_label_ids=target_labels,
        target_a_id=0,
        target_b_id=1,
    )

    assert torch.allclose(actual, 0.5 * base)


def test_checkpointed_tracin_rejects_nonpositive_producing_lr() -> None:
    view = _view(learning_rate=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        checkpointed_last_layer_influence(
            checkpoints=(view,),
            train_label_ids=torch.tensor([0, 2]),
            target_label_ids=torch.tensor([0, 1]),
            target_a_id=0,
            target_b_id=1,
        )
