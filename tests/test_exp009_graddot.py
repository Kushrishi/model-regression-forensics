from __future__ import annotations

import copy

import pytest

from model_forensics.exp009_graddot import (
    PairwiseMarginLoss,
    last_layer_grad_dot_influence,
    suspiciousness_from_influence,
)

torch = pytest.importorskip("torch")


def test_pairwise_margin_loss_matches_manual_reference() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, -2.0],
            [0.5, 2.5, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    losses = PairwiseMarginLoss(0, 1)(logits, labels)

    expected = torch.tensor([-2.0, -2.0])
    torch.testing.assert_close(losses, expected)

    summed = PairwiseMarginLoss(0, 1, reduction="sum")(logits, labels)
    assert float(summed) == pytest.approx(float(expected.sum()))


def test_pairwise_margin_loss_rejects_protected_label() -> None:
    logits = torch.zeros((2, 3), dtype=torch.float32)
    labels = torch.tensor([0, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="non-target"):
        PairwiseMarginLoss(0, 1)(logits, labels)


def test_suspiciousness_is_negative_mean_captum_influence() -> None:
    influence = torch.tensor(
        [
            [2.0, -1.0, 0.5],
            [4.0, -3.0, 1.5],
        ],
        dtype=torch.float32,
    )

    suspiciousness = suspiciousness_from_influence(influence)

    torch.testing.assert_close(
        suspiciousness,
        torch.tensor([-3.0, 2.0, -1.0]),
    )


def test_captum_one_checkpoint_matches_manual_last_layer_grad_dot() -> None:
    captum = pytest.importorskip("captum")
    assert captum is not None
    from captum.influence import TracInCP
    from torch import nn
    from torch.utils.data import TensorDataset

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Linear(2, 3)
            self.classifier = nn.Linear(3, 3)

        def forward(self, x):
            return self.classifier(torch.tanh(self.hidden(x)))

    torch.manual_seed(7)
    model = ToyModel()

    train_x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    train_y = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    target_x = torch.tensor(
        [
            [0.75, 0.25],
            [0.25, 0.75],
        ],
        dtype=torch.float32,
    )
    target_y = torch.tensor([0, 1], dtype=torch.long)

    checkpoint = copy.deepcopy(model.state_dict())

    def load_checkpoint(module, state):
        module.load_state_dict(state)
        return 1.0

    tracin = TracInCP(
        model=model,
        train_dataset=TensorDataset(train_x, train_y),
        checkpoints=[checkpoint],
        checkpoints_load_func=load_checkpoint,
        layers=["classifier"],
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
        test_loss_fn=PairwiseMarginLoss(0, 1, reduction="none"),
        batch_size=4,
        sample_wise_grads_per_batch=False,
    )

    observed = tracin.influence((target_x, target_y), k=None)
    assert observed.shape == (2, 4)

    model.load_state_dict(checkpoint)
    expected = torch.empty_like(observed)

    for target_index in range(len(target_x)):
        model.zero_grad(set_to_none=True)
        target_logits = model(target_x[target_index : target_index + 1])
        target_loss = PairwiseMarginLoss(0, 1, reduction="sum")(
            target_logits,
            target_y[target_index : target_index + 1],
        )
        target_grads = torch.autograd.grad(
            target_loss,
            (model.classifier.weight, model.classifier.bias),
            retain_graph=False,
        )
        target_vector = torch.cat([grad.reshape(-1) for grad in target_grads])

        for train_index in range(len(train_x)):
            model.zero_grad(set_to_none=True)
            train_logits = model(train_x[train_index : train_index + 1])
            train_loss = nn.functional.cross_entropy(
                train_logits,
                train_y[train_index : train_index + 1],
                reduction="sum",
            )
            train_grads = torch.autograd.grad(
                train_loss,
                (model.classifier.weight, model.classifier.bias),
                retain_graph=False,
            )
            train_vector = torch.cat([grad.reshape(-1) for grad in train_grads])
            expected[target_index, train_index] = torch.dot(
                target_vector,
                train_vector,
            )

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        suspiciousness_from_influence(observed),
        -expected.mean(dim=0),
        rtol=1e-5,
        atol=1e-6,
    )


def test_analytic_last_layer_grad_dot_matches_autograd() -> None:
    torch.manual_seed(23)
    classifier = torch.nn.Linear(4, 3)
    train_features = torch.randn(5, 4)
    train_logits = classifier(train_features).detach()
    train_labels = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
    target_features = torch.randn(3, 4)
    target_labels = torch.tensor([0, 1, 0], dtype=torch.long)

    observed = last_layer_grad_dot_influence(
        train_features=train_features,
        train_logits=train_logits,
        train_label_ids=train_labels,
        target_features=target_features,
        target_label_ids=target_labels,
        target_a_id=0,
        target_b_id=1,
    )

    expected = torch.empty((3, 5), dtype=torch.float32)
    for target_index in range(3):
        classifier.zero_grad(set_to_none=True)
        target_loss = PairwiseMarginLoss(0, 1, reduction="sum")(
            classifier(target_features[target_index : target_index + 1]),
            target_labels[target_index : target_index + 1],
        )
        target_grad = torch.autograd.grad(
            target_loss,
            (classifier.weight, classifier.bias),
        )
        target_vector = torch.cat([value.reshape(-1) for value in target_grad])

        for train_index in range(5):
            classifier.zero_grad(set_to_none=True)
            train_loss = torch.nn.functional.cross_entropy(
                classifier(train_features[train_index : train_index + 1]),
                train_labels[train_index : train_index + 1],
                reduction="sum",
            )
            train_grad = torch.autograd.grad(
                train_loss,
                (classifier.weight, classifier.bias),
            )
            train_vector = torch.cat([value.reshape(-1) for value in train_grad])
            expected[target_index, train_index] = torch.dot(
                target_vector,
                train_vector,
            )

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
