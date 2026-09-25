# ruff: noqa: E402
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

trak = pytest.importorskip("trak")
torch = pytest.importorskip("torch")

from model_forensics.exp009_candidates import build_opaque_candidate_manifests
from model_forensics.exp009_data import content_id_for_record
from model_forensics.exp009_release import Exp009ReleaseSlot
from model_forensics.exp009_trak import (
    EXP009_DISTILBERT_HEAD_PARAMETERS,
    Exp009PairwiseTRAKModelOutput,
    SelectedParameterFunctionalGradientComputer,
    exp009_head_parameter_names,
    selected_parameter_count,
    trak_candidate_scores,
    trak_scores_to_slot_suspiciousness,
    validate_traker_version,
)


class ToyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.pre_classifier = torch.nn.Linear(4, 4)
        self.classifier = torch.nn.Linear(4, 3)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = torch.tanh(self.encoder(input_ids.float()))
        hidden = torch.relu(self.pre_classifier(hidden))
        return self.classifier(hidden)


def _selected_weights(model: ToyClassifier):
    names = exp009_head_parameter_names(model)
    available = dict(model.named_parameters())
    return {name: available[name] for name in names}


def _slot(slot_id: str, label: str, text: str) -> Exp009ReleaseSlot:
    content_id = content_id_for_record(label=label, text=text)
    return Exp009ReleaseSlot(
        slot_id=slot_id,
        source_content_id=content_id,
        model_content_id=content_id,
        text=text,
        label=label,
    )


def _replace_label(
    baseline: tuple[Exp009ReleaseSlot, ...],
    *,
    slot_id: str,
    label: str,
) -> tuple[Exp009ReleaseSlot, ...]:
    output = []
    for slot in baseline:
        if slot.slot_id != slot_id:
            output.append(slot)
            continue
        output.append(
            replace(
                slot,
                label=label,
                model_content_id=content_id_for_record(label=label, text=slot.text),
            )
        )
    return tuple(output)


def test_traker_version_is_pinned() -> None:
    assert validate_traker_version() == "0.3.2"


def test_exp009_head_parameter_set_is_exact_and_encoder_is_excluded() -> None:
    model = ToyClassifier()

    names = exp009_head_parameter_names(model)

    assert names == EXP009_DISTILBERT_HEAD_PARAMETERS
    assert all(not name.startswith("encoder.") for name in names)
    assert selected_parameter_count(model, names) == sum(
        parameter.numel() for name, parameter in model.named_parameters() if name in set(names)
    )


def test_distilbert_head_contract_matches_exp009_dimensions() -> None:
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    config = DistilBertConfig(
        vocab_size=100,
        max_position_embeddings=32,
        dim=768,
        hidden_dim=3072,
        n_layers=1,
        n_heads=12,
        num_labels=77,
    )
    model = DistilBertForSequenceClassification(config)
    names = exp009_head_parameter_names(model)

    assert names == EXP009_DISTILBERT_HEAD_PARAMETERS
    assert selected_parameter_count(model, names) == 649_805


def test_selected_gradient_computer_supports_transformers_distilbert() -> None:
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    config = DistilBertConfig(
        vocab_size=64,
        max_position_embeddings=16,
        dim=12,
        hidden_dim=24,
        n_layers=1,
        n_heads=3,
        num_labels=3,
        dropout=0.0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    model = DistilBertForSequenceClassification(config)
    names = exp009_head_parameter_names(model)
    task = Exp009PairwiseTRAKModelOutput()
    computer = SelectedParameterFunctionalGradientComputer(
        model=model,
        task=task,
        grad_dim=selected_parameter_count(model, names),
        dtype=torch.float32,
        device="cpu",
        grad_wrt=names,
    )
    batch = (
        torch.randint(0, config.vocab_size, (2, 5)),
        torch.ones((2, 5), dtype=torch.long),
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    )

    gradients = computer.compute_per_sample_grad(batch)

    assert set(gradients) == set(names)
    assert all(value.shape[0] == 2 for value in gradients.values())
    assert all(torch.isfinite(value).all() for value in gradients.values())


def test_pairwise_model_output_uses_frozen_target_margin() -> None:
    model = ToyClassifier()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([4.0, 1.0, -2.0]))

    task = Exp009PairwiseTRAKModelOutput()
    output = task.get_output(
        model,
        _selected_weights(model),
        dict(model.named_buffers()),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.ones(3),
        torch.tensor(0),
        torch.tensor(1),
    )

    assert float(output.detach()) == pytest.approx(3.0)


def test_training_sentinel_uses_standard_multiclass_margin() -> None:
    model = ToyClassifier()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias.copy_(torch.tensor([4.0, 1.0, -2.0]))

    task = Exp009PairwiseTRAKModelOutput()
    output = task.get_output(
        model,
        _selected_weights(model),
        dict(model.named_buffers()),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.ones(3),
        torch.tensor(0),
        torch.tensor(-1),
    )

    expected = 4.0 - float(torch.logsumexp(torch.tensor([1.0, -2.0]), dim=0))
    assert float(output.detach()) == pytest.approx(expected)


def test_selected_gradient_computer_returns_only_head_gradients() -> None:
    torch.manual_seed(0)
    model = ToyClassifier()
    names = exp009_head_parameter_names(model)
    task = Exp009PairwiseTRAKModelOutput()
    computer = SelectedParameterFunctionalGradientComputer(
        model=model,
        task=task,
        grad_dim=selected_parameter_count(model, names),
        dtype=torch.float32,
        device="cpu",
        grad_wrt=names,
    )

    batch = (
        torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
            ]
        ),
        torch.ones((2, 3)),
        torch.tensor([0, 1]),
        torch.tensor([-1, -1]),
    )

    gradients = computer.compute_per_sample_grad(batch)

    assert set(gradients) == set(names)
    assert all(value.shape[0] == 2 for value in gradients.values())
    assert model.encoder.weight.requires_grad is False
    assert model.pre_classifier.weight.requires_grad is True


def test_selected_gradient_computer_pairwise_target_grad_is_supported() -> None:
    torch.manual_seed(0)
    model = ToyClassifier()
    names = exp009_head_parameter_names(model)
    task = Exp009PairwiseTRAKModelOutput()
    computer = SelectedParameterFunctionalGradientComputer(
        model=model,
        task=task,
        grad_dim=selected_parameter_count(model, names),
        dtype=torch.float32,
        device="cpu",
        grad_wrt=names,
    )

    batch = (
        torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, -0.5],
            ]
        ),
        torch.ones((2, 3)),
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
    )

    gradients = computer.compute_per_sample_grad(batch)

    assert set(gradients) == set(names)
    assert all(torch.isfinite(value).all() for value in gradients.values())


def test_trak_score_orientation_negates_equal_weight_target_mean() -> None:
    scores = np.asarray(
        [
            [2.0, 4.0],
            [-1.0, -3.0],
        ]
    )

    suspiciousness = trak_scores_to_slot_suspiciousness(
        scores,
        ("slot_0000", "slot_0001"),
    )

    assert suspiciousness == {
        "slot_0000": -3.0,
        "slot_0001": 2.0,
    }


def test_trak_candidate_scores_consume_opaque_manifest() -> None:
    baseline = (
        _slot("slot_0000", "A", "alpha"),
        _slot("slot_0001", "B", "beta"),
    )
    candidate_a = _replace_label(baseline, slot_id="slot_0000", label="B")
    candidate_b = _replace_label(baseline, slot_id="slot_0001", label="A")
    diagnostic, _truth = build_opaque_candidate_manifests(
        baseline,
        {"internal_a": candidate_a, "internal_b": candidate_b},
    )

    scores = np.asarray(
        [
            [-2.0, -2.0],
            [1.0, 1.0],
        ]
    )
    candidate_scores = trak_candidate_scores(
        scores,
        ("slot_0000", "slot_0001"),
        diagnostic,
    )

    assert sorted(candidate_scores.values()) == [-1.0, 2.0]
