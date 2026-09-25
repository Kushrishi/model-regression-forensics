from __future__ import annotations

import pytest

from model_forensics.exp009_trak import (
    DISTILBERT_HEAD_PARAMETER_NAMES,
    configure_distilbert_for_trak,
    distilbert_head_parameter_names,
    make_distilbert_standard_trak_model_output,
    parameter_count_for_names,
    standard_correct_class_log_odds_margin,
)

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_distilbert() -> object:
    config = transformers.DistilBertConfig(
        vocab_size=101,
        max_position_embeddings=32,
        n_layers=1,
        n_heads=2,
        dim=32,
        hidden_dim=64,
        dropout=0.0,
        attention_dropout=0.0,
        num_labels=3,
    )
    model = transformers.DistilBertForSequenceClassification(config)
    configure_distilbert_for_trak(model)
    model.eval()
    return model


def test_standard_trak_margin_matches_manual_reference() -> None:
    logits = torch.tensor(
        [
            [3.0, 1.0, -2.0],
            [-1.0, 2.5, 0.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    observed = standard_correct_class_log_odds_margin(logits, labels)
    expected = torch.stack(
        (
            logits[0, 0] - torch.logsumexp(logits[0, 1:], dim=0),
            logits[1, 1] - torch.logsumexp(logits[1, [0, 2]], dim=0),
        )
    )

    torch.testing.assert_close(observed, expected)


def test_unpadded_no_mask_matches_padded_masked_logits() -> None:
    torch.manual_seed(0)
    model = _tiny_distilbert()

    unpadded = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    padded = torch.tensor([[1, 4, 5, 2, 0, 0, 0, 0]], dtype=torch.long)
    padded_mask = (padded != 0).to(torch.long)

    with torch.no_grad():
        unpadded_logits = model(input_ids=unpadded).logits
        padded_logits = model(input_ids=padded, attention_mask=padded_mask).logits

    torch.testing.assert_close(unpadded_logits, padded_logits, rtol=1e-5, atol=1e-6)


def test_head_parameter_selection_is_exact() -> None:
    model = _tiny_distilbert()

    names = distilbert_head_parameter_names(model)

    assert names == DISTILBERT_HEAD_PARAMETER_NAMES
    expected = (
        model.pre_classifier.weight.numel()
        + model.pre_classifier.bias.numel()
        + model.classifier.weight.numel()
        + model.classifier.bias.numel()
    )
    assert parameter_count_for_names(model, names) == expected


def test_tiny_distilbert_standard_trak_integration(tmp_path) -> None:
    trak = pytest.importorskip("trak")
    assert trak is not None
    from trak import TRAKer

    torch.manual_seed(0)
    model = _tiny_distilbert()
    names = distilbert_head_parameter_names(model)
    task = make_distilbert_standard_trak_model_output()

    traker = TRAKer(
        model=model,
        task=task,
        train_set_size=4,
        save_dir=tmp_path / "trak-smoke",
        device="cpu",
        proj_dim=128,
        projector_seed=0,
        grad_wrt=names,
    )

    checkpoint = model.state_dict()
    traker.load_checkpoint(checkpoint, model_id=0)

    train_input_ids = torch.tensor(
        [
            [1, 4, 5, 2, 3],
            [1, 8, 9, 2, 3],
            [1, 3, 7, 6, 2],
            [1, 5, 5, 5, 2],
        ],
        dtype=torch.long,
    )
    train_labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    traker.featurize(
        batch=(train_input_ids, train_labels),
        num_samples=4,
    )
    traker.finalize_features()

    target_input_ids = train_input_ids[:2].clone()
    target_labels = train_labels[:2].clone()

    traker.start_scoring_checkpoint(
        exp_name="tiny-distilbert",
        checkpoint=checkpoint,
        model_id=0,
        num_targets=2,
    )
    traker.score(
        batch=(target_input_ids, target_labels),
        num_samples=2,
    )
    scores = traker.finalize_scores(exp_name="tiny-distilbert")

    assert scores.shape == (4, 2)
    assert np.isfinite(scores).all()
