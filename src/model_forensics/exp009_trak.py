from __future__ import annotations

from typing import Any


DISTILBERT_HEAD_PARAMETER_NAMES = (
    "pre_classifier.weight",
    "pre_classifier.bias",
    "classifier.weight",
    "classifier.bias",
)


def standard_correct_class_log_odds_margin(logits: Any, labels: Any) -> Any:
    """Return TRAK's standard multiclass correct-vs-all-other log-odds margin."""

    import torch

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch]")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must have the same batch size")
    if logits.shape[1] < 2:
        raise ValueError("classification margin requires at least two classes")

    labels = labels.to(dtype=torch.long, device=logits.device)
    row_index = torch.arange(logits.shape[0], device=logits.device)
    correct = logits[row_index, labels]

    other_logits = logits.clone()
    other_logits[row_index, labels] = -torch.inf
    return correct - torch.logsumexp(other_logits, dim=-1)


def distilbert_head_parameter_names(model: Any) -> tuple[str, ...]:
    """Return the exact head-only gradient parameter names used by the smoke test."""

    available = dict(model.named_parameters())
    missing = [name for name in DISTILBERT_HEAD_PARAMETER_NAMES if name not in available]
    if missing:
        raise ValueError(
            "model is missing required DistilBERT classification-head parameters: "
            + ", ".join(missing)
        )
    return DISTILBERT_HEAD_PARAMETER_NAMES


def parameter_count_for_names(model: Any, names: tuple[str, ...]) -> int:
    available = dict(model.named_parameters())
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError("unknown model parameter names: " + ", ".join(missing))
    return sum(int(available[name].numel()) for name in names)


def configure_distilbert_for_trak(model: Any) -> Any:
    """Use a stable eager-attention backend for TRAK compatibility tests."""

    setter = getattr(model, "set_attn_implementation", None)
    if setter is None or not callable(setter):
        raise ValueError("DistilBERT model does not expose set_attn_implementation")
    setter("eager")
    return model


def make_distilbert_standard_trak_model_output() -> Any:
    """Construct a DistilBERT-compatible standard-classification TRAK task.

    This object is for infrastructure feasibility only. Its correct-vs-all
    output is TRAK's standard multiclass formulation and is not the frozen
    Exp009 pairwise target in research/ATTRIBUTION_TARGET.md.
    """

    import torch
    from trak.modelout_functions import AbstractModelOutput

    class DistilBertTextClassificationModelOutput(AbstractModelOutput):
        def __init__(self, temperature: float = 1.0) -> None:
            super().__init__()
            self.softmax = torch.nn.Softmax(dim=-1)
            self.loss_temperature = temperature

        @staticmethod
        def get_output(
            model: Any,
            weights: Any,
            buffers: Any,
            input_id: Any,
            label: Any,
        ) -> Any:
            outputs = torch.func.functional_call(
                model,
                (weights, buffers),
                args=(),
                kwargs={"input_ids": input_id.unsqueeze(0)},
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            return standard_correct_class_log_odds_margin(
                logits,
                label.unsqueeze(0),
            ).sum()

        def get_out_to_loss_grad(
            self,
            model: Any,
            weights: Any,
            buffers: Any,
            batch: Any,
        ) -> Any:
            input_ids, labels = batch
            outputs = torch.func.functional_call(
                model,
                (weights, buffers),
                args=(),
                kwargs={"input_ids": input_ids},
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probabilities = self.softmax(logits / self.loss_temperature)
            row_index = torch.arange(logits.shape[0], device=logits.device)
            p_correct = probabilities[row_index, labels]
            return (1.0 - p_correct).clone().detach().unsqueeze(-1)

    return DistilBertTextClassificationModelOutput()
