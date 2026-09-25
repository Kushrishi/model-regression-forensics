from __future__ import annotations

from typing import Any

DISTILBERT_HEAD_PARAMETER_NAMES = (
    "pre_classifier.weight",
    "pre_classifier.bias",
    "classifier.weight",
    "classifier.bias",
)


def correct_class_log_odds_margin(logits: Any, labels: Any) -> Any:
    """Return the multiclass correct-class log-odds margin for each row."""

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


def distilbert_additive_attention_mask(attention_mask: Any, *, dtype: Any) -> Any:
    """Build a vmap-safe 4-D additive key-padding mask for DistilBERT."""

    import torch

    if attention_mask.ndim not in {1, 2}:
        raise ValueError("attention_mask must have shape [seq] or [batch, seq]")

    mask_2d = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
    valid_keys = mask_2d.to(dtype=torch.bool)
    batch_size, sequence_length = valid_keys.shape
    valid_keys = valid_keys[:, None, None, :].expand(
        batch_size,
        1,
        sequence_length,
        sequence_length,
    )

    zero = torch.zeros((), dtype=dtype, device=attention_mask.device)
    blocked = torch.full(
        (),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=attention_mask.device,
    )
    return torch.where(valid_keys, zero, blocked)


def distilbert_head_parameter_names(model: Any) -> tuple[str, ...]:
    """Return the exact frozen head-only gradient parameter names."""

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
    """Force eager attention for the explicit additive-mask TRAK path."""

    setter = getattr(model, "set_attn_implementation", None)
    if setter is None or not callable(setter):
        raise ValueError("DistilBERT model does not expose set_attn_implementation")
    setter("eager")
    return model


def make_distilbert_trak_model_output() -> Any:
    """Construct a TRAK model-output instance lazily."""

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
            attention_mask: Any,
            label: Any,
        ) -> Any:
            prepared_mask = distilbert_additive_attention_mask(
                attention_mask,
                dtype=next(iter(weights.values())).dtype,
            )
            outputs = torch.func.functional_call(
                model,
                (weights, buffers),
                args=(),
                kwargs={
                    "input_ids": input_id.unsqueeze(0),
                    "attention_mask": prepared_mask,
                },
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            return correct_class_log_odds_margin(logits, label.unsqueeze(0)).sum()

        def get_out_to_loss_grad(
            self,
            model: Any,
            weights: Any,
            buffers: Any,
            batch: Any,
        ) -> Any:
            input_ids, attention_mask, labels = batch
            prepared_mask = distilbert_additive_attention_mask(
                attention_mask,
                dtype=next(iter(weights.values())).dtype,
            )
            outputs = torch.func.functional_call(
                model,
                (weights, buffers),
                args=(),
                kwargs={
                    "input_ids": input_ids,
                    "attention_mask": prepared_mask,
                },
            )
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            probabilities = self.softmax(logits / self.loss_temperature)
            row_index = torch.arange(logits.shape[0], device=logits.device)
            p_correct = probabilities[row_index, labels]
            return (1.0 - p_correct).clone().detach().unsqueeze(-1)

    return DistilBertTextClassificationModelOutput()
