from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GradDotSummary:
    """One trajectory's frozen last-layer Grad-Dot output."""

    slot_suspiciousness: tuple[float, ...]
    target_example_count: int


class PairwiseMarginLoss:
    """Negative correct-vs-paired-target margin for influence test examples."""

    def __init__(
        self,
        target_a_id: int,
        target_b_id: int,
        *,
        reduction: str = "none",
    ) -> None:
        if target_a_id == target_b_id:
            raise ValueError("pairwise target IDs must be distinct")
        if reduction not in {"none", "sum", "mean"}:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")
        self.target_a_id = int(target_a_id)
        self.target_b_id = int(target_b_id)
        self.reduction = reduction

    def __call__(self, logits: Any, labels: Any) -> Any:
        import torch

        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, classes]")
        if labels.ndim != 1:
            raise ValueError("labels must have shape [batch]")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("logits and labels must have the same batch size")

        labels = labels.to(dtype=torch.long, device=logits.device)
        allowed = (labels == self.target_a_id) | (labels == self.target_b_id)
        if not bool(torch.all(allowed).item()):
            raise ValueError("pairwise target loss received a non-target label")

        other = torch.where(
            labels == self.target_a_id,
            torch.full_like(labels, self.target_b_id),
            torch.full_like(labels, self.target_a_id),
        )
        row = torch.arange(logits.shape[0], device=logits.device)
        losses = -(logits[row, labels] - logits[row, other])
        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return losses.sum()
        return losses.mean()


def suspiciousness_from_influence(influence: Any) -> Any:
    """Convert Captum influence to the frozen MRF suspiciousness orientation.

    Captum returns shape [target_examples, training_examples]. Positive values
    are proponents of the test objective. Since Exp009 uses negative pairwise
    margin as the test loss, harmful current training contributions are
    opponents. Suspiciousness is therefore negative mean influence.
    """

    import torch

    tensor = torch.as_tensor(influence)
    if tensor.ndim != 2:
        raise ValueError("influence must have shape [target_examples, training_examples]")
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError("influence matrix must be non-empty")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("influence matrix contains non-finite values")
    return -tensor.mean(dim=0)


class DistilBertLogitsWrapper:
    """Factory namespace for a logits-only DistilBERT wrapper.

    Defined lazily so importing the core package does not require research
    dependencies in lightweight CI.
    """

    @staticmethod
    def build(model: Any) -> Any:
        import torch

        class _Wrapper(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner
                self.classifier = inner.classifier

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                return self.inner(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

        return _Wrapper(model)


def last_layer_grad_dot_influence(
    *,
    train_features: Any,
    train_logits: Any,
    train_label_ids: Any,
    target_features: Any,
    target_label_ids: Any,
    target_a_id: int,
    target_b_id: int,
) -> Any:
    """Exact final-linear-layer one-checkpoint Grad-Dot influence matrix."""

    import torch

    train_features = torch.as_tensor(train_features)
    train_logits = torch.as_tensor(train_logits)
    train_label_ids = torch.as_tensor(train_label_ids, dtype=torch.long)
    target_features = torch.as_tensor(target_features)
    target_label_ids = torch.as_tensor(target_label_ids, dtype=torch.long)

    if train_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError("features must be rank-2")
    if train_logits.ndim != 2:
        raise ValueError("train_logits must be rank-2")
    if train_features.shape[0] != train_logits.shape[0]:
        raise ValueError("training feature/logit counts differ")
    if train_label_ids.shape != (train_logits.shape[0],):
        raise ValueError("training label shape mismatch")
    if target_label_ids.shape != (target_features.shape[0],):
        raise ValueError("target label shape mismatch")
    if train_features.shape[1] != target_features.shape[1]:
        raise ValueError("feature widths differ")
    if target_a_id == target_b_id:
        raise ValueError("target IDs must differ")

    num_labels = train_logits.shape[1]
    allowed = (target_label_ids == target_a_id) | (target_label_ids == target_b_id)
    if not bool(torch.all(allowed).item()):
        raise ValueError("target labels must belong to the frozen pair")

    train_delta = torch.softmax(train_logits, dim=1).clone()
    train_rows = torch.arange(train_logits.shape[0])
    train_delta[train_rows, train_label_ids] -= 1.0

    target_delta = torch.zeros(
        (target_features.shape[0], num_labels),
        dtype=train_delta.dtype,
    )
    target_rows = torch.arange(target_features.shape[0])
    other = torch.where(
        target_label_ids == target_a_id,
        torch.full_like(target_label_ids, target_b_id),
        torch.full_like(target_label_ids, target_a_id),
    )
    target_delta[target_rows, target_label_ids] = -1.0
    target_delta[target_rows, other] = 1.0

    influence = (target_delta @ train_delta.T) * (target_features @ train_features.T + 1.0)
    if not bool(torch.isfinite(influence).all().item()):
        raise ValueError("Grad-Dot influence contains non-finite values")
    return influence
