from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import version
from typing import Any

import numpy as np
import torch
from torch import Tensor
from trak.gradient_computers import AbstractGradientComputer
from trak.modelout_functions import AbstractModelOutput

from model_forensics.exp009_attribution import aggregate_candidate_suspiciousness
from model_forensics.exp009_candidates import candidate_slots_from_manifest

TRAKER_FEASIBILITY_VERSION = "0.3.2"
EXP009_DISTILBERT_HEAD_PARAMETERS = (
    "pre_classifier.weight",
    "pre_classifier.bias",
    "classifier.weight",
    "classifier.bias",
)


def validate_traker_version() -> str:
    """Require the exact TRAK distribution version used by this feasibility baseline."""

    observed = version("traker")
    if observed != TRAKER_FEASIBILITY_VERSION:
        raise RuntimeError(
            "Exp009 TRAK feasibility requires an exact traker version: "
            f"expected={TRAKER_FEASIBILITY_VERSION} observed={observed}"
        )
    return observed


def exp009_head_parameter_names(model: torch.nn.Module) -> tuple[str, ...]:
    """Validate and return the exact DistilBERT classification-head parameters."""

    available = dict(model.named_parameters())
    missing = [name for name in EXP009_DISTILBERT_HEAD_PARAMETERS if name not in available]
    if missing:
        raise ValueError(f"missing frozen Exp009 head parameters: {missing}")
    return EXP009_DISTILBERT_HEAD_PARAMETERS


def selected_parameter_count(
    model: torch.nn.Module,
    parameter_names: Sequence[str],
) -> int:
    """Count a prospectively named parameter subset and reject unknown names."""

    available = dict(model.named_parameters())
    names = tuple(parameter_names)
    if not names:
        raise ValueError("at least one selected parameter is required")
    if len(names) != len(set(names)):
        raise ValueError("selected parameter names must be unique")

    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"selected parameters are missing from the model: {missing}")
    return sum(int(available[name].numel()) for name in names)


def _model_logits(
    model: torch.nn.Module,
    weights: Mapping[str, Tensor],
    buffers: Mapping[str, Tensor],
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    outputs = torch.func.functional_call(
        model,
        (dict(weights), dict(buffers)),
        args=(),
        kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        strict=False,
    )
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if logits.ndim != 2:
        raise ValueError(f"classifier logits must be rank 2; observed shape={tuple(logits.shape)}")
    return logits


class Exp009PairwiseTRAKModelOutput(AbstractModelOutput):
    """TRAK output preserving standard training features and pairwise target scoring.

    Batch contract is input_ids, attention_mask, label, paired_label.

    A negative paired_label means an ordinary training example and uses TRAK's
    standard correct-vs-all-other multiclass margin.

    A non-negative paired_label means a target example and uses the frozen
    Exp009 correct-vs-other-target pairwise margin. TRAK target scoring does not
    call get_out_to_loss_grad.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.loss_temperature = float(temperature)

    @staticmethod
    def get_output(
        model: torch.nn.Module,
        weights: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
        input_id: Tensor,
        attention_mask: Tensor,
        label: Tensor,
        paired_label: Tensor,
    ) -> Tensor:
        logits = _model_logits(
            model,
            weights,
            buffers,
            input_ids=input_id.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        )

        label_index = label.to(dtype=torch.long).reshape(1, 1)
        correct = logits.gather(1, label_index).reshape(())

        class_ids = torch.arange(logits.shape[-1], device=logits.device)
        other_mask = class_ids.reshape(1, -1) != label.to(dtype=torch.long)
        standard_other = logits.masked_fill(~other_mask, -torch.inf)
        standard_margin = correct - standard_other.logsumexp(dim=-1).reshape(())

        safe_pair_index = paired_label.to(dtype=torch.long).clamp(min=0).reshape(1, 1)
        paired = logits.gather(1, safe_pair_index).reshape(())
        pairwise_margin = correct - paired

        return torch.where(paired_label >= 0, pairwise_margin, standard_margin)

    def get_out_to_loss_grad(
        self,
        model: torch.nn.Module,
        weights: Mapping[str, Tensor],
        buffers: Mapping[str, Tensor],
        batch: Iterable[Tensor],
    ) -> Tensor:
        input_ids, attention_mask, labels, _paired_labels = batch
        logits = _model_logits(
            model,
            weights,
            buffers,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        probabilities = torch.softmax(logits / self.loss_temperature, dim=-1)
        correct_probability = probabilities[
            torch.arange(logits.shape[0], device=logits.device),
            labels.to(dtype=torch.long),
        ]
        return (1.0 - correct_probability).detach().unsqueeze(-1)


class SelectedParameterFunctionalGradientComputer(AbstractGradientComputer):
    """TRAK gradient computer that differentiates only a named parameter subset.

    Upstream TRAK grad_wrt currently filters gradients after differentiating the
    full parameter dictionary. This feasibility computer instead presents only
    the selected parameter dictionary as the differentiable PyTree.

    Omitted parameters remain on the loaded module and are used by
    torch.func.functional_call with strict=False as fixed state.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        task: AbstractModelOutput,
        grad_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        grad_wrt: Iterable[str] | None = None,
    ) -> None:
        super().__init__(model, task, grad_dim, dtype, device)
        if grad_wrt is None:
            raise ValueError("selected-parameter TRAK requires an explicit grad_wrt list")
        self.grad_wrt = tuple(grad_wrt)
        if not self.grad_wrt:
            raise ValueError("selected-parameter TRAK requires at least one parameter")
        if len(self.grad_wrt) != len(set(self.grad_wrt)):
            raise ValueError("grad_wrt must contain unique parameter names")

        expected_dim = selected_parameter_count(model, self.grad_wrt)
        if int(grad_dim) != expected_dim:
            raise ValueError(
                f"selected gradient dimension mismatch: expected={expected_dim} observed={grad_dim}"
            )
        self.load_model_params(model)

    def load_model_params(self, model: torch.nn.Module) -> None:
        self.model = model
        available = dict(model.named_parameters())

        missing = [name for name in self.grad_wrt if name not in available]
        if missing:
            raise ValueError(f"selected gradient parameters disappeared: {missing}")

        selected = set(self.grad_wrt)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in selected)

        self.func_weights = {name: available[name] for name in self.grad_wrt}
        self.func_buffers = dict(model.named_buffers())

    def compute_per_sample_grad(self, batch: Iterable[Tensor]) -> dict[str, Tensor]:
        batch_tuple = tuple(batch)
        if not batch_tuple:
            raise ValueError("TRAK gradient batch must be non-empty")

        grad_fn = torch.func.grad(self.modelout_fn.get_output, argnums=1)
        gradients = torch.func.vmap(
            grad_fn,
            in_dims=(None, None, None, *([0] * len(batch_tuple))),
            randomness="different",
        )(
            self.model,
            self.func_weights,
            self.func_buffers,
            *batch_tuple,
        )

        observed = tuple(gradients)
        if set(observed) != set(self.grad_wrt):
            raise RuntimeError(
                "selected-parameter gradient keys drifted: "
                f"expected={sorted(self.grad_wrt)} observed={sorted(observed)}"
            )
        return gradients

    def compute_loss_grad(self, batch: Iterable[Tensor]) -> Tensor:
        return self.modelout_fn.get_out_to_loss_grad(
            self.model,
            self.func_weights,
            self.func_buffers,
            batch,
        )


def trak_scores_to_slot_suspiciousness(
    trak_scores: Any,
    slot_ids: Sequence[str],
) -> dict[str, float]:
    """Convert TRAK support scores to the frozen MRF suspiciousness orientation."""

    scores = np.asarray(trak_scores, dtype=np.float64)
    ids = tuple(slot_ids)
    if scores.ndim != 2:
        raise ValueError(f"TRAK scores must be train-by-target rank 2; observed={scores.shape}")
    if scores.shape[0] != len(ids):
        raise ValueError(
            f"TRAK training-row count mismatch: scores={scores.shape[0]} slots={len(ids)}"
        )
    if scores.shape[1] == 0:
        raise ValueError("TRAK scores must contain at least one target example")
    if len(ids) != len(set(ids)):
        raise ValueError("slot_ids must be unique")
    if not np.isfinite(scores).all():
        raise ValueError("TRAK scores must be finite")

    means = scores.mean(axis=1)
    return {slot_id: -float(score) for slot_id, score in zip(ids, means, strict=True)}


def trak_candidate_scores(
    trak_scores: Any,
    slot_ids: Sequence[str],
    diagnostic_candidate_manifest: Mapping[str, object],
) -> dict[str, float]:
    """Aggregate frozen-orientation TRAK slot scores to opaque candidates."""

    slot_scores = trak_scores_to_slot_suspiciousness(trak_scores, slot_ids)
    candidates = candidate_slots_from_manifest(diagnostic_candidate_manifest)
    return aggregate_candidate_suspiciousness(slot_scores, candidates)
