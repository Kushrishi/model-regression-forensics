from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_forensics.exp009_data import DevelopmentPartition
from model_forensics.exp009_trajectory import (
    Exp009TrainingSlot,
    build_stable_training_slots,
    derive_trajectory_seeds,
    epoch_slot_order,
    slot_schedule_sha256,
)

EXP009_PILOT_MODEL_NAME = "distilbert/distilbert-base-uncased"
EXP009_PILOT_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
EXP009_PILOT_MODEL_SAFETENSORS_SHA256 = (
    "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
)
EXP009_FROZEN_DEVELOPMENT_PARTITION_SHA256 = (
    "61ecbdd9224cf2bbeafcf5ceb9164cfc3fc74eb449129136198646385aaa8e88"
)


@dataclass(frozen=True)
class Exp009ClassifierPilotConfig:
    """Development-only sequence-classifier settings; not confirmatory-frozen."""

    model_name: str = EXP009_PILOT_MODEL_NAME
    model_revision: str = EXP009_PILOT_MODEL_REVISION
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_length: int = 128
    max_grad_norm: float = 1.0

    def validate(self) -> None:
        if self.model_name != EXP009_PILOT_MODEL_NAME:
            raise ValueError("initial Exp009 pilot supports only the pinned DistilBERT candidate")
        if self.model_revision != EXP009_PILOT_MODEL_REVISION:
            raise ValueError("initial Exp009 pilot revision does not match the pinned candidate")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")


def development_partition_manifest_text(partition: DevelopmentPartition) -> str:
    """Serialize the frozen development membership without exposing example text."""

    rows = [
        {
            "content_id": record.content_id,
            "label": record.label,
            "split": split_name,
        }
        for split_name, records in (
            ("development_train", partition.development_train),
            ("development_eval", partition.development_eval),
        )
        for record in records
    ]
    rows.sort(key=lambda row: str(row["content_id"]))
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def development_partition_sha256(partition: DevelopmentPartition) -> str:
    return hashlib.sha256(development_partition_manifest_text(partition).encode("utf-8")).hexdigest()


def validate_frozen_development_partition(partition: DevelopmentPartition) -> str:
    observed = development_partition_sha256(partition)
    if observed != EXP009_FROZEN_DEVELOPMENT_PARTITION_SHA256:
        raise ValueError(
            "Exp009 development partition does not match the frozen substrate: "
            f"expected={EXP009_FROZEN_DEVELOPMENT_PARTITION_SHA256} observed={observed}"
        )
    return observed


def label_vocabulary(partition: DevelopmentPartition) -> tuple[str, ...]:
    train_labels = {record.label for record in partition.development_train}
    eval_labels = {record.label for record in partition.development_eval}
    if train_labels != eval_labels:
        raise ValueError("development train/eval label vocabularies must match exactly")
    labels = tuple(sorted(train_labels))
    if len(labels) != 77:
        raise ValueError(f"Exp009 Banking77 requires exactly 77 labels; observed {len(labels)}")
    return labels


def _set_torch_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(torch: Any) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _tensor_state_sha256(state_dict: dict[str, Any]) -> str:
    """Hash tensor names, metadata, and raw values in deterministic key order."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch_uint8(tensor)).numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def torch_uint8(tensor: Any) -> Any:
    """Resolve torch.uint8 from a tensor without importing torch at module import time."""

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("state hash expects torch tensors")
    return torch.uint8


def _batch_indices(indices: list[int], batch_size: int) -> list[list[int]]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def _runtime_metadata(torch: Any, transformers: Any, *, device: str) -> dict[str, object]:
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "device": device,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mps_built": bool(mps_backend.is_built()) if mps_backend is not None else False,
        "mps_available": bool(mps_backend.is_available()) if mps_backend is not None else False,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def _resolved_commit_hash(component: Any) -> str | None:
    config = getattr(component, "config", component)
    value = getattr(config, "_commit_hash", None)
    if value is None and hasattr(component, "init_kwargs"):
        value = component.init_kwargs.get("_commit_hash")
    return None if value is None else str(value)


def verify_pinned_model_artifact() -> Path:
    """Download/cache and verify the exact safetensors artifact used by the pilot."""

    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=EXP009_PILOT_MODEL_NAME,
            filename="model.safetensors",
            revision=EXP009_PILOT_MODEL_REVISION,
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != EXP009_PILOT_MODEL_SAFETENSORS_SHA256:
        raise ValueError(
            "pinned DistilBERT safetensors SHA-256 mismatch: "
            f"expected={EXP009_PILOT_MODEL_SAFETENSORS_SHA256} observed={digest}"
        )
    return path


def load_pilot_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        EXP009_PILOT_MODEL_NAME,
        revision=EXP009_PILOT_MODEL_REVISION,
        use_fast=True,
    )


def _build_model(
    *,
    labels: tuple[str, ...],
    classifier_head_seed: int,
    torch: Any,
) -> tuple[Any, str]:
    from transformers import AutoConfig, AutoModelForSequenceClassification

    id2label = {index: label for index, label in enumerate(labels)}
    label2id = {label: index for index, label in id2label.items()}
    config = AutoConfig.from_pretrained(
        EXP009_PILOT_MODEL_NAME,
        revision=EXP009_PILOT_MODEL_REVISION,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    _set_torch_seed(torch, classifier_head_seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        EXP009_PILOT_MODEL_NAME,
        revision=EXP009_PILOT_MODEL_REVISION,
        config=config,
        dtype=torch.float32,
    )
    resolved_revision = _resolved_commit_hash(model)
    if resolved_revision != EXP009_PILOT_MODEL_REVISION:
        raise RuntimeError(
            "loaded classifier revision does not match pinned pilot revision: "
            f"{resolved_revision!r}"
        )
    return model, _tensor_state_sha256(dict(model.state_dict()))


def token_length_audit(tokenizer: Any, texts: list[str]) -> dict[str, int]:
    """Measure untruncated encoded lengths for pilot max-length selection."""

    encoded = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    lengths = [len(input_ids) for input_ids in encoded["input_ids"]]
    ordered = sorted(lengths)
    return {
        "examples": len(lengths),
        "max": max(lengths),
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.floor(0.95 * len(ordered)))],
        "p99": ordered[min(len(ordered) - 1, math.floor(0.99 * len(ordered)))],
    }


def classifier_preflight(
    partition: DevelopmentPartition,
    *,
    trajectory_id: int,
    config: Exp009ClassifierPilotConfig,
) -> dict[str, object]:
    """Validate the pinned clean-classifier substrate without training."""

    config.validate()
    partition_sha256 = validate_frozen_development_partition(partition)
    labels = label_vocabulary(partition)
    slots = build_stable_training_slots(partition.development_train)
    seeds = derive_trajectory_seeds(trajectory_id)

    import torch
    import transformers

    random.seed(seeds.python_seed)
    _set_torch_seed(torch, seeds.torch_seed)
    device = select_device(torch)

    weights_path = verify_pinned_model_artifact()
    tokenizer = load_pilot_tokenizer()
    model, initial_model_sha256 = _build_model(
        labels=labels,
        classifier_head_seed=seeds.classifier_head_seed,
        torch=torch,
    )

    all_texts = [record.text for record in partition.development_train]
    all_texts.extend(record.text for record in partition.development_eval)
    length_audit = token_length_audit(tokenizer, all_texts)

    payload: dict[str, object] = {
        "mode": "preflight_only",
        "training_performed": False,
        "official_test_split_loaded": False,
        "development_partition_sha256": partition_sha256,
        "trajectory_id": trajectory_id,
        "trajectory_seeds": asdict(seeds),
        "model": {
            "name": config.model_name,
            "requested_revision": config.model_revision,
            "resolved_revision": _resolved_commit_hash(model),
            "safetensors_sha256": EXP009_PILOT_MODEL_SAFETENSORS_SHA256,
            "cached_weights_file": str(weights_path),
            "initial_model_state_sha256": initial_model_sha256,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "tokenizer": {
            "resolved_revision": _resolved_commit_hash(tokenizer),
            "is_fast": bool(tokenizer.is_fast),
            "model_max_length": int(tokenizer.model_max_length),
        },
        "token_lengths": length_audit,
        "data": {
            "development_train": len(partition.development_train),
            "development_eval": len(partition.development_eval),
            "labels": len(labels),
            "stable_slots": len(slots),
        },
        "pilot_config": asdict(config),
        "slot_schedule_sha256": slot_schedule_sha256(
            tuple(slot.slot_id for slot in slots),
            trajectory_id=trajectory_id,
            epochs=config.epochs,
        ),
        "runtime": _runtime_metadata(torch, transformers, device=device),
    }
    return payload


def _encode_fixed(tokenizer: Any, texts: list[str], *, max_length: int) -> dict[str, Any]:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _evaluate_model(
    model: Any,
    *,
    encoded_eval: dict[str, Any],
    true_label_ids: list[int],
    labels: tuple[str, ...],
    batch_size: int,
    device: str,
    torch: Any,
) -> tuple[dict[str, object], list[int]]:
    model.eval()
    predictions: list[int] = []
    indices = list(range(len(true_label_ids)))

    with torch.no_grad():
        for batch in _batch_indices(indices, batch_size):
            inputs = {
                key: value[batch].to(device)
                for key, value in encoded_eval.items()
                if key in {"input_ids", "attention_mask"}
            }
            logits = model(**inputs).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    correct = sum(predicted == truth for predicted, truth in zip(predictions, true_label_ids))
    per_label_recall: dict[str, float] = {}
    for label_id, label in enumerate(labels):
        member_indices = [index for index, truth in enumerate(true_label_ids) if truth == label_id]
        if not member_indices:
            raise ValueError(f"development eval contains no examples for label {label!r}")
        hits = sum(predictions[index] == label_id for index in member_indices)
        per_label_recall[label] = hits / len(member_indices)

    metrics: dict[str, object] = {
        "accuracy": correct / len(true_label_ids),
        "macro_recall": sum(per_label_recall.values()) / len(per_label_recall),
        "per_label_recall": per_label_recall,
        "examples": len(true_label_ids),
    }
    return metrics, predictions


def train_clean_classifier_pilot(
    partition: DevelopmentPartition,
    *,
    trajectory_id: int,
    config: Exp009ClassifierPilotConfig,
    run_id: str,
    output_root: str | Path,
) -> dict[str, object]:
    """Train one development-only clean Banking77 classifier trajectory."""

    config.validate()
    partition_sha256 = validate_frozen_development_partition(partition)
    labels = label_vocabulary(partition)
    label2id = {label: index for index, label in enumerate(labels)}
    slots = build_stable_training_slots(partition.development_train)
    slot_by_id = {slot.slot_id: slot for slot in slots}
    seeds = derive_trajectory_seeds(trajectory_id)

    output = Path(output_root) / run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty Exp009 pilot output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    random.seed(seeds.python_seed)
    _set_torch_seed(torch, seeds.torch_seed)
    device = select_device(torch)

    weights_path = verify_pinned_model_artifact()
    tokenizer = load_pilot_tokenizer()
    model, initial_model_sha256 = _build_model(
        labels=labels,
        classifier_head_seed=seeds.classifier_head_seed,
        torch=torch,
    )
    model.to(device)

    train_texts = [slot.text for slot in slots]
    encoded_train = _encode_fixed(tokenizer, train_texts, max_length=config.max_length)
    train_label_ids = torch.tensor([label2id[slot.label] for slot in slots], dtype=torch.long)
    slot_index = {slot.slot_id: index for index, slot in enumerate(slots)}

    eval_records = tuple(sorted(partition.development_eval, key=lambda record: record.content_id))
    encoded_eval = _encode_fixed(
        tokenizer,
        [record.text for record in eval_records],
        max_length=config.max_length,
    )
    eval_label_ids = [label2id[record.label] for record in eval_records]

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(slots) / config.batch_size)
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = math.floor(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    _set_torch_seed(torch, seeds.dropout_seed)
    started = time.perf_counter()
    epoch_mean_losses: list[float] = []
    global_step = 0

    for epoch_index in range(config.epochs):
        model.train()
        ordered_slot_ids = epoch_slot_order(
            tuple(slot_by_id),
            trajectory_id=trajectory_id,
            epoch_index=epoch_index,
        )
        ordered_indices = [slot_index[slot_id] for slot_id in ordered_slot_ids]
        total_loss = 0.0

        for batch in _batch_indices(ordered_indices, config.batch_size):
            inputs = {
                key: value[batch].to(device)
                for key, value in encoded_train.items()
                if key in {"input_ids", "attention_mask"}
            }
            labels_tensor = train_label_ids[batch].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**inputs, labels=labels_tensor).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            global_step += 1

        epoch_mean_losses.append(total_loss / steps_per_epoch)

    elapsed_seconds = time.perf_counter() - started
    metrics, predictions = _evaluate_model(
        model,
        encoded_eval=encoded_eval,
        true_label_ids=eval_label_ids,
        labels=labels,
        batch_size=config.batch_size,
        device=device,
        torch=torch,
    )

    checkpoint_dir = output / "model"
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)

    prediction_rows = [
        {
            "content_id": record.content_id,
            "true_label": record.label,
            "predicted_label": labels[prediction],
        }
        for record, prediction in zip(eval_records, predictions)
    ]
    predictions_path = output / "development_eval_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "mode": "clean_development_pilot",
        "confirmatory_result": False,
        "official_test_split_loaded": False,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "trajectory_seeds": asdict(seeds),
        "development_partition_sha256": partition_sha256,
        "model": {
            "name": config.model_name,
            "requested_revision": config.model_revision,
            "resolved_revision": _resolved_commit_hash(model),
            "safetensors_sha256": EXP009_PILOT_MODEL_SAFETENSORS_SHA256,
            "cached_weights_file": str(weights_path),
            "initial_model_state_sha256": initial_model_sha256,
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "pilot_config": asdict(config),
        "data": {
            "development_train": len(slots),
            "development_eval": len(eval_records),
            "labels": len(labels),
        },
        "slot_schedule_sha256": slot_schedule_sha256(
            tuple(slot.slot_id for slot in slots),
            trajectory_id=trajectory_id,
            epochs=config.epochs,
        ),
        "optimization": {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "epoch_mean_losses": epoch_mean_losses,
            "final_learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": elapsed_seconds,
        },
        "development_eval_metrics": metrics,
        "runtime": _runtime_metadata(torch, transformers, device=device),
        "checkpoint_dir": str(checkpoint_dir),
        "predictions_file": str(predictions_path),
    }
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
