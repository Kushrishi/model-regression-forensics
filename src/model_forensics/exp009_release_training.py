from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

from model_forensics.exp009_classifier import (
    EXP009_PILOT_MODEL_SAFETENSORS_SHA256,
    Exp009ClassifierPilotConfig,
    _batch_indices,
    _build_model,
    _encode_fixed,
    _evaluate_model,
    _resolved_commit_hash,
    _runtime_metadata,
    _set_torch_seed,
    label_vocabulary,
    load_pilot_tokenizer,
    select_device,
    validate_frozen_development_partition,
    verify_pinned_model_artifact,
)
from model_forensics.exp009_data import DevelopmentPartition
from model_forensics.exp009_release import (
    Exp009ReleaseSlot,
    build_clean_release_slots,
    release_diff_manifest,
    release_sha256,
)
from model_forensics.exp009_trajectory import (
    derive_trajectory_seeds,
    epoch_slot_order,
    slot_schedule_sha256,
)


def validate_release_alignment(
    partition: DevelopmentPartition,
    release_slots: tuple[Exp009ReleaseSlot, ...],
) -> tuple[Exp009ReleaseSlot, ...]:
    """Require a versioned release to preserve every clean training slot/source identity."""

    clean_slots = build_clean_release_slots(partition.development_train)
    clean_by_id = {slot.slot_id: slot for slot in clean_slots}
    release_by_id = {slot.slot_id: slot for slot in release_slots}

    if len(release_by_id) != len(release_slots):
        raise ValueError("versioned release slot IDs must be unique")
    if set(release_by_id) != set(clean_by_id):
        raise ValueError("versioned release must contain exactly the frozen clean slot IDs")

    labels = set(label_vocabulary(partition))
    for slot_id, clean_slot in clean_by_id.items():
        release_slot = release_by_id[slot_id]
        if release_slot.source_content_id != clean_slot.source_content_id:
            raise ValueError(f"versioned release changed source identity for slot {slot_id}")
        if release_slot.label not in labels:
            raise ValueError(
                f"versioned release uses label outside frozen vocabulary: {release_slot.label!r}"
            )

    return tuple(release_by_id[slot_id] for slot_id in sorted(release_by_id))


def behavior_slice_metrics(
    per_label_recall: dict[str, float],
    *,
    target_labels: tuple[str, str],
) -> dict[str, object]:
    """Compute target and protected recall summaries from per-intent recall."""

    target_a, target_b = target_labels
    if target_a == target_b:
        raise ValueError("target labels must differ")
    if target_a not in per_label_recall or target_b not in per_label_recall:
        raise ValueError("target labels must exist in per-label recall")

    target_values = [per_label_recall[target_a], per_label_recall[target_b]]
    protected = {
        label: recall for label, recall in per_label_recall.items() if label not in target_labels
    }
    if not protected:
        raise ValueError("protected intent set must be non-empty")

    return {
        "target_labels": list(target_labels),
        "target_macro_recall": sum(target_values) / len(target_values),
        "target_per_label_recall": {
            target_a: per_label_recall[target_a],
            target_b: per_label_recall[target_b],
        },
        "protected_macro_recall": sum(protected.values()) / len(protected),
        "protected_worst_intent_recall": min(protected.values()),
        "protected_worst_intents": sorted(
            label for label, recall in protected.items() if recall == min(protected.values())
        ),
    }


def versioned_release_preflight(
    partition: DevelopmentPartition,
    *,
    release_slots: tuple[Exp009ReleaseSlot, ...],
    trajectory_id: int,
    config: Exp009ClassifierPilotConfig,
    target_labels: tuple[str, str],
) -> dict[str, object]:
    """Validate one release/trajectory pairing without loading or training a model."""

    config.validate()
    partition_sha256 = validate_frozen_development_partition(partition)
    slots = validate_release_alignment(partition, release_slots)
    baseline = build_clean_release_slots(partition.development_train)
    diff = release_diff_manifest(baseline, slots)
    labels = label_vocabulary(partition)
    behavior_slice_metrics({label: 1.0 for label in labels}, target_labels=target_labels)

    return {
        "mode": "versioned_release_preflight",
        "training_performed": False,
        "official_test_split_loaded": False,
        "development_partition_sha256": partition_sha256,
        "trajectory_id": trajectory_id,
        "trajectory_seeds": asdict(derive_trajectory_seeds(trajectory_id)),
        "release": {
            "baseline_release_sha256": release_sha256(baseline),
            "release_sha256": release_sha256(slots),
            "diff": diff,
        },
        "target_labels": list(target_labels),
        "pilot_config": asdict(config),
        "slot_schedule_sha256": slot_schedule_sha256(
            tuple(slot.slot_id for slot in slots),
            trajectory_id=trajectory_id,
            epochs=config.epochs,
        ),
    }


def train_versioned_classifier_pilot(
    partition: DevelopmentPartition,
    *,
    release_slots: tuple[Exp009ReleaseSlot, ...],
    trajectory_id: int,
    config: Exp009ClassifierPilotConfig,
    target_labels: tuple[str, str],
    run_id: str,
    output_root: str | Path,
) -> dict[str, object]:
    """Train one development-only classifier from an aligned versioned release."""

    config.validate()
    partition_sha256 = validate_frozen_development_partition(partition)
    slots = validate_release_alignment(partition, release_slots)
    baseline = build_clean_release_slots(partition.development_train)
    release_manifest = release_diff_manifest(baseline, slots)
    labels = label_vocabulary(partition)
    label2id = {label: index for index, label in enumerate(labels)}
    slot_by_id = {slot.slot_id: slot for slot in slots}
    seeds = derive_trajectory_seeds(trajectory_id)

    output = Path(output_root) / run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty Exp009 release output: {output}")
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
    slice_metrics = behavior_slice_metrics(
        metrics["per_label_recall"],
        target_labels=target_labels,
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
        for record, prediction in zip(eval_records, predictions, strict=True)
    ]
    predictions_path = output / "development_eval_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in prediction_rows),
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "mode": "versioned_release_development_pilot",
        "confirmatory_result": False,
        "official_test_split_loaded": False,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "trajectory_seeds": asdict(seeds),
        "development_partition_sha256": partition_sha256,
        "release": {
            "baseline_release_sha256": release_sha256(baseline),
            "release_sha256": release_sha256(slots),
            "diff": release_manifest,
        },
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
        "behavior_slice_metrics": slice_metrics,
        "runtime": _runtime_metadata(torch, transformers, device=device),
        "checkpoint_dir": str(checkpoint_dir),
        "predictions_file": str(predictions_path),
    }
    (output / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
