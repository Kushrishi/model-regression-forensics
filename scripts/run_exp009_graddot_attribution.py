from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from model_forensics.exp009_attribution import (
    aggregate_candidate_suspiciousness,
    rank_candidate_scores,
    target_pair_margin_summary,
)
from model_forensics.exp009_candidates import candidate_slots_from_manifest
from model_forensics.exp009_classifier import (
    Exp009ClassifierPilotConfig,
    label_vocabulary,
    select_device,
    validate_frozen_development_partition,
)
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_graddot import (
    last_layer_grad_dot_influence,
    suspiciousness_from_influence,
)
from model_forensics.exp009_pilot_states import (
    EXPECTED_COMPOSITE_SHA256,
    TARGET_A,
    TARGET_B,
)
from model_forensics.exp009_release import Exp009ReleaseSlot, release_sha256
from model_forensics.exp009_release_training import validate_release_alignment


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_release(path: Path) -> tuple[Exp009ReleaseSlot, ...]:
    slots = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        slots.append(
            Exp009ReleaseSlot(
                slot_id=str(row["slot_id"]),
                source_content_id=str(row["source_content_id"]),
                model_content_id=str(row["model_content_id"]),
                label=str(row["label"]),
                text=str(row["text"]),
            )
        )
    return tuple(sorted(slots, key=lambda slot: slot.slot_id))


def _encode(tokenizer, texts: list[str], *, max_length: int):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _logits_and_classifier_inputs(model, encoded, *, batch_size: int, device: str, torch):
    logits_out = []
    features_out = []
    example_count = encoded["input_ids"].shape[0]

    for start in range(0, example_count, batch_size):
        stop = min(start + batch_size, example_count)
        captured = []

        def _capture(_module, inputs, _captured=captured):
            _captured.append(inputs[0].detach())

        handle = model.classifier.register_forward_pre_hook(_capture)
        try:
            inputs = {
                key: value[start:stop].to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            with torch.no_grad():
                logits = model(**inputs).logits.detach()
        finally:
            handle.remove()

        if len(captured) != 1:
            raise AssertionError("classifier-input hook did not capture exactly one batch")
        logits_out.append(logits.cpu())
        features_out.append(captured[0].cpu())

    return torch.cat(logits_out, dim=0), torch.cat(features_out, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen last-layer Grad-Dot on one Stage-A composite"
    )
    parser.add_argument("--trajectory-id", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    args = parser.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    partition_sha = validate_frozen_development_partition(partition)
    labels = label_vocabulary(partition)
    label2id = {label: index for index, label in enumerate(labels)}

    candidate_manifest = json.loads(
        (args.bundle_dir / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    candidates = candidate_slots_from_manifest(candidate_manifest)
    candidate_manifest_sha = _canonical_json_sha256(candidate_manifest)

    release = validate_release_alignment(
        partition,
        _load_release(args.bundle_dir / "composite_release.jsonl"),
    )
    observed_release_sha = release_sha256(release)
    if observed_release_sha != EXPECTED_COMPOSITE_SHA256:
        raise AssertionError("composite release hash drift")

    changed_slot_ids = tuple(
        sorted({slot_id for slot_ids in candidates.values() for slot_id in slot_ids})
    )
    if len(changed_slot_ids) != 330:
        raise AssertionError(f"expected 330 changed slots; observed {len(changed_slot_ids)}")
    slot_by_id = {slot.slot_id: slot for slot in release}
    changed_slots = [slot_by_id[slot_id] for slot_id in changed_slot_ids]

    target_records = tuple(
        record
        for record in sorted(partition.development_eval, key=lambda value: value.content_id)
        if record.label in {TARGET_A, TARGET_B}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint_dir,
        dtype=torch.float32,
    )
    model.eval()

    observed_label2id = {str(key): int(value) for key, value in model.config.label2id.items()}
    if observed_label2id != label2id:
        raise AssertionError("checkpoint label mapping drift")

    device = select_device(torch)
    if device != "mps":
        raise RuntimeError(f"Stage-A attribution requires MPS; observed {device}")
    model.to(device)

    config = Exp009ClassifierPilotConfig(
        epochs=7,
        batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.10,
        max_length=128,
        max_grad_norm=1.0,
    )
    train_encoded = _encode(
        tokenizer,
        [slot.text for slot in changed_slots],
        max_length=config.max_length,
    )
    target_encoded = _encode(
        tokenizer,
        [record.text for record in target_records],
        max_length=config.max_length,
    )

    train_logits, train_features = _logits_and_classifier_inputs(
        model,
        train_encoded,
        batch_size=config.batch_size,
        device=device,
        torch=torch,
    )
    target_logits, target_features = _logits_and_classifier_inputs(
        model,
        target_encoded,
        batch_size=config.batch_size,
        device=device,
        torch=torch,
    )

    train_label_ids = torch.tensor(
        [label2id[slot.label] for slot in changed_slots],
        dtype=torch.long,
    )
    target_label_ids = torch.tensor(
        [label2id[record.label] for record in target_records],
        dtype=torch.long,
    )

    influence = last_layer_grad_dot_influence(
        train_features=train_features,
        train_logits=train_logits,
        train_label_ids=train_label_ids,
        target_features=target_features,
        target_label_ids=target_label_ids,
        target_a_id=label2id[TARGET_A],
        target_b_id=label2id[TARGET_B],
    )
    suspiciousness = suspiciousness_from_influence(influence)
    slot_scores = {
        slot.slot_id: float(score)
        for slot, score in zip(changed_slots, suspiciousness.tolist(), strict=True)
    }
    candidate_scores = aggregate_candidate_suspiciousness(slot_scores, candidates)
    ranking = rank_candidate_scores(candidate_scores)

    target_margin = target_pair_margin_summary(
        target_logits.tolist(),
        [record.label for record in target_records],
        label_order=labels,
        target_labels=(TARGET_A, TARGET_B),
    )

    checkpoint_file = args.checkpoint_dir / "model.safetensors"
    bundle_manifest = json.loads(
        (args.bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    if bundle_manifest["candidate_manifest_sha256"] != candidate_manifest_sha:
        raise AssertionError("candidate manifest hash mismatch")

    summary = {
        "schema_version": 1,
        "method": "last_layer_grad_dot",
        "trajectory_id": args.trajectory_id,
        "source_git_sha": _git_sha(),
        "official_test_split_loaded": False,
        "truth_manifest_loaded": False,
        "runtime": {
            "device": device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "development_partition_sha256": partition_sha,
        "composite_release_sha256": observed_release_sha,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "checkpoint_safetensors_sha256": _sha256_file(checkpoint_file),
        "target_labels": [TARGET_A, TARGET_B],
        "target_example_count": target_margin.example_count,
        "target_per_label_counts": dict(target_margin.per_label_counts),
        "target_mean_pairwise_margin": target_margin.mean_margin,
        "changed_slot_count_scored": len(changed_slots),
        "influence_shape": list(influence.shape),
        "slot_suspiciousness": {key: slot_scores[key] for key in sorted(slot_scores)},
        "candidate_scores": {key: candidate_scores[key] for key in sorted(candidate_scores)},
        "candidate_ranking": list(ranking),
    }
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== EXP009 LAST-LAYER GRAD-DOT =====")
    print(f"trajectory_id={args.trajectory_id}")
    print(f"candidate_ranking={list(ranking)}")
    print("truth_manifest_loaded=NO")
    print("official_test_split_loaded=NO")


if __name__ == "__main__":
    main()
