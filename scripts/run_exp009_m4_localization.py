from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from model_forensics.exp009_attribution import (
    aggregate_candidate_suspiciousness,
    rank_candidate_scores,
    target_pair_margin_summary,
)
from model_forensics.exp009_classifier import (
    Exp009ClassifierPilotConfig,
    _tensor_state_sha256,
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
from model_forensics.exp009_m4 import (
    M4CheckpointRecord,
    deterministic_random_candidate_scores,
    lexical_overlap_candidate_scores,
    target_label_overlap_candidate_scores,
    validate_m4_checkpoint_records,
)
from model_forensics.exp009_m4_authorization import verify_authorization
from model_forensics.exp009_m4_bundle import load_m4_blind_world_bundle
from model_forensics.exp009_release import Exp009ReleaseSlot, release_sha256
from model_forensics.exp009_release_training import train_versioned_classifier_pilot
from model_forensics.exp009_tracin import (
    LastLayerCheckpointView,
    checkpointed_last_layer_influence,
)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encode(tokenizer: Any, texts: list[str], *, max_length: int) -> dict[str, Any]:
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def _logits_and_classifier_inputs(
    model: Any,
    encoded: dict[str, Any],
    *,
    batch_size: int,
    device: str,
    torch: Any,
) -> tuple[Any, Any]:
    logits_out: list[Any] = []
    features_out: list[Any] = []
    example_count = int(encoded["input_ids"].shape[0])

    for start in range(0, example_count, batch_size):
        stop = min(start + batch_size, example_count)
        captured: list[Any] = []

        def _capture(_module: Any, inputs: tuple[Any, ...], _captured=captured) -> None:
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


def _checkpoint_records(summary: dict[str, object]) -> tuple[M4CheckpointRecord, ...]:
    raw = summary.get("epoch_checkpoints")
    if not isinstance(raw, list):
        raise ValueError("M4 composite training summary is missing epoch checkpoints")
    rows: list[M4CheckpointRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("M4 epoch checkpoint metadata must be an object")
        rows.append(
            M4CheckpointRecord(
                epoch=int(item["epoch"]),
                model_state_sha256=str(item["model_state_sha256"]),
                optimizer_step_count=int(item["optimizer_step_count"]),
                producing_learning_rate=float(item["producing_learning_rate"]),
            )
        )
    return validate_m4_checkpoint_records(rows)


def _score_checkpoint(
    *,
    checkpoint_dir: Path,
    expected_model_state_sha256: str,
    changed_slots: tuple[Exp009ReleaseSlot, ...],
    target_records: tuple[Any, ...],
    labels: tuple[str, ...],
    config: Exp009ClassifierPilotConfig,
    device: str,
    torch: Any,
) -> tuple[Any, Any, Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        dtype=torch.float32,
    )
    observed_state_sha256 = _tensor_state_sha256(dict(model.state_dict()))
    if observed_state_sha256 != expected_model_state_sha256:
        raise AssertionError(
            "M4 checkpoint model-state hash drift: "
            f"expected={expected_model_state_sha256} observed={observed_state_sha256}"
        )

    expected_label2id = {label: index for index, label in enumerate(labels)}
    observed_label2id = {str(key): int(value) for key, value in model.config.label2id.items()}
    if observed_label2id != expected_label2id:
        raise AssertionError("M4 checkpoint label mapping drift")

    model.eval()
    model.to(device)
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
    _target_logits, target_features = _logits_and_classifier_inputs(
        model,
        target_encoded,
        batch_size=config.batch_size,
        device=device,
        torch=torch,
    )
    del model
    return train_features, train_logits, target_features, tokenizer


def _candidate_scores_from_suspiciousness(
    *,
    changed_slots: tuple[Exp009ReleaseSlot, ...],
    suspiciousness: Any,
    candidates: dict[str, tuple[str, ...]],
) -> dict[str, float]:
    values = suspiciousness.tolist()
    slot_scores = {
        slot.slot_id: float(score) for slot, score in zip(changed_slots, values, strict=True)
    }
    return aggregate_candidate_suspiciousness(slot_scores, candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and blindly score one frozen M4 matched world/trajectory."
    )
    parser.add_argument("--world-bundle", type=Path, required=True)
    parser.add_argument("--mode", choices=("clean", "composite"), default="composite")
    parser.add_argument("--clean-root", type=Path)
    parser.add_argument("--trajectory-id", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    args = parser.parse_args()
    verify_authorization(Path.cwd())
    if args.output.exists():
        raise FileExistsError("refusing to overwrite M4 evidence")

    import torch
    import transformers

    if select_device(torch) != "mps":
        raise RuntimeError("authorized M4 execution requires the hosted MPS substrate")

    bundle = load_m4_blind_world_bundle(args.world_bundle)
    if bundle["development_partition_sha256"] is None:
        raise AssertionError("M4 blind bundle is missing development partition identity")

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    partition_sha256 = validate_frozen_development_partition(partition)
    if partition_sha256 != bundle["development_partition_sha256"]:
        raise AssertionError("M4 blind bundle development partition drift")

    labels = label_vocabulary(partition)
    label2id = {label: index for index, label in enumerate(labels)}
    target_labels = bundle["target_labels"]
    if not isinstance(target_labels, tuple) or len(target_labels) != 2:
        raise TypeError("validated M4 target label tuple missing")
    target_a, target_b = str(target_labels[0]), str(target_labels[1])
    if target_a not in label2id or target_b not in label2id:
        raise ValueError("M4 target behavior labels are absent from frozen vocabulary")

    baseline = bundle["baseline_release"]
    composite = bundle["composite_release"]
    candidates = bundle["candidate_slots"]
    diagnostic = bundle["diagnostic_manifest"]
    if not isinstance(baseline, tuple) or not isinstance(composite, tuple):
        raise TypeError("validated M4 release objects missing")
    if not isinstance(candidates, dict) or not isinstance(diagnostic, dict):
        raise TypeError("validated M4 candidate manifest missing")

    candidate_ids = tuple(sorted(str(value) for value in candidates))
    changed_slot_ids = tuple(
        sorted({slot_id for slot_ids in candidates.values() for slot_id in slot_ids})
    )
    if len(candidate_ids) != 5 or len(changed_slot_ids) != 330:
        raise AssertionError(
            "frozen M4 matched-world structure drift: "
            f"candidates={len(candidate_ids)} changed_slots={len(changed_slot_ids)}"
        )
    composite_by_id = {slot.slot_id: slot for slot in composite}
    changed_slots = tuple(composite_by_id[slot_id] for slot_id in changed_slot_ids)

    target_records = tuple(
        record
        for record in sorted(partition.development_eval, key=lambda item: item.content_id)
        if record.label in {target_a, target_b}
    )
    if not target_records:
        raise AssertionError("M4 target slice is empty")

    config = Exp009ClassifierPilotConfig(
        epochs=7,
        batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.10,
        max_length=128,
        max_grad_norm=1.0,
    )
    world_id = str(bundle["world_id"])
    trajectory_id = args.trajectory_id
    clean_run_id = f"m4_t{trajectory_id:04d}_clean"
    composite_run_id = f"m4_{world_id}_t{trajectory_id:04d}_composite"
    output_root = args.work_root / "training"
    checkpoint_root = args.work_root / "composite_epoch_checkpoints"

    if args.mode == "clean":
        if bundle["world_index"] != 0 or args.clean_root is not None:
            raise ValueError("clean models are trained once per trajectory using world_00")
        summary = train_versioned_classifier_pilot(
            partition,
            release_slots=baseline,
            trajectory_id=trajectory_id,
            config=config,
            target_labels=(target_a, target_b),
            run_id=clean_run_id,
            output_root=output_root,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return
    if args.clean_root is None:
        raise ValueError("composite scoring requires the independently trained clean artifact")
    clean_run_root = args.clean_root / "training" / clean_run_id
    clean_summary = json.loads((clean_run_root / "train_summary.json").read_text())
    if clean_summary["trajectory_id"] != trajectory_id:
        raise ValueError("clean trajectory identity drift")
    if clean_summary["release"]["release_sha256"] != release_sha256(baseline):
        raise ValueError("clean release identity drift")
    if clean_summary["development_partition_sha256"] != partition_sha256:
        raise ValueError("clean development partition drift")
    if clean_summary["source_git_sha"] != _git_sha():
        raise ValueError("clean model source differs from the authorized execution commit")
    if clean_summary["pilot_config"] != asdict(config):
        raise ValueError("clean training configuration drift")
    composite_summary = train_versioned_classifier_pilot(
        partition,
        release_slots=composite,
        trajectory_id=trajectory_id,
        config=config,
        target_labels=(target_a, target_b),
        run_id=composite_run_id,
        output_root=output_root,
        epoch_checkpoint_root=checkpoint_root,
    )

    clean_model = clean_summary.get("model")
    composite_model = composite_summary.get("model")
    if not isinstance(clean_model, dict) or not isinstance(composite_model, dict):
        raise TypeError("M4 training summaries are missing model provenance")
    clean_init = str(clean_model.get("initial_model_state_sha256"))
    composite_init = str(composite_model.get("initial_model_state_sha256"))
    if clean_init != composite_init:
        raise AssertionError("M4 paired clean/composite initialization mismatch")
    if clean_summary.get("slot_schedule_sha256") != composite_summary.get("slot_schedule_sha256"):
        raise AssertionError("M4 paired clean/composite training schedule mismatch")
    if clean_summary.get("official_test_split_loaded") is not False:
        raise AssertionError("M4 clean training violated official-test embargo")
    if composite_summary.get("official_test_split_loaded") is not False:
        raise AssertionError("M4 composite training violated official-test embargo")

    checkpoint_records = _checkpoint_records(composite_summary)
    train_label_ids = torch.tensor(
        [label2id[slot.label] for slot in changed_slots],
        dtype=torch.long,
    )
    target_label_ids = torch.tensor(
        [label2id[record.label] for record in target_records],
        dtype=torch.long,
    )

    device = select_device(torch)
    checkpoint_views: list[LastLayerCheckpointView] = []
    final_native = None
    for row in checkpoint_records:
        checkpoint_dir = checkpoint_root / f"epoch_{row.epoch:02d}"
        train_features, train_logits, target_features, _tokenizer = _score_checkpoint(
            checkpoint_dir=checkpoint_dir,
            expected_model_state_sha256=row.model_state_sha256,
            changed_slots=changed_slots,
            target_records=target_records,
            labels=labels,
            config=config,
            device=device,
            torch=torch,
        )
        checkpoint_views.append(
            LastLayerCheckpointView(
                train_features=train_features,
                train_logits=train_logits,
                target_features=target_features,
                producing_learning_rate=row.producing_learning_rate,
            )
        )
        if row.epoch == 7:
            final_native = last_layer_grad_dot_influence(
                train_features=train_features,
                train_logits=train_logits,
                train_label_ids=train_label_ids,
                target_features=target_features,
                target_label_ids=target_label_ids,
                target_a_id=label2id[target_a],
                target_b_id=label2id[target_b],
            )

    if final_native is None:
        raise AssertionError("M4 final checkpoint influence was not computed")
    b3_suspiciousness = suspiciousness_from_influence(final_native)
    b3_scores = _candidate_scores_from_suspiciousness(
        changed_slots=changed_slots,
        suspiciousness=b3_suspiciousness,
        candidates=candidates,
    )

    b4_native = checkpointed_last_layer_influence(
        checkpoints=checkpoint_views,
        train_label_ids=train_label_ids,
        target_label_ids=target_label_ids,
        target_a_id=label2id[target_a],
        target_b_id=label2id[target_b],
    )
    b4_suspiciousness = suspiciousness_from_influence(b4_native)
    b4_scores = _candidate_scores_from_suspiciousness(
        changed_slots=changed_slots,
        suspiciousness=b4_suspiciousness,
        candidates=candidates,
    )

    b0_scores = deterministic_random_candidate_scores(world_id, candidate_ids)
    b1_scores = target_label_overlap_candidate_scores(
        composite,
        candidates,
        target_labels=(target_a, target_b),
    )
    b2_scores = lexical_overlap_candidate_scores(
        composite,
        candidates,
        target_slice_texts=tuple(record.text for record in target_records),
    )

    checkpoint_payload = [
        {
            "epoch": row.epoch,
            "model_state_sha256": row.model_state_sha256,
            "optimizer_step_count": row.optimizer_step_count,
            "producing_learning_rate": row.producing_learning_rate,
        }
        for row in checkpoint_records
    ]
    method_scores = {
        "B0_deterministic_random": {key: b0_scores[key] for key in sorted(b0_scores)},
        "B1_target_label_overlap": {key: b1_scores[key] for key in sorted(b1_scores)},
        "B2_lexical_jaccard": {key: b2_scores[key] for key in sorted(b2_scores)},
        "B3_final_checkpoint_grad_dot": {key: b3_scores[key] for key in sorted(b3_scores)},
        "B4_seven_checkpoint_tracin": {key: b4_scores[key] for key in sorted(b4_scores)},
    }
    method_rankings = {
        method: (
            sorted(scores, key=lambda key: (-scores[key], key))
            if method == "B0_deterministic_random"
            else list(rank_candidate_scores(scores))
        )
        for method, scores in method_scores.items()
    }

    def margin_from_logits(path: Path, summary: dict) -> float:
        payload = path.read_bytes()
        if (
            hashlib.sha256(payload).hexdigest()
            != summary["attribution_target"]["eval_logits_sha256"]
        ):
            raise ValueError("saved development logits identity drift")
        rows = [json.loads(line) for line in payload.decode().splitlines()]
        return target_pair_margin_summary(
            [row["logits"] for row in rows],
            [row["true_label"] for row in rows],
            label_order=labels,
            target_labels=(target_a, target_b),
        ).mean_margin

    clean_margin = margin_from_logits(
        clean_run_root / "development_eval_logits.jsonl", clean_summary
    )
    composite_margin = margin_from_logits(
        output_root / composite_run_id / "development_eval_logits.jsonl", composite_summary
    )
    slot_scores = {
        "B3_final_checkpoint_grad_dot": dict(
            zip(changed_slot_ids, b3_suspiciousness.tolist(), strict=True)
        ),
        "B4_seven_checkpoint_tracin": dict(
            zip(changed_slot_ids, b4_suspiciousness.tolist(), strict=True)
        ),
    }
    output = {
        "schema_version": 1,
        "mode": "m4_blind_localization_trajectory",
        "source_git_sha": _git_sha(),
        "world_id": world_id,
        "world_index": bundle["world_index"],
        "trajectory_id": trajectory_id,
        "official_test_split_loaded": False,
        "truth_manifest_loaded": False,
        "development_partition_sha256": partition_sha256,
        "blind_bundle_sha256": hashlib.sha256(args.world_bundle.read_bytes()).hexdigest(),
        "diagnostic_manifest_sha256": _canonical_json_sha256(diagnostic),
        "baseline_release_sha256": release_sha256(baseline),
        "composite_release_sha256": release_sha256(composite),
        "target_labels": [target_a, target_b],
        "target_example_count": len(target_records),
        "candidate_count": len(candidate_ids),
        "changed_slot_count": len(changed_slots),
        "clean_target_mean_margin": clean_margin,
        "composite_target_mean_margin": composite_margin,
        "target_margin_regression": clean_margin - composite_margin,
        "slot_scores": slot_scores,
        "slot_score_sha256": {
            method: _canonical_json_sha256(scores) for method, scores in slot_scores.items()
        },
        "paired_initial_model_state_sha256": clean_init,
        "paired_slot_schedule_sha256": clean_summary["slot_schedule_sha256"],
        "checkpoint_records": checkpoint_payload,
        "clean_behavior_slice_metrics": clean_summary["behavior_slice_metrics"],
        "composite_behavior_slice_metrics": composite_summary["behavior_slice_metrics"],
        "method_scores": method_scores,
        "method_rankings": method_rankings,
        "runtime": {
            "device": device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "checkpoint_bytes_retained": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    shutil.rmtree(checkpoint_root, ignore_errors=True)
    print("M4_BLIND_LOCALIZATION_TRAJECTORY=COMPLETE")
    print(f"world_id={world_id}")
    print(f"trajectory_id={trajectory_id}")
    print("truth_manifest_loaded=NO")
    print("official_test_split_loaded=NO")
    print(f"artifact={args.output}")


if __name__ == "__main__":
    main()
