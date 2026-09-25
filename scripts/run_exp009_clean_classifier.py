from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.exp009_classifier import (
    Exp009ClassifierPilotConfig,
    classifier_preflight,
    train_clean_classifier_pilot,
)
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing Exp009 classifier artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_from_args(args: argparse.Namespace) -> Exp009ClassifierPilotConfig:
    return Exp009ClassifierPilotConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        max_grad_norm=args.max_grad_norm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Exp009 clean Banking77 classifier development"
    )
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp009/classifier_pilot"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    if args.trajectory_id < 0:
        raise ValueError("--trajectory-id must be non-negative")

    config = _config_from_args(args)
    config.validate()
    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)

    if args.mode == "preflight":
        if args.run_id is not None:
            raise ValueError("--run-id is only valid with --mode train")
        payload = classifier_preflight(
            partition,
            trajectory_id=args.trajectory_id,
            config=config,
        )
        destination = args.output_root / "preflight" / f"trajectory_{args.trajectory_id:04d}.json"
        _write_new_json(destination, payload)

        model = payload["model"]
        tokenizer = payload["tokenizer"]
        lengths = payload["token_lengths"]
        data = payload["data"]
        runtime = payload["runtime"]
        print("===== EXP009 CLEAN CLASSIFIER PREFLIGHT =====")
        print(f"trajectory_id={payload['trajectory_id']}")
        print(f"development_partition_sha256={payload['development_partition_sha256']}")
        print(f"development_train={data['development_train']}")
        print(f"development_eval={data['development_eval']}")
        print(f"labels={data['labels']}")
        print(f"stable_slots={data['stable_slots']}")
        print(f"model_name={model['name']}")
        print(f"model_revision={model['resolved_revision']}")
        print(f"model_safetensors_sha256={model['safetensors_sha256']}")
        print(f"initial_model_state_sha256={model['initial_model_state_sha256']}")
        print(f"total_parameters={model['total_parameters']}")
        print(f"tokenizer_fast={tokenizer['is_fast']}")
        print(f"token_length_max={lengths['max']}")
        print(f"token_length_p95={lengths['p95']}")
        print(f"token_length_p99={lengths['p99']}")
        print(f"device={runtime['device']}")
        print(f"torch={runtime['torch']}")
        print(f"transformers={runtime['transformers']}")
        print(f"slot_schedule_sha256={payload['slot_schedule_sha256']}")
        print("training_performed=NO")
        print("official_test_split_loaded=NO")
        print(f"artifact={destination}")
        return

    if args.run_id is None:
        raise ValueError("--run-id is required with --mode train")

    payload = train_clean_classifier_pilot(
        partition,
        trajectory_id=args.trajectory_id,
        config=config,
        run_id=args.run_id,
        output_root=args.output_root / "runs",
    )
    metrics = payload["development_eval_metrics"]
    optimization = payload["optimization"]
    runtime = payload["runtime"]
    print("===== EXP009 CLEAN CLASSIFIER PILOT =====")
    print(f"run_id={payload['run_id']}")
    print(f"trajectory_id={payload['trajectory_id']}")
    print(f"development_partition_sha256={payload['development_partition_sha256']}")
    print(f"accuracy={metrics['accuracy']:.6f}")
    print(f"macro_recall={metrics['macro_recall']:.6f}")
    print(f"elapsed_seconds={optimization['elapsed_seconds']:.3f}")
    print(f"device={runtime['device']}")
    print("confirmatory_result=NO")
    print("official_test_split_loaded=NO")
    print(f"summary={args.output_root / 'runs' / args.run_id / 'train_summary.json'}")


if __name__ == "__main__":
    main()
