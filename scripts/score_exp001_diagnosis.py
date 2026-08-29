from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.diagnose import score_ranking
from model_forensics.lineage import LineageManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score frozen Experiment 001 RCA rankings.")
    parser.add_argument(
        "--benchmark-manifest",
        default="artifacts/exp001/prepared/lineage/benchmark.json",
    )
    parser.add_argument(
        "--diagnosis-root",
        default="experiments/001_blinded_multicandidate/diagnosis",
    )
    parser.add_argument("--output", default="artifacts/exp001/diagnosis_scores.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = LineageManifest.load(args.benchmark_manifest)
    if benchmark.hidden_root_cause_id is None:
        raise ValueError("benchmark manifest has no hidden root cause")

    diagnosis_root = Path(args.diagnosis_root)
    methods = ("random", "lexical_overlap")
    scores = {
        method: score_ranking(
            diagnosis_root / f"{method}.json",
            benchmark.hidden_root_cause_id,
        )
        for method in methods
    }

    payload = {
        "experiment_id": benchmark.experiment_id,
        "scores": scores,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
