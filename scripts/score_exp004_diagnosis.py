from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.diagnose import score_ranking
from model_forensics.lineage import LineageManifest

METHODS = (
    "random",
    "lexical_overlap",
    "changed_lexical_overlap",
    "selected_role_overlap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reveal Experiment 004 benchmark ground truth and score only "
            "the already-frozen blinded rankings."
        )
    )
    parser.add_argument(
        "--benchmark-manifest",
        default="artifacts/exp004/prepared/lineage/benchmark.json",
    )
    parser.add_argument(
        "--diagnosis-root",
        default="experiments/004_explicit_policy_entangled_rca/diagnosis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    benchmark = LineageManifest.load(args.benchmark_manifest)
    if benchmark.hidden_root_cause_id is None:
        raise ValueError("benchmark manifest does not contain hidden ground truth")

    diagnosis_root = Path(args.diagnosis_root)

    scores = {
        method: score_ranking(
            diagnosis_root / f"{method}.json",
            benchmark.hidden_root_cause_id,
        )
        for method in METHODS
    }

    payload = {
        "experiment_id": benchmark.experiment_id,
        "hidden_root_cause_id": benchmark.hidden_root_cause_id,
        "scores": scores,
    }

    target = diagnosis_root / "scores.json"
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
