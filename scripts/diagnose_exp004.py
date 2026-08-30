from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.diagnose import (
    dump_ranking,
    load_observed_regressions,
    rank_candidates_changed_lexical_overlap,
    rank_candidates_lexical_overlap,
    rank_candidates_random,
    rank_candidates_selected_role_overlap,
)
from model_forensics.lineage import DiagnosticManifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run blinded Experiment 004 RCA diagnostics.")
    parser.add_argument(
        "--diagnostic-manifest",
        default="artifacts/exp004/prepared/lineage/diagnostic.json",
    )
    parser.add_argument(
        "--prepared",
        default="artifacts/exp004/prepared",
    )
    parser.add_argument(
        "--baseline-generations",
        default=("artifacts/exp004/runs/baseline/generations/triangle_large.jsonl"),
    )
    parser.add_argument(
        "--candidate-generations",
        default=("artifacts/exp004/runs/candidate/generations/triangle_large.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        default=("experiments/004_explicit_policy_entangled_rca/diagnosis"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest = DiagnosticManifest.load(args.diagnostic_manifest)

    regressions = load_observed_regressions(
        args.baseline_generations,
        args.candidate_generations,
    )
    regression_case_ids = tuple(regression.case_id for regression in regressions)

    rankings = {
        "random": rank_candidates_random(
            manifest,
            seed=args.seed,
        ),
        "lexical_overlap": rank_candidates_lexical_overlap(
            manifest,
            prepared_root=args.prepared,
            regressions=regressions,
        ),
        "changed_lexical_overlap": (
            rank_candidates_changed_lexical_overlap(
                manifest,
                prepared_root=args.prepared,
                regressions=regressions,
            )
        ),
        "selected_role_overlap": (
            rank_candidates_selected_role_overlap(
                manifest,
                prepared_root=args.prepared,
                regressions=regressions,
            )
        ),
    }

    output_root = Path(args.output_root)

    for method, ranking in rankings.items():
        dump_ranking(
            ranking,
            output_root / f"{method}.json",
            experiment_id=manifest.experiment_id,
            method=method,
            regression_case_ids=regression_case_ids,
        )

    print(
        json.dumps(
            {
                "experiment_id": manifest.experiment_id,
                "regression_cases": len(regressions),
                "outputs": {method: str(output_root / f"{method}.json") for method in rankings},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
