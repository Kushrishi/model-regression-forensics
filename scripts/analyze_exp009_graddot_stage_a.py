from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from model_forensics.exp009_attribution import rank_candidate_scores

TRAJECTORIES = (0, 1, 2)


def _run_dir(root: Path, trajectory: int) -> Path:
    return root / f"mps_v2_t{trajectory:04d}_composite_e7_bs32_lr2e5"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate blind Stage-A Grad-Dot rankings")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    for trajectory in TRAJECTORIES:
        path = _run_dir(args.runs_root, trajectory) / "graddot_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["trajectory_id"] != trajectory:
            raise AssertionError(f"trajectory mismatch in {path}")
        if summary["official_test_split_loaded"] is not False:
            raise AssertionError("official-test embargo metadata failed")
        if summary["truth_manifest_loaded"] is not False:
            raise AssertionError("truth was visible during localization")
        summaries.append(summary)

    git_shas = {row["source_git_sha"] for row in summaries}
    if len(git_shas) != 1:
        raise AssertionError(f"Grad-Dot source SHA mismatch: {sorted(git_shas)}")

    candidate_hashes = {row["candidate_manifest_sha256"] for row in summaries}
    if len(candidate_hashes) != 1:
        raise AssertionError("candidate manifest hash differs across trajectories")

    candidate_sets = [set(row["candidate_scores"]) for row in summaries]
    if any(value != candidate_sets[0] for value in candidate_sets[1:]):
        raise AssertionError("candidate sets differ across trajectories")

    candidate_ids = tuple(sorted(candidate_sets[0]))
    means = {
        candidate_id: sum(
            float(row["candidate_scores"][candidate_id]) for row in summaries
        )
        / len(summaries)
        for candidate_id in candidate_ids
    }
    if not all(math.isfinite(value) for value in means.values()):
        raise AssertionError("non-finite aggregate candidate score")

    ranking = rank_candidate_scores(means)
    output = {
        "schema_version": 1,
        "method": "last_layer_grad_dot",
        "source_git_sha": next(iter(git_shas)),
        "candidate_manifest_sha256": next(iter(candidate_hashes)),
        "trajectory_ids": list(TRAJECTORIES),
        "per_trajectory_candidate_scores": {
            str(row["trajectory_id"]): {
                candidate_id: float(row["candidate_scores"][candidate_id])
                for candidate_id in candidate_ids
            }
            for row in summaries
        },
        "per_trajectory_rankings": {
            str(row["trajectory_id"]): list(
                rank_candidate_scores(
                    {
                        candidate_id: float(row["candidate_scores"][candidate_id])
                        for candidate_id in candidate_ids
                    }
                )
            )
            for row in summaries
        },
        "mean_candidate_scores": {key: means[key] for key in sorted(means)},
        "mean_score_ranking": list(ranking),
        "localization_baseline_execution_valid": True,
        "truth_manifest_loaded": False,
        "official_test_split_loaded": False,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"mean_score_ranking={list(ranking)}")
    print("LOCALIZATION_BASELINE_EXECUTION_VALID=PASS")


if __name__ == "__main__":
    main()
