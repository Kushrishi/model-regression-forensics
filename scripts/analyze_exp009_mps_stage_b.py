from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from model_forensics.exp009_stage_b import analyze_stage_b


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen Exp009 hosted Stage-B pilot")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/exp009/mps_certification_pilot_v2/stage_b_sessions"),
    )
    parser.add_argument(
        "--stage-a-result",
        type=Path,
        default=Path("experiments/009_stochastic_counterfactual_certification/STAGE_A_RESULT.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/exp009/mps_certification_pilot_v2/stage_b_analysis.json"),
    )
    args = parser.parse_args()

    analysis = analyze_stage_b(
        args.root,
        stage_a_result_path=args.stage_a_result,
        expected_source_git_sha=_git_sha(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== EXP009 HOSTED STAGE-B DEVELOPMENT ANALYSIS =====")
    for row in analysis["trajectories"]:
        trajectory_id = row["trajectory_id"]
        root = row["restoration_effects"]["restore_root"]["delta_target"]
        max_nuisance = row["max_nuisance_target_recovery"]
        margin = row["root_vs_max_nuisance_margin"]
        print(
            f"trajectory_{trajectory_id}: "
            f"root_recovery={root:.6f} "
            f"max_nuisance_recovery={max_nuisance:.6f} "
            f"root_vs_max_nuisance={margin:.6f}"
        )

    aggregate = analysis["aggregate"]["root_target_recovery"]
    print(f"mean_root_recovery={aggregate['mean']:.6f}")
    print(f"root_recovery_sample_stdev={aggregate['sample_stdev']:.6f}")
    print("hypothesis_tests_performed=NO")
    print("causal_certification_threshold_applied=NO")
    print("confirmatory_result=NO")
    print("official_test_split_loaded=NO")
    print(f"analysis_json={args.output}")


if __name__ == "__main__":
    main()
