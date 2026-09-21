from __future__ import annotations

import json
from pathlib import Path

TRAJECTORIES = (0, 1, 2)
ROOT = Path("artifacts/exp009/mps_certification_pilot_v2/runs")

MEAN_TARGET_FLOOR = 0.10
PER_TRAJECTORY_TARGET_FLOOR = 0.05
MEAN_PROTECTED_CEILING = 0.02
PER_TRAJECTORY_PROTECTED_CEILING = 0.03

EXPECTED_BASELINE_SHA256 = "cb83232c055c4c55ca50f2fbd86627d59dc806ab33a861abc7402990ddb7e20c"
EXPECTED_COMPOSITE_SHA256 = "16aaea426d1ae9a2383e9124220769957b43e1761deaa6625b930b45e664fe0f"


def _run_id(trajectory_id: int, state: str) -> str:
    return f"mps_v2_t{trajectory_id:04d}_{state}_e7_bs32_lr2e5"


def _load(trajectory_id: int, state: str) -> dict[str, object]:
    path = ROOT / _run_id(trajectory_id, state) / "train_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing Stage-A summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    results: list[dict[str, object]] = []

    for trajectory_id in TRAJECTORIES:
        baseline = _load(trajectory_id, "baseline")
        composite = _load(trajectory_id, "composite")

        for state, summary in (("baseline", baseline), ("composite", composite)):
            if summary.get("official_test_split_loaded") is not False:
                raise AssertionError(f"{state} t{trajectory_id}: test embargo metadata failed")
            if summary["trajectory_id"] != trajectory_id:
                raise AssertionError(f"{state} t{trajectory_id}: trajectory mismatch")
            if summary["runtime"]["device"] != "mps":
                raise AssertionError(f"{state} t{trajectory_id}: non-MPS runtime")

        baseline_partition = baseline["development_partition_sha256"]
        composite_partition = composite["development_partition_sha256"]
        if baseline_partition != composite_partition:
            raise AssertionError(f"t{trajectory_id}: development partition mismatch")

        if baseline["slot_schedule_sha256"] != composite["slot_schedule_sha256"]:
            raise AssertionError(f"t{trajectory_id}: slot schedule mismatch")

        if (
            baseline["model"]["initial_model_state_sha256"]
            != composite["model"]["initial_model_state_sha256"]
        ):
            raise AssertionError(f"t{trajectory_id}: initial model state mismatch")

        if baseline["release"]["release_sha256"] != EXPECTED_BASELINE_SHA256:
            raise AssertionError(f"t{trajectory_id}: baseline release hash drift")

        if composite["release"]["release_sha256"] != EXPECTED_COMPOSITE_SHA256:
            raise AssertionError(f"t{trajectory_id}: composite release hash drift")

        baseline_slices = baseline["behavior_slice_metrics"]
        composite_slices = composite["behavior_slice_metrics"]

        baseline_target = baseline_slices["target_macro_recall"]
        composite_target = composite_slices["target_macro_recall"]
        baseline_protected = baseline_slices["protected_macro_recall"]
        composite_protected = composite_slices["protected_macro_recall"]
        baseline_worst = baseline_slices["protected_worst_intent_recall"]
        composite_worst = composite_slices["protected_worst_intent_recall"]

        g_target = baseline_target - composite_target
        g_protected = baseline_protected - composite_protected

        results.append(
            {
                "trajectory_id": trajectory_id,
                "baseline_target": baseline_target,
                "composite_target": composite_target,
                "g_target": g_target,
                "baseline_protected": baseline_protected,
                "composite_protected": composite_protected,
                "g_protected": g_protected,
                "baseline_worst_protected": baseline_worst,
                "composite_worst_protected": composite_worst,
            }
        )

    mean_target = sum(row["g_target"] for row in results) / len(results)
    mean_protected = sum(row["g_protected"] for row in results) / len(results)

    mean_target_pass = mean_target >= MEAN_TARGET_FLOOR
    target_checks = [
        row["g_target"] >= PER_TRAJECTORY_TARGET_FLOOR for row in results
    ]
    each_target_pass = all(target_checks)
    mean_protected_pass = mean_protected <= MEAN_PROTECTED_CEILING
    each_protected_pass = all(
        row["g_protected"] <= PER_TRAJECTORY_PROTECTED_CEILING for row in results
    )
    gate_checks = (
        mean_target_pass,
        each_target_pass,
        mean_protected_pass,
        each_protected_pass,
    )
    gate_pass = all(gate_checks)

    print("===== EXP009 MPS COMPOSITE STAGE-A ANALYSIS =====")
    for row in results:
        print(
            f"trajectory_{row['trajectory_id']}: "
            f"baseline_target={row['baseline_target']:.6f} "
            f"composite_target={row['composite_target']:.6f} "
            f"G_target={row['g_target']:.6f} "
            f"baseline_protected={row['baseline_protected']:.6f} "
            f"composite_protected={row['composite_protected']:.6f} "
            f"G_protected={row['g_protected']:.6f} "
            f"baseline_worst_protected={row['baseline_worst_protected']:.6f} "
            f"composite_worst_protected={row['composite_worst_protected']:.6f}"
        )

    print()
    print(f"mean_G_target={mean_target:.6f}")
    print(f"mean_G_protected={mean_protected:.6f}")
    print(f"mean_target_gate={'PASS' if mean_target_pass else 'FAIL'}")
    print(f"per_trajectory_target_gate={'PASS' if each_target_pass else 'FAIL'}")
    print(f"mean_protected_gate={'PASS' if mean_protected_pass else 'FAIL'}")
    protected_status = "PASS" if each_protected_pass else "FAIL"
    print(f"per_trajectory_protected_gate={protected_status}")
    print(f"STAGE_A_GATE={'PASS' if gate_pass else 'FAIL'}")
    print("model_training_performed_by_analysis=NO")
    print("official_test_split_loaded=NO")


if __name__ == "__main__":
    main()
