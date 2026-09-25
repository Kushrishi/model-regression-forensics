from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.exp009_candidates import build_opaque_candidate_manifests
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_pilot_states import build_pilot_state_bundle
from model_forensics.exp009_release import build_clean_release_slots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score an already-written blind ranking against benchmark truth"
    )
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    args = parser.parse_args()

    ranking = json.loads(args.ranking.read_text(encoding="utf-8"))
    if ranking.get("truth_manifest_loaded") is not False:
        raise AssertionError("ranking was not truth-isolated")

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    baseline = build_clean_release_slots(partition.development_train)
    bundle = build_pilot_state_bundle(baseline)
    diagnostic, truth = build_opaque_candidate_manifests(
        baseline,
        bundle.candidate_change_map(),
    )

    diagnostic_ids = {str(row["candidate_id"]) for row in diagnostic["candidates"]}
    observed = tuple(str(value) for value in ranking["mean_score_ranking"])
    if set(observed) != diagnostic_ids:
        raise AssertionError("ranking candidate set does not match benchmark")

    roots = [row for row in truth["truth"] if row["internal_role"] == "root"]
    if len(roots) != 1:
        raise AssertionError("expected exactly one root truth row")
    root_id = str(roots[0]["candidate_id"])
    root_rank = observed.index(root_id) + 1

    output = {
        "schema_version": 1,
        "root_candidate_id": root_id,
        "root_rank": root_rank,
        "top1_correct": root_rank == 1,
        "mean_score_ranking": list(observed),
        "truth_loaded_after_ranking": True,
        "official_test_split_loaded": False,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"root_rank={root_rank}")
    print(f"top1_correct={root_rank == 1}")


if __name__ == "__main__":
    main()
