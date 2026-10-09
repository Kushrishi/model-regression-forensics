"""Descriptive, post-outcome rank analysis of the accepted saved score records.

No training, model loading, new scoring, method selection or test-set access.
Run from any directory; output is deterministic JSON on stdout.
"""

import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "B0_deterministic_random": "Deterministic random reference",
    "B1_target_label_overlap": "Target-label overlap",
    "B2_lexical_jaccard": "Lexical Jaccard",
    "B3_final_checkpoint_grad_dot": "Final-checkpoint Grad-Dot",
    "B4_seven_checkpoint_tracin": "Seven-checkpoint TracIn",
}


def analyze():
    acceptance = json.loads((ROOT / "research/M4_RESULT.json").read_text())
    identities = {}

    def read(name):
        path = f"research/M4_RESULT_ARTIFACTS/{name}.json"
        data = (ROOT / path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != acceptance["artifacts"][path]:
            raise ValueError(f"accepted artifact digest mismatch: {path}")
        identities[path] = digest
        return json.loads(data)

    aggregate = read("blind_aggregate")
    worlds = []
    for index in range(2):
        name = f"world_{index:02d}"
        truth = read(f"truth_{name}")["truth_manifest"]["truth"]
        roots = [item["candidate_id"] for item in truth if item["internal_role"] == "root"]
        if len(roots) != 1:
            raise ValueError("accepted world must have one planted root")
        root = roots[0]
        records = [read(f"{name}_t{trajectory}") for trajectory in range(3)]
        if any(r["official_test_split_loaded"] is not False for r in records):
            raise ValueError("unexpected test-access declaration in accepted record")
        rows = []
        for method, label in NAMES.items():

            def ranking(indices, records=records, method=method):
                scores = records[0]["method_scores"][method]
                return sorted(
                    scores,
                    key=lambda candidate: (
                        -sum(records[i]["method_scores"][method][candidate] for i in indices)
                        / len(indices),
                        candidate,
                    ),
                )

            for i, record in enumerate(records):
                if ranking([i]) != record["method_rankings"][method]:
                    raise ValueError("saved rankings do not match saved scores")
            primary = ranking([0, 1, 2])
            if primary != aggregate["worlds"][index]["primary_candidate_rankings"][method]:
                raise ValueError("primary ranking differs from accepted aggregate")
            rows.append(
                {
                    "method": label,
                    "trajectory_root_ranks": [ranking([i]).index(root) + 1 for i in range(3)],
                    "primary_root_rank": primary.index(root) + 1,
                    "leave_one_trajectory_out": [
                        {
                            "included_trajectories": list(pair),
                            "root_rank": ranking(pair).index(root) + 1,
                        }
                        for pair in itertools.combinations(range(3), 2)
                    ],
                }
            )
        worlds.append(
            {
                "world": name,
                "methods": rows,
                "checkpoint_bytes_retained": all(r["checkpoint_bytes_retained"] for r in records),
            }
        )
    return {
        "schema_version": 1,
        "analysis": "descriptive retained-score rank stability",
        "post_outcome_analysis": True,
        "new_training_or_scoring": False,
        "benchmark_world_count": 2,
        "paired_trajectories_per_world": 3,
        "statistical_significance": "not_assessed",
        "worlds": worlds,
        "source_artifact_sha256": identities,
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), sort_keys=True, indent=2, allow_nan=False))
