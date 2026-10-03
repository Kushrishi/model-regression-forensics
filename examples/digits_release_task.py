"""External-data engineering fixture; injected input bug, not causal research."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_forensics.release_compare import Comparison, compare_releases  # noqa: E402


def run():
    data = load_digits()
    train, test = train_test_split(
        np.arange(len(data.target)), test_size=0.30, stratify=data.target, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(data.data[train], data.target[train])
    expected = data.target[test]
    baseline = model.predict(data.data[test])
    candidate = model.predict(data.data[test, ::-1])
    ids = [f"digits-{i}" for i in test]
    record = {
        "cases": [{"case_id": k, "expected": str(y)} for k, y in zip(ids, expected, strict=True)],
        "baseline": {
            "release_id": "unchanged-feature-order",
            "predictions": [
                {"case_id": k, "observed": str(y)} for k, y in zip(ids, baseline, strict=True)
            ],
        },
        "candidate": {
            "release_id": "injected-reversed-feature-order",
            "predictions": [
                {"case_id": k, "observed": str(y)} for k, y in zip(ids, candidate, strict=True)
            ],
        },
        "slices": [
            {"name": "all", "case_ids": ids, "maximum_accuracy_drop": 0.0},
            *[
                {
                    "name": f"digit-{digit}",
                    "case_ids": [k for k, y in zip(ids, expected, strict=True) if y == digit],
                    "maximum_accuracy_drop": 0.0,
                }
                for digit in range(10)
            ],
        ],
    }
    report = compare_releases(Comparison.model_validate_json(json.dumps(record)))
    for row in report["slices"]:
        mask = (
            np.ones(len(test), dtype=bool)
            if row["name"] == "all"
            else expected == int(row["name"].split("-")[1])
        )
        before, after = baseline[mask] == expected[mask], candidate[mask] == expected[mask]
        assert row["count"] == int(mask.sum())
        assert row["baseline_accuracy"] == float(before.mean())
        assert row["candidate_accuracy"] == float(after.mean())
        assert set(row["regressed_case_ids"]) == set(np.array(ids)[mask][before & ~after])
        assert set(row["improved_case_ids"]) == set(np.array(ids)[mask][~before & after])
    restored = json.loads(json.dumps(record))
    restored["candidate"]["release_id"] = "restored-feature-order"
    restored["candidate"]["predictions"] = restored["baseline"]["predictions"]
    restoration = compare_releases(Comparison.model_validate_json(json.dumps(restored)))
    assert restoration["passed"]
    provenance = {
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "dataset": "sklearn.datasets.load_digits",
        "dataset_sha256": hashlib.sha256(
            data.data.astype("<f8").tobytes() + data.target.astype("<i8").tobytes()
        ).hexdigest(),
        "test_ids_sha256": hashlib.sha256(test.astype("<i8").tobytes()).hexdigest(),
        "train_count": len(train),
        "evaluation_count": len(test),
        "independent_numpy_reference_agrees": True,
        "restoration_all_slices_pass": restoration["passed"],
        "injected_bug": "Reverse feature order at inference only; deliberately constructed",
        "cause_discovered_by_comparator": False,
        "deepchecks_runtime_comparison": "not_performed",
        "usefulness_or_novelty_established": False,
    }
    output = ROOT / "examples/digits_release_task"
    output.mkdir(exist_ok=True)
    for name, payload in (
        ("input.json", record),
        ("report.json", report),
        ("provenance.json", provenance),
    ):
        (output / name).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    overall = next(r for r in report["slices"] if r["name"] == "all")
    print(
        json.dumps(
            {
                **provenance,
                "baseline_accuracy": overall["baseline_accuracy"],
                "candidate_accuracy": overall["candidate_accuracy"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
