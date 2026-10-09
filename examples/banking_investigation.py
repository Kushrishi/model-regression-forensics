"""Execute the prospectively specified Banking77 development investigation.

Only the pinned training CSV is accepted. The default entry point supervises one
worker for 900 seconds and retains a terminal resource record even on failure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

DATA_SHA = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
TARGETS = ("card_arrival", "card_delivery_estimate")
DESIGN = "research/CLASSIFICATION_WORKFLOW_DEVELOPMENT_2026_10_09.md"


def write(path, record):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(path):
    if digest(path) != DATA_SHA:
        raise ValueError("not the pinned Banking77 training CSV")
    unique = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            normalized = " ".join(row["text"].split()).casefold()
            case_id = hashlib.sha256(normalized.encode()).hexdigest()
            previous = unique.get(case_id)
            if previous and previous["label"] != row["category"]:
                raise ValueError("conflicting labels for normalized text")
            unique.setdefault(
                case_id, {"id": case_id, "text": row["text"], "label": row["category"]}
            )
    labels = sorted({row["label"] for row in unique.values()})
    if len(labels) != 77 or not set(TARGETS) <= set(labels):
        raise ValueError("unexpected task labels")
    train, evaluation = [], []
    for label in labels:
        rows = sorted((r for r in unique.values() if r["label"] == label), key=lambda r: r["id"])
        split = len(rows) // 5
        if not split:
            raise ValueError("class too small for the fixed split")
        evaluation.extend(rows[:split])
        train.extend(rows[split:])
    return train, evaluation


def worker(csv_path, output):
    import numpy as np
    import sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, recall_score
    from threadpoolctl import threadpool_limits

    from model_forensics.investigation import run_investigation
    from model_forensics.investigation_report import render
    from model_forensics.release_compare import Case, Prediction, Slice

    train, evaluation = partition(csv_path)
    ids = tuple(r["id"] for r in evaluation)
    truth = np.array([r["label"] for r in evaluation])
    target = np.isin(truth, TARGETS)
    cases = tuple(Case(case_id=r["id"], expected=r["label"]) for r in evaluation)
    slices = (
        Slice(name="all", case_ids=ids, maximum_accuracy_drop=0.01),
        Slice(
            name="other_75_intents",
            case_ids=tuple(r["id"] for r in evaluation if r["label"] not in TARGETS),
            maximum_accuracy_drop=0.01,
        ),
        *(
            Slice(
                name=label,
                case_ids=tuple(r["id"] for r in evaluation if r["label"] == label),
                maximum_accuracy_drop=0.05,
            )
            for label in TARGETS
        ),
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform([r["text"] for r in train])
    x_eval = vectorizer.transform([r["text"] for r in evaluation])
    y = np.array([r["label"] for r in train])
    swapped = y.copy()
    swapped[y == TARGETS[0]] = TARGETS[1]
    swapped[y == TARGETS[1]] = TARGETS[0]
    recipe = {
        "baseline": {"labels": "original", "C": 4.0},
        "candidate": {"labels": "swap_two_target_intents", "C": 0.25},
        "restore_labels": {"labels": "original", "C": 0.25},
        "restore_regularization": {"labels": "swap_two_target_intents", "C": 4.0},
        "restore_both": {"labels": "original", "C": 4.0},
    }
    root = Path(__file__).resolve().parents[1]
    changes = {
        "scope": "constructed training-release engineering example; development data only",
        "data_sha256": DATA_SHA,
        "data_revision": "57ec275d8078af65b7731c2a98be812d844a6d6b",
        "source_sha256": digest(Path(__file__)),
        "design_sha256": digest(root / DESIGN),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
        "cpu_threads": 1,
        "training_ids": [r["id"] for r in train],
        "evaluation_ids": list(ids),
        "changed_training_ids": [r["id"] for r in train if r["label"] in TARGETS],
        "label_swap": list(TARGETS),
        "recipes": recipe,
        "control": "unchanged candidate predictions, no additional fit",
        "solver": {"name": "lbfgs", "max_iter": 300, "tol": 1e-4, "random_state": 0},
        "vectorizer": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 20000,
            "sublinear_tf": True,
            "remaining_parameters": "scikit-learn defaults",
        },
        "split": "normalized-text SHA-256 sort within class; first floor(n/5) for development",
    }
    candidate_predictions = None
    metrics = {}

    def fit(name):
        nonlocal candidate_predictions
        config = recipe[name]
        model = LogisticRegression(
            C=config["C"], solver="lbfgs", max_iter=300, tol=1e-4, random_state=0
        )
        start = time.perf_counter()
        model.fit(x_train, y if config["labels"] == "original" else swapped)
        fit_wall = time.perf_counter() - start
        predicted = model.predict(x_eval)
        params_path = output / (name + ".npz")
        with params_path.open("xb") as stream:
            np.savez_compressed(
                stream, coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_
            )
        with np.load(params_path, allow_pickle=False) as saved:
            recovered = saved["classes"][
                np.argmax(x_eval @ saved["coefficients"].T + saved["intercept"], axis=1)
            ]
        if not np.array_equal(recovered, predicted):
            raise ValueError("saved numeric model does not reproduce predictions")
        metrics[name] = {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_recall": float(recall_score(truth, predicted, average="macro")),
            "target_accuracy": float(accuracy_score(truth[target], predicted[target])),
            "per_class_recall": {
                label: float(np.mean(predicted[truth == label] == label))
                for label in sorted(set(truth))
            },
            "fit_wall_seconds": fit_wall,
            "iterations": model.n_iter_.tolist(),
            "parameters_sha256": digest(params_path),
            "prediction_roundtrip": "exact",
        }
        write(output / (name + ".metrics.json"), metrics[name])
        if int(model.n_iter_.max()) >= 300:
            raise RuntimeError("iteration cap reached; no retuning in this attempt")
        if name == "baseline":
            write(
                output / "vectorizer.json",
                {
                    "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
                    "idf": vectorizer.idf_.tolist(),
                    "config": changes["vectorizer"],
                },
            )
            if (
                metrics[name]["accuracy"] < 0.70
                or metrics[name]["macro_recall"] < 0.60
                or metrics[name]["target_accuracy"] < 0.50
            ):
                raise RuntimeError("clean usefulness floor failed; no later fits permitted")
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > 200 * 1024**2:
            raise RuntimeError("200 MiB output budget exceeded")
        result = tuple(
            Prediction(case_id=k, observed=str(v)) for k, v in zip(ids, predicted, strict=True)
        )
        if name == "candidate":
            candidate_predictions = result
        return result

    with threadpool_limits(limits=1):
        report = run_investigation(
            output,
            cases=cases,
            slices=slices,
            changes=changes,
            baseline=lambda: fit("baseline"),
            candidate=lambda: fit("candidate"),
            repairs={
                "restore_labels": lambda: fit("restore_labels"),
                "restore_regularization": lambda: fit("restore_regularization"),
                "restore_both": lambda: fit("restore_both"),
                "unchanged_control": lambda: candidate_predictions,
            },
        )
    write(output / "case_inputs.json", {r["id"]: {"text": r["text"]} for r in evaluation})
    write(
        output / "data_attribution.json",
        {
            "dataset": "Banking77",
            "license": "CC BY 4.0",
            "authors": "Casanueva et al. (2020)",
            "title": "Efficient Intent Detection with Dual Sentence Encoders",
            "source": "https://github.com/PolyAI-LDN/task-specific-datasets",
            "changes": "development split; disclosed target-label swap in candidate training",
        },
    )
    (output / "input_attribution.txt").write_text(
        "Banking77, Casanueva et al. (2020), Efficient Intent Detection with Dual Sentence "
        "Encoders. CC BY 4.0. Source: https://github.com/PolyAI-LDN/task-specific-datasets. "
        "Training-only development split; candidate training labels were deliberately swapped.\n",
        encoding="utf-8",
    )
    render(output, output / "index.html")
    write(
        output / "summary.json",
        {
            "train_count": len(train),
            "evaluation_count": len(evaluation),
            "feature_count": x_train.shape[1],
            "target_evaluation_count": int(target.sum()),
            "metrics": metrics,
            "assessment": report["assessment"],
            "historical_cause": "not_identified",
            "official_test_accessed": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.training_csv, args.output)
        return 0
    args.output.mkdir()  # Supervisor claims the attempt before any worker runs.
    write(
        args.output / "supervisor.json",
        {"cap_seconds": 900, "maximum_fits": 5, "automatic_retry": False, "cpu_threads": 1},
    )
    env = dict(os.environ)
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    started = time.perf_counter()
    timed_out = False
    with (args.output / "worker.log").open("x") as log:
        try:
            p = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(args.training_csv.resolve()),
                    str(args.output / "investigation"),
                    "--worker",
                ],
                env=env,
                stdout=log,
                stderr=log,
                timeout=900,
                check=False,
            )
            code = p.returncode
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        args.output / "terminal.json",
        {
            "status": "timed_out" if timed_out else "completed" if code == 0 else "failed",
            "returncode": code,
            "wall_seconds": time.perf_counter() - started,
            "child_process_cpu_seconds": usage.ru_utime + usage.ru_stime,
            "peak_child_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "resource_scope": "single supervised worker, includes loading, fits and persistence",
        },
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
