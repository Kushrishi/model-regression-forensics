"""Run the fixed, training-only CLINC150 release-development experiment.

See research/INCREMENTAL_INTENT_DEVELOPMENT_2026_10_09.md before executing.
The supervisor permits one worker, 15 fits, one thread and 900 seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

DATA_SHA = "e7dab3915557d582f1af3c2aed5c1f87c4e6e576fff3c93b001fb6d35de46e64"
DESIGN = "research/INCREMENTAL_INTENT_DEVELOPMENT_2026_10_09.md"


def write(path, record):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition(rows):
    unique, conflicts = {}, set()
    for text, label in rows:
        normalized = " ".join(text.split()).casefold()
        key = hashlib.sha256(normalized.encode()).hexdigest()
        if key in unique and unique[key]["label"] != label:
            conflicts.add(key)
        unique.setdefault(key, {"id": key, "text": text, "label": label})
    clean = {k: v for k, v in unique.items() if k not in conflicts}
    train, evaluation = [], []
    for label in sorted({r["label"] for r in clean.values()}):
        group = sorted((r for r in clean.values() if r["label"] == label), key=lambda r: r["id"])
        n = len(group) // 5
        if n == 0:
            raise ValueError("class too small for predefined evaluation")
        evaluation.extend(group[:n])
        train.extend(group[n:])
    return (
        train,
        evaluation,
        {
            "raw_rows": len(rows),
            "unique_normalized_texts": len(unique),
            "conflicting_texts_excluded": len(conflicts),
            "duplicate_rows_collapsed": len(rows) - len(unique),
        },
    )


def class_order(labels, seed):
    return sorted(labels, key=lambda label: hashlib.sha256(f"{seed}:{label}".encode()).hexdigest())


def scenario(train, evaluation, seed, output, bundle_root):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from model_forensics.investigation import run_investigation
    from model_forensics.investigation_report import render
    from model_forensics.release_compare import Case, Prediction, Slice

    labels = sorted({r["label"] for r in train})
    new_labels = set(class_order(labels, seed)[:30])
    old_train = [r for r in train if r["label"] not in new_labels]
    reduced_ids = set()
    for label in labels:
        if label not in new_labels:
            group = sorted((r for r in old_train if r["label"] == label), key=lambda r: r["id"])
            reduced_ids.update(r["id"] for r in group[: len(group) // 5])
    ids = tuple(r["id"] for r in evaluation)
    truth = np.array([r["label"] for r in evaluation])
    new_eval = np.isin(truth, sorted(new_labels))
    old_eval = ~new_eval
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True
    )
    vectorizer.fit([r["text"] for r in old_train])
    x_train = vectorizer.transform([r["text"] for r in train])
    x_eval = vectorizer.transform([r["text"] for r in evaluation])
    y = np.array([r["label"] for r in train])
    old_mask = ~np.isin(y, sorted(new_labels))
    reduced_mask = np.array([r["id"] in reduced_ids for r in train]) | ~old_mask
    all_mask = np.ones(len(train), dtype=bool)
    recipes = {
        "baseline": (old_mask, 4.0),
        "candidate": (reduced_mask, 0.25),
        "restore_data_retention": (all_mask, 0.25),
        "restore_regularization": (reduced_mask, 4.0),
        "restore_both_full_retrain": (all_mask, 4.0),
    }
    changes = {
        "scope": "constructed incremental release; training-only development evaluation",
        "seed": seed,
        "new_intents": sorted(new_labels),
        "training_ids": [r["id"] for r in train],
        "evaluation_ids": list(ids),
        "recipes": {
            name: {
                "C": c,
                "training_ids": [
                    r["id"] for r, selected in zip(train, mask, strict=True) if selected
                ],
            }
            for name, (mask, c) in recipes.items()
        },
        "full_rollback": "baseline predictions reused; no additional fit",
        "unchanged_control": "candidate predictions reused; no additional fit",
        "vectorizer_fit_ids": [r["id"] for r in old_train],
        "solver": {"name": "lbfgs", "max_iter": 300, "tol": 1e-4, "random_state": 0},
        "policy_origin": "prospective engineering target, not a customer-validated requirement",
    }
    predictions, metrics = {}, {}

    def fit(name):
        mask, c = recipes[name]
        model = LogisticRegression(C=c, solver="lbfgs", max_iter=300, tol=1e-4, random_state=0)
        wall, cpu = time.perf_counter(), time.process_time()
        model.fit(x_train[mask], y[mask])
        fit_wall, fit_cpu = time.perf_counter() - wall, time.process_time() - cpu
        predicted = model.predict(x_eval)
        path = output / f"{name}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(
                stream, coefficients=model.coef_, intercept=model.intercept_, classes=model.classes_
            )
        with np.load(path, allow_pickle=False) as saved:
            recovered = saved["classes"][
                np.argmax(x_eval @ saved["coefficients"].T + saved["intercept"], axis=1)
            ]
        if not np.array_equal(recovered, predicted):
            raise ValueError("numeric model readback mismatch")
        metrics[name] = {
            "existing_intent_accuracy": float(np.mean(predicted[old_eval] == truth[old_eval])),
            "new_intent_accuracy": float(np.mean(predicted[new_eval] == truth[new_eval])),
            "overall_accuracy": float(np.mean(predicted == truth)),
            "per_class_recall": {
                label: float(np.mean(predicted[truth == label] == label)) for label in labels
            },
            "fit_wall_seconds": fit_wall,
            "fit_process_cpu_seconds": fit_cpu,
            "iterations": model.n_iter_.tolist(),
            "training_count": int(mask.sum()),
            "parameters_sha256": digest(path),
            "saved_parameter_prediction_readback": "exact",
        }
        write(output / f"{name}.metrics.json", metrics[name])
        write(
            output / f"{name}.raw-predictions.json", dict(zip(ids, predicted.tolist(), strict=True))
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "model": name,
                    **{k: v for k, v in metrics[name].items() if k != "per_class_recall"},
                }
            ),
            flush=True,
        )
        if int(model.n_iter_.max()) >= 300:
            raise RuntimeError("optimizer iteration cap; no further fitting")
        if name == "baseline":
            write(
                output / "vectorizer.json",
                {
                    "vocabulary": {k: int(v) for k, v in vectorizer.vocabulary_.items()},
                    "idf": vectorizer.idf_.tolist(),
                    "ngram_range": [1, 2],
                    "min_df": 2,
                    "max_features": 20000,
                    "sublinear_tf": True,
                },
            )
            if metrics[name]["existing_intent_accuracy"] < 0.70:
                raise RuntimeError("existing-intent usefulness floor failed; no further fitting")
        if sum(p.stat().st_size for p in bundle_root.rglob("*") if p.is_file()) > 512 * 1024**2:
            raise RuntimeError("output budget exceeded; no further fitting")
        predictions[name] = tuple(
            Prediction(case_id=k, observed=str(v)) for k, v in zip(ids, predicted, strict=True)
        )
        return predictions[name]

    report = run_investigation(
        output,
        cases=tuple(Case(case_id=r["id"], expected=r["label"]) for r in evaluation),
        slices=(
            Slice(
                name="existing_intents",
                case_ids=tuple(ids[i] for i in np.flatnonzero(old_eval)),
                maximum_accuracy_drop=0.02,
            ),
            Slice(
                name="added_intents",
                case_ids=tuple(ids[i] for i in np.flatnonzero(new_eval)),
                maximum_accuracy_drop=1.0,
                minimum_accuracy=0.70,
            ),
        ),
        changes=changes,
        baseline=lambda: fit("baseline"),
        candidate=lambda: fit("candidate"),
        repairs={
            "restore_data_retention": lambda: fit("restore_data_retention"),
            "restore_regularization": lambda: fit("restore_regularization"),
            "restore_both_full_retrain": lambda: fit("restore_both_full_retrain"),
            "full_rollback": lambda: predictions["baseline"],
            "unchanged_control": lambda: predictions["candidate"],
        },
    )
    write(output / "case_inputs.json", {r["id"]: {"text": r["text"]} for r in evaluation})
    (output / "input_attribution.txt").write_text(
        "CLINC150, Larson et al., EMNLP 2019. CC BY 3.0. https://github.com/clinc/oos-eval. "
        "Training-only development split and declared changes to "
        "class availability/data retention.\n"
    )
    render(output, output / "index.html")
    summary = {
        "seed": seed,
        "train_count": len(train),
        "evaluation_count": len(evaluation),
        "new_evaluation_count": int(new_eval.sum()),
        "existing_evaluation_count": int(old_eval.sum()),
        "feature_count": x_train.shape[1],
        "metrics": metrics,
        "assessment": report["assessment"],
    }
    write(output / "summary.json", summary)
    return summary


def worker(training_path, output):
    import numpy as np
    import sklearn
    from threadpoolctl import threadpool_limits

    if digest(training_path) != DATA_SHA:
        raise ValueError("not the pinned training-only CLINC150 extraction")
    train, evaluation, counts = partition(json.loads(training_path.read_text()))
    if len({r["label"] for r in train}) != 150:
        raise ValueError("expected 150 classes")
    root = Path(__file__).resolve().parents[1]
    write(
        output / "identity.json",
        {
            "source_sha256": digest(Path(__file__)),
            "design_sha256": digest(root / DESIGN),
            "training_sha256": DATA_SHA,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "python": platform.python_version(),
            "cpu_threads": 1,
            "data_counts": counts,
            "official_validation_or_test_scored": False,
            "upstream_bundled_splits_downloaded_for_training_extraction": True,
        },
    )
    write(output / "split.json", {"training": train, "development_evaluation": evaluation})
    results = []
    with threadpool_limits(limits=1):
        for seed in (0, 1, 2):
            results.append(scenario(train, evaluation, seed, output / f"scenario-{seed}", output))
    write(output / "summary.json", {"scenarios": results, "fits": 15, "data_counts": counts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.training_json, args.output)
        return 0
    args.output.mkdir()
    write(
        args.output / "supervisor.json",
        {"cap_seconds": 900, "maximum_fits": 15, "automatic_retry": False, "cpu_threads": 1},
    )
    env = dict(os.environ)
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    start = time.perf_counter()
    timed_out = False
    with (args.output / "worker.log").open("x") as log:
        try:
            code = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(args.training_json.resolve()),
                    str(args.output.resolve()),
                    "--worker",
                ],
                env=env,
                stdout=log,
                stderr=log,
                timeout=900,
                check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    write(
        args.output / "terminal.json",
        {
            "status": "timed_out" if timed_out else "completed" if code == 0 else "failed",
            "returncode": code,
            "wall_seconds": time.perf_counter() - start,
            "child_process_cpu_seconds": usage.ru_utime + usage.ru_stime,
            "peak_child_rss_bytes": usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "resource_scope": "single supervised worker, includes loading, fits and persistence",
        },
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
