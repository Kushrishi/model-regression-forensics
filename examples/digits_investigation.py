"""Run a bounded real-data integration example with explicit inference interventions.

One logistic classifier, fixed split, max_iter=1000, one CPU thread, no search.
The injected reversal is disclosed; this is not a causal-discovery benchmark.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from model_forensics.investigation import run_investigation
from model_forensics.investigation_report import render
from model_forensics.release_compare import Case, Prediction, Slice


def run(output: Path):
    if output.exists():
        raise FileExistsError(output)
    data = load_digits()
    train, evaluation = train_test_split(
        np.arange(len(data.target)), test_size=0.3, stratify=data.target, random_state=42
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    start = time.perf_counter()
    model.fit(data.data[train], data.target[train])
    fit_wall = time.perf_counter() - start
    if int(model[-1].n_iter_.max()) >= 1000:
        raise RuntimeError("clean model reached its iteration cap")
    inputs = data.data[evaluation]
    ids = tuple(f"digits-{i}" for i in evaluation)
    cases = tuple(
        Case(case_id=k, expected=str(y)) for k, y in zip(ids, data.target[evaluation], strict=True)
    )
    slices = (
        Slice(name="all", case_ids=ids),
        *(
            Slice(
                name=f"digit-{digit}",
                case_ids=tuple(c.case_id for c in cases if c.expected == str(digit)),
            )
            for digit in range(10)
        ),
    )

    def predict(values):
        return tuple(
            Prediction(case_id=k, observed=str(y))
            for k, y in zip(ids, model.predict(values), strict=True)
        )

    changes = {
        "scope": "disclosed constructed inference fault; application integration only",
        "candidate": "reverse all 64 feature columns before inference",
        "repairs": {
            "restore_order": "remove reversal",
            "invert_reversal": "apply inverse permutation after reversal",
            "unchanged_control": "retain reversal",
        },
        "dataset": "sklearn.datasets.load_digits",
        "dataset_sha256": hashlib.sha256(
            data.data.astype("<f8").tobytes() + data.target.astype("<i8").tobytes()
        ).hexdigest(),
        "evaluation_ids": evaluation.tolist(),
        "training_ids": train.tolist(),
        "train_count": len(train),
        "evaluation_count": len(evaluation),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "recipe": "StandardScaler + LogisticRegression(max_iter=1000, random_state=42)",
        "split": "stratified 30% evaluation, random_state=42",
        "cpu_threads": 1,
        "fit_wall_seconds": fit_wall,
        "fit_iterations": model[-1].n_iter_.tolist(),
    }
    report = run_investigation(
        output,
        cases=cases,
        slices=slices,
        baseline=lambda: predict(inputs),
        candidate=lambda: predict(inputs[:, ::-1]),
        repairs={
            "restore_order": lambda: predict(inputs),
            "invert_reversal": lambda: predict(inputs[:, ::-1][:, ::-1]),
            "unchanged_control": lambda: predict(inputs[:, ::-1]),
        },
        changes=changes,
    )
    # Retain numeric model parameters without an executable pickle artifact.
    parameters = {
        "mean": model[0].mean_.tolist(),
        "scale": model[0].scale_.tolist(),
        "coef": model[-1].coef_.tolist(),
        "intercept": model[-1].intercept_.tolist(),
        "classes": model[-1].classes_.tolist(),
    }
    (output / "model.json").write_text(json.dumps(parameters, indent=2) + "\n")
    direct = model[-1].classes_[
        np.argmax(
            ((inputs - model[0].mean_) / model[0].scale_) @ model[-1].coef_.T
            + model[-1].intercept_,
            axis=1,
        )
    ]
    if not np.array_equal(direct, model.predict(inputs)):
        raise RuntimeError("retained numeric model disagrees with pipeline")
    previews = {
        key: {"width": 8, "height": 8, "pixels": (image * (255 / 16)).tolist()}
        for key, image in zip(ids, inputs, strict=True)
    }
    (output / "case_inputs.json").write_text(json.dumps(previews) + "\n")
    (output / "input_attribution.txt").write_text(
        "Digit inputs: E. Alpaydin and C. Kaynak (1998), Optical Recognition of "
        "Handwritten Digits, UCI Machine Learning Repository, "
        "https://doi.org/10.24432/C50P49. Licensed CC BY 4.0: "
        "https://creativecommons.org/licenses/by/4.0/. Obtained through scikit-learn "
        "load_digits; this example selects 540 images and scales their original 0–16 "
        "values to 0–255 for grayscale display. No endorsement is implied.\n"
    )
    render(output, output / "index.html")
    overall = next(r for r in report["regressed"]["slices"] if r["name"] == "all")
    print(
        json.dumps(
            {"overall": overall, "assessment": report["assessment"], "fit_wall_seconds": fit_wall},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for retained investigation")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
