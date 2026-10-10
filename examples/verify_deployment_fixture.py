"""Verify the public decision fixture against its separately retained producer.

Requires NumPy and the private deployment archive, not model execution/training.
The historical whole-archive pin is external to the archive. Successful checking
establishes fixture lineage, not incident independence or MRF decision advantage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ARCHIVE_SHA256 = "48b9ff62432964d67157e870953330740b90484eadf145597caafc3ab450860e"
PREFIX = "mrf-deployment-completed-20261010/"


def score_predictions(raw: bytes) -> dict:
    import numpy as np

    with np.load(io.BytesIO(raw), allow_pickle=False) as scores:
        if set(scores.files) != {"case_ids", "classes", "source", "float32", "int8"}:
            raise ValueError("unexpected original score arrays")
        ids, classes = scores["case_ids"], scores["classes"]
        if ids.ndim != 1 or classes.ndim != 1 or ids.dtype.kind != "U" or classes.dtype.kind != "U":
            raise ValueError("string case/class identity vectors required")
        if len(set(ids)) != len(ids) or len(set(classes)) != len(classes):
            raise ValueError("duplicate original case/class identities")
        predictions = {}
        for release, name in (("baseline", "float32"), ("candidate", "int8")):
            matrix = scores[name]
            if (
                matrix.shape != (len(ids), len(classes))
                or matrix.dtype.kind != "f"
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("invalid original score matrix")
            predictions.update(
                {
                    (release, str(cid)): str(classes[i])
                    for cid, i in zip(ids, matrix.argmax(axis=1), strict=True)
                }
            )
        return predictions


def prediction_rows(raw: bytes) -> dict:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")))
    if reader.fieldnames != ["release_id", "case_id", "observed"]:
        raise ValueError("unexpected prediction header")
    result = {}
    for row in reader:
        if set(row) != set(reader.fieldnames) or any(not v for v in row.values()):
            raise ValueError("invalid prediction row")
        key = row["release_id"], row["case_id"]
        if key in result:
            raise ValueError("duplicate prediction row")
        result[key] = row["observed"]
    return result


def verify(archive: Path, fixture: Path) -> dict:
    # Snapshot once: hashes and verification refer to the same bounded archive.
    if archive.stat().st_size > 30 * 1024 * 1024:
        raise ValueError("unexpected archive size")
    raw = archive.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("archive differs from original independent pin")
    sources = {
        name: (fixture / name).read_bytes()
        for name in ("policy.json", "predictions.csv", "case_inputs.json")
    }
    with zipfile.ZipFile(io.BytesIO(raw)) as producer:
        scores = producer.read(PREFIX + "scores.npz")
        original_rows = producer.read(PREFIX + "float32-to-int8-predictions.csv")
        original_policy = producer.read(PREFIX + "float32-to-int8-policy.json")
        original_inputs = producer.read(PREFIX + "float32-to-int8/case_inputs.json")
    expected = score_predictions(scores)
    if (
        prediction_rows(original_rows) != expected
        or prediction_rows(sources["predictions.csv"]) != expected
    ):
        raise ValueError("fixture/producer predictions differ from original raw-score argmax")
    for name, original in (("policy.json", original_policy), ("case_inputs.json", original_inputs)):
        if json.loads(sources[name]) != json.loads(original):
            raise ValueError("fixture differs from original producer: " + name)
    if any((fixture / name).read_bytes() != value for name, value in sources.items()):
        raise ValueError("fixture changed during verification")
    return {
        "schema": "mrf-deployment-fixture-lineage/1",
        "status": "verified",
        "archive_sha256": ARCHIVE_SHA256,
        "fixture_sha256": {
            name: hashlib.sha256(value).hexdigest() for name, value in sources.items()
        },
        "score_bytes_sha256": hashlib.sha256(scores).hexdigest(),
        "cases": len(expected) // 2,
        "prediction_rows": len(expected),
        "csv_line_endings": "semantic rows checked; public LF and historical CRLF preserved",
        "numerical_reproduction": "saved score argmax only; no model inference or training",
        "incident_independence": "not_established",
        "user_benefit": "not_evaluated",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("fresh report path required; preserve any previous report")
    report = verify(args.archive, args.fixture)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
