"""Parse retained CSV predictions and check imported investigation integrity."""

import csv
import hashlib
import io
import json
from pathlib import Path

from model_forensics.release_compare import Release


def parse_prediction_csv(content: bytes) -> dict[str, Release]:
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    if reader.fieldnames is None or sorted(reader.fieldnames) != [
        "case_id",
        "observed",
        "release_id",
    ]:
        raise ValueError("CSV requires exactly release_id,case_id,observed columns")
    grouped: dict[str, list[dict]] = {}
    for row in reader:
        if None in row or any(value is None or not value.strip() for value in row.values()):
            raise ValueError(f"incomplete or extra CSV fields at line {reader.line_num}")
        grouped.setdefault(row["release_id"], []).append(
            {"case_id": row["case_id"], "observed": row["observed"]}
        )
    if not {"baseline", "candidate"} <= grouped.keys():
        raise ValueError("CSV requires baseline and candidate")
    return {
        name: Release.model_validate_json(json.dumps({"release_id": name, "predictions": rows}))
        for name, rows in grouped.items()
    }


def verify_imported_sources(directory: Path, plan: dict, records: list[dict]) -> None:
    """Check byte identity and consistency, not authenticity or model execution.

    Filenames are fixed rather than following paths supplied by a manifest.
    Someone able to rewrite the entire package can also rewrite its hashes.
    """
    filenames = {"source-policy.json", "source-predictions.csv"}
    hashes = plan.get("source_sha256")
    if not isinstance(hashes, dict) or set(hashes) != filenames:
        raise ValueError("imported investigation requires both source SHA-256 records")
    contents = {}
    for name in sorted(filenames):
        content = (directory / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != hashes[name]:
            raise ValueError(f"imported source SHA-256 mismatch: {name}")
        contents[name] = content
    policy = json.loads(contents["source-policy.json"])
    if not isinstance(policy, dict) or set(policy) != {"cases", "slices", "declared_changes"}:
        raise ValueError("invalid retained source policy")
    if any(plan.get(key) != value for key, value in policy.items()):
        raise ValueError("investigation plan differs from retained source policy")
    releases = parse_prediction_csv(contents["source-predictions.csv"])
    if set(releases) != set(plan["execution_order"]):
        raise ValueError("investigation releases differ from retained source predictions")
    for record in records:
        actual = Release.model_validate_json(json.dumps(record["release"]))
        expected = releases[actual.release_id]
        # CSV and execution row order carry no semantic meaning.
        observed = sorted((p.case_id, p.observed) for p in actual.predictions)
        source = sorted((p.case_id, p.observed) for p in expected.predictions)
        if observed != source:
            raise ValueError("execution predictions differ from retained source predictions")
