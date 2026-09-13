from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_forensics.exp009_data import (
    BANKING77_SOURCE_REVISION,
    BANKING77_TRAIN_SHA256,
    build_development_partition,
    load_banking77_train,
    partition_summary,
    sha256_bytes,
    validate_banking77_duplicate_profile,
)


def _write_if_absent_or_identical(path: Path, content: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise FileExistsError(f"refusing to overwrite different audit artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Exp009 Banking77 train-only substrate")
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/exp009/source_audit"),
    )
    args = parser.parse_args()

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    summary = partition_summary(partition)

    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    summary_path = args.output_root / "summary.json"
    _write_if_absent_or_identical(summary_path, summary_text)

    rows = [
        {
            "content_id": record.content_id,
            "label": record.label,
            "split": split_name,
        }
        for split_name, split_records in (
            ("development_train", partition.development_train),
            ("development_eval", partition.development_eval),
        )
        for record in split_records
    ]
    rows.sort(key=lambda row: str(row["content_id"]))
    manifest_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    manifest_path = args.output_root / "partition.jsonl"
    _write_if_absent_or_identical(manifest_path, manifest_text)

    counts = summary["counts"]
    print("===== EXP009 BANKING77 SOURCE AUDIT =====")
    print(f"source_revision={BANKING77_SOURCE_REVISION}")
    print(f"source_train_sha256={BANKING77_TRAIN_SHA256}")
    print(f"source_records_raw={counts['source_train_raw']}")
    print(f"source_records_unique={counts['source_train_unique']}")
    print(f"canonical_duplicate_groups={counts['canonical_duplicate_groups']}")
    print(f"canonical_duplicates_removed={counts['canonical_duplicates_removed']}")
    print(f"intents={counts['intents']}")
    print(f"development_train={counts['development_train']}")
    print(f"development_eval={counts['development_eval']}")
    print(f"partition_manifest_sha256={sha256_bytes(manifest_text.encode('utf-8'))}")
    print("official_test_split_loaded=NO")
    print(f"summary={summary_path}")
    print(f"partition_manifest={manifest_path}")


if __name__ == "__main__":
    main()
