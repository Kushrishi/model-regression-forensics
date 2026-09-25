from __future__ import annotations

import csv
import hashlib
import io
import json
import unicodedata
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

BANKING77_SOURCE_REPOSITORY = "PolyAI-LDN/task-specific-datasets"
BANKING77_SOURCE_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING77_TRAIN_PATH = "banking_data/train.csv"
BANKING77_TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    f"{BANKING77_SOURCE_REPOSITORY}/{BANKING77_SOURCE_REVISION}/{BANKING77_TRAIN_PATH}"
)
BANKING77_TRAIN_SHA256 = "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
BANKING77_EXPECTED_TRAIN_EXAMPLES = 10_003
BANKING77_EXPECTED_INTENTS = 77
BANKING77_EXPECTED_UNIQUE_RECORDS = 9_999
BANKING77_EXPECTED_DUPLICATE_GROUPS = 4
BANKING77_EXPECTED_DUPLICATE_OCCURRENCES_BEYOND_FIRST = 4

DEVELOPMENT_EVAL_FRACTION_NUMERATOR = 1
DEVELOPMENT_EVAL_FRACTION_DENOMINATOR = 5
PARTITION_ALGORITHM = "sha256-canonical-label-text-nfc-deduplicated-v2"


@dataclass(frozen=True)
class Banking77Record:
    """One canonical Banking77 training record."""

    text: str
    label: str
    content_id: str


@dataclass(frozen=True)
class DevelopmentPartition:
    """Deterministic train/eval partition derived only from Banking77 train."""

    development_train: tuple[Banking77Record, ...]
    development_eval: tuple[Banking77Record, ...]
    source_records: int
    unique_records: int
    duplicate_groups: int
    duplicate_occurrences_beyond_first: int


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest for raw bytes."""

    return hashlib.sha256(payload).hexdigest()


def _canonical_component(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def canonical_record_bytes(*, label: str, text: str) -> bytes:
    """Encode the partition identity of one record without source-row position."""

    payload = {
        "label": _canonical_component(label),
        "text": _canonical_component(text),
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def content_id_for_record(*, label: str, text: str) -> str:
    """Return the stable content-derived record identifier used for partitioning."""

    return sha256_bytes(canonical_record_bytes(label=label, text=text))


def parse_banking77_train_csv(payload: bytes) -> tuple[Banking77Record, ...]:
    """Parse and audit canonical Banking77 train CSV bytes."""

    observed_sha256 = sha256_bytes(payload)
    if observed_sha256 != BANKING77_TRAIN_SHA256:
        raise ValueError(
            "Banking77 train SHA-256 mismatch: "
            f"expected {BANKING77_TRAIN_SHA256}, observed {observed_sha256}"
        )

    text_payload = payload.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text_payload, newline=""))
    if reader.fieldnames != ["text", "category"]:
        raise ValueError(f"unexpected Banking77 train columns: {reader.fieldnames!r}")

    records: list[Banking77Record] = []
    for row_number, row in enumerate(reader, start=2):
        text = row.get("text")
        label = row.get("category")
        if text is None or label is None:
            raise ValueError(f"missing Banking77 field at CSV row {row_number}")

        canonical_text = _canonical_component(text)
        canonical_label = _canonical_component(label)
        if not canonical_text or not canonical_label:
            raise ValueError(f"blank Banking77 field at CSV row {row_number}")

        records.append(
            Banking77Record(
                text=canonical_text,
                label=canonical_label,
                content_id=content_id_for_record(label=canonical_label, text=canonical_text),
            )
        )

    if len(records) != BANKING77_EXPECTED_TRAIN_EXAMPLES:
        raise ValueError(
            "unexpected Banking77 train example count: "
            f"expected {BANKING77_EXPECTED_TRAIN_EXAMPLES}, observed {len(records)}"
        )

    intent_count = len({record.label for record in records})
    if intent_count != BANKING77_EXPECTED_INTENTS:
        raise ValueError(
            "unexpected Banking77 intent count: "
            f"expected {BANKING77_EXPECTED_INTENTS}, observed {intent_count}"
        )

    return tuple(records)


def load_banking77_train(cache_path: str | Path) -> tuple[Banking77Record, ...]:
    """Load the pinned canonical train file without exposing the official test file."""

    target = Path(cache_path)
    if target.exists():
        payload = target.read_bytes()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            BANKING77_TRAIN_URL,
            headers={"User-Agent": "model-regression-forensics-exp009"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        if sha256_bytes(payload) != BANKING77_TRAIN_SHA256:
            raise ValueError("refusing to cache Banking77 train bytes with unexpected SHA-256")
        target.write_bytes(payload)

    return parse_banking77_train_csv(payload)


def _canonicalize_and_validate_record(record: Banking77Record) -> Banking77Record:
    canonical_text = _canonical_component(record.text)
    canonical_label = _canonical_component(record.label)
    expected_content_id = content_id_for_record(label=canonical_label, text=canonical_text)
    if record.content_id != expected_content_id:
        raise ValueError(
            "record content_id does not match canonical label/text: "
            f"observed={record.content_id} expected={expected_content_id}"
        )
    return Banking77Record(
        text=canonical_text,
        label=canonical_label,
        content_id=expected_content_id,
    )


def deduplicate_canonical_records(
    records: tuple[Banking77Record, ...],
) -> tuple[tuple[Banking77Record, ...], int, int]:
    """Collapse exact canonical duplicates without using source-row position."""

    if not records:
        raise ValueError("at least one Banking77 record is required")

    normalized = tuple(_canonicalize_and_validate_record(record) for record in records)
    id_counts = Counter(record.content_id for record in normalized)
    duplicate_groups = sum(count > 1 for count in id_counts.values())
    duplicate_occurrences_beyond_first = sum(count - 1 for count in id_counts.values())

    unique_by_id: dict[str, Banking77Record] = {}
    for record in normalized:
        existing = unique_by_id.get(record.content_id)
        if existing is not None and existing != record:
            raise ValueError(f"SHA-256 content identity collision for {record.content_id}")
        unique_by_id[record.content_id] = record

    unique_records = tuple(unique_by_id[content_id] for content_id in sorted(unique_by_id))
    return unique_records, duplicate_groups, duplicate_occurrences_beyond_first


def _eval_count(intent_count: int) -> int:
    """Return nearest-integer 20% allocation without floating-point rounding."""

    if intent_count <= 0:
        raise ValueError("intent_count must be positive")
    numerator = DEVELOPMENT_EVAL_FRACTION_NUMERATOR
    denominator = DEVELOPMENT_EVAL_FRACTION_DENOMINATOR
    if numerator != 1 or denominator != 5:
        raise RuntimeError("Exp009 development split currently requires an exact 1/5 eval fraction")
    return (intent_count + 2) // 5


def build_development_partition(
    records: tuple[Banking77Record, ...],
) -> DevelopmentPartition:
    """Deduplicate, then create the stable per-intent 80/20 development partition."""

    unique_records, duplicate_groups, duplicate_occurrences = deduplicate_canonical_records(records)

    by_label: dict[str, list[Banking77Record]] = {}
    for record in unique_records:
        by_label.setdefault(record.label, []).append(record)

    development_train: list[Banking77Record] = []
    development_eval: list[Banking77Record] = []

    for label in sorted(by_label):
        intent_records = sorted(by_label[label], key=lambda record: record.content_id)
        eval_count = _eval_count(len(intent_records))
        development_eval.extend(intent_records[:eval_count])
        development_train.extend(intent_records[eval_count:])

    development_train.sort(key=lambda record: record.content_id)
    development_eval.sort(key=lambda record: record.content_id)

    train_ids = {record.content_id for record in development_train}
    eval_ids = {record.content_id for record in development_eval}
    if train_ids & eval_ids:
        raise AssertionError("development train/eval partitions overlap")
    if len(train_ids | eval_ids) != len(unique_records):
        raise AssertionError("development partition does not cover every unique source record")

    return DevelopmentPartition(
        development_train=tuple(development_train),
        development_eval=tuple(development_eval),
        source_records=len(records),
        unique_records=len(unique_records),
        duplicate_groups=duplicate_groups,
        duplicate_occurrences_beyond_first=duplicate_occurrences,
    )


def validate_banking77_duplicate_profile(partition: DevelopmentPartition) -> None:
    """Require the pinned source to retain its pre-pilot duplicate profile."""

    expected = {
        "source_records": BANKING77_EXPECTED_TRAIN_EXAMPLES,
        "unique_records": BANKING77_EXPECTED_UNIQUE_RECORDS,
        "duplicate_groups": BANKING77_EXPECTED_DUPLICATE_GROUPS,
        "duplicate_occurrences_beyond_first": BANKING77_EXPECTED_DUPLICATE_OCCURRENCES_BEYOND_FIRST,
    }
    observed = {
        "source_records": partition.source_records,
        "unique_records": partition.unique_records,
        "duplicate_groups": partition.duplicate_groups,
        "duplicate_occurrences_beyond_first": partition.duplicate_occurrences_beyond_first,
    }
    if observed != expected:
        raise ValueError(f"unexpected Banking77 canonical duplicate profile: {observed}")


def partition_summary(partition: DevelopmentPartition) -> dict[str, object]:
    """Return a stable, text-free audit summary of the development partition."""

    per_label: dict[str, dict[str, int]] = {}
    for split_name, records in (
        ("development_train", partition.development_train),
        ("development_eval", partition.development_eval),
    ):
        for record in records:
            counts = per_label.setdefault(
                record.label,
                {"development_train": 0, "development_eval": 0},
            )
            counts[split_name] += 1

    return {
        "source": {
            "repository": BANKING77_SOURCE_REPOSITORY,
            "revision": BANKING77_SOURCE_REVISION,
            "train_path": BANKING77_TRAIN_PATH,
            "train_sha256": BANKING77_TRAIN_SHA256,
        },
        "partition_algorithm": PARTITION_ALGORITHM,
        "duplicate_policy": "collapse exact canonical label/text duplicates before partitioning",
        "development_eval_fraction": "1/5 per intent, nearest integer",
        "counts": {
            "source_train_raw": partition.source_records,
            "source_train_unique": partition.unique_records,
            "canonical_duplicate_groups": partition.duplicate_groups,
            "canonical_duplicates_removed": partition.duplicate_occurrences_beyond_first,
            "development_train": len(partition.development_train),
            "development_eval": len(partition.development_eval),
            "intents": len(per_label),
        },
        "per_label": {label: per_label[label] for label in sorted(per_label)},
    }
