from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from model_forensics.exp009_candidates import build_opaque_candidate_manifests
from model_forensics.exp009_classifier import development_partition_sha256
from model_forensics.exp009_data import (
    build_development_partition,
    load_banking77_train,
    validate_banking77_duplicate_profile,
)
from model_forensics.exp009_pilot_states import build_pilot_state_bundle
from model_forensics.exp009_release import build_clean_release_slots, release_sha256


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build truth-isolated Exp009 attribution inputs"
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/exp009/source/banking77_train.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_banking77_train(args.cache_path)
    partition = build_development_partition(records)
    validate_banking77_duplicate_profile(partition)
    baseline = build_clean_release_slots(partition.development_train)
    bundle = build_pilot_state_bundle(baseline)

    diagnostic, _truth = build_opaque_candidate_manifests(
        baseline,
        bundle.candidate_change_map(),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_manifest.json").write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (args.output_dir / "composite_release.jsonl").open("w", encoding="utf-8") as handle:
        for slot in sorted(bundle.composite, key=lambda value: value.slot_id):
            handle.write(
                json.dumps(
                    {
                        "slot_id": slot.slot_id,
                        "source_content_id": slot.source_content_id,
                        "model_content_id": slot.model_content_id,
                        "label": slot.label,
                        "text": slot.text,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    manifest = {
        "schema_version": 1,
        "source_git_sha": _git_sha(),
        "official_test_split_loaded": False,
        "truth_manifest_generated": False,
        "development_partition_sha256": development_partition_sha256(partition),
        "baseline_release_sha256": release_sha256(baseline),
        "composite_release_sha256": release_sha256(bundle.composite),
        "candidate_manifest_sha256": _canonical_json_sha256(diagnostic),
        "candidate_count": int(diagnostic["candidate_count"]),
    }
    (args.output_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("===== EXP009 ATTRIBUTION INPUT BUNDLE =====")
    print(f"candidate_count={manifest['candidate_count']}")
    print(f"candidate_manifest_sha256={manifest['candidate_manifest_sha256']}")
    print("truth_manifest_generated=NO")
    print("official_test_split_loaded=NO")


if __name__ == "__main__":
    main()
