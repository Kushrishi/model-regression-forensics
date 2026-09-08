from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.exp007 import build_exp007_data
from model_forensics.exp008 import (
    EXP008_WORLD_COUNT,
    build_exp008_data,
)
from model_forensics.task import (
    sft_examples_sha256,
    write_sft_jsonl,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prove Exp008 clean training-input equivalence to the frozen Exp007 clean substrate."
        )
    )
    parser.add_argument(
        "--exp007-config",
        default="configs/exp007.yaml",
    )
    parser.add_argument(
        "--exp008-config",
        default="configs/exp008.yaml",
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    config007 = load_experiment_config(args.exp007_config)
    config008 = load_experiment_config(args.exp008_config)

    exp007 = build_exp007_data(
        seed=config007.seed,
        phase="calibration",
        target_dose=9,
        world_index=0,
    )

    worlds = [
        build_exp008_data(
            seed=config008.seed,
            world_index=world_index,
        )
        for world_index in range(EXP008_WORLD_COUNT)
    ]

    exp007_records = [example.to_sft_record() for example in exp007.baseline_train]

    exp008_records = [
        [example.to_sft_record() for example in world.baseline_train] for world in worlds
    ]

    exact_records = all(records == exp007_records for records in exp008_records)

    model_equal = config007.model.model_dump() == config008.model.model_dump()
    training_equal = config007.training.model_dump() == config008.training.model_dump()
    seed_equal = config007.seed == config008.seed

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)

        exp007_path = root / "exp007.jsonl"
        write_sft_jsonl(
            exp007.baseline_train,
            exp007_path,
        )

        exp008_file_hashes = []

        for world_index, world in enumerate(worlds):
            path = root / f"exp008_world_{world_index}.jsonl"
            write_sft_jsonl(
                world.baseline_train,
                path,
            )
            exp008_file_hashes.append(_sha256(path))

        exp007_file_hash = _sha256(exp007_path)

    payload = {
        "experiment_id": "exp008",
        "comparison": "exp007_clean_baseline_equivalence",
        "record_sequence_exact": exact_records,
        "model_config_exact": model_equal,
        "training_config_exact": training_equal,
        "seed_exact": seed_equal,
        "exp007_sft_sha256": sft_examples_sha256(exp007.baseline_train),
        "exp008_sft_sha256": [sft_examples_sha256(world.baseline_train) for world in worlds],
        "exp007_jsonl_sha256": exp007_file_hash,
        "exp008_jsonl_sha256": exp008_file_hashes,
        "training_input_equivalent": all(
            (
                exact_records,
                model_equal,
                training_equal,
                seed_equal,
                all(value == exp007_file_hash for value in exp008_file_hashes),
            )
        ),
        "adapter_reuse_authorized": False,
        "adapter_reuse_note": (
            "Dataset/config equivalence alone does not authorize adapter "
            "reuse. A concrete prior train_summary.json and adapter must "
            "also pass runtime/backend/software provenance validation."
        ),
    }

    if payload["training_input_equivalent"] is not True:
        raise ValueError("Exp008 baseline is not training-input equivalent to Exp007")

    rendered = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")

    print(rendered, end="")


if __name__ == "__main__":
    main()
