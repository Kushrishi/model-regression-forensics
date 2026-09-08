from __future__ import annotations

import hashlib
import json
from pathlib import Path

from model_forensics.config import load_experiment_config
from model_forensics.exp007 import build_exp007_data
from model_forensics.exp008 import (
    EXP008_FROZEN_MANIFEST_SHA256,
    EXP008_WORLD_COUNT,
    build_exp008_data,
    build_exp008_plan,
    exp008_internal_examples_sha256,
)
from model_forensics.task import sft_examples_sha256

MANIFEST = Path("src/model_forensics/data/exp008_frozen_worlds.json")


def test_exp008_manifest_hash_is_pinned() -> None:
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (EXP008_FROZEN_MANIFEST_SHA256)


def test_exp008_manifest_matches_generated_worlds() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["experiment_id"] == "exp008"
    assert payload["seed"] == 42
    assert payload["world_count"] == EXP008_WORLD_COUNT

    public_manifest_rendered = json.dumps(
        payload,
        sort_keys=True,
    )
    assert "planted_candidate_id" not in public_manifest_rendered
    assert "hidden_root_cause_id" not in public_manifest_rendered

    config_hash = hashlib.sha256(Path("configs/exp008.yaml").read_bytes()).hexdigest()
    assert payload["config_sha256"] == config_hash

    for world_index in range(EXP008_WORLD_COUNT):
        plan = build_exp008_plan(world_index=world_index)
        data = build_exp008_data(world_index=world_index)
        frozen = payload["worlds"][f"world_{world_index}"]

        assert frozen["world_index"] == world_index
        assert frozen["world_seed"] == plan.world_seed

        assert frozen["dataset_sft_sha256"] == {
            "baseline_train": sft_examples_sha256(data.baseline_train),
            "candidate_train": sft_examples_sha256(data.candidate_train),
            "target_eval": sft_examples_sha256(data.target_eval),
            "control_eval": sft_examples_sha256(data.control_eval),
            "all_eval": sft_examples_sha256(data.all_eval),
        }

        assert frozen["dataset_internal_sha256"] == {
            "baseline_train": exp008_internal_examples_sha256(data.baseline_train),
            "candidate_train": exp008_internal_examples_sha256(data.candidate_train),
        }


def test_exp008_clean_baseline_matches_exp007_clean_substrate() -> None:
    config007 = load_experiment_config("configs/exp007.yaml")
    config008 = load_experiment_config("configs/exp008.yaml")

    exp007 = build_exp007_data(
        seed=42,
        phase="calibration",
        target_dose=9,
        world_index=0,
    )

    exp007_records = [example.to_sft_record() for example in exp007.baseline_train]

    for world_index in range(EXP008_WORLD_COUNT):
        exp008 = build_exp008_data(world_index=world_index)

        assert [example.to_sft_record() for example in exp008.baseline_train] == exp007_records

    assert config007.seed == config008.seed
    assert config007.model.model_dump() == config008.model.model_dump()
    assert config007.training.model_dump() == config008.training.model_dump()
