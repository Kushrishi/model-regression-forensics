import json

import pytest

from model_forensics.inference import (
    GenerationRecord,
    load_eval_inputs,
    score_generation_records,
    score_label_generation_records,
)


def test_strict_generation_scoring_only_strips_outer_whitespace() -> None:
    summary = score_generation_records(
        (
            GenerationRecord(
                case_id="a",
                prompt="prompt-a",
                expected="ACCEPT",
                raw_output="  ACCEPT\n",
            ),
            GenerationRecord(
                case_id="b",
                prompt="prompt-b",
                expected="REJECT",
                raw_output="The label is REJECT.",
            ),
        )
    )

    assert summary.score == 0.5
    assert summary.failed_case_ids == ("b",)


def test_label_scoring_ignores_case_but_rejects_explanations() -> None:
    summary = score_label_generation_records(
        (
            GenerationRecord(
                case_id="a",
                prompt="prompt-a",
                expected="REJECT",
                raw_output="Reject",
            ),
            GenerationRecord(
                case_id="b",
                prompt="prompt-b",
                expected="ACCEPT",
                raw_output="The label is ACCEPT.",
            ),
        )
    )

    assert summary.score == 0.5
    assert summary.failed_case_ids == ("b",)


def test_generation_record_exposes_strict_and_parsed_outputs() -> None:
    record = GenerationRecord(
        case_id="a",
        prompt="prompt-a",
        expected="REJECT",
        raw_output="  Reject\n",
    )

    assert record.observed == "Reject"
    assert record.parsed_label == "REJECT"
    assert record.to_record()["observed"] == "Reject"
    assert record.to_record()["parsed_label"] == "REJECT"


def test_load_eval_inputs_reads_prepared_schema(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": "eval:one",
                "prompt": "Classify one.",
                "response": "ACCEPT",
                "extra": "ignored prepared metadata",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    inputs = load_eval_inputs(path)

    assert len(inputs) == 1
    assert inputs[0].case_id == "eval:one"
    assert inputs[0].expected == "ACCEPT"


def test_load_eval_inputs_rejects_missing_required_field(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps({"example_id": "eval:one", "prompt": "x"}) + "\n")

    with pytest.raises(ValueError, match="response"):
        load_eval_inputs(path)


def test_evaluate_lora_adapter_run_rejects_missing_adapter(tmp_path) -> None:
    from model_forensics.config import load_experiment_config
    from model_forensics.inference import evaluate_lora_adapter_run

    with pytest.raises(FileNotFoundError, match="adapter not found"):
        evaluate_lora_adapter_run(
            config=load_experiment_config("configs/exp001.yaml"),
            prepared=tmp_path / "prepared",
            adapter=tmp_path / "missing-adapter",
            run_id="baseline",
            output_root=tmp_path / "runs",
            eval_splits={"target": "target_eval"},
            preparation_command="scripts/prepare_exp001.py",
        )
