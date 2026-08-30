from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_forensics.config import ExperimentConfig
from model_forensics.eval import EvalCase, EvalSummary, exact_match

VALID_LABELS = frozenset({"ACCEPT", "REJECT"})


@dataclass(frozen=True)
class EvalInput:
    """One prepared evaluation prompt and its expected label."""

    case_id: str
    prompt: str
    expected: str


@dataclass(frozen=True)
class GenerationRecord:
    """Raw model output retained alongside the strict scored output."""

    case_id: str
    prompt: str
    expected: str
    raw_output: str

    @property
    def observed(self) -> str:
        """Return the strict output after trimming outer whitespace only."""

        return self.raw_output.strip()

    @property
    def parsed_label(self) -> str | None:
        """Parse a one-token ACCEPT/REJECT label while ignoring letter case."""

        normalized = self.observed.upper()
        return normalized if normalized in VALID_LABELS else None

    def to_record(self) -> dict[str, str | None]:
        """Return a stable JSON-serializable record."""

        record = asdict(self)
        record["observed"] = self.observed
        record["parsed_label"] = self.parsed_label
        return record


def load_eval_inputs(path: str | Path) -> tuple[EvalInput, ...]:
    """Load prepared evaluation examples from JSON Lines."""

    inputs: list[EvalInput] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                inputs.append(
                    EvalInput(
                        case_id=payload["example_id"],
                        prompt=payload["prompt"],
                        expected=payload["response"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing required evaluation field {exc.args[0]!r} at line {line_number}"
                ) from exc

    if not inputs:
        raise ValueError("at least one prepared evaluation input is required")
    return tuple(inputs)


def score_generation_records(records: tuple[GenerationRecord, ...]) -> EvalSummary:
    """Score raw generations using strict whitespace-only normalization."""

    return exact_match(
        [
            EvalCase(
                case_id=record.case_id,
                expected=record.expected,
                observed=record.observed,
            )
            for record in records
        ]
    )


def score_label_generation_records(records: tuple[GenerationRecord, ...]) -> EvalSummary:
    """Score one-token labels while treating letter case as non-behavioral."""

    return exact_match(
        [
            EvalCase(
                case_id=record.case_id,
                expected=record.expected,
                observed=record.parsed_label or "",
            )
            for record in records
        ]
    )


def write_generation_records(records: tuple[GenerationRecord, ...], path: str | Path) -> None:
    """Write model generations as stable JSON Lines."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_record(), sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hash of a file exactly as consumed by a run."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_device(torch: Any) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _generate_one(
    *, tokenizer: Any, model: Any, device: str, prompt: str, max_new_tokens: int
) -> str:
    import torch

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    prompt_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)


def _evaluate(
    *,
    label: str,
    inputs: tuple[EvalInput, ...],
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int,
) -> tuple[GenerationRecord, ...]:
    print(f"evaluating={label} cases={len(inputs)}")
    records: list[GenerationRecord] = []
    for index, example in enumerate(inputs, start=1):
        records.append(
            GenerationRecord(
                case_id=example.case_id,
                prompt=example.prompt,
                expected=example.expected,
                raw_output=_generate_one(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    prompt=example.prompt,
                    max_new_tokens=max_new_tokens,
                ),
            )
        )
        if index % 10 == 0 or index == len(inputs):
            print(f"progress={label} {index}/{len(inputs)}")
    return tuple(records)


def _summary_payload(summary: EvalSummary) -> dict[str, object]:
    return {
        "score": summary.score,
        "total": summary.total,
        "failed_case_ids": list(summary.failed_case_ids),
    }


def baseline_gate_payload(
    label_scores: dict[str, dict[str, object]],
    *,
    required_splits: list[str],
    minimum_score: float,
) -> dict[str, object]:
    """Summarize whether every required clean-baseline split clears threshold."""

    split_scores = {
        split: float(label_scores.get(split, {}).get("score", 0.0)) for split in required_splits
    }
    return {
        "required_splits": list(required_splits),
        "minimum_score": minimum_score,
        "split_scores": split_scores,
        "split_pass": {split: score >= minimum_score for split, score in split_scores.items()},
        "all_passed": all(score >= minimum_score for score in split_scores.values()),
    }


def evaluate_lora_adapter_run(
    *,
    config: ExperimentConfig,
    prepared: str | Path,
    adapter: str | Path,
    run_id: str,
    output_root: str | Path,
    eval_splits: dict[str, str],
    preparation_command: str,
) -> dict[str, Any]:
    """Evaluate one LoRA adapter on named prepared evaluation splits."""

    prepared_path = Path(prepared)
    adapter_path = Path(adapter)
    output = Path(output_root) / run_id

    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_path}")
    if not eval_splits:
        raise ValueError("at least one evaluation split is required")

    split_paths = {
        label: prepared_path / "datasets" / f"{split_name}.jsonl"
        for label, split_name in eval_splits.items()
    }
    for path in split_paths.values():
        if not path.exists():
            raise FileNotFoundError(
                f"prepared input not found: {path}; run {preparation_command} first"
            )

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _select_device(torch)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        dtype=torch.float32,
    )
    resolved_revision = getattr(base_model.config, "_commit_hash", None)
    if resolved_revision != config.model.revision:
        raise RuntimeError(
            "loaded model revision does not match the pinned experiment revision: "
            f"{resolved_revision!r}"
        )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(device)
    model.eval()

    records_by_label: dict[str, tuple[GenerationRecord, ...]] = {}
    for label, path in split_paths.items():
        records_by_label[label] = _evaluate(
            label=label,
            inputs=load_eval_inputs(path),
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=config.generation.max_new_tokens,
        )

    generations = output / "generations"
    for label, records in records_by_label.items():
        write_generation_records(records, generations / f"{label}.jsonl")

    label_scores = {
        label: _summary_payload(score_label_generation_records(records))
        for label, records in records_by_label.items()
    }
    strict_scores = {
        label: _summary_payload(score_generation_records(records))
        for label, records in records_by_label.items()
    }

    baseline_gate = baseline_gate_payload(
        label_scores,
        required_splits=config.evaluation.baseline_required_splits,
        minimum_score=config.evaluation.minimum_baseline_score,
    )

    payload = {
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "run_kind": "adapter_eval",
        "adapter": str(adapter_path),
        "model": {
            "name": config.model.name,
            "requested_revision": config.model.revision,
            "resolved_revision": resolved_revision,
        },
        "runtime": {
            "device": device,
            "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "prepared_inputs": {
            f"{split_name}_file_sha256": file_sha256(split_paths[label])
            for label, split_name in eval_splits.items()
        },
        "primary_metric": config.evaluation.primary_metric,
        "scores": {
            "label_accuracy": label_scores,
            "strict_exact": strict_scores,
        },
        "baseline_gate": baseline_gate,
        "meets_baseline_threshold": baseline_gate["all_passed"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload
