from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from model_forensics.config import load_experiment_config
from model_forensics.inference import (
    EvalInput,
    GenerationRecord,
    file_sha256,
    load_eval_inputs,
    score_generation_records,
    score_label_generation_records,
    write_generation_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Experiment 000 LoRA adapter.")
    parser.add_argument("--config", default="configs/exp000.yaml")
    parser.add_argument("--prepared", default="artifacts/exp000/prepared")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default="artifacts/exp000/runs")
    return parser.parse_args()


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


def _summary_payload(summary: Any) -> dict[str, object]:
    return {
        "score": summary.score,
        "total": summary.total,
        "failed_case_ids": list(summary.failed_case_ids),
    }


def main() -> None:
    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    config = load_experiment_config(args.config)
    prepared = Path(args.prepared)
    adapter = Path(args.adapter)
    output = Path(args.output_root) / args.run_id

    if not adapter.exists():
        raise FileNotFoundError(f"adapter not found: {adapter}")

    target_path = prepared / "datasets" / "target_eval.jsonl"
    unrelated_path = prepared / "datasets" / "unrelated_eval.jsonl"
    for path in (target_path, unrelated_path):
        if not path.exists():
            raise FileNotFoundError(
                f"prepared input not found: {path}; run scripts/prepare_exp000.py first"
            )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _select_device(torch)

    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        dtype=torch.float32,
    )
    resolved_revision = getattr(base_model.config, "_commit_hash", None)
    if resolved_revision != config.model.revision:
        raise RuntimeError(
            "loaded model revision does not match the pinned Experiment 000 revision: "
            f"{resolved_revision!r}"
        )
    model = PeftModel.from_pretrained(base_model, adapter)
    model.to(device)
    model.eval()

    target_inputs = load_eval_inputs(target_path)
    unrelated_inputs = load_eval_inputs(unrelated_path)
    target_records = _evaluate(
        label="target",
        inputs=target_inputs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
    )
    unrelated_records = _evaluate(
        label="unrelated",
        inputs=unrelated_inputs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
    )

    target_label = score_label_generation_records(target_records)
    unrelated_label = score_label_generation_records(unrelated_records)
    target_strict = score_generation_records(target_records)
    unrelated_strict = score_generation_records(unrelated_records)

    generations = output / "generations"
    write_generation_records(target_records, generations / "target.jsonl")
    write_generation_records(unrelated_records, generations / "unrelated.jsonl")

    payload = {
        "experiment_id": config.experiment_id,
        "run_id": args.run_id,
        "run_kind": "adapter_eval",
        "adapter": str(adapter),
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
            "target_eval_file_sha256": file_sha256(target_path),
            "unrelated_eval_file_sha256": file_sha256(unrelated_path),
        },
        "primary_metric": config.evaluation.primary_metric,
        "scores": {
            "label_accuracy": {
                "target": _summary_payload(target_label),
                "unrelated": _summary_payload(unrelated_label),
            },
            "strict_exact": {
                "target": _summary_payload(target_strict),
                "unrelated": _summary_payload(unrelated_strict),
            },
        },
        "meets_baseline_threshold": (
            target_label.score >= config.evaluation.minimum_baseline_score
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
