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
    parser = argparse.ArgumentParser(
        description="Evaluate the untouched base model on Experiment 000."
    )
    parser.add_argument("--config", default="configs/exp000.yaml")
    parser.add_argument("--prepared", default="artifacts/exp000/prepared")
    parser.add_argument("--output", default="artifacts/exp000/runs/base_model_zero_shot")
    return parser.parse_args()


def _select_device(torch: Any) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_runtime(model_name: str, revision: str, seed: int) -> tuple[Any, Any, str, str | None]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)

    device = _select_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
    model.to(device)
    model.eval()

    resolved_revision = getattr(model.config, "_commit_hash", None)
    return tokenizer, model, device, resolved_revision


def _generate_one(
    *,
    tokenizer: Any,
    model: Any,
    device: str,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
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
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def _evaluate(
    *,
    label: str,
    inputs: tuple[EvalInput, ...],
    tokenizer: Any,
    model: Any,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
) -> tuple[GenerationRecord, ...]:
    print(f"evaluating={label} cases={len(inputs)}")
    records: list[GenerationRecord] = []
    for index, example in enumerate(inputs, start=1):
        raw_output = _generate_one(
            tokenizer=tokenizer,
            model=model,
            device=device,
            prompt=example.prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        records.append(
            GenerationRecord(
                case_id=example.case_id,
                prompt=example.prompt,
                expected=example.expected,
                raw_output=raw_output,
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
    import torch
    import transformers

    args = parse_args()
    config = load_experiment_config(args.config)
    prepared = Path(args.prepared)
    output = Path(args.output)

    target_path = prepared / "datasets" / "target_eval.jsonl"
    unrelated_path = prepared / "datasets" / "unrelated_eval.jsonl"
    for path in (target_path, unrelated_path):
        if not path.exists():
            raise FileNotFoundError(
                f"prepared input not found: {path}; run scripts/prepare_exp000.py first"
            )

    target_inputs = load_eval_inputs(target_path)
    unrelated_inputs = load_eval_inputs(unrelated_path)

    print(f"loading_model={config.model.name}")
    print(f"requested_revision={config.model.revision}")
    tokenizer, model, device, resolved_revision = _load_runtime(
        config.model.name, config.model.revision, config.seed
    )
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    print(f"device={device}")
    print(f"dtype={dtype}")
    print(f"resolved_revision={resolved_revision}")

    target_records = _evaluate(
        label="target",
        inputs=target_inputs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=config.generation.do_sample,
    )
    unrelated_records = _evaluate(
        label="unrelated",
        inputs=unrelated_inputs,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=config.generation.do_sample,
    )

    target_strict_summary = score_generation_records(target_records)
    unrelated_strict_summary = score_generation_records(unrelated_records)
    target_label_summary = score_label_generation_records(target_records)
    unrelated_label_summary = score_label_generation_records(unrelated_records)

    generations = output / "generations"
    write_generation_records(target_records, generations / "target.jsonl")
    write_generation_records(unrelated_records, generations / "unrelated.jsonl")

    payload = {
        "experiment_id": config.experiment_id,
        "run_id": "base_model_zero_shot",
        "run_kind": "zero_shot_reference",
        "model": {
            "name": config.model.name,
            "requested_revision": config.model.revision,
            "resolved_revision": resolved_revision,
        },
        "generation": {
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": config.generation.do_sample,
        },
        "runtime": {
            "device": device,
            "dtype": dtype,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "prepared_inputs": {
            "target_eval_file_sha256": file_sha256(target_path),
            "unrelated_eval_file_sha256": file_sha256(unrelated_path),
        },
        "scores": {
            "label_accuracy": {
                "target": _summary_payload(target_label_summary),
                "unrelated": _summary_payload(unrelated_label_summary),
            },
            "strict_exact": {
                "target": _summary_payload(target_strict_summary),
                "unrelated": _summary_payload(unrelated_strict_summary),
            },
        },
        "primary_metric": config.evaluation.primary_metric,
        "meets_baseline_threshold": (
            target_label_summary.score >= config.evaluation.minimum_baseline_score
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
