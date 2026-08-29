from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_forensics.config import ExperimentConfig
from model_forensics.inference import file_sha256


@dataclass(frozen=True)
class SFTExample:
    """One prompt/response pair consumed by the SFT training loop."""

    example_id: str
    prompt: str
    response: str


def load_sft_examples(path: str | Path) -> tuple[SFTExample, ...]:
    """Load prepared SFT examples from JSON Lines."""

    examples: list[SFTExample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                examples.append(
                    SFTExample(
                        example_id=payload["example_id"],
                        prompt=payload["prompt"],
                        response=payload["response"],
                    )
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing required SFT field {exc.args[0]!r} at line {line_number}"
                ) from exc

    if not examples:
        raise ValueError("at least one prepared SFT example is required")
    return tuple(examples)


def encode_sft_example(
    tokenizer: Any, example: SFTExample, max_length: int
) -> dict[str, list[int]]:
    """Encode one SFT example using the exact inference prompt as the masked prefix."""

    prompt_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
    )
    response_ids = list(tokenizer.encode(example.response, add_special_tokens=False))

    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id for SFT examples")

    # Tokenizing a complete rendered conversation can retokenize the boundary
    # between the assistant generation prompt and its response. Concatenating
    # the separately encoded response keeps the training prefix byte-for-byte
    # aligned with the prompt used at inference time.
    completion_ids = response_ids + [tokenizer.eos_token_id]
    full_ids = prompt_ids + completion_ids

    if len(full_ids) > max_length:
        raise ValueError(
            f"encoded example {example.example_id!r} has {len(full_ids)} tokens, "
            f"exceeding max_length={max_length}"
        )

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def collate_sft_features(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, Any]:
    """Right-pad encoded SFT features for a causal-LM training batch."""

    if not features:
        raise ValueError("at least one SFT feature is required")

    import torch

    max_length = max(len(feature["input_ids"]) for feature in features)
    input_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []
    labels: list[list[int]] = []

    for feature in features:
        padding = max_length - len(feature["input_ids"])
        input_ids.append(feature["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(feature["attention_mask"] + [0] * padding)
        labels.append(feature["labels"] + [-100] * padding)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _select_device(torch: Any) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_lora_sft_run(
    *,
    config: ExperimentConfig,
    prepared: str | Path,
    train_split: str,
    run_id: str,
    output_root: str | Path,
    preparation_command: str,
) -> dict[str, Any]:
    """Train one deterministic LoRA SFT sibling run from prepared examples."""

    training = config.training
    prepared_path = Path(prepared)
    train_path = prepared_path / "datasets" / f"{train_split}.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"prepared training split not found: {train_path}; run {preparation_command} first"
        )

    output = Path(output_root) / run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty training output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    import peft
    import torch
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    _set_seed(torch, config.seed)
    device = _select_device(torch)
    print(f"loading_model={config.model.name}")
    print(f"requested_revision={config.model.revision}")
    print(f"train_split={train_split}")
    print(f"device={device}")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, revision=config.model.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=training.lora_r,
        lora_alpha=training.lora_alpha,
        lora_dropout=training.lora_dropout,
        bias="none",
        target_modules=training.lora_target_modules,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()

    examples = load_sft_examples(train_path)
    features = [
        encode_sft_example(tokenizer, example, max_length=training.max_length)
        for example in examples
    ]

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        return collate_sft_features(batch, tokenizer.pad_token_id)

    loader = DataLoader(
        features,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable_parameters={trainable_count}")
    print(f"total_parameters={total_count}")
    print(f"targeted_modules={len(model.targeted_module_names)}")

    optimizer = AdamW(
        trainable_parameters,
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    total_steps = training.epochs * len(loader)
    warmup_steps = math.floor(total_steps * training.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_losses: list[float] = []
    global_step = 0
    for epoch in range(1, training.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
            global_step += 1

        mean_loss = total_loss / len(loader)
        epoch_losses.append(mean_loss)
        print(
            f"epoch={epoch}/{training.epochs} "
            f"mean_loss={mean_loss:.6f} step={global_step}/{total_steps}"
        )

    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    payload = {
        "experiment_id": config.experiment_id,
        "run_id": run_id,
        "run_kind": "lora_sft",
        "train_split": train_split,
        "seed": config.seed,
        "model": {
            "name": config.model.name,
            "requested_revision": config.model.revision,
            "resolved_revision": resolved_revision,
            "dtype": "float32",
        },
        "training": training.model_dump(),
        "runtime": {
            "device": device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "prepared_input": {
            "file": str(train_path),
            "file_sha256": file_sha256(train_path),
            "examples": len(examples),
        },
        "parameters": {
            "trainable": trainable_count,
            "targeted_module_names": list(model.targeted_module_names),
            "total": total_count,
            "trainable_fraction": trainable_count / total_count,
        },
        "optimization": {
            "steps_per_epoch": len(loader),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "epoch_mean_losses": epoch_losses,
            "final_learning_rate": scheduler.get_last_lr()[0],
        },
        "adapter_path": str(adapter_dir),
    }
    (output / "train_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload
