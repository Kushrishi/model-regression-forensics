"""Bounded development-only LoRA feasibility; never M5 or official test access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
LABELS = ("card_arrival", "card_delivery_estimate", "order_physical_card", "card_linking")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--seed", type=int, choices=(11, 23), required=True)
    p.add_argument("--release", choices=("clean", "swapped"), required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    # Exclusive reservation preserves a failed or timed-out development attempt.
    a.out.mkdir()
    start = time.monotonic()
    torch.set_num_threads(2)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    model_manifest = json.loads((a.model / "manifest.json").read_text())
    if model_manifest["revision"] != MODEL_REVISION:
        raise ValueError("model/tokenizer must match the declared immutable revision")
    for f in model_manifest["files"]:
        if hashlib.sha256((a.model / f["name"]).read_bytes()).hexdigest() != f["sha256"]:
            raise ValueError("model artifact identity mismatch")
    if a.train_csv.name != "train.csv":
        raise ValueError("only train.csv is allowed; no official test access")
    if (
        hashlib.sha256(a.train_csv.read_bytes()).hexdigest()
        != "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"
    ):
        raise ValueError("train.csv does not match the pinned source identity")
    rows = list(csv.DictReader(a.train_csv.open(newline="")))
    grouped = {label: [r for r in rows if r["category"] == label] for label in LABELS}
    if any(len(v) < 24 for v in grouped.values()):
        raise ValueError("declared development cohort is incomplete")
    train = [(r["text"], label) for label in LABELS for r in grouped[label][:16]]
    dev = [(r["text"], label) for label in LABELS for r in grouped[label][16:24]]
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=0.003)

    def prompt(text):
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Choose exactly one intent. Output only its name. Choices: "
                    + ", ".join(LABELS)
                    + ". Request: "
                    + text,
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
        )

    encoded = []
    for text, label in train:
        target = label
        if a.release == "swapped" and label in LABELS[:2]:
            target = LABELS[1] if label == LABELS[0] else LABELS[0]
        prefix = prompt(text)
        answer = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = prefix + answer
        if len(ids) > 256:
            raise ValueError("declared context bound exceeded; no silent truncation")
        encoded.append((ids, [-100] * len(prefix) + answer))
    model.train()
    updates = []
    for _epoch in range(3):
        order = list(range(len(encoded)))
        random.shuffle(order)
        for offset in range(0, len(order), 4):
            chosen = [encoded[i] for i in order[offset : offset + 4]]
            size = max(len(x[0]) for x in chosen)
            ids = torch.tensor(
                [x[0] + [tokenizer.pad_token_id] * (size - len(x[0])) for x in chosen]
            )
            labels = torch.tensor([x[1] + [-100] * (size - len(x[1])) for x in chosen])
            mask = torch.tensor([[1] * len(x[0]) + [0] * (size - len(x[0])) for x in chosen])
            t = time.monotonic()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates.append(time.monotonic() - t)
            print(
                json.dumps(
                    {
                        "update": len(updates),
                        "seconds": updates[-1],
                        "loss": float(loss.detach()),
                        "elapsed": time.monotonic() - start,
                    }
                ),
                flush=True,
            )
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (text, label) in enumerate(dev):
            ids = torch.tensor([prompt(text)])
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=False,
                max_new_tokens=16,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            output = tokenizer.decode(
                generated[0, ids.shape[1] :], skip_special_tokens=True
            ).strip()
            predictions.append(
                {
                    "development_id": i,
                    "label": label,
                    "output": output,
                    "correct": output == label,
                    "valid": output in LABELS,
                }
            )
    groups = {
        "all": predictions,
        "target": [r for r in predictions if r["label"] in LABELS[:2]],
        "protected": [r for r in predictions if r["label"] in LABELS[2:]],
    }
    report = {
        "schema": "mrf-development-feasibility-v1",
        "confirmatory": False,
        "model_revision": MODEL_REVISION,
        "train_sha256": hashlib.sha256(a.train_csv.read_bytes()).hexdigest(),
        "seed": a.seed,
        "release": a.release,
        "updates": len(updates),
        "fit_seconds": sum(updates),
        "wall_seconds": time.monotonic() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "accuracy": {
            name: sum(r["correct"] for r in values) / len(values) for name, values in groups.items()
        },
        "predictions": predictions,
        "official_test_accessed": False,
        "process": "LoRA, not full-model retraining",
    }
    (a.out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "complete": True,
                "accuracy": report["accuracy"],
                "wall_seconds": report["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
