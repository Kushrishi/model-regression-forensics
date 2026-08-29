# Experiment 000 — One planted regression

Status: **protocol implementation; model run not started**

## Purpose

Validate the research protocol with one deliberately introduced, reproducible post-training regression whose true cause is known to the experiment harness but structurally hidden from the diagnostic method.

This experiment does **not** establish novelty or diagnostic quality. With only one changed training artifact, its purpose is to validate the end-to-end benchmark mechanics before introducing a non-trivial candidate-cause search space.

## Candidate model

Initial target: `HuggingFaceTB/SmolLM2-360M-Instruct`

Reasons:
- 360M parameters;
- Transformers support;
- Apache-2.0 license;
- practical for repeated intervention/retraining experiments.

We may switch if baseline behavior or Apple Silicon training support is poor.

## Controlled task

Experiment 000 uses a synthetic object-classification task with four observable attributes:

- material;
- color;
- shape;
- size.

The canonical policy maps circles and triangles to `ACCEPT` and squares to `REJECT`. Training and evaluation use disjoint material vocabularies so the target regression must transfer to held-out examples rather than merely reproducing training records.

The planted regression changes only the `triangle_large` SFT shard. Baseline and recovery data retain the canonical labels; candidate data flips the labels for that shard. Unrelated evaluation cases exclude the target slice.

The task is intentionally simple. It is a protocol unit test, not the eventual benchmark.

## Prepared artifacts

Run:

```bash
uv run python scripts/prepare_exp000.py
```

This materializes:

- `baseline_train.jsonl`;
- `candidate_train.jsonl`;
- `recovery_train.jsonl`;
- `target_eval.jsonl`;
- `unrelated_eval.jsonl`;
- benchmark lineage containing hidden ground truth;
- diagnostic lineage with the ground-truth field absent by construction;
- content hashes and dataset counts.

All generated outputs live under ignored `artifacts/`.

## Experimental shape

Create:

- baseline run;
- candidate run with one planted change;
- recovery run where that change is reversed.

The model-training stage has not yet been implemented.

## Success criteria

Experiment 000 passes only if:

1. baseline meets the declared target;
2. candidate regresses by the declared minimum;
3. regression holds on held-out examples;
4. planted change is present in recorded lineage;
5. diagnostic code receives no hidden benchmark label;
6. intervention produces meaningful recovery;
7. unrelated evals remain approximately stable;
8. run is reproducible from a clean checkout.

A later experiment must introduce multiple plausible changed artifacts before candidate-ranking accuracy is treated as meaningful research evidence.
