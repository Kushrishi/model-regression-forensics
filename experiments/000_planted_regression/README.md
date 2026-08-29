# Experiment 000 — One planted regression

Status: **zero-shot reference complete; clean baseline SFT implemented**

## Purpose

Validate the research protocol with one deliberately introduced, reproducible post-training regression whose true cause is known to the experiment harness but structurally hidden from the diagnostic method.

This experiment does **not** establish novelty or diagnostic quality. With only one changed training artifact, its purpose is to validate the end-to-end benchmark mechanics before introducing a non-trivial candidate-cause search space.

## Candidate model

Initial target: `HuggingFaceTB/SmolLM2-360M-Instruct` at pinned revision `a10cc1512eabd3dde888204e902eca88bddb4951`

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

Before training, run `scripts/eval_exp000_zero_shot.py` against the prepared held-out sets. Generation is greedy and raw generations are retained. The primary behavioral metric is one-token label accuracy: outer whitespace and letter case are ignored, but explanations such as `The label is ACCEPT` remain failures. Strict exact match is reported separately as a format-compliance metric.

The first zero-shot pilot used strict exact match and scored 0.00 on the target slice and 0.05 on unrelated cases. Inspection showed outputs such as `REJECT` and `Reject`, revealing that strict exact match confounded label behavior with capitalization. Before any SFT baseline, candidate, or recovery run, the protocol was therefore refined prospectively to use label accuracy as the primary behavioral metric while preserving strict exact match as a secondary metric.

The untouched instruct model still misses the configured baseline threshold on the target slice, so Experiment 000 creates a clean LoRA SFT baseline checkpoint before applying the planted regression. Baseline, candidate, and recovery are sibling runs from the same pinned parent model with the same optimizer, schedule, seed, epoch count, data order, prompt formatting, and LoRA configuration. Only the prepared training split changes.

The initial SFT protocol uses float32 model loading and LoRA adapters over the attention and MLP projection layers. Hyperparameters are declared in `configs/exp000.yaml` before the clean baseline run. If the baseline misses the declared threshold, any pilot adjustment must be recorded before candidate or recovery training.

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
