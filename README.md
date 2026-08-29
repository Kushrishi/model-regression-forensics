# Model Regression Forensics

> Working research direction. Novelty is **not established**.

This repository investigates a narrow question:

**Can an automated debugger localize the training change responsible for an observed model regression and verify that diagnosis through controlled intervention rather than correlation alone?**

## Current wedge

Adjacent work already covers major parts of the problem: behavioral model diffing, training-data attribution, predictive data debugging, ML pipeline root-cause analysis, and intervention.

The working wedge is narrower:

**post-hoc behavioral regression → structured training lineage → candidate-cause ranking across multiple artifact types → controlled intervention → verified root cause**

The project proceeds only if the related-work review supports a meaningful gap.

## Experiment 000

Experiment 000 is a protocol-validation run, not a novelty result. It uses a deterministic synthetic classification task with one planted SFT-shard regression and held-out evaluation materials.

1. Prepare clean baseline, corrupted candidate, and recovery datasets.
2. Train and evaluate a reproducible baseline checkpoint.
3. Introduce the known shard change.
4. Confirm a held-out behavioral regression on the target slice.
5. Record the change in a benchmark-owned lineage manifest.
6. Pass only a structurally redacted manifest to diagnostic code.
7. Intervene on the suspected cause and re-run training/evaluation.
8. Require held-out recovery with stable unrelated evals.

Prepare the deterministic inputs without installing the heavy ML stack:

```bash
uv run python scripts/prepare_exp000.py
```

Generated datasets and manifests are written under `artifacts/`, which is intentionally ignored by Git.

Evaluate the untouched instruct model before defining the trained baseline checkpoint:

```bash
uv run python scripts/eval_exp000_zero_shot.py
```

The run records the pinned model revision, runtime versions, prepared-input file hashes, raw generations, primary one-token label accuracy, and secondary strict exact-match scores. The zero-shot run is a reference measurement; it is not the Experiment 000 baseline checkpoint.

Train the clean sibling run from the same pinned parent checkpoint:

```bash
uv run python scripts/train_exp000_sft.py \
  --train-split baseline_train \
  --run-id baseline

uv run python scripts/eval_exp000_adapter.py \
  --adapter artifacts/exp000/checkpoints/baseline/adapter \
  --run-id baseline
```

Baseline, candidate, and recovery runs use the same LoRA SFT configuration. The prepared dataset is already deterministically shuffled, so the training loader does not reshuffle it.

See `experiments/000_planted_regression/README.md`.

## Experiment 001

Experiment 001 introduces five opaque, observable data-shard changes and asks
which one explains a specified held-out regression. The benchmark keeps the
target-causal designation private; diagnostic code receives only the redacted
lineage and per-change artifacts. A separate intervention dataset restores only
the predicted target cause while preserving the other candidate changes.

Prepare the deterministic inputs:

```bash
uv run python scripts/prepare_exp001.py
```

Experiment 001 reuses the frozen sibling LoRA SFT protocol from Experiment 000.
The baseline, five-change candidate, and selective intervention runs are trained
with `scripts/train_exp001_sft.py` and evaluated with
`scripts/eval_exp001_adapter.py` on target, control, and full held-out splits.

See `experiments/001_blinded_multicandidate/README.md`.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Install the heavy ML stack only when model training starts:

```bash
uv sync --extra dev --extra research
```

## Research rules

- Do not call the method novel until the primary-source review supports it.
- Do not report planned or synthetic results as completed evidence.
- Keep planted root causes hidden from diagnostic methods.
- Separate diagnosis from verification.
- Verify causes by intervention, not attribution score alone.
- Evaluate held-out recovery and unrelated-capability regressions.
- Keep every experiment reproducible from config + seed + lineage.
