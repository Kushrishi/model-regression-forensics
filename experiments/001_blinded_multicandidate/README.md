# Experiment 001 — Blinded multi-candidate diagnosis

Status: **model validation passed; blinded diagnosis pending**

## Purpose

Move from Experiment 000's single obvious changed artifact to a target-specific
root-cause localization problem with five opaque, observable data-shard changes.

The benchmark privately designates the shard responsible for the observed
`triangle_large` regression. Diagnostic code receives the same five changed
artifacts with that designation structurally absent.

## Candidate-change design

All five observable changes are equal-sized SFT shards and all contain genuine
label changes. Their IDs are deliberately opaque (`shard_delta_01` through
`shard_delta_05`) so the identifier itself does not reveal behavioral semantics.

The other changed shards are distractors **for the specified target regression**.
They may affect other behavioral slices; Experiment 001 does not pretend they are
globally inert. The held-out `square_small` slice is unchanged and serves as the
negative-control behavior.

This design asks a realistic question: given one observed regression and several
real training changes, which change explains that regression?

## Intervention

The intervention dataset restores only the benchmark-owned target-causal shard
while leaving every other candidate change in place. Successful target recovery
therefore tests the diagnosis rather than simply returning to the clean baseline
dataset.

## Preparation

```bash
uv run python scripts/prepare_exp001.py
```

Generated artifacts include baseline, candidate, intervention, target, control,
and full held-out datasets; per-change before/after shard files; a benchmark
lineage manifest; and a structurally redacted diagnostic manifest.

## What this experiment does not yet establish

Experiment 001 is the first non-trivial localization setting, but one successful
instance is not evidence of general RCA performance. Later experiments must vary
the target slice, hidden cause, distractor structure, and random seed.

## Model validation

Experiment 001 reuses the frozen Experiment 000 LoRA SFT protocol. Train three
fresh sibling adapters from the same pinned parent revision:

```bash
uv run python scripts/train_exp001_sft.py \
  --train-split baseline_train \
  --run-id baseline

uv run python scripts/train_exp001_sft.py \
  --train-split candidate_train \
  --run-id candidate

uv run python scripts/train_exp001_sft.py \
  --train-split intervention_train \
  --run-id intervention
```

Evaluate each adapter on the target slice, unchanged control slice, and complete
96-case held-out set:

```bash
uv run python scripts/eval_exp001_adapter.py \
  --adapter artifacts/exp001/checkpoints/baseline/adapter \
  --run-id baseline
```

Repeat the evaluation command for `candidate` and `intervention` using the
corresponding adapter paths. The declared gates are read from
`configs/exp001.yaml`; they must not be tuned after observing these runs.

## Blinded diagnosis

The first diagnostic stage intentionally starts with simple baselines. Diagnosis
and benchmark scoring are separate commands so hidden ground truth is not
available while producing a ranking.

```bash
uv run python scripts/diagnose_exp001.py --method random
uv run python scripts/diagnose_exp001.py --method lexical_overlap
```

Only after those ranking files exist, score them against benchmark-owned ground
truth with `scripts/score_exp001_diagnosis.py`. Experiment 001 is deliberately an
easy localization sanity check; later experiments must use entangled distractors.
