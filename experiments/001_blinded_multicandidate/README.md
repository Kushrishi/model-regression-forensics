# Experiment 001 — Blinded multi-candidate diagnosis

Status: **complete; blinded diagnosis scored**

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

The diagnostic implementation was committed before any ranking was produced. The
diagnostic command read only the redacted lineage, visible changed-shard
artifacts, and observed baseline/candidate target generations. The two ranking
files were then committed and pushed before benchmark-owned ground truth was
revealed to the scorer.

Both the seeded-random baseline and lexical-overlap baseline placed the hidden
root cause at rank 1. The random result is a chance success on one five-candidate
instance; its Top-1 reference probability is 0.20. The lexical method assigned the
causal shard a score of 0.9091 versus 0.8261 for the next-ranked candidates.

Experiment 001 therefore closes as a blinded multi-candidate sanity check, not as
evidence of general RCA performance. Experiment 002 must entangle target-relevant
content across multiple candidate shards so simple lexical overlap is no longer
sufficient by construction. See `RESULTS.md` for the frozen interpretation.
