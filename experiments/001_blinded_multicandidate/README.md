# Experiment 001 — Blinded multi-candidate diagnosis

Status: **protocol defined; inputs not yet trained**

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
