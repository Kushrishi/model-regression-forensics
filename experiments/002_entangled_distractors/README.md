# Experiment 002 — Entangled distractors

Status: **benchmark construction in progress; no model results yet**

## Purpose

Experiment 002 tests whether the blinded RCA workflow still works after removing
Experiment 001's easiest shortcut. The observed target remains `triangle_large`,
but target-relevant prompts are now distributed across all five visible changed
shards. Only one benchmark-owned shard flips target labels; the other four flip
different behavioral slices.

The immediate construction goal is narrower than solving RCA: the exact
artifact-level mean-best Jaccard lexical-overlap baseline used in Experiment 001
must assign all five candidates the same target-overlap score before any model is
trained.

## Frozen construction

- five opaque `dataset_shard` candidates;
- 48 records per candidate;
- 32 label changes per candidate;
- one generated benchmark-private causal shard contains 32 changed
  `triangle_large` records;
- every distractor contains four unchanged `triangle_large` records spanning all
  four colors and 32 changed records from another behavioral slice;
- the remaining records are clean filler from the other changed slices;
- all 48 `square_small` records remain outside the changed candidates as the
  negative-control slice;
- the selective intervention restores only the 32 target-label changes in the
  benchmark-private causal shard and leaves 128 other label changes in place.

The causal shard is generated deterministically from the experiment seed but is
not declared in `configs/exp002.yaml` and is not printed by the preparation
summary. Diagnostic code must continue to receive only the redacted lineage.

## Pre-training difficulty gate

Before model training, the preparation command runs the existing Experiment 001
artifact-level lexical-overlap scorer against the held-out target prompts. The
maximum candidate-score range must be at most the tolerance declared in
`configs/exp002.yaml` (currently `1e-12`). If that gate fails, the benchmark must
be redesigned before training.

This gate defeats the exact coarse lexical-overlap shortcut observed in
Experiment 001. It does **not** claim that every cheap heuristic is neutralized:
the causal shard intentionally contains more target records than each distractor
so that a strong planted regression remains plausible. A target-record-count or
change-aware method may therefore remain strong; if so, that becomes the next
shortcut to remove in Experiment 003.

## Model protocol

If and only if the construction gate passes, Experiment 002 will reuse the
frozen sibling LoRA SFT protocol from Experiments 000–001. No candidate-specific
hyperparameter tuning is allowed.

Predeclared behavioral gates are in `configs/exp002.yaml`:

- clean target accuracy >= 0.95;
- baseline-to-candidate target regression >= 0.15;
- candidate-to-intervention target recovery >= 0.15;
- control-slice drift <= 0.05.

No model result has been observed yet.
