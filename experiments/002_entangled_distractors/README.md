# Experiment 002 — Entangled distractors

Status: **complete**

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

The construction gate passed with an exact five-way lexical-overlap tie, so
Experiment 002 now reuses the frozen sibling LoRA SFT protocol from Experiments
000–001. No candidate-specific hyperparameter tuning is allowed.

Predeclared behavioral gates are in `configs/exp002.yaml`:

- clean target accuracy >= 0.95;
- baseline-to-candidate target regression >= 0.15;
- candidate-to-intervention target recovery >= 0.15;
- control-slice drift <= 0.05.

Model validation passed all predeclared target-regression, recovery, and control-drift gates. The oracle intervention recovered the target completely but also produced partial canonical recovery on 12 non-target cases, so the stronger no-spillover prediction was not supported. See `RESULTS.md` for the frozen measurements. The private benchmark manifest remained unopened until all three blinded ranking artifacts were committed and pushed. Ground-truth scoring was performed only after that freeze.

## Blinded diagnosis protocol

After model validation, Experiment 002 freezes three model-free RCA baselines
before any private ground-truth scoring:

- `random`: deterministic seeded random ranking;
- `lexical_overlap`: the unchanged Experiment 001 artifact-level scorer, which
  is expected to remain an exact five-way score tie by construction;
- `changed_lexical_overlap`: the same lexical score applied only to records that
  differ between each debugger-visible `before` and `after` shard artifact.

The third baseline tests whether simple lineage differencing is sufficient once
whole-artifact lexical similarity has been neutralized. It is not intended as a
novel attribution method. Diagnostic code continues to consume only the
redacted manifest and observed baseline/candidate target generations.

Scoring is a separate post-freeze step. Because deterministic sorting can assign
arbitrary ordinal positions inside equal-score groups, the scorer uses tie-aware
rank bounds for Top-1, Top-3, and reciprocal-rank metrics while retaining the
ordinal position for auditability.


## Blinded diagnosis outcome

The hidden benchmark cause was revealed only after the three ranking JSON files
were committed and pushed. The scorer identified `shard_mix_01` as the hidden
root cause.

- `random` placed the cause fifth (reciprocal rank `0.20`).
- `lexical_overlap` assigned all five candidates the same score. Tie-aware
  scoring therefore reports an average tied rank of `3.0`, a five-way tie, no
  unique Top-1 localization, and no guaranteed Top-3 localization.
- `changed_lexical_overlap` uniquely ranked `shard_mix_01` first with score
  `0.9090909091`; the next-best candidates scored `0.8260869565`. Its tie size
  was one, so Top-1 and Top-3 both pass without relying on deterministic
  identifier ordering.

Experiment 002 therefore establishes that coarse whole-artifact lexical
similarity is insufficient on the entangled construction, while simple
debugger-visible lineage differencing is sufficient on this single instance.
This is a benchmark-progression result, not evidence of a novel RCA method. The
next benchmark must neutralize the changed-record lexical shortcut before
introducing stronger attribution methods.
