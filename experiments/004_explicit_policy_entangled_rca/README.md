# Experiment 004 — Explicit-Policy Entangled RCA

Status: **prospective protocol — no Experiment 004 model result observed**

## Research question

Can blinded training-change attribution and selective causal intervention work
on the stronger Experiment 003 role-binding benchmark when the clean task is
made prospectively learnable using the explicit policy validated in
Experiment 003-D?

Target slice: `triangle_large`

Untouched negative-control slice: `square_small`

Canonical policy:

- circle -> ACCEPT
- triangle -> ACCEPT
- square -> REJECT

## Frozen construction

Experiment 004 combines:

1. Experiment 003's five-candidate role-binding RCA construction.
2. Experiment 003-D's exact explicit-policy prompt prefix.

The experiment uses fresh opaque Experiment 004 candidate identities and a
fresh deterministic benchmark-private plan.

There are exactly five changed candidate shards.

Per candidate:

- 48 records
- 36 changed labels
- 12 unchanged records
- each of the six selected slots appears exactly six times among changed records

The candidate training set flips all 36 designated labels in all five shards.

The selective intervention is diagnosis-driven rather than oracle-driven.

After the diagnostic ranking is frozen, the intervention restores only the 36
changes belonging to the uniquely top-ranked candidate selected by the primary
`selected_role_overlap` diagnostic. The other 144 corruptions remain present.

The hidden root-cause identity must not be used to choose the intervention
candidate.

If `selected_role_overlap` has no unique top-ranked candidate, localization is
inconclusive. Private ground truth must not be used to break the tie, and no
diagnosis-driven intervention is run for Experiment 004.

Corruption strength may not be changed after model results are observed.

## Explicit-policy parity

The clean model-facing Experiment 004 task must be equivalent to Experiment
003-D's validated clean task.

The only intended clean-task difference from Experiment 003 is the already
validated explicit-policy prefix:

`Explicit policy: shape=circle -> ACCEPT; shape=triangle -> ACCEPT; shape=square -> REJECT.`

Panel construction, selected-slot semantics, labels, nuisance attributes and
held-out evaluation structure remain otherwise unchanged.

## Hidden-ground-truth and anti-leak rules

The hidden root-cause identity:

- must not appear in `configs/exp004.yaml`;
- must not appear in the diagnostic manifest;
- must not be printed by ordinary preparation summaries;
- must not be available to diagnostic ranking code.

Debugger-visible shard records may contain only:

- `example_id`
- `prompt`
- `response`

Example IDs must be deterministic and opaque.

## Pre-training construction gates

Before any model training, preparation must verify:

- exactly 5 observable candidates;
- exactly 48 records per candidate;
- exactly 36 changed records per candidate;
- whole-artifact lexical score range <= `1e-12`;
- changed-record lexical score range <= `1e-12`;
- equal target-descriptor surface exposure across candidates;
- equal changed-record target-descriptor surface exposure across candidates;
- every changed candidate selects each slot exactly 6 times;
- public record schema contains only the declared fields;
- public IDs are opaque;
- diagnostic lineage contains no root-cause field;
- clean model-facing data passes Experiment 003-D parity checks;
- generated counts match `configs/exp004.yaml`.

Any failed construction gate stops training until the implementation is repaired
within this prospectively declared design.

## Frozen model and training protocol

Experiment 004 keeps the existing pinned setup:

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- revision: `a10cc1512eabd3dde888204e902eca88bddb4951`
- seed: 42
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0
- batch size: 8
- epochs: 10
- learning rate: `5e-4`
- weight decay: 0
- warmup ratio: 0.05
- max length: 192
- max gradient norm: 1
- assistant-answer-only supervision
- no response loss weighting

Baseline, candidate and intervention are fresh sibling runs from the same pinned
parent model.

## Clean-baseline gate

The clean baseline is evaluated on:

- `circle_small`
- `circle_large`
- `square_small`
- `square_large`
- `triangle_small`
- `triangle_large`
- `all`

Every required split must achieve accuracy >= `0.95`.

Because each semantic slice contains 16 examples, this effectively requires
16/16 on every individual slice.

If any clean split fails:

**STOP.**

Do not train candidate or intervention siblings.

Any model, prompt, seed, optimizer, LoRA, weighting or training-protocol change
motivated by that result requires a separately named follow-up experiment.

## Behavioral RCA gates

If the clean baseline passes:

- required target regression: >= `0.15`
- required target recovery after intervention: >= `0.15`
- maximum absolute negative-control drift: <= `0.05`, measured across both
  candidate-vs-baseline and intervention-vs-baseline

All six semantic slices plus aggregate accuracy must be reported for baseline,
candidate and intervention.

Spillover is reported rather than hidden.

If the candidate does not produce the required regression, record the negative
result without strengthening the corruption.

If intervention does not produce the required recovery, record the negative
result without broadening the intervention.

## Frozen blinded diagnostics

After a qualifying regression is observed, but before private truth is revealed,
run four frozen model-free diagnostics.

The first three are comparison baselines:

1. `random`
2. `lexical_overlap`
3. `changed_lexical_overlap`

The prospectively declared primary, intervention-driving diagnostic is:

4. `selected_role_overlap`

`selected_role_overlap` is a transparent task-aware baseline. It parses the
public `selected_slot`, resolves that slot to the selected
`shape=<...>,size=<...>` descriptor, and compares selected descriptors between
observed regression prompts and changed candidate records.

For a regression prompt and changed record:

`match = 1` when their selected descriptors are equal, otherwise `0`.

A candidate's score is the mean across regression cases of its best
changed-record match.

The diagnostic may use only debugger-visible information. It may not use:

- benchmark-private annotations;
- hidden shard semantics;
- root-cause identity;
- model activations;
- model gradients;
- intervention results.

Its parser and ranking logic must be unit-tested on synthetic fixtures before
being applied to Experiment 004 results.

This diagnostic is intentionally transparent and task-aware. It is not claimed
as a novel general RCA algorithm.

## Freeze-before-reveal rule

The private root cause remains unopened while diagnostic rankings are generated.

All ranking artifacts must first be generated, committed and pushed.

If and only if `selected_role_overlap` has a unique top-ranked candidate, that
public candidate ID becomes the frozen intervention target.

The intervention dataset restores that candidate's 36 changed records without
consulting private ground truth.

The intervention must then be trained, evaluated, recorded, committed and pushed
while the private root cause is still unopened.

Only after both the diagnosis and intervention result are frozen may the private
benchmark manifest be opened and the rankings scored against the hidden cause.

If `selected_role_overlap` is tied for the top score, localization is
inconclusive. Private truth must not be used to select an intervention candidate.

Tie-aware scoring must prevent deterministic identifier ordering inside an
equal-score group from being counted as successful localization.

## Interpretation and negative results

A complete positive result requires:

1. clean baseline pass;
2. qualifying target regression;
3. a unique `selected_role_overlap` prediction frozen before truth reveal;
4. qualifying recovery from intervention on that predicted candidate;
5. subsequent truth reveal showing that the frozen prediction equals the hidden
   causal candidate.

Even then, the result is controlled evidence within this synthetic benchmark,
not proof of novelty or general-purpose model debugging.

Do not hide or silently repair:

- baseline failure;
- insufficient regression;
- ties;
- incorrect localization;
- failed recovery;
- control drift;
- spillover;
- unexpected failures.

Any protocol change motivated by observed results becomes a separately documented
experiment rather than replacing Experiment 004.
