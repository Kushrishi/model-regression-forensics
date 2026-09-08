# Experiment 007 — Sensitivity-Calibrated Causal RCA

Status: **prospective design scaffold — no Experiment 007 model result observed**

## Research question

Can a behaviorally effective corruption strength be selected prospectively on
development-only calibration data and then transfer to untouched certification
worlds strongly enough to support causal model-regression RCA?

Experiment 007 addresses the benchmark-construction failure observed in
Experiment 006.

Experiment 006 controlled corruption in semantic space, but all five frozen
candidate worlds produced exactly zero regression on the intended
`triangle_large` target.

Experiment 007 therefore separates behavioral sensitivity calibration from
final benchmark certification.

## Fixed task and model

The following remain unchanged from Experiment 006:

- explicit-policy six-object role-binding task;
- primary target: `triangle_large`;
- five protected non-target semantic slices;
- pinned `HuggingFaceTB/SmolLM2-360M-Instruct` revision;
- seed 42;
- LoRA rank 16 and alpha 32;
- 10 epochs;
- batch size 8;
- learning rate `5e-4`;
- optimizer and schedule settings;
- deterministic generation;
- label accuracy as the primary metric;
- clean-baseline threshold 0.95;
- target-regression threshold 0.15;
- target-recovery threshold 0.15;
- maximum protected-slice drift 0.05.

Experiment 007 is a benchmark-construction follow-up, not a new-model,
new-optimizer, new-task, or new-diagnostic experiment.

## Calibration and certification separation

Calibration and certification evaluation must use disjoint model-facing
examples.

The existing Experiment 003-D / Experiments 004-006 held-out material family
remains reserved for certification:

- bamboo
- ceramic
- marble
- wool

Experiment 007 calibration uses a new development-only material family:

- bronze
- cotton
- quartz
- velvet

Calibration examples must have distinct opaque IDs from training and
certification examples.

Each evaluation family contains the same six semantic slices with 16 examples
per slice and 96 examples total.

Certification evaluation may not influence corruption-strength selection.

No candidate model may be evaluated on certification data until calibration has
selected and frozen a strength.

## Prospective corruption-strength grid

Experiment 007 freezes exactly four candidate strengths:

- 12 changed labels per candidate;
- 24 changed labels per candidate;
- 36 changed labels per candidate;
- 48 changed labels per candidate.

No intermediate or additional strength may be added after any model result is
observed.

These values are chosen because the 48-example training capacity of each
semantic slice permits exact balancing at every strength across:

- six selected slots;
- four colors;
- twelve training materials.

For strength `k`, every candidate must therefore contain:

- exactly `k` changed records;
- exactly `48 - k` unchanged records;
- exactly `k / 6` changed examples per selected slot;
- exactly `k / 4` changed examples per color;
- exactly `k / 12` changed examples per training material.

## Five-candidate construction

Every calibration and certification world contains five opaque candidate
dataset shards.

All five candidates use the same strength within a world.

One benchmark-private candidate is the planted target-associated candidate.

Its changed records must all:

- belong to `triangle_large`;
- be clean `ACCEPT` examples;
- become `REJECT` examples.

The four distractor candidates may not contain `triangle_large` changes.

Distractor changes must come only from the five protected semantic slices.

Their semantic allocation must be solver-certified so that:

- all five protected slices are represented in every distractor whenever
  mathematically possible;
- per-distractor protected-slice counts differ by at most one;
- aggregate protected-slice counts across all four distractors differ by at
  most one;
- selected-slot, color, and material balance remain exact;
- changed-record sets do not overlap;
- all five observable candidates have the same record count and changed-record
  count.

The exact solver-certified record identities must be frozen before any
Experiment 007 model training.

## Frozen world families

Before result-bearing training, the implementation must materialize all
candidate constructions for every allowed strength.

Calibration uses exactly two deterministic worlds per strength.

Calibration world seeds are derived from the frozen namespace:

`sha256("exp007-calibration|42|strength|world_index")`

Certification uses exactly five deterministic worlds per strength.

Certification world seeds are derived from the frozen namespace:

`sha256("exp007-certification|42|strength|world_index")`

The implementation must freeze the digest-to-integer conversion, generated
world manifests, and manifest hash before any model result is observed.

No additional calibration or certification world may be generated after model
results are available.

## Calibration clean baseline

The clean model-facing training set is identical across calibration strengths
and worlds.

A single fresh deterministic calibration clean sibling may therefore be reused
across calibration candidate runs only if preparation verifies exact
model-facing clean-training parity.

That baseline is evaluated only on the calibration evaluation family during
strength selection.

Every calibration semantic slice and aggregate accuracy must be at least 0.95.

If the calibration clean baseline fails, Experiment 007 stops before candidate
calibration.

## Calibration candidate gate

For each strength, evaluate both frozen calibration worlds.

A calibration world passes only when:

1. `triangle_large` regression relative to the calibration clean baseline is
   at least 0.15;
2. absolute drift on every protected non-target slice is at most 0.05;
3. all six semantic slices and aggregate accuracy are reported.

A strength qualifies only if both calibration worlds pass.

The selected strength is the smallest qualifying strength.

Strengths are considered strictly in increasing order:

12, 24, 36, 48.

Once a strength qualifies, larger strengths are not trained or inspected.

If no strength qualifies, Experiment 007 stops with a benchmark-construction
negative result.

The grid may not be extended and thresholds may not be weakened.

## Freeze-before-certification boundary

The selected calibration strength and its supporting calibration gate results
must be materialized and frozen before certification evaluation is used.

After that boundary:

- calibration results may not change the selected strength;
- certification failures may not trigger recalibration;
- no new calibration world may be generated;
- no new corruption strength may be introduced.

## Certification worlds

Only the five certification worlds corresponding to the frozen selected
strength are eligible for result-bearing certification.

They are considered strictly in world-index order.

Each world must first pass:

1. construction gates;
2. the clean capability gate on certification evaluation;
3. the localized target-regression and protected-behavior gate.

The first world passing those prerequisites may enter the existing private
causal-certification procedure.

If a world fails before causal certification, continue to the next frozen
certification world.

If all five fail, stop with a certification-transfer negative result.

No sixth world is permitted.

## Private causal certification

The existing Experiment 006 causal-certification logic is retained in
principle.

Only after a certification world passes the localized-regression gate may the
private harness train one restoration sibling for each of the five candidates.

The planted candidate qualifies only if:

- its restoration improves `triangle_large` by at least 0.15;
- protected slices remain within 0.05 of the clean baseline;
- non-planted restorations do not materially recover the target;
- the planted recovery is unique under the frozen rule.

A separately frozen alternative training-order control must also reproduce the
candidate regression and planted recovery before the benchmark is declared
causally certified.

The exact reusable certification API and order-control implementation will be
bound to Experiment 007 before execution.

## Blinded RCA boundary

No blinded diagnostic is run during calibration, certification-world
selection, or private causal certification.

Only a causally certified world may enter blinded RCA.

The benchmark-private root identity and private restoration evidence remain
unavailable to diagnostic methods.

Diagnosis, intervention, truth reveal, and scoring retain the freeze-before-
reveal discipline established in Experiments 004-006.

## Negative-result policy

Valid outcomes include:

- calibration clean baseline failure;
- no strength passes calibration;
- calibration selects a strength but no certification world reproduces the
  localized regression;
- a certification world passes behaviorally but fails private causal
  certification;
- causal certification succeeds but blinded localization fails;
- localization succeeds but diagnosis-driven recovery fails;
- complete end-to-end success.

None of these outcomes authorizes post-result changes to the grid, thresholds,
world counts, target, protected slices, model, optimizer, training protocol, or
certification evaluation.

## Claim boundary

A positive Experiment 007 result would support only a narrow controlled claim:

under this frozen synthetic task and model regime, a corruption strength chosen
prospectively on disjoint calibration data transferred to untouched
certification data and enabled a causally certified training-data regression
benchmark on which blinded RCA could be evaluated.

It would not establish general causal attribution for neural networks or
production-scale model debugging.

## Current execution boundary

No Experiment 007 model has been trained.

The next allowed work is construction implementation and solver feasibility.

The protocol is not considered fully frozen until:

- the calibration-data generator is implemented and tested;
- all calibration and certification world manifests are generated;
- every construction gate passes;
- the manifest hash is recorded here;
- the certification API binding is tested;
- the complete pre-model quality gate passes.
