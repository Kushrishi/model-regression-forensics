# Experiment 007 — Sensitivity-Calibrated Causal RCA

Status: **prospective design scaffold — no Experiment 007 model result observed**

## Research question

Can a behaviorally effective target corruption dose be selected prospectively on
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

Certification evaluation may not influence target-dose selection.

No candidate model may be evaluated on certification data until calibration has
selected and frozen a target dose.

## Pre-model feasibility amendment

The initial Experiment 007 scaffold proposed varying total corruption strength
over 12, 24, 36, and 48 changed labels per candidate.

Before any Experiment 007 model was trained, solver-only feasibility analysis
identified two avoidable observability problems:

- varying total changed-record count could reveal corruption magnitude directly;
- the planted candidate could have a different flip-direction signature from
  distractors.

The protocol is therefore amended prospectively, before any behavioral result,
to hold total corruption magnitude and flip-direction totals fixed while
varying only the target-specific corruption dose.

Solver analysis also showed that exact simultaneous balancing of selected slot,
color, and material is structurally infeasible under five-way changed-record
non-overlap. Exact direction-specific slot and color balance are retained.
Material is instead controlled by requiring the same bounded material histogram
across all five candidates within a world.

An exhaustive admissible target-dose scan tested:

3, 6, 9, 12, 15, 18, 21, 24.

Only target doses 9 and 18 were feasible under the amended construction.

An initial local-trade capacity proof produced seven distinct valid worlds at
dose 9 and seven distinct valid worlds at dose 18 without model training or
certification evaluation. A subsequent pre-model identity-overlap audit showed
that mere distinctness could still leave those worlds too similar.

The frozen construction therefore adds a pairwise diversity gate: among the
seven worlds at a fixed target dose, any two worlds may share at most 144 of
their 180 changed-record identities. Equivalently, at least 36 changed records
must differ between every pair of worlds at that dose. This criterion is frozen
before any Experiment 007 behavioral result.

## Prospective target-dose grid

Experiment 007 freezes exactly two target doses:

- 9 `triangle_large` changes in the planted candidate;
- 18 `triangle_large` changes in the planted candidate.

Every candidate at either dose contains exactly:

- 36 changed records and 12 unchanged records;
- 24 clean `ACCEPT` records changed to `REJECT`;
- 12 clean `REJECT` records changed to `ACCEPT`;
- four `ACCEPT` to `REJECT` changes per selected slot;
- two `REJECT` to `ACCEPT` changes per selected slot;
- six `ACCEPT` to `REJECT` changes per color;
- three `REJECT` to `ACCEPT` changes per color.

For each training material, every candidate must contain between two and four
changed records. Within a world, all five candidates must have exactly the same
twelve-material changed-record histogram.

No additional target dose may be introduced after any model result is observed.

## Five-candidate construction

Every calibration and certification world contains five opaque candidate
dataset shards.

One benchmark-private candidate is the planted target-associated candidate.

At target dose 9, its `ACCEPT` to `REJECT` allocation is:

- 9 `triangle_large`;
- 5 `circle_small`;
- 5 `circle_large`;
- 5 `triangle_small`.

At target dose 18, its `ACCEPT` to `REJECT` allocation is:

- 18 `triangle_large`;
- 2 `circle_small`;
- 2 `circle_large`;
- 2 `triangle_small`.

Each of the four distractors contains:

- zero `triangle_large` changes;
- 8 `circle_small` changes;
- 8 `circle_large` changes;
- 8 `triangle_small` changes.

Every candidate also contains exactly:

- 6 `square_small` `REJECT` to `ACCEPT` changes;
- 6 `square_large` `REJECT` to `ACCEPT` changes.

Within every world:

- all five candidates have the same total changed-record count;
- all five candidates have the same flip-direction totals;
- direction-specific selected-slot balance is exact;
- direction-specific color balance is exact;
- each material contributes between two and four changed records per candidate;
- the complete material histogram is identical across all five candidates;
- changed-record sets are mutually non-overlapping across all five candidates;
- candidate identifiers are opaque and the planted role is benchmark-private.

The exact solver-certified record identities and private role assignment must be
frozen before any Experiment 007 model training.

## Frozen world families

Before result-bearing training, the implementation must materialize all
candidate constructions for every allowed target dose.

Calibration uses exactly two deterministic worlds per target dose.

Calibration world seeds are derived from the frozen namespace:

`sha256("exp007-calibration|42|target_dose|world_index")`

Certification uses exactly five deterministic worlds per target dose.

Certification world seeds are derived from the frozen namespace:

`sha256("exp007-certification|42|target_dose|world_index")`

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

For each target dose, evaluate both frozen calibration worlds.

A calibration world passes only when:

1. `triangle_large` regression relative to the calibration clean baseline is
   at least 0.15;
2. absolute drift on every protected non-target slice is at most 0.05;
3. all six semantic slices and aggregate accuracy are reported.

A target dose qualifies only if both calibration worlds pass.

The selected target dose is the smallest qualifying dose.

Target doses are considered strictly in increasing order:

9, 18.

Once a target dose qualifies, the larger dose is not trained or inspected.

If neither target dose qualifies, Experiment 007 stops with a benchmark-
construction negative result.

The grid may not be extended and thresholds may not be weakened.

## Freeze-before-certification boundary

The selected calibration target dose and its supporting calibration gate results
must be materialized and frozen before certification evaluation is used.

After that boundary:

- calibration results may not change the selected target dose;
- certification failures may not trigger recalibration;
- no new calibration world may be generated;
- no new target dose may be introduced.

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
- no target dose passes calibration;
- calibration selects a target dose but no certification world reproduces the
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

under this frozen synthetic task and model regime, a target-specific
corruption dose chosen prospectively on disjoint calibration data transferred to untouched
certification data and enabled a causally certified training-data regression
benchmark on which blinded RCA could be evaluated.

It would not establish general causal attribution for neural networks or
production-scale model debugging.

## Current execution boundary

No Experiment 007 model has been trained.

Solver feasibility is complete. The next allowed work is deterministic
construction implementation, manifest generation, and certification API binding.

Frozen construction manifest SHA-256:

`a6c5be745c5f4f4a97db7bf882651591886d25a0f125d967a147b861b19bdc28`

The protocol is not considered fully frozen until:

- the calibration-data generator is implemented and tested;
- all calibration and certification world manifests are generated;
- every construction gate passes;
- the manifest hash is recorded here;
- the certification API binding is tested;
- the complete pre-model quality gate passes.

## Pre-model execution-boundary freeze

The Experiment 007 execution layer was implemented and validated before any
model training or certification model evaluation.

The frozen workflow enforces these boundaries programmatically:

- calibration evaluation uses only the dedicated calibration materials;
- certification preparation requires a frozen calibration-selection artifact
  bound to the frozen manifest and Experiment 007 config;
- dose 18 cannot be supplied to the selection step after dose 9 qualifies;
- each candidate gate is bound to the exact prepared phase, target dose, and
  world index;
- baseline and candidate adapters are verified against the exact training-file
  SHA-256 values from that prepared world;
- baseline and candidate evaluations are verified against both their matching
  trained adapters and every frozen evaluation-file SHA-256 value;
- restoration and order-control preparation require a passed localized-
  regression gate from the matching certification world and manifest;
- benchmark-private planted-candidate identity remains excluded from
  diagnostic lineage;
- calibration, certification, and training/evaluation identities remain
  disjoint according to the frozen protocol.

Experiment 007 reuses the existing generic certification abstraction for
candidate gating, causal certification, order control, and public
certification output.

No Experiment 007 behavioral result was observed before this execution-layer
freeze.
