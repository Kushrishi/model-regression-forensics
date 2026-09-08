# Experiment 008 — Selective Causal RCA

Status: prospective protocol implementation — no Experiment 008 model has been trained.

## Research question

> Given several concurrent debugger-visible training-lineage changes, can a
> behavior-specific regression be localized to the responsible change and then
> counterfactually verified by selectively restoring that change?

Experiment 008 is the final major synthetic benchmark-design iteration on the
current explicit-policy shape task.

Its purpose is to establish one prospectively frozen end-to-end causal
regression-forensics benchmark without repeating the structural failures exposed
by Experiments 004–007.

## Frozen clean substrate

Experiment 008 retains the Experiment 003-D explicit-policy capability substrate.

Canonical policy:

- circle -> ACCEPT
- triangle -> ACCEPT
- square -> REJECT

Target:

- `triangle_large`

Protected slices:

- `circle_small`
- `circle_large`
- `triangle_small`
- `square_small`
- `square_large`

Frozen model/training regime:

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- revision: `a10cc1512eabd3dde888204e902eca88bddb4951`
- seed: 42
- LoRA SFT
- epochs: 10
- batch size: 8
- learning rate: 5e-4
- LoRA r: 16
- LoRA alpha: 32
- LoRA dropout: 0

## Causal structure

Five concurrent dataset-shard changes are debugger-visible.

Exactly one is the benchmark-private root cause.

Each candidate shard contains:

- 48 baseline records;
- 36 genuinely changed model-facing records;
- 12 unchanged filler records.

The five 36-record changed sets are mutually disjoint.

Every candidate has exactly:

- 36 prompt changes;
- 36 response changes.

Changed-record count, prompt-change count, and response-change count therefore
cannot identify the root by themselves.

## Important model-facing invariant

The baseline training set contains 288 prompts.

Experiment 008 requires:

> the complete candidate prompt multiset to be exactly identical to the complete
> baseline prompt multiset.

No new prompt is introduced.

No prompt is deleted.

No prompt is duplicated or upweighted relative to the baseline prompt
multiset.

The candidate differs in where model-facing examples occur in deterministic
training order and, for the root only, in the labels attached to the target
prompts.

This requirement prevents a nuisance intervention from silently creating
duplicate prompts or changing semantic frequency while still appearing balanced
in an aggregate table.

## Root intervention

The root operates on exactly 36 of the 48 `triangle_large` training records.

The 36 root records are selected prospectively with exact balance:

- 3 per training material;
- 9 per original color;
- 6 per original selected slot.

The selected root prompts are deterministically permuted among those 36 record
positions.

Thus:

- every selected root prompt changes at its record position;
- the root prompt multiset is exactly preserved;
- selected semantic role remains `triangle_large`;
- target prompt frequency is unchanged.

For all 36 root records, the canonical label:

`ACCEPT`

becomes the policy-inconsistent label:

`REJECT`.

Root properties:

- changed records: 36;
- prompt changes: 36;
- response changes: 36;
- source semantic: 36 `triangle_large`;
- result semantic: 36 `triangle_large`;
- policy-inconsistent target examples: 36;
- policy-inconsistent protected examples: 0;
- prompt multiset preserved: yes;
- prompt/response-pair multiset preserved: no.

The direct target corruption is therefore:

- 36 / 48 target training examples;
- 75 percent of the target slice.

## Nuisance interventions

Each of the four nuisance candidates changes exactly 36 protected records.

A nuisance uses nine deterministic panels.

Within each selected panel it changes exactly four records:

- two canonical-ACCEPT protected records;
- `square_small`;
- `square_large`.

Across the nine panels, each nuisance uses exactly:

- `circle_small`: 6
- `circle_large`: 6
- `triangle_small`: 6
- `square_small`: 9
- `square_large`: 9
- `triangle_large`: 0

Thus every nuisance starts with exactly:

- 18 ACCEPT records;
- 18 REJECT records.

Nuisance panel allocation is prospectively constrained so that the 36 changed
records cover all six selected slots, with every selected slot occurring between
4 and 8 times.

This constraint is solved before training and without observing any model
behavior.

Within each selected panel, the two ACCEPT records are paired with the two
square records.

Each pair swaps its complete model-facing task content.

For example, conceptually:

- ACCEPT example A receives the square example's prompt and canonical REJECT;
- the square example receives example A's prompt and canonical ACCEPT.

The transformation is reciprocal.

Therefore every changed nuisance record has:

- a different prompt;
- a different response;
- a different selected protected semantic role.

But the four-record prompt/response multiset inside that panel is exactly
unchanged.

Consequently, for every nuisance candidate:

- changed records = 36;
- prompt changes = 36;
- response changes = 36;
- source semantic histogram = result semantic histogram;
- source ACCEPT/REJECT mass = result ACCEPT/REJECT mass;
- prompt multiset is exactly preserved;
- prompt/response-pair multiset is exactly preserved;
- policy-inconsistent labels = 0;
- target interventions = 0.

The nuisance interventions are not claimed to be optimization no-ops.

The training loop consumes rows deterministically with `shuffle=False`, so
permuting correct model-facing examples across row positions can alter the SGD
trajectory.

That is intentional: they are real competing lineage changes that remain
policy-correct.

## Why this differs from Experiment 007

Experiment 007 combined 180 incorrect labels into one 288-example candidate
dataset.

Protected semantic families were directly corrupted more heavily than the
intended target while the behavioral gate simultaneously required them to remain
stable.

Experiment 008 forbids that structure.

The exact combined model-facing candidate contains:

- 36 policy-inconsistent labels total;
- all 36 on `triangle_large`;
- zero policy-inconsistent protected labels.

The aggregate semantic histogram remains exactly:

- 48 records per semantic slice.

The aggregate prompt histogram is also exactly preserved.

## Mandatory pre-model construction audit

Before any Experiment 008 model is trained, programmatic validation must prove:

1. baseline training size = 288;
2. candidate training size = 288;
3. every candidate shard contains 48 records;
4. every candidate has exactly 36 changed records;
5. every candidate has exactly 36 prompt changes;
6. every candidate has exactly 36 response changes;
7. changed-record sets are mutually disjoint;
8. baseline semantic count = 48 per slice;
9. candidate semantic count = 48 per slice;
10. complete baseline/candidate prompt multisets are identical;
11. no prompt-frequency inflation is introduced;
12. root source semantic = 36 `triangle_large`;
13. root result semantic = 36 `triangle_large`;
14. root policy-inconsistent target count = 36;
15. root policy-inconsistent protected count = 0;
16. every nuisance policy-inconsistent count = 0;
17. every nuisance directly touches `triangle_large` zero times;
18. every nuisance source/result semantic histogram is identical;
19. every nuisance source/result label mass is identical;
20. every nuisance changed prompt/response-pair multiset is identical before and
    after;
21. the root changed prompt multiset is identical before and after;
22. the root changed prompt/response-pair multiset is not identical before and
    after.

A failure of any structural invariant stops the experiment before model
training.

## No sensitivity search

Experiment 008 freezes one target intervention prospectively:

- 36 / 48 target records.

There is no dose grid.

After Experiment 008 model behavior is observed, the experiment may not:

- change the target dose;
- weaken the behavioral thresholds;
- search additional world seeds;
- choose only a favorable world.

## Frozen worlds

Two deterministic benchmark-private worlds are defined before training.

World seed namespace:

`sha256("exp008-world|42|world_index")`

The worlds differ in:

- hidden root candidate identity;
- exact root record allocation;
- nuisance panel allocation;
- nuisance pairing/order.

They do not differ in scientific thresholds or intervention strength.

Both worlds must satisfy the candidate behavioral gate.

## Candidate behavioral gate

Clean baseline capability:

- accuracy >= 0.95 on every required split.

Target materiality:

- baseline minus candidate `triangle_large` accuracy >= 0.15.

Protected locality:

- absolute candidate-vs-baseline drift <= 0.05 on every protected slice.

Both frozen worlds must pass.

Certification remains closed unless this gate passes.

## Counterfactual certification

For each qualified world, restore each lineage candidate independently.

Restoration reverses the complete frozen candidate-specific intervention.

For the root this restores:

- original prompt placement;
- canonical response;
- original internal task metadata.

For a nuisance it restores:

- the original policy-correct prompt/response placement;
- original internal task metadata.

True-root restoration must:

- recover `triangle_large` by >= 0.15 relative to the candidate;
- leave every protected split within 0.05 of clean baseline.

Every non-root restoration must:

- recover `triangle_large` by <= 0.05;
- leave every protected split within 0.05 of clean baseline.

Exactly one candidate may satisfy the target-recovery threshold.

## Alternative-order robustness

The primary training implementation uses deterministic row order and
`shuffle=False`.

After primary causal certification succeeds, the prospectively defined
alternative-order robustness control is retained for the primary world.

It tests whether the causal conclusion survives a second frozen ordering.

It is not a mechanism for choosing a favorable result.

## Truth-isolated diagnostic evaluation

Experiment 008 does not claim investigator blinding.

During prospective benchmark construction, the benchmark-private root identities
were visible in construction-validation output. That procedural history cannot
be undone and is recorded explicitly rather than being described as a blinded
human experiment.

The diagnostic evaluation itself remains truth-isolated.

The diagnostic implementation under evaluation may receive only:

- the redacted diagnostic lineage manifest;
- debugger-visible before/after training artifacts referenced by that manifest;
- baseline/candidate behavioral regression evidence authorized by the protocol.

It may not receive:

- the private world artifact;
- an explicit planted-root field;
- truth-scoring metadata.

The public frozen-world manifest does not contain an explicit root identity.

The diagnostic ranking is generated and frozen before automated comparison with
the private benchmark truth.

Possible diagnostic methods include:

- deterministic random;
- lexical overlap;
- changed-record lexical overlap;
- selected-role semantic overlap;
- later Grad-Dot;
- TracIn;
- TRAK;
- influence-function backends.

A diagnostic may infer the responsible shard from the debugger-visible evidence.
That is allowed and is part of the task.

A ranking remains localization/attribution evidence rather than causal evidence
by itself.

The causal conclusion requires measured counterfactual restoration.

A future benchmark may strengthen the protocol further through an independent
evaluation process in which the investigator never sees the private world
mapping.

## Execution portability

Experiment 008 scientific parameters are hardware-independent.

The training implementation selects the best supported runtime backend in this
order:

1. Apple Metal Performance Shaders (`mps`);
2. CUDA (`cuda`);
3. CPU (`cpu`).

Machine-specific operating-system tuning, including Windows WSL memory,
processor, or swap configuration, is external to the repository and is not part
of the Experiment 008 scientific protocol.

Every trained adapter records execution provenance including:

- selected device backend;
- operating system;
- platform release;
- machine architecture;
- Python version;
- logical CPU count visible to Python;
- PyTorch intra-op thread count;
- PyTorch inter-op thread count;
- MPS build/availability;
- CUDA availability;
- PyTorch version;
- Transformers version;
- PEFT version.

Primary sibling comparisons must not silently mix hardware backends.

A clean baseline, candidate, and its counterfactual restorations used in one
primary causal comparison must use the same execution backend.

A CPU, MPS, or CUDA replication may be performed separately as a robustness
replication, but hardware backend is not a world-selection mechanism and may not
be changed to rescue a failed behavioral gate.

## Stopping rule

Experiment 008 is the final major benchmark-design iteration on this synthetic
shape substrate.

If the frozen construction cannot produce the required localized and uniquely
restorable target regression, the project will not create an Exp009 whose
purpose is to tune:

- corruption dose;
- thresholds;
- seeds;
- nuisance allocation

on the same substrate.

Instead, this synthetic substrate is retired.

The project then proceeds toward:

- stable `model_forensics` APIs;
- mature attribution backends;
- a different and more realistic regression mechanism.

## Intended claim boundary

A positive result would support the narrow statement:

> Under a frozen controlled training regime, behavioral regression evidence and
> debugger-visible training lineage can rank plausible causes, while selective
> counterfactual restoration can prospectively test whether the proposed cause
> is genuinely responsible for the target regression.

Experiment 008 would not establish universal causal attribution for neural
networks.
