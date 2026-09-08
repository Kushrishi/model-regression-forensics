# Experiment 006 Results — Semantic-Balanced Causal RCA

Status: **complete — benchmark-construction negative result; no world certified**

Experiment 006 prospectively tested whether semantic-balanced corruption could
produce a localized `triangle_large` regression suitable for causal root-cause
analysis while preserving all five protected semantic slices.

The protocol allowed exactly five frozen worlds, evaluated in order. Each world
had to pass:

1. pre-model construction gates;
2. a fresh clean-baseline capability gate;
3. a localized `triangle_large` regression gate;
4. only then, private five-way causal restoration;
5. training-order robustness;
6. only after certification, blinded diagnosis and diagnosis-driven
   intervention.

No result-motivated changes to corruption strength, thresholds, model,
optimizer, target slice, protected slices, world count, world order, or
diagnostics were permitted.

## Result summary

All five frozen worlds passed their prospective construction gates.

All five fresh clean siblings achieved perfect evaluation accuracy:

| Split | Accuracy |
| --- | ---: |
| circle_small | 1.0000 |
| circle_large | 1.0000 |
| square_small | 1.0000 |
| square_large | 1.0000 |
| triangle_small | 1.0000 |
| triangle_large | 1.0000 |
| all | 1.0000 |

The clean task therefore remained fully learnable in every world. There was no
clean-baseline capability confound.

The corrupted candidate siblings produced three distinct observed phenotypes.

### Worlds 0 and 2

Worlds 0 and 2 produced the same evaluation outcome:

| Split | Clean | Candidate | Delta |
| --- | ---: | ---: | ---: |
| circle_small | 1.0000 | 1.0000 | +0.0000 |
| circle_large | 1.0000 | 1.0000 | +0.0000 |
| square_small | 1.0000 | 0.8750 | -0.1250 |
| square_large | 1.0000 | 0.8125 | -0.1875 |
| triangle_small | 1.0000 | 1.0000 | +0.0000 |
| triangle_large | 1.0000 | 1.0000 | +0.0000 |
| all | 1.0000 | 0.9479 | -0.0521 |

The intended target did not regress, while both protected square slices
degraded beyond the frozen `0.05` protected-drift limit.

### Worlds 1 and 4

Worlds 1 and 4 produced no observed evaluation regression:

| Split | Clean | Candidate | Delta |
| --- | ---: | ---: | ---: |
| circle_small | 1.0000 | 1.0000 | +0.0000 |
| circle_large | 1.0000 | 1.0000 | +0.0000 |
| square_small | 1.0000 | 1.0000 | +0.0000 |
| square_large | 1.0000 | 1.0000 | +0.0000 |
| triangle_small | 1.0000 | 1.0000 | +0.0000 |
| triangle_large | 1.0000 | 1.0000 | +0.0000 |
| all | 1.0000 | 1.0000 | +0.0000 |

These worlds satisfied protected-behavior stability but failed the required
target-regression criterion.

### World 3

World 3 produced the strongest protected-behavior failure:

| Split | Clean | Candidate | Delta |
| --- | ---: | ---: | ---: |
| circle_small | 1.0000 | 1.0000 | +0.0000 |
| circle_large | 1.0000 | 1.0000 | +0.0000 |
| square_small | 1.0000 | 0.0000 | -1.0000 |
| square_large | 1.0000 | 0.0000 | -1.0000 |
| triangle_small | 1.0000 | 1.0000 | +0.0000 |
| triangle_large | 1.0000 | 1.0000 | +0.0000 |
| all | 1.0000 | 0.6667 | -0.3333 |

Again, `triangle_large` remained perfect while both protected square slices
failed completely.

## Five-world outcome

| World | Construction | Clean gate | Target regression | Max protected drift | Localized-regression gate |
| --- | --- | --- | ---: | ---: | --- |
| 0 | pass | pass | 0.0000 | 0.1875 | **fail** |
| 1 | pass | pass | 0.0000 | 0.0000 | **fail** |
| 2 | pass | pass | 0.0000 | 0.1875 | **fail** |
| 3 | pass | pass | 0.0000 | 1.0000 | **fail** |
| 4 | pass | pass | 0.0000 | 0.0000 | **fail** |

Observed `triangle_large` regression:

`0.0000` in all five worlds.

Frozen minimum required target regression:

`0.1500`

Final world-selection result:

**0/5 frozen worlds passed the localized-regression gate.**

## Stopping rule

The prospective protocol required stopping after the fifth world if no world
reached causal certification.

That stopping rule now applies.

Experiment 006 therefore does **not**:

- generate an additional world;
- increase corruption strength;
- weaken the target-regression threshold;
- relax protected-slice drift;
- change the semantic quotas;
- change the model or optimizer;
- change the target slice;
- alter training order to rescue the primary result;
- run a private restoration sweep;
- run order-control certification;
- run blinded diagnostics;
- reveal or use the planted candidate identity.

Any redesign belongs in a separately named follow-up experiment.

## Causal-certification status

Private causal certification was **not reached**.

The frozen protocol authorized restoration training only after a world passed
the localized-regression gate. No world did.

Accordingly:

- unique causal root certification: **not evaluated**
- five-way restoration sweep: **not evaluated**
- training-order robustness: **not evaluated**
- benchmark certification: **false**
- blinded localization: **not evaluated**
- diagnosis-driven intervention: **not evaluated**
- end-to-end RCA: **not evaluated**

The private planted root remained unopened throughout world selection.

This result therefore does not show that the causal-certification procedure
fails. The prerequisite benchmark phenotype was never produced.

## Interpretation

Experiment 006 addressed the specific benchmark-construction problem identified
after Experiment 005: corruption was controlled directly in semantic space
instead of preserving only aggregate label counts.

That intervention changed the observed failure pattern, but it did not produce
the required localized target regression.

Unlike Experiment 005, the five candidate worlds did not all collapse to one
identical behavior. Worlds 1 and 4 remained perfect, Worlds 0 and 2 showed
moderate square-slice degradation, and World 3 showed complete square-slice
collapse.

Despite that variation, one result was invariant:

> `triangle_large` remained at 1.0000 accuracy in every corrupted candidate
> world.

Thus semantic balancing was insufficient, under the frozen Experiment 006
model, task, training regime, and corruption strength, to make the planted
`triangle_large` corruption behaviorally material at the required evaluation
threshold.

The repeated protected-square degradation in three worlds also shows that the
candidate corruption can materially alter model behavior without producing the
intended target effect. This is an observed benchmark property, not evidence of
a proven optimization or representation-level mechanism.

The next benchmark-design problem is therefore not merely balancing corruption
statistics. A follow-up experiment should prospectively construct corruption
whose behavioral influence on the intended target has been demonstrated without
using the final certification evaluation to tune the result.

## What Experiment 006 establishes

Supported conclusions:

- all five prospective construction worlds were successfully executed;
- all five clean siblings achieved 96/96 evaluation accuracy;
- all five candidate worlds produced exactly zero `triangle_large` regression;
- protected behavior ranged from perfectly stable to severe square-slice
  degradation;
- semantic-balanced corruption did not reliably produce an admissible localized
  target regression under this frozen setup;
- the preregistered five-world stopping rule was followed;
- causal certification was correctly not entered.

Not supported:

- that the private causal-certification procedure succeeds or fails;
- that any observable candidate was proven to be the causal root;
- that a blinded diagnostic succeeds or fails;
- that square behavior is intrinsically more vulnerable than triangle behavior
  in other tasks or models;
- that the observed behavior has a proven optimizer-level or
  representation-level explanation;
- that the result generalizes beyond this synthetic task, pinned model, LoRA
  configuration, or corruption construction.

## Final status

**Experiment 006: COMPLETE — NEGATIVE BENCHMARK-CONSTRUCTION RESULT**

Five of five frozen worlds were executed.

Zero of five passed the localized-regression gate.

Causal certification was never reached.

The next experiment may prospectively redesign how behaviorally effective
corruptions are selected, but Experiment 006 remains frozen as a negative
result.
