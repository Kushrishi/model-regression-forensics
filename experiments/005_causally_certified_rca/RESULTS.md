# Experiment 005 Results — Causally Certified RCA

Status: **complete — benchmark-construction negative result; no world certified**

Experiment 005 prospectively required a generated world to pass, in order:

1. pre-model construction gates;
2. a fresh clean-baseline capability gate;
3. a localized `triangle_large` regression gate;
4. private five-way causal certification;
5. one frozen training-order robustness control;
6. only then, blinded diagnosis and diagnosis-driven intervention.

The protocol allowed at most five deterministic candidate worlds. No threshold,
corruption-strength, model, optimizer, seed, diagnostic, protected-slice, or
world-count change was permitted after results were observed.

## Result summary

All five allowed worlds passed the declared pre-model construction checks.

All five fresh clean siblings then achieved:

| Split | Accuracy |
| --- | ---: |
| circle_small | 1.0000 |
| circle_large | 1.0000 |
| square_small | 1.0000 |
| square_large | 1.0000 |
| triangle_small | 1.0000 |
| triangle_large | 1.0000 |
| all | 1.0000 |

Thus the clean task remained fully learnable under every world attempt. There
was no clean-baseline capability confound.

All five corrupted candidate siblings then converged to the same observed
behavior:

| Split | Clean | Candidate | Delta |
| --- | ---: | ---: | ---: |
| circle_small | 1.0000 | 1.0000 | +0.0000 |
| circle_large | 1.0000 | 1.0000 | +0.0000 |
| square_small | 1.0000 | 0.0000 | -1.0000 |
| square_large | 1.0000 | 0.0000 | -1.0000 |
| triangle_small | 1.0000 | 1.0000 | +0.0000 |
| triangle_large | 1.0000 | 1.0000 | +0.0000 |
| all | 1.0000 | 0.6667 | -0.3333 |

This pattern is behaviorally consistent with an `ACCEPT`-only collapse: all
canonical ACCEPT slices remained correct and both canonical REJECT slices
failed completely.

The intended target, `triangle_large`, did not regress in any world.

Observed target regression in every candidate world:

`1.0000 - 1.0000 = 0.0000`

Frozen minimum required target regression:

`0.1500`

Observed protected drift on both square slices in every world:

`1.0000`

Frozen maximum protected-slice drift:

`0.0500`

Therefore every candidate world failed both:

- the target-regression requirement; and
- the protected-behavior requirement.

## Five-world outcome

| Allowed world | Construction | Clean gate | Target regression | Protected behavior | Localized-regression gate |
| --- | --- | --- | ---: | --- | --- |
| 1 | pass | pass | 0.0000 | fail | **fail** |
| 2 | pass | pass | 0.0000 | fail | **fail** |
| 3 | pass | pass | 0.0000 | fail | **fail** |
| 4 | pass | pass | 0.0000 | fail | **fail** |
| 5 | pass | pass | 0.0000 | fail | **fail** |

The five candidate datasets were distinct, while the clean model-facing dataset,
training protocol, pinned model revision, seed, and evaluation task remained
fixed.

Final benchmark-construction result:

**0/5 allowed worlds passed the localized-regression gate.**

## Stopping rule

The prospective protocol explicitly required stopping if none of the five
allowed worlds certified.

That stopping rule now applies.

Experiment 005 therefore does **not**:

- generate a sixth world;
- increase corruption strength;
- weaken the target-regression threshold;
- relax protected-slice drift;
- change the target slice;
- change the model or optimizer;
- alter training order to rescue the primary result;
- run private restoration sweeps;
- run the order-control certification stage;
- run blinded diagnostics;
- reveal or use private planted-candidate identity.

Any result-motivated redesign belongs in a separately named follow-up
experiment.

## Causal-certification status

Private causal certification was **not reached**.

The protocol allowed five-way restoration training only after a world passed
the clean and localized-regression gates. Because all five worlds failed the
localized-regression prerequisite, no restoration sweep was authorized.

Accordingly:

- unique causal root certification: **not evaluated**
- training-order robustness: **not evaluated**
- benchmark certification: **false**
- blinded localization: **not evaluated**
- diagnosis-driven intervention: **not evaluated**
- end-to-end RCA: **not evaluated**

This distinction matters: Experiment 005 is not evidence that the frozen causal
certification procedure itself fails. It is evidence that the prospectively
frozen world generator did not produce an admissible localized regression on
which that procedure could be tested.

## Interpretation

Experiment 005 successfully removed one known confound from Experiment 004:
aggregate clean and corrupted training-label counts were exactly preserved.

That change was not sufficient to produce the desired localized
`triangle_large` regression.

Instead, five distinct balanced corruption worlds produced the same broad,
label-aligned failure phenotype: canonical ACCEPT behavior remained intact while
canonical REJECT behavior collapsed.

The repeatability across all five prospectively allowed worlds suggests that,
under this frozen task and training setup, the behavioral outcome is dominated
by structure shared across the corruption construction rather than by the exact
world-specific record assignment.

That is an empirical observation about this benchmark design, not a general
claim about transformers, LoRA, neural-network optimization, or label noise.

The experiment therefore exposes a new benchmark-design requirement:

> Preserving global class counts is not sufficient. A causally certifiable RCA
> benchmark must also control the semantic effect of the corruption so that the
> induced failure is localized to the intended behavior while protected
> behaviors remain stable.

## What Experiment 005 establishes

Supported conclusions:

- the explicit-policy clean task remained stable at 96/96 across all five fresh
  clean runs;
- all five candidate constructions satisfied the prospectively frozen
  balancing and anti-shortcut checks;
- all five distinct candidate worlds failed the same localized-regression gate;
- global label-count preservation alone did not prevent a broad label-aligned
  collapse under this construction;
- the preregistered stopping rule was followed without result-motivated tuning.

Not supported:

- that the private causal-certification procedure succeeds or fails;
- that the frozen diagnostic succeeds or fails;
- that one candidate shard was the causal root;
- that the observed collapse has a proven optimizer-level or
  representation-level mechanism;
- that the result generalizes beyond this synthetic task, model, LoRA setup, or
  corruption construction.

## Final status

**Experiment 005: COMPLETE — NEGATIVE BENCHMARK-CONSTRUCTION RESULT**

No world was certified.

The next experiment may redesign the corruption mechanism prospectively, but
Experiment 005 remains frozen as a negative result.
