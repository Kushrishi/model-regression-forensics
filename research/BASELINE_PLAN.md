# Baseline plan

**Updated:** 2026-09-25
**Status:** prospective development plan; confirmatory baseline set not yet frozen

The baseline layer asks whether a method can rank version changes associated
with the observed regression. The separate intervention layer asks whether a
ranked change can be causally certified against alternatives.

No localization baseline may use planted-root identity.

## Baseline families

B0: deterministic random candidate ranking.

B1: structural and lexical shortcut baselines, including changed-slot count,
text-change count, label-change count, and lexical overlap with the complete
target development-eval slice.

B2: modern data attribution. TRAK is the first serious family because it is a
well-established counterfactual attribution baseline with a public BERT
classification implementation.

B3: a second attribution family such as TracIn-style checkpoint-gradient
attribution if it adds meaningful methodological coverage.

B4: exact candidate restoration as the empirical intervention reference.

B5: MRF certification using matched stochastic restoration trajectories and
explicit abstention when root and nuisance effects are not distinguishable.

## Canonical target and aggregation

The only current target contract is research/ATTRIBUTION_TARGET.md.

Primary target:

- every dev-eval example in the two incident intents;
- correct-class log-odds margin against all other classes;
- equal example weight within intent;
- equal weight across intents.

Primary attribution state:

- the regressed composite release/model.

Primary trajectory aggregation:

- class-balance target-example support within each composite trajectory;
- average support across all three composite trajectories.

Primary candidate aggregation:

- convert helpful-positive support to detracting-positive suspiciousness;
- sum suspiciousness over all changed slots in the opaque candidate manifest.

Baseline-minus-composite attribution differences are not part of the primary
ranking.

## Fairness requirements

Record model/checkpoint access, training-data access, gradients/checkpoints,
target examples, visible candidate metadata, runtime/device, number of trained
models, method version, and code revision.

Any method with privileged truth information is an oracle/reference, not a fair
localization baseline.

## Completion criteria

Development baseline work is complete when:

1. at least one serious attribution baseline runs reproducibly;
2. opaque candidate truth isolation is verified;
3. target, sign, trajectory aggregation, and candidate aggregation are frozen;
4. structural shortcut baselines are recorded;
5. compute and information access are documented;
6. certification remains separable from ranking.

Official-test outcomes may not select these choices.
