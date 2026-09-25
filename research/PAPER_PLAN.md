# Publication plan

Last updated: 2026-09-24

This is a planning document, not a manuscript and not a novelty claim.

## Working paper question

**Can a suspected versioned training change be causally certified as responsible
for a behavioral regression when plausible nuisance interventions and ordinary
training-path variability are explicitly controlled?**

## Working title

**Counterfactual Certification of Training-Data Regressions Under Stochastic
Retraining**

The title is provisional and must be changed if the final evidence supports a
narrower conclusion.

## Intended contribution shape

The paper should aim to contribute a rigorous evaluation framework and empirical
finding, not merely a new attribution score.

The core distinction is:

```
localization
    != restorative influence
    != uniquely specific causal evidence
```

The paper is useful only if readers can learn when those notions separate and
how a regression-debugging workflow can test the separation prospectively.

## Candidate venue strategy

A rolling venue such as TMLR is a strong candidate if the final contribution is
primarily an ML methodology/evaluation result. Venue choice remains open until
the research gap and evidence are stable.

Do not change the scientific question merely to fit a venue deadline.

## Required evidence before manuscript freeze

### A. Related-work gate

Refresh the related-work matrix through the current date and explicitly compare
the project with:

- modern training-data attribution/influence estimation;
- distributional/stochastic attribution;
- further-training or retraining-based attribution gold standards;
- predictive data debugging / model-diff systems;
- provenance-aware debugging; and
- regression/root-cause analysis literature outside ML where directly relevant.

For every close work record:

- problem unit;
- information available to the method;
- causal estimand;
- whether the true cause is hidden;
- whether all candidate interventions are replayed;
- how training randomness is treated;
- benchmark construction;
- scale/cost; and
- what question remains unanswered.

### B. Baseline gate

The final evaluation must include a defensible set of baselines spanning:

1. trivial/random ranking;
2. simple change-level heuristics;
3. at least one modern gradient/influence/data-attribution family when
   technically applicable;
4. a retraining/further-training reference where appropriate; and
5. the full intervention-based certification procedure.

Exact methods are selected only after the refreshed literature/feasibility
audit. A baseline may be excluded for incompatibility, but the reason must be
documented before confirmatory results.

### C. Benchmark gate

The benchmark must establish before confirmatory use:

- trustworthy clean behavior;
- localized and material target regression;
- protected-behavior stability;
- non-trivial candidate set;
- nuisance changes that are real and plausible;
- no obvious metadata/lexical shortcut revealing the root;
- deterministic construction/provenance; and
- a frozen test split not consumed during development.

### D. Stochastic evidence gate

Root restoration, nuisance restorations, and relevant controls must be compared
under matched stochastic trajectories.

Primary analysis must report distributions/effect uncertainty rather than a
single favorable retraining run.

### E. Confirmatory gate

Before loading the official confirmatory test split:

- freeze the candidate set;
- freeze the diagnostic inputs;
- freeze the ranking method(s);
- freeze the primary estimand/statistic;
- freeze success/failure thresholds;
- freeze trajectory/seed schedule;
- freeze exclusion/failure rules; and
- commit the protocol.

No threshold or candidate redesign may be performed after confirmatory outcomes
are visible.

### F. Reviewer-resistance gate

Before submission, perform an adversarial review addressing at least:

- novelty overlap;
- benchmark artificiality;
- leakage/shortcut risks;
- stochastic uncertainty;
- baseline fairness;
- compute parity;
- multiple-comparison or selection bias;
- generalization limits;
- negative/null results;
- reproducibility; and
- claims that exceed the evaluated substrate.

## Paper structure

1. Problem and motivating distinction
2. Related work and exact gap
3. Formal regression-forensics setup
4. Benchmark / versioned-change construction
5. Diagnostic and intervention framework
6. Stochastic matched-trajectory evaluation
7. Baselines
8. Results
9. Failure modes and negative results
10. Limitations / scope
11. Reproducibility statement

## Reproducibility package

A submission-quality release should make it possible to regenerate the main
tables/figures from committed configuration plus documented data access.

Required artifacts:

- exact environment/dependency lock;
- model revision;
- dataset revision/split hashes;
- all seeds/trajectory schedules;
- candidate-release hashes;
- baseline configurations;
- structured per-run metrics;
- aggregation/statistical-analysis code;
- figure/table generation code; and
- a one-command or one-workflow reproduction path for each primary result.

## Stop / pivot rules

Pivot or narrow the paper if:

- refreshed related work removes the working gap;
- plausible nuisances cannot be constructed without an obvious structural
  giveaway;
- root effects cannot be distinguished from nuisance/retraining effects under a
  scientifically reasonable design;
- expected baselines make the proposed framework redundant; or
- the final evidence supports only a benchmark-design lesson rather than a
  regression-certification method.

A narrower correct paper is preferable to a larger unsupported one.
