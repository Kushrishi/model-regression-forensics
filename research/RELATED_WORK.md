# Related work and contribution boundary

**Updated:** 2026-09-24  
**Status:** active literature audit

The broad idea "trace model behavior or failures back to training data" is not
novel. Neither are stochastic training-data attribution, counterfactual
retraining, planted-data validation, or intervention-based repair by themselves.

The current MRF opportunity is narrower:

> **Given a known-good release, a regressed release, and a finite set of
> versioned candidate changes, can a suspected cause be *certified* by
> counterfactual restoration evidence that is distinguishable from plausible
> non-root interventions and ordinary retraining variability?**

This is a working contribution boundary, not a novelty claim.

## Closest work

| Work | Primary unit | Counterfactual / intervention | Training stochasticity | Relationship to MRF |
| --- | --- | --- | --- | --- |
| Ilyas et al., *Datamodels* (ICML 2022) | Training-example subsets | Predicts dataset counterfactuals; evaluates repeated retraining | Repeated trials used to reduce training noise | Establishes that training-set counterfactual effects can be modeled; not release-change causal certification. |
| Park et al., *TRAK* (ICML 2023) | Training examples | Evaluated against counterfactual retraining-style attribution targets | Uses multiple trained models | Strong scalable attribution baseline; MRF cannot compete by merely producing another attribution score. |
| Wei et al., *Final-Model-Only Data Attribution* (NeurIPS 2025) | Training examples | Further training is used as an attribution reference | Explicit averaging over training randomness | Particularly close to the idea that empirical intervention is a gold standard. |
| Mlodozeniec et al., *Distributional Training Data Attribution* (NeurIPS 2025) | Training examples / output distribution | Distributional attribution | **Central contribution** | Removes any novelty claim based on handling stochastic retraining alone. |
| Goodfire, *Predictive Data Debugging / Anatomy of Post-Training* (2026) | Preference examples and clusters | Predicts training effects, traces responsible data, and demonstrates targeted interventions including planted validation | Not the primary framing | Very close on behavior-to-data debugging and intervention; MRF must distinguish release-diff certification rather than claim causal data debugging broadly. |
| Mo et al., *DebugLM* (2026) | Learned provenance sources | Source-specific test-time remediation | Not central | Solves provenance by learning traceability into the model; MRF is post-hoc on ordinary releases. |
| Dapaah & Grabowski, *From diagnosis to repair* (2026) | Dataset/configuration descriptors and hyperparameters | SCM-based what-if repair validated by rerunning pipelines | Not central | Shows ML-pipeline RCA plus counterfactual rerun validation already exists outside TDA. |
| Deng et al., *DeMix* (KDD 2026) | Erroneous samples + error types | Intervention-based learning and data repair | Not central | Training-data debugging with mixed error types is already an explicit research problem. |
| Model-update / negative-flip literature | Updated-model predictions | Usually mitigation rather than root replay | Varies | Establishes model regression after updates as a mature phenomenon; MRF focuses on causal origin among version changes. |

## What MRF must not claim

MRF must not claim novelty for:

- tracing model behavior to training examples;
- identifying erroneous training data;
- predicting the effect of training-data changes;
- validating attribution through retraining;
- planted-data debugging;
- stochastic/distributional attribution;
- model-update regression detection;
- pipeline root-cause analysis;
- provenance-aware debugging in general.

## Residual research question

The potentially distinct object is a **release-change certification problem**:

```text
known-good release
      +
regressed release
      +
finite versioned change set
      ↓
candidate diagnosis
      ↓
restore every candidate separately
      ↓
repeat under matched stochastic trajectories
      ↓
compare root recovery with nuisance-recovery distribution
      ↓
certify / abstain
```

The causal claim is intentionally narrower than generic causal identification.
It asks whether a candidate change's restoration effect is specifically
distinguishable from realistic alternatives under the frozen benchmark.

## Consequences for Exp009

### Nuisance v2 remains development-only

Nuisance v2 is structurally mismatched with the planted root:

- root: label changes with unchanged text;
- nuisances: label and text changes.

Therefore nuisance v2 must **not** be used to claim difficult blinded
localization.

It can still answer a narrower development question:

> Is the root restoration effect materially separable from the restoration
> effects of plausible non-root release changes under matched training
> trajectories?

That makes the existing 21-run MPS plan a **certification-layer development
pilot**, not a paper-ready end-to-end benchmark.

### Paper architecture

A defensible paper should likely separate:

1. **candidate generation / localization** — supplied by simple heuristics and
   modern attribution baselines;
2. **counterfactual certification** — MRF's root-vs-nuisance repeated
   intervention layer;
3. **abstention** — failure to certify is a valid result when interventions are
   not distinguishable.

MRF should be evaluated partly by whether it refuses to over-certify ambiguous
cases.

## Baseline requirement

Before confirmatory evidence, the benchmark must include a prospectively chosen
set of baselines spanning:

1. random/change-size controls;
2. simple lexical/change-locality heuristics;
3. at least one modern gradient/influence attribution method applicable to the
   DistilBERT classifier;
4. a retraining/further-training reference where computationally feasible; and
5. exhaustive counterfactual certification.

Candidate methods should be aggregated from example-level attribution to
version-change-level scores using a rule frozen before confirmatory outcomes.

See `research/BASELINE_PLAN.md`.

## Decision on the existing development pilot

The current literature audit does **not** kill the nuisance-v2 MPS pilot.

It changes its purpose.

The pilot is worth running only as a bounded development study of:

- root-vs-nuisance recovery effect structure;
- paired stochastic variability;
- protected-behavior movement;
- feasibility/power for a later confirmatory certification rule.

It is not sufficient for a publication-level localization benchmark.

No publication claim should depend on v2 structural matching.

## Primary sources

- Ilyas et al. (2022), *Datamodels: Understanding Predictions with Data and
  Data with Predictions*, ICML: https://proceedings.mlr.press/v162/ilyas22a.html
- Park et al. (2023), *TRAK: Attributing Model Behavior at Scale*, ICML:
  https://proceedings.mlr.press/v202/park23c
- Wei et al. (2025), *Final-Model-Only Data Attribution with a Unifying View of
  Gradient-Based Methods*, NeurIPS:
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/99d7326032bbed26de1b244beaff6a84-Abstract-Conference.html
- Mlodozeniec et al. (2025), *Distributional Training Data Attribution*,
  NeurIPS:
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html
- Goodfire (2026), *Predictive Data Debugging*:
  https://www.goodfire.com/research/predictive-data-debugging
- Mo et al. (2026), *DebugLM*: https://arxiv.org/abs/2603.17884
- Dapaah & Grabowski (2026), *From diagnosis to repair*:
  https://doi.org/10.1007/s11334-026-00642-8
- Deng et al. (2026), *DeMix*, KDD:
  https://doi.org/10.1145/3770855.3817774

## Open questions before confirmatory work

1. Which modern attribution baseline is technically fair and reproducible on
   the frozen DistilBERT/Banking77 substrate?
2. What change-level aggregation converts example-level attribution scores into
   a candidate release-change ranking without using hidden truth?
3. What structurally matched candidate construction is needed for a future
   end-to-end localization benchmark?
4. What paired effect statistic and multiplicity procedure are appropriate for
   root-vs-multiple-nuisance certification?
5. How many trajectories are required for useful confirmatory uncertainty?
6. Does a directly overlapping release-diff benchmark appear before submission?

If the literature answers the core question more directly, narrow or stop the
project rather than overstate novelty.
