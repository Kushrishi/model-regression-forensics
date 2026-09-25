# Related-work matrix

Status: **in progress**

Last refreshed: **2026-09-12**

## Current conclusion

The broad idea "find model failures and trace them to training data" is **not novel**.

The bar is now substantially higher than it was when this project began:

- Goodfire's 2026 predictive data debugging work predicts behavioral changes from
  preference data, traces them to responsible examples/clusters, and demonstrates
  targeted interventions.
- Distributional Training Data Attribution (Mlodozeniec et al., NeurIPS 2025)
  explicitly treats stochastic training as a distribution over possible model
  outcomes rather than as one privileged retraining run.
- Final-Model-Only Data Attribution (Wei et al.) uses further training with
  adjustment and averaging as an attribution gold standard and studies how
  gradient methods approximate that counterfactual quantity.
- DebugLM (Mo et al., 2026) trains models to expose provenance tags linking
  behavior to training sources, although provenance is deliberately learned into
  the model rather than reconstructed post hoc from an ordinary training run.
- Anthropic's 2026 model-diff work automatically surfaces behavioral differences
  between models, but explicitly does not establish the training origin of those
  differences.

The working gap still to validate is therefore narrower:

> **Post-hoc regression forensics over versioned training-lineage changes,
> evaluated using hidden known causes and full counterfactual retraining, where
> causal certification requires the planted cause's intervention effect to be
> distinguishable from nuisance interventions and ordinary training variability.**

This is a working gap, not a novelty claim.

| Work | Behavior diff / target behavior | Training-data attribution / provenance | Heterogeneous config or training-stage causes | Intervention / counterfactual verification | Hidden known-cause regression benchmark | Explicit stochastic-training treatment |
|---|---:|---:|---:|---:|---:|---:|
| Anthropic, *A "diff" tool for AI* (2026) | Yes | No | No | Feature steering | No | No |
| Goodfire, *Predictive Data Debugging / Anatomy of Post-Training* (2026) | Yes | Yes | Primarily preference-data / reward shaping | Yes | Planted validation, not a general version-regression RCA benchmark | Not the central framing |
| Mlodozeniec et al., *Distributional Training Data Attribution* (NeurIPS 2025) | Target measurement | Yes | No | Distributional attribution framework | No versioned regression-forensics benchmark | **Yes** |
| Wei et al., *Final-Model-Only Data Attribution* (2024) | Evaluation function | Yes | No | Further-training gold standard | No | Averages over training randomness |
| Mo et al., *DebugLM* (2026) | Yes | Learned source provenance | Multi-stage data sources | Test-time source-specific remediation | No ordinary post-hoc hidden-root benchmark | Not the central framing |
| Li et al., *Do Influence Functions Work on LLMs?* (Findings of EMNLP 2025) | Eval-conditioned | Yes | No | Attribution evaluation | No | Discusses fine-tuning/convergence limitations |
| Choe et al., *What is Your Data Worth to GPT?* (2025) | No | Yes | No | Data valuation focus | No | No |
| Dapaah & Grabowski, *From diagnosis to repair* (2026) | Pipeline performance | Dataset/config descriptors | Selected hyperparameters | Counterfactual what-if analysis | No LLM provenance benchmark | Not the central framing |
| Anthropic Petri 2.0 (2026) | Automated behavior auditing | No | No | No | Scenario suite | No |

## Consequences for Experiment 009

Experiment 009 must not claim novelty for any of the following ideas in isolation:

- accounting for stochastic training variability;
- averaging attribution effects over repeated runs;
- tracing undesirable behavior to training data;
- planted-data validation;
- intervening on training data and observing behavior change;
- attaching provenance to training sources;
- behavioral model diffing.

The experiment is scientifically useful only if the combined regression-forensics
problem remains distinct:

1. start from a clean model release and a regressed model release;
2. expose several versioned debugger-visible training changes;
3. keep one known root hidden from diagnosis;
4. rank plausible causes post hoc from ordinary model/lineage evidence;
5. retrain every candidate intervention rather than only the suspected cause;
6. repeat interventions over paired training trajectories;
7. require the root's localized recovery to exceed nuisance-intervention and
   retraining variability under a prospectively frozen rule.

## Primary sources

- https://www.anthropic.com/research/diff-tool
- https://www.goodfire.com/research/predictive-data-debugging
- https://arxiv.org/abs/2606.12360
- https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html
- https://arxiv.org/abs/2506.12965
- https://arxiv.org/abs/2412.03906
- https://arxiv.org/abs/2603.17884
- https://aclanthology.org/2025.findings-emnlp.775/
- https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6d26053b977f8c589669fd201615119-Abstract-Conference.html
- https://doi.org/10.1007/s11334-026-00642-8
- https://alignment.anthropic.com/2026/petri-v2/

## Questions that must be answered before a novelty claim

1. Has anyone benchmarked post-hoc root-cause localization of **model-version
   regressions** with hidden known training-lineage causes?
2. Has anyone treated a set of versioned training artifacts as a causal search
   space and required every candidate to be replayed/retrained under matched
   stochastic trajectories?
3. Has anyone used root-vs-nuisance **intervention-effect separation** rather than
   one successful retraining outcome as the causal correctness criterion?
4. Are there public benchmarks whose unit is baseline checkpoint + regressed
   checkpoint + lineage diff + hidden root + repeated recovery distributions?
5. Can the eventual protocol extend beyond data changes to prompt/template,
   optimizer/schedule, checkpoint, or multi-stage training changes without
   changing the causal estimand?
6. Does a directly overlapping benchmark or method appear before publication?

If prior work answers these strongly, pivot the claim rather than overstating
novelty.
