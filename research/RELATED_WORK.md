# Related work

**Updated:** October 5, 2026

Model behavior can already be traced to training examples through data attribution, counterfactual retraining and intervention-based debugging. MRF studies a narrower question: can reversal and repeated retraining distinguish a responsible release change from other changes that also repair the same behavior?

## Closest work

| Work | Primary unit | Counterfactual / intervention | Training stochasticity | Relationship to MRF |
| --- | --- | --- | --- | --- |
| Ilyas et al., *Datamodels* (ICML 2022) | Training-example subsets | Predicts dataset counterfactuals; evaluates repeated retraining | Repeated trials used to reduce training noise | Establishes that training-set counterfactual effects can be modeled; not release-change causal certification. |
| Park et al., *TRAK* (ICML 2023) | Training examples | Evaluated against counterfactual retraining-style attribution targets | Uses multiple trained models | Scalable attribution baseline for a future comparison. |
| Wei et al., *Final-Model-Only Data Attribution* (NeurIPS 2025) | Training examples | Further training is used as an attribution reference | Explicit averaging over training randomness | Particularly close to the idea that empirical intervention is a gold standard. |
| Mlodozeniec et al., *Distributional Training Data Attribution* (NeurIPS 2025) | Training examples / output distribution | Distributional attribution | **Central contribution** | Studies attribution under training randomness. |
| Goodfire, *Predictive Data Debugging / Anatomy of Post-Training* (2026) | Preference examples and clusters | Predicts training effects, traces responsible data, and demonstrates targeted interventions including planted validation | Not the primary framing | Closely related behavior-to-data debugging and targeted intervention. |
| Mo et al., *DebugLM* (2026) | Learned provenance sources | Source-specific test-time remediation | Not central | Solves provenance by learning traceability into the model; MRF is post-hoc on ordinary releases. |
| Dapaah & Grabowski, *From diagnosis to repair* (2026) | Dataset/configuration descriptors and hyperparameters | SCM-based what-if repair validated by rerunning pipelines | Not central | Shows ML-pipeline RCA plus counterfactual rerun validation already exists outside TDA. |
| Deng et al., *DeMix* (KDD 2026) | Erroneous samples + error types | Intervention-based learning and data repair | Not central | Training-data debugging with mixed error types is already an explicit research problem. |
| Model-update / negative-flip literature | Updated-model predictions | Usually mitigation rather than root replay | Varies | Establishes model regression after updates as a mature phenomenon; MRF focuses on causal origin among version changes. |

## Implications of the completed studies

The early Banking77 pilot used candidates with different edit structures. Its restoration results therefore remain development evidence. The later matched benchmark removed that structural difference, but target-label and lexical baselines still identified both planted changes. Grad-Dot and TracIn added no first-place ranking benefit.

The deterministic competing-repair example illustrates a separate limitation: two interventions can restore the same predictions without identifying a unique historical cause.

A useful follow-up would need candidates that remain plausible after simple semantic checks, competitive attribution baselines and a defined rule for reporting ambiguity. Repeated retraining alone does not supply a distinct contribution. The current studies are documented in the [technical report](M4_TECHNICAL_REPORT.md); the [roadmap](ROADMAP.md) describes the next design work.

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
