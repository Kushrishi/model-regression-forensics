# Related-work matrix

Status: **in progress**

## Current conclusion

The broad idea "find model failures and trace them to training data" is **not novel**.

The strongest overlap found so far is Goodfire's 2026 predictive data debugging work, which predicts behavioral changes from preference data, traces them to responsible examples/clusters, and demonstrates targeted interventions. Anthropic's 2026 model-diff work automatically surfaces behavioral differences between models, but explicitly notes that the diff does not determine the origin of those behaviors.

The working gap still to validate is:

> **Post-hoc, provenance-aware causal regression forensics across heterogeneous training changes, evaluated on a benchmark with hidden planted root causes and intervention-verified diagnoses.**

| Work | Behavior diff | Training-data attribution | Config/training-stage causes | Intervention verification | Ground-truth RCA benchmark |
|---|---:|---:|---:|---:|---:|
| Anthropic, *A "diff" tool for AI* (2026) | Yes | No | No | Feature steering | No |
| Goodfire, *Anatomy of Post-Training / Predictive Data Debugging* (2026) | Yes | Yes | Primarily preference-data / reward shaping | Yes | Planted validation, not a general RCA benchmark |
| Li et al., *Do Influence Functions Work on LLMs?* (2025) | Eval-conditioned | Yes | No | Attribution evaluation | No |
| Choe et al., *What is Your Data Worth to GPT?* (2025) | No | Yes | No | Data valuation focus | No |
| Dapaah & Grabowski, *From diagnosis to repair* (2026) | Pipeline performance | Dataset/config descriptors | Selected hyperparameters | Counterfactual | No LLM provenance benchmark |
| Anthropic Petri 2.0 (2026) | Automated behavior auditing | No | No | No | Scenario suite |

## Primary sources

- https://www.anthropic.com/research/diff-tool
- https://arxiv.org/abs/2606.12360
- https://www.goodfire.com/research/predictive-data-debugging
- https://aclanthology.org/2025.findings-emnlp.775/
- https://proceedings.neurips.cc/paper_files/paper/2025/hash/d6d26053b977f8c589669fd201615119-Abstract-Conference.html
- https://doi.org/10.1007/s11334-026-00642-8
- https://alignment.anthropic.com/2026/petri-v2/

## Questions that must be answered before a novelty claim

1. Has anyone benchmarked root-cause localization of model regressions with hidden known causes?
2. Has anyone treated training provenance as a heterogeneous causal search space spanning data, config, optimizer/schedule, prompt/template, and training stage?
3. Has anyone coupled candidate ranking with actual replay/retraining interventions as the criterion for correctness?
4. Are there public benchmarks whose unit is baseline checkpoint + regressed checkpoint + lineage + hidden cause + recovery test?
5. Can one protocol span SFT, preference optimization, and non-data changes?

If prior work answers these strongly, pivot.
