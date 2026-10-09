# Bounded vNext contribution audit — 7 October 2026

Status: discovery / engineering only. **No established scientific gap or training
proposal.** This extends the existing October dossier rather than reopening v2.
The dossier's earlier attribution review remains useful; this pass adds software
regression isolation, stochastic testing and an externally grounded case screen.
No independent review, training, official test access or outreach occurred.

## Proposed object and estimand

The proposed object is a versioned training-pipeline release incident, with the
complete engineer-visible changes between known-good and regressed releases.
For a declared change-set intervention and declared stochastic retraining process,
measure the distribution of target behavior and protected-behavior restoration.
A responsible set is meaningful only relative to this specification. A historical
fix, a restorative intervention and an uniquely responsible set are different
objects. No numerical restoration/certification thresholds are selected here.

A useful comparison would ask whether a policy reduces wrong-specific diagnoses
at useful incident-level commitment coverage and matched intervention cost, versus
complete diff inspection, rank-then-intervene and delta-debugging policies. Seeds,
examples and candidates are not independent incidents. Returning all candidates
or abstaining everywhere cannot alone support a useful contribution.

## Closest-work comparison

Reading scope is explicit: existing dossier full-paper notes are retained, current
proceedings/author sources were rechecked where accessible; new delta-debugging,
TERA and Gopher sources were read at paper/abstract or author-artifact level. This
is a bounded assessment, not an exhaustive novelty proof or reproduction.

| Primary work | Established component / evidence access | Relation to the proposed decision |
| --- | --- | --- |
| [DATE-LM, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e1ebda145808ca45774993fb67314894-Abstract-Datasets_and_Benchmarks_Track.html) | Task-specific LLM attribution evaluation, selection/filtering/factual tasks, strong simpler competitors | A new ranking benchmark or adding an LLM is insufficient. Release incidents, complete diffs and decision risk would have to add measured value. |
| [Which Influence Are We Estimating?, v1](https://arxiv.org/html/2609.31214v1) | Behavior, intervention and counterfactual training-process specification; expectation over randomness; mismatched estimands change exact rankings | Explicit estimands are required hygiene, already prior art. “Counterfactual responsibility” wording does not create novelty. |
| [Distributional TDA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html) | How output distributions across training runs depend on training data | Accounting for seeds or distributions is not a contribution. A release decision requires a compatible stochastic target and evidence budget. |
| [Group-level synthetic-data signals, v1](https://arxiv.org/abs/2610.00779v1) | Interacting groups, individual-signal insufficiency and a cheap compute-budget diagnostic for group estimation | Group units, interactions and budget awareness are occupied. This source addresses curation, not by itself unique release responsibility; that difference remains unvalidated. |
| [DeMix, v2](https://arxiv.org/abs/2606.11616v2) | Mixed data-error types and influence-vector diagnosis | Diagnosing labels/features/spurious correlations is prior art. Existing dossier full-paper notes retained; current v2 HTML fetch failed, so no new implementation/cost assertion is made. |
| [Gopher: Interpretable Data-Based Explanations for Fairness Debugging](https://arxiv.org/abs/2112.09745) | Causal responsibility defined through removing/updating coherent training subsets; approximate top-k explanation search | Intervention-defined responsibility and data-debugging repair are established. MRF must not relabel these as a new causal framework. |
| [mlwhatif, SIGMOD 2023 author paper](https://ssc.io/pdf/mlwhatif.pdf), [pinned artifact](https://github.com/stefan-grafberger/mlwhatif/tree/90bd5003c1e1ef0a51545455383d89e7e26a6d01); [ArgusEyes](https://github.com/amsterdata/arguseyes) | Extract native pipeline plans, execute/optimize what-if analyses; declarative pipeline issue checks | A generic diagnosis-to-repair harness is useful engineering, not an unoccupied research contribution. Need compare actual evidence/cost rather than claim all pipeline tools identify unique release causes. |
| [DeepFD author artifact](https://github.com/ArabelaTso/DeepFD/tree/9fe089510cf10af35b0c2cba024d9732a1048df4), [DEFault artifact](https://github.com/SigmaJahan/DEFault-DNN-Debugging/tree/9a1f12d8b7a355f07ca99cd34fd6322e0b8130c3) | Runtime/static training features to multi-label fault diagnosis/localization and repair suggestions | Multi-fault outputs and pipeline localization are already studied. Type classification differs from a fully specified version-change counterfactual, but this distinction alone has no demonstrated benefit. |
| [Classical delta debugging: Simplifying and Isolating Failure-Inducing Input](https://www.st.cs.uni-saarland.de/papers/tse2002/) | Repeated tests of subsets/complements isolate a 1-minimal failure-inducing input/change set; unresolved tests can occur | Intervention search and minimal failure-inducing sets are prior art. A 1-minimal set is not necessarily globally minimum, uniquely responsible or historically causal. |
| [ProbDD author artifact](https://github.com/Amocy-Wang/ProbDD), [analysis / CDD](https://arxiv.org/abs/2408.04735) | Probabilistic relevance model schedules reduction tests; simpler skipping policies can explain gains | ProbDD's probabilistic search model is **not automatically** a model of stochastic training outcomes. Efficient group queries and test budgets are not new; distinguish search randomness from a flaky property oracle. |
| [Delta Debugging for CPS with Flaky Test Executions](https://arxiv.org/abs/2607.25695) | Repeated statistical failure analysis plus reduction under nondeterministic simulation | “Delta debugging plus repetitions” is occupied even outside ML. Domain transfer alone is weak; applicability to declared retraining must be checked before claiming guarantees. |
| [TERA, ISSTA 2021 author PDF](https://www.cs.cornell.edu/~saikatd/papers/tera-issta21.pdf); [FLASH](https://www.cs.cornell.edu/~saikatd/papers/flash-issta20.pdf) | Reliability/runtime tradeoffs in stochastic ML regression tests; algorithmic randomness and flaky tests | Stochastic ML tests and cost-aware repeat choices are prior art. Optimizing tests differs from naming a change, but cost/uncertainty alone cannot establish distinctness. |
| [git bisect official documentation](https://git-scm.com/docs/git-bisect) | Good/bad versions, binary localization along history, automated property tests, skip untestable commits | Mandatory cheap baseline when chronological/monotonic assumptions hold. Retraining flakiness, interacting changes and equivalent repairs can break simple semantics; these must be observed in real cases, not stipulated to defeat bisect. |
| [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a.html), direct diff / rank-then-intervene | Reject-option prediction; ordinary engineering inspection and low-cost targeted reversions | Abstention/risk–coverage/set reporting are established ingredients. No advantage exists if complete diffs or one intervention safely settle every case. |

## Contribution assessment

1. **Distinctness:** not established. None of the bounded sources checked here was
shown to solve exactly the complete-diff, stochastic release-responsibility decision
at an equivalent intervention budget. That is absence of a demonstrated collapse
in this pass, **not evidence of novelty**. The combined prior-art burden is high.
2. **Nonclaims:** ranking, influence approximations, counterfactual specification,
stochastic attribution, groups/interactions, retraining repair, pipeline what-if,
delta debugging, probabilistic test scheduling, abstention and risk–coverage are
not MRF inventions. Completed v1 evidence does not supply a difficult new regime.
3. **Plausible value:** a carefully sourced incident collection with complete diffs,
explicit process-dependent responsibility sets and matched-cost comparisons could
be useful if realistic nontrivial cases exist. An engineering ledger is useful
without a scientific novelty claim. A new selective decision protocol or method
would require improvement over established policies; current evidence supplies none.
4. **Falsification:** stop a scientific proposal if realistic incidents require
hiding normal engineer evidence or fabricated distractors; if diffs/rank-one/
bisect/ddmin solve them safely at equal cost; if responsibility cannot be defined
under repeated interventions; or if existing tools provide the same useful risk,
coverage and cost. No broader rebranding should rescue that result.

## Decision at this boundary

The [incident screen](VNEXT_INCIDENT_AUDIT_2026_10_07.md) yields **zero qualified
training-release benchmark incidents** in this bounded pass. No three-case gate,
nontrivial diff case, stochastic oracle or measured intervention budget has been
established. **NO-GO for new training or a confirmatory benchmark proposal now.**
Retain minimal infrastructure and the passive public submission path. Future
public evidence may justify reopening the incident screen; endless literature
churn or a new model search is not the next action.

Source-access note: the old mlwhatif author PDF URL returned404; the coauthor
PDF and pinned artifact provide the current route. The classical delta-debugging
author landing page was verified through search after a direct fetch timeout.
No inaccessible page is treated as a newly reproduced result.

## 9 October: retained real-defect workflow decision

This assessment uses the accepted [LFQ fixed-tensor record](LFQ_FIXED_TENSOR_REPLAY_2026_10_09.json), its [scope and reproduction instructions](LFQ_FIXED_TENSOR_REPLAY_2026_10_09.md), the three source versions and the explicit squared-error/gradient oracle. It does not rerun models or turn this software case into a trained-task incident.

The engineer decision is whether a candidate commitment-loss repair preserves both masked and unmasked behavior on the declared tensors. Evidence is available to all workflows: complete pinned source files/history, both input tensors, masks, exact configuration and the oracle. No changes are concealed.

| Candidate/action | Evidence from the retained replay | Ordinary baseline action | Decision supported |
| --- | --- | --- | --- |
| Reported source | Masked loss and gradients disagree with the oracle; two-batch unmasked call raises RuntimeError | Direct squared-error/gradient checks and basic shape tests | Reproduce the defect; no training required to establish this local failure |
| First repair | Both masked fixtures match the oracle; both unmasked fixtures raise UnboundLocalError | Test mask=None in addition to the repaired branch; inspect initialization of input_for_entropy | Reject this candidate as a complete repair of the declared paths |
| Follow-up repair | All four declared fixtures return and match the loss/gradient oracle | Run the same four checks against the candidate | Accept these exercised paths only; broader correctness and trained-task restoration remain unknown |
| MRF-specific diagnosis | No separate algorithm was compared on this case | Give source review and targeted tests the same evidence | No demonstrated incremental diagnosis benefit or cost reduction |

The source changes explain the observed sequence: the reported implementation reuses an input after entropy-related reshaping/masking; the first repair separates that input but leaves the entropy variable uninitialized when no mask is supplied; the follow-up initializes it for both paths. This is source-grounded interpretation consistent with retained numerical outputs, not an independently timed developer study.

Local source SHA-256 readback matches all three accepted digests. The retained result has twelve version/fixture checks, zero optimizer updates and zero trainable parameters. No additional test results are claimed by this assessment.

### Continuation decision

**Keep this as a bounded regression-testing and repair-review example. Do not use it to justify a new training run, automated diagnosis method or causal-specificity benchmark.** The observed practical decision is already settled by ordinary branch coverage and the explicit oracle. A replay/evidence package can be useful without claiming novel diagnosis.

A historical known-good/regressed trained-task pair, licensed pinned task/data, repeated restoration evidence and a surviving comparison against complete diff inspection remain missing. They cannot be supplied by the existing fixed tensors. No further general incident search or artificial difficulty is justified by this case.

Research continuation remains conditional on concrete new evidence supporting a different unresolved engineer decision. Until then, maintain the completed localization study and this executable example, and allocate substantial experiment effort elsewhere. This decision does not claim that all release debugging is easy or that ML regression diagnosis is solved.
