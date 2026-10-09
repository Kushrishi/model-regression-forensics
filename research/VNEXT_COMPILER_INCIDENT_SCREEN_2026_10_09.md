# Bounded follow-up: compiler-related public regression leads

Date: 2026-10-09 UTC. Metadata/source inspection only; no reproductions, model
downloads, training, intervention measurements or new scientific results.

This supplements, rather than replaces, the earlier 12-row incident audit and
DeepFD provenance rejection. A bounded search of public accuracy-regression
reports yielded three PyTorch leads worth reading. Search hits alone are not
qualified incidents. These cases are compiler/numerical pipeline changes, not
training-data edits. No category is silently pooled with the completed M4 study.

## Evidence and dispositions

### PyTorch #188912: masking and unmasking in SigLIP AMP training

[Primary report](https://github.com/pytorch/pytorch/issues/188912), inspected
2026-10-09. The reporter describes 2.12.1 passing, 2.13.0 failing, and subsequent
main passing a compiled-gradient/reference accuracy check. The chronology names
`b2bc2b83bce` / PR #182897, `93cce594152` / #183680,
`a8e63623df8` / #185856, and `f41aca09a57` / #183661: a reported initial
regression, masking change, unmasking change, and later incidental restoration.
The reproducer requires a full-SM H100 and a particular cuDNN-enabled path;
reduced-SM/MIG or cuDNN-disabled builds reportedly pass. These are reporter
observations, not independent verification here.

**Retain as a provenance lead, not a qualified benchmark.** Masking makes it
conceptually relevant to distinguishing restoration from responsibility. But the
metric is compiler numerical agreement, not demonstrated task-performance loss
after repeated stochastic training. Full immutable versions, model/input artifact
identities, upstream dependency licenses and run costs still need qualification.
No defensible CPU-only reproduction or counterfactual training cost is established.
Published root/fix chronology must stay evaluator-only in a future design; public
availability and prior knowledge create answer-leakage risks that require disclosure.

### PyTorch #188261: two precision paths

[Primary report](https://github.com/pytorch/pytorch/issues/188261), inspected
2026-10-09. It identifies commit
`6f94c6e16e1bf562f8f5514953cf9b8d47d07cdd` / #184366, good and bad nightly
boundaries, and an Intel Arc B580 reproducer. The reporter describes an AMP
gradient mismatch, backend isolation, and restoration on a later nightly requiring
both a narrow-cast optimization change and disabling a separate saved-tensor
precision path. This is an externally reported multi-change repair lead.

**Retain as a provenance lead only.** Existing bisect and source inspection already
localize the issue; no incremental diagnostic advantage is demonstrated. The
reported precision metric is not repeated trained-model behavior. Nightly artifacts,
hardware access, dependency licenses, complete candidate diffs and intervention
cost remain unqualified. No repro was run, and no stochastic responsibility set
can be inferred from the report alone.

### PyTorch #186084: fusion and downstream numerical effects

[Primary report](https://github.com/pytorch/pytorch/issues/186084), inspected
2026-10-09. The supplied two-rank GPU reproducer checks kernel-count changes
and a small fp32 output difference. A separate production report describes roughly
2 dB PSNR loss in low-precision video inference. The compact reproducer does not
itself establish that production behavior. It toggles an already named compiler
option rather than presenting multiple unresolved training-release candidates.

**Reject as currently suitable.** This is inference/compiler behavior, not a
qualified stochastic training-release incident. Production model/data identities
and licenses are not established here; specialized hardware and a complete
behavioral reproducer are absent. A direct known-option intervention is already
the reported remedy. No new cost or performance claim is made.

## Contribution implications and stopping decision

These reports weaken the overly broad proposition that real multi-change
regressions cannot be found. They do not overturn the earlier zero-qualified
result: this follow-up adds **three screened leads, zero qualified incidents**.
Two are retained for provenance investigation only, not proposed training cases.
Compiler numerical agreement must not be renamed task accuracy or training
stochasticity. Any future category broadening beyond training releases needs an
explicit prospective owner decision, not a convenient substitute benchmark.

Masking, bisect, group reverts and repeat testing remain prior-art mechanisms;
finding examples does not make them MRF contributions. The defensible near-term
software role is an auditable evidence ledger: explicit visible changes,
counterfactual semantics, failed/timeout attempts, repeat identities and supplied
set-valued decisions. Hash verification improves its engineering reliability,
not scientific novelty. A new diagnosis algorithm or publication claim is not
justified. New training remains unauthorized.

A future research proposal must demonstrate task-behavior relevance, executable
counterfactuals at proportionate cost and a decision gap after complete visible
diff inspection and cheap intervention baselines. If those baselines safely solve
the cases, the scientific need is falsified, even if the ledger remains useful.
