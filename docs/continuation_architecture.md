# MRF utility and continuation architecture

October 3, 2026. The matched localization study stays frozen. This document plans engineering and a falsification task; it is not an implemented causal certificate or an authorization for more transformer training.

## Current implementation

`model_forensics.release_compare` validates exact-label records, declared slices and accuracy-drop tolerances. It emits deterministic input hashes and per-case differences, with exit codes 0 (pass), 1 (regression) and 2 (invalid input). The external digits fixture agrees with an independent NumPy reference. It is an author-injected feature-order bug, not a discovered cause, independent replication or benchmark win.

## Separate three layers

| Layer | Input | Output / boundary |
| --- | --- | --- |
| Regression detector (implemented) | Aligned known-good/candidate predictions, truth and policy | Declared slice regressions; no causal interpretation |
| Intervention ledger (proposed) | Candidate manifests, paired trajectory IDs, evaluation identities, restoration outputs | Provenance-linked repair evidence and completeness |
| Specificity assessment (proposed) | Complete ledger and a policy frozen before outcomes | Supported repair, ambiguous repairs, or insufficient evidence within the declared candidate set |

Keep the CLI small and framework-independent. Reuse its identity validation and canonical hashing. Do not silently extend exact labels to continuous score vectors. Viewer adapters remain optional; a dashboard is not the contribution.

## Next milestone: falsify unique attribution cheaply

Define a deterministic external debugging task with a known historical change and a distinct compensating repair that also restores accuracy. For example, a feature-order regression can be repaired by restoring input order or inversely remapping model coefficients. Those interventions can produce the same predictions despite different history. Include both in the declared candidate set, plus an ineffective intervention and incomplete evidence. Freeze identities, slices, zero-drop policy and expected ambiguity before running the fixture.

Acceptance: the same predictions yield the same regression report; two successful repairs are explicitly ambiguous; missing pairing/provenance yields insufficient evidence; a truth field never enters debugger-facing ranking or decisions; and the simple reference supplied the same intervention evidence reaches the same conclusion. This tests bookkeeping and refusal to overclaim, not new causal statistics. An intervention that repairs performance is not automatically the historical cause.

A small evidence-ledger implementation is justified only if the contract reduces a concrete debugging error or effort beyond a reference script. Compare against existing evaluation tools on the same workflow, including setup and report interpretation. Do not call a documentation comparison a runtime benchmark.

## Research gate

A paper requires a distinct contribution beyond TracIn, ordinary model comparison and supplied restoration evidence. A new study must remove the known-label semantic shortcut, include interactions and plausible alternative repairs, isolate truth, predeclare false unique-attribution and useful-abstention metrics, and justify independent worlds/sample size. No official Banking77 test access or expensive training is needed to falsify the cheap fixture.

The repository has no established open-source distribution license. Resolve rights and installability before a reusable public release. Preserve the completed M4 report even if the utility does not justify expansion.
