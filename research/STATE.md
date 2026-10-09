# Research status

Updated October 9, 2026. This page is the current status; dated records below describe their original scopes.

The matched Banking77 localization study is complete. The repository also has an
executable classification investigation workflow: model callbacks, retained
predictions/costs, repair assessment and portable input-level inspection.
The next task is the [bounded real-text training-release example](CLASSIFICATION_WORKFLOW_DEVELOPMENT_2026_10_09.md).
That prospective engineering scope follows the owner's continuation decision;
no new confirmatory study or official-test access follows. No paper is published.

## Question

After a model regresses, can controlled retraining distinguish the responsible training-data change from other changes that also improve the failed behavior?

## Completed studies

The earlier Banking77 development study used 8,001 training examples, 1,998 development examples and a pinned DistilBERT classifier. Three paired training runs produced a mean target regression of 0.129289 and a mean protected-behavior regression of 0.002711. Reversing the planted change produced greater recovery than any nuisance restoration in all three runs. However, the root and nuisance changes had different structures, so this was preliminary evidence.

The matched follow-up used two constructed worlds with five candidates each. Every candidate made symmetric label swaps on 66 records without changing text or total label counts. Three paired training runs were recorded per world; those runs are repetitions within a world, rather than six independent benchmark worlds.

| Diagnostic | World 00 root rank | World 01 root rank |
| --- | ---: | ---: |
| Deterministic random reference | 3 | 3 |
| Target-label overlap | 1 | 1 |
| Lexical Jaccard | 1 | 1 |
| Final-checkpoint Grad-Dot | 1 | 5 |
| Seven-checkpoint TracIn | 1 | 5 |

Label-overlap and lexical checks found the planted change in both worlds. The evaluated model-based methods provided no top-1 improvement in this setting. Known target labels and disjoint candidate label pairs made the simple checks particularly informative.

The matched study measures localization. It does not test whether reversing a matched candidate uniquely explains the regression. The official Banking77 test split has not been used.

See the [technical report](M4_TECHNICAL_REPORT.md), [result record](M4_RESULT.json), [replay instructions](REPRODUCE_M4.md) and [experiment history](EXPERIMENT_HISTORY.md) for the complete evidence.

## Software

PR63/64 added actual release/repair execution and interactive inspection. The
digit walkthrough retains 540 evaluation cases, model parameters and five
executions. Two repairs restore its disclosed input-order fault; ambiguity is
reported. This is application evidence, not a new diagnosis result.


The exact-label comparator checks record alignment, declared evaluation slices and accuracy-drop tolerances. Its eleven reports on the handwritten-digits example agree with an independent NumPy calculation. The example's input-order bug is supplied by the author; the comparator measures its effect rather than discovering its cause.

The [ambiguous-repair example](../docs/ambiguous-repairs.md) reverses a linear classifier's input feature order. Restoring the inputs or reversing the weights both recover the original predictions. The assessment reports `ambiguous_repairs` and leaves the historical cause `not_identified`. This demonstrates the difference between a successful repair and evidence for a unique cause.

## Historical v2 closure — 7 October

The owner [amended the governance rule](GOVERNANCE_AMENDMENT_2026_10_07.md) without
pretending independent review occurred. A development-only SmolLM2-135M-Instruct
feasibility attempt then failed its declared clean protected-behavior floor in
one of two seeds. See [the bounded closure](DEVELOPMENT_CLOSURE_2026_10_07.md).
No regressed fits, new incidents, interventions or official test access followed.
This is not confirmatory evidence and does not disprove the general diagnosis
question. The tested v2 continuation is closed; the completed negative v1 and
unsuccessful development feasibility record are preserved. The no-automatic-v3 decision was subsequently amended prospectively by the owner, without reopening v2.

## Evidence and limitations

The results concern the constructed tasks tested here. They do not establish general causal identification, superiority to modern attribution methods or cross-model generalization. Frozen protocols and result records define each study; the [claims summary](CLAIMS.md) and this page describe their current interpretation.

## Historical discovery decision — 7 October

[The dated amendment](VNEXT_OWNER_AMENDMENT_2026_10_07.md) permits a separate
discovery/infrastructure track. It broadens candidate categories prospectively to
versioned training-pipeline changes. No new model training, official test access,
confirmatory protocol, large benchmark or scientific contribution is authorized or
established. See the current contribution and incident audits.

The [October 9 provenance follow-up](VNEXT_PROVENANCE_FOLLOWUP_2026_10_09.md)
traced the two DeepFD leads to their original reports and found additional source
reconstruction differences. Neither qualifies as a release-regression incident;
the screened set still contains zero qualified cases. This does not close the
maintained discovery/software track or authorize a new training study.

A [bounded compiler-incident screen](VNEXT_COMPILER_INCIDENT_SCREEN_2026_10_09.md)
adds three public leads, including reported masking and multi-change repairs.
Two remain provenance leads only; none qualifies for a new training study. Their
compiler numerical checks must not be presented as task-performance evidence.
The ledger now offers opt-in bounded local payload hashing, which checks byte
identity rather than run authenticity or scientific responsibility.

The [ten-lead pipeline screen](VNEXT_PIPELINE_SCREEN_2026_10_09.md) is now complete:
zero additional qualified incidents, two provenance leads only, and no new
training. Several reported defects have direct code-level oracles; none supplied
the required repeated good-versus-regressed task evidence. Further expansion is
paused pending concrete new qualification evidence rather than an unbounded search.

A source-only follow-up now pins both masked-LFQ maintainer repairs and the
DeepSpeed merge parent/complete fix scope. Repair provenance improved; neither
lead supplies a reconstructed repeated training-task regression. Qualification
remains zero and the bounded follow-up is closed without training.

A [fixed-tensor LFQ replay](LFQ_FIXED_TENSOR_REPLAY_2026_10_09.md) now independently
reproduces incorrect commitment losses/gradients, the first repair's unmasked
exception and the follow-up repair's agreement with a direct numerical oracle.
Twelve CPU checks use zero trainable parameters and zero optimizer updates. This
adds executable software evidence, not trained-task restoration or a qualified
release-regression incident. No new training is authorized by this result.
