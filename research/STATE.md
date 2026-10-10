# Research status

Updated October 10, 2026. This page is the current status; dated records below describe their original scopes.

The matched Banking77 localization study is complete. The repository also has an
executable classification investigation workflow: model callbacks, retained
predictions/costs, repair assessment and portable input-level inspection.
The [real-text training-release example](../docs/banking-investigation.md) is
complete: five fits, 1,969 development cases, and 11,814 independently reproduced
predictions across six executions. Only the combined rollback met the declared
policy. The [incremental intent release](../docs/incremental-intent-release.md)
also completed: 15 fits across three fixed CLINC150 development scenarios.
Ordinary full-data retraining was the only acceptable repair in each scenario;
partial rollbacks failed. These constructed, overlapping scenarios establish
application behavior, not a novel diagnostic advantage. Next: independent first
use with an engineer's own saved predictions and, separately, qualified real
incidents for a comparative research design.
These are development results; no confirmatory study or official-test result is
claimed. No paper is published.

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

The application executes release/repair functions and provides interactive inspection. The
digit walkthrough retains 540 evaluation cases, model parameters and five
executions. Two repairs restore its disclosed input-order fault; ambiguity is
reported. This is application evidence, not a new diagnosis result.


The exact-label comparator checks record alignment, declared evaluation slices and accuracy-drop tolerances. Its eleven reports on the handwritten-digits example agree with an independent NumPy calculation. The example's input-order bug is supplied by the author; the comparator measures its effect rather than discovering its cause.

The [ambiguous-repair example](../docs/ambiguous-repairs.md) reverses a linear classifier's input feature order. Restoring the inputs or reversing the weights both recover the original predictions. The assessment reports `ambiguous_repairs` and leaves the historical cause `not_identified`. This demonstrates the difference between a successful repair and evidence for a unique cause.

## Research history and limits

The [experiment history](EXPERIMENT_HISTORY.md) retains the original study sequence.
The [small-model feasibility attempt](DEVELOPMENT_CLOSURE_2026_10_07.md) failed its
clean-baseline requirement and remains closed. Later incident screens found no
qualified training-release case. A [fixed-tensor software-defect replay](LFQ_FIXED_TENSOR_REPLAY_2026_10_09.md)
reproduced a loss/gradient defect and repairs without training; it is not evidence
of trained-task recovery or an MRF-specific advantage.

Current work follows the [roadmap](ROADMAP.md). The findings do not establish
causal identification, superiority to modern attribution methods or cross-model
generalization. Frozen protocols and result records remain the evidence source.

## Reproduction gap

The Banking77 investigation bundle is retained. Persistent upload of the complete
incremental-intent experiment archive failed; its recovery remains unresolved.
Compact protocol and result records are available, but they are not a substitute
for the complete numeric models, inputs and predictions. Independent first use
has not been completed.

## Saved-prediction import — 2026-10-09

The installed `mrf-import` command accepts a declared evaluation policy and a
long-form prediction CSV. It validates identical case sets before creating output,
retains exact source bytes and hashes, and builds a self-contained inspection
report. Costs remain unknown for imported records. Reimporting the retained
Banking77 example reproduced all 11,814 predictions and the complete recomputed
assessment without additional fits. Independent human first use and comparative
advantage over ordinary rollback remain unmeasured. See
[the user walkthrough](../docs/import-predictions.md).

## Release acceptance requirements — 2026-10-09

A slice can now require `minimum_accuracy` as well as a maximum baseline-relative
accuracy drop. Both conditions are enforced and included in policy identity. This
allows a real new-capability requirement to disqualify complete rollback when it
removes that capability. The four-case hand-authored fixture verifies this
contract; it is not trained-model or research evidence. Requirements must precede
outcome inspection, and ordinary complete/targeted rollback remain mandatory
comparators. Existing Banking77 assessments and digests remain unchanged. No
new fit or comparative study was executed. See
[release acceptance](../docs/release-acceptance.md).
