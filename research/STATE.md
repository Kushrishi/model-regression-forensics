# Research status

Updated October 4, 2026.

The matched Banking77 localization study is complete. The repository also includes a prediction-comparison utility and an example in which two different repairs recover the same behavior. No paper has been submitted or published.

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

The exact-label comparator checks record alignment, declared evaluation slices and accuracy-drop tolerances. Its eleven reports on the handwritten-digits example agree with an independent NumPy calculation. The example's input-order bug is supplied by the author; the comparator measures its effect rather than discovering its cause.

The [ambiguous-repair example](../docs/ambiguous-repairs.md) reverses a linear classifier's input feature order. Restoring the inputs or reversing the weights both recover the original predictions. The assessment reports `ambiguous_repairs` and leaves the historical cause `not_identified`. This demonstrates the difference between a successful repair and evidence for a unique cause.

## Next research decision

A useful follow-up would test situations where several plausible changes or repairs affect the same behavior. It needs a debugging decision that existing attribution and counterfactual methods do not already resolve, with a simple baseline receiving the same evidence.

The [continuation review](M4_CONTRIBUTION_REVIEW.md) explains the overlap with prior work. Further matched-world certification training remains paused under the recorded [M4 continuation requirements](M4_CONTINUATION_GATE.md). An independent scientific assessment is still required before that study resumes. The completed report and software remain available.

## Evidence and limitations

The results concern the constructed tasks tested here. They do not establish general causal identification, superiority to modern attribution methods or cross-model generalization. Frozen protocols and result records define each study; the [claims summary](CLAIMS.md) and this page describe their current interpretation.
