# Research direction

Updated October 4, 2026.

## Completed

- Development retraining and restoration studies, including negative results.
- Structurally matched Banking77 candidates and a two-world localization comparison.
- A technical report with complete rankings and reproducible aggregation.
- An exact-label prediction comparator, an external digits example and an ambiguous-repair example.

The [experiment history](EXPERIMENT_HISTORY.md) records how the design developed. The [technical report](M4_TECHNICAL_REPORT.md) presents the matched study.

## Next

The next research question is whether controlled repair evidence can improve a specific debugging decision when several changes are plausible. A follow-up design needs:

1. A task where target-label overlap does not immediately reveal the planted change.
2. Alternative repairs or interactions that make explanation different from recovery.
3. A simple baseline using the same prediction and intervention evidence.
4. A defined measure of incorrect unique attribution and useful abstention.
5. A comparison with the closest existing methods.

These are design requirements, not an approved experiment. Further matched-world certification training remains paused under the [continuation requirements](M4_CONTINUATION_GATE.md), which require independent scientific review.

## Possible outputs

The completed evidence already supports a technical report and a reproducible case study. A workshop paper would require a suitable venue and an assessment of what the case study adds. A broader method paper would require a distinct method and stronger evidence across tasks.

The comparator could become a small maintained package if external users find its record validation and repair reporting useful. General prediction comparison alone is already available elsewhere; a package release needs a clear purpose and settled licensing.

If a follow-up cannot establish a useful research question, the completed report and software remain the project's outcome. More training runs alone would not resolve that issue.
