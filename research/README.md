# Research evidence guide

The [current status](STATE.md) and [roadmap](ROADMAP.md) govern ongoing work.
The [technical report](M4_TECHNICAL_REPORT.md) presents the completed matched
localization study. For the application, start with
[saved predictions](../docs/import-predictions.md) or the
[software architecture](../docs/architecture.md).

| Record group | What it establishes |
| --- | --- |
| [Experiment history](EXPERIMENT_HISTORY.md) | Sequence of constructions, failures, amendments and results |
| [Matched study and reproduction](REPRODUCE_M4.md) | Retained score aggregation; simple baselines outperform or match the tested attribution rankings |
| [Contribution review](M4_CONTRIBUTION_REVIEW.md) and [claims](CLAIMS.md) | Limits of the completed evidence |
| [Small-model closure](DEVELOPMENT_CLOSURE_2026_10_07.md) | A failed development feasibility attempt, not an active experiment |
| [Incident screen](VNEXT_PIPELINE_SCREEN_2026_10_09.md) and [fixed-tensor replay](LFQ_FIXED_TENSOR_REPLAY_2026_10_09.md) | Source leads and a reproduced software defect; no qualified training-release benchmark follows |
| [Banking77 application development](CLASSIFICATION_WORKFLOW_DEVELOPMENT_2026_10_09.md) | A prospectively specified integration example with actual fitted models |
| [Incremental-intent development](INCREMENTAL_INTENT_DEVELOPMENT_2026_10_09.md) | Constructed release requirements for which ordinary full retraining won |

## Reading historical names

`Exp000`–`Exp008` name successive constructed shape-task experiments.
`Exp009` names the Banking77 development series. Within that series, `M4` is the
matched label-swap localization study. `v2` and `vNext` in dated filenames refer
to proposals or development stages, not published software versions.

The September research plan, thesis dossier, baseline plans and benchmark draft
record hypotheses made before later results. They are historical design material,
not parallel current roadmaps. Original protocols, result records and failed
attempts remain available because they explain the decisions and permit replay.
The [decision log](DECISION_LOG.md) preserves chronology; it is not the entry point
for using the software.
