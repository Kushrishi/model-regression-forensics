# Research status

The classification-comparison utility is in maintenance. The matched localization study is complete. No new training, scorer development or comparative study is active.

## Main finding

Two constructed Banking77 benchmarks each contained five candidate label-swap changes. Three paired training trajectories were recorded per benchmark; these are repeated trainings, not six independent benchmarks.

| Diagnostic | Benchmark A root rank | Benchmark B root rank |
| --- | ---: | ---: |
| Deterministic random reference | 3 | 3 |
| Target-label overlap | 1 | 1 |
| Lexical Jaccard | 1 | 1 |
| Final-checkpoint Grad-Dot | 1 | 5 |
| Seven-checkpoint TracIn | 1 | 5 |

The model-based methods provided no consistent top-1 benefit. Known target labels and disjoint candidate pairs made the simple baselines particularly informative. This study measures localization in a narrow construction; it does not establish general causal identification or superiority to attribution methods.

The official Banking77 test split was not used. See the [technical report](M4_TECHNICAL_REPORT.md), [result record](M4_RESULT.json) and [replay instructions](REPRODUCE_M4.md).

## Implemented software

The application imports saved predictions or runs supplied model functions, validates record identity, checks declared release requirements and creates a local HTML report. Supplied repairs are assessed against the same evaluation policy.

The [saved-prediction example](../docs/deployment-comparison.md) compares 1,969 development predictions. INT8 conversion changes eight labels while overall accuracy remains 88.07% and all declared slice limits pass. Ordinary prediction comparison reaches the same conclusions. Repairs are not evaluated in this example.

The [ambiguity example](../docs/ambiguous-repairs.md) demonstrates two different interventions restoring identical predictions. It reports `ambiguous_repairs`, not a uniquely identified historical cause.

## Additional development work

- The [Banking77 training-release walkthrough](../docs/banking-investigation.md) completed five fits. Only the combined rollback met its declared policy.
- The [incremental-intent example](../docs/incremental-intent-release.md) completed fifteen fits across three fixed CLINC150 scenarios. Full-data retraining was the only acceptable repair in those scenarios.
- The [fixed-tensor replay](LFQ_FIXED_TENSOR_REPLAY_2026_10_09.md) reproduced a software loss/gradient defect without training. It does not establish trained-task recovery or an MRF-specific advantage.

These examples are development evidence, not independent confirmatory benchmarks. Earlier failed attempts and confounds remain in the [experiment history](EXPERIMENT_HISTORY.md).

## Remaining limits

Independent first use and comparative user benefit have not been measured. No qualified real-incident comparative study or peer-reviewed publication is available.

The complete incremental-intent archive was not durably recovered after an upload failure. Its compact protocol and result records do not replace the full models, inputs and predictions. The Banking77 investigation bundle is retained separately.

Original project software is available under the [MIT License](../LICENSE). Dataset and other third-party terms remain separate.

## Maintenance

Address concrete defects before adding features. Reopen comparative research only with a suitable task, outcome evidence and fair baselines. A passing engineering policy is not a significance test or proof of causation.

The [previous detailed status](https://github.com/Kushrishi/model-regression-forensics/blob/5c6f05beb7c110ab178b5beb3a369f1e3c880261/research/STATE.md) remains in repository history. This summary changes presentation only; protocols, code and recorded results are unchanged.
