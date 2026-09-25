# Experiment 009 — Pilot dose 1/4 trajectory-1 replication

Status: development pilot replication result — not confirmatory.

## Context

This result records the second intentionally corrupted Banking77 model trained at the provisionally selected `1/4` pilot dose under the predeclared Exp009 dose-selection rule.

- target pair: `Refund_not_showing_up` ↔ `request_refund`
- root mechanism: symmetric label-mapping fault
- dose: `1/4`
- trajectory: `1`
- changed slots: `66` total (`33` per direction)
- candidate release SHA-256: `7ba6b58723c30d17bdcc424058cf81c45a600f4959386329976565e64a6101d8`
- changed-slot IDs SHA-256: `48a27a97c1d74ff0ff4daa6738c8b45aa315ce3c110174f5f8bdc683e494a127`
- slot schedule SHA-256: `4d3c1bee75dbd6d888bd764ea143b1146141af81990a411d4e19ca6835d276e4`
- initial model state SHA-256: `a027051c41f0f09eabe50ef2c676cd9c6bc437c26e93521a9325d953b85f9494`

The candidate matched its paired clean trajectory on both initial model state and slot schedule.

## Development-evaluation result

Clean paired trajectory-1 target macro recall: `0.954963`

Candidate target macro recall: `0.836397`

Target regression:

`G_target = 0.118566`

Clean protected macro recall: `0.886711`

Candidate protected macro recall: `0.884843`

Protected regression:

`G_protected = 0.001869`

Per-target recall:

- `Refund_not_showing_up`: `0.968750` → `0.937500`
- `request_refund`: `0.941176` → `0.735294`

Candidate overall metrics:

- accuracy: `0.893393`
- macro recall: `0.883584`
- elapsed training time: `130.61` minutes

## Predeclared Stage-1 screen

Requirements frozen before corrupted-model training:

- target drop must be at least `0.10`
- protected macro-recall drop must be at most `0.02`

Observed:

- target gate: **PASS** (`0.118566 >= 0.10`)
- protected gate: **PASS** (`0.001869 <= 0.02`)
- trajectory-1 replication result: **PASS**

Interpretation: the provisionally selected `1/4` symmetric label fault again produced a material target-localized regression under a second controlled stochastic trajectory while protected behavior remained stable. This is development replication evidence, not confirmatory causal certification.

Per the frozen pilot rule, the next required action is to run the same `1/4` dose on trajectory `2` before finalizing pilot-dose replication.

## Integrity status

- confirmatory result: `NO`
- official Banking77 test split loaded: `NO`
- no post-result change was made to the target pair, dose thresholds, clean model configuration, or Stage-1 selection rule.
