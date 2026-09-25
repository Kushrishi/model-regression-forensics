# Experiment 009 — Pilot dose 1/4 result

Status: development pilot result — not confirmatory.

## Context

This result records the second intentionally corrupted Banking77 model trained under the predeclared Exp009 pilot dose-selection rule after dose `1/8` failed the target-materiality gate.

- target pair: `Refund_not_showing_up` ↔ `request_refund`
- root mechanism: symmetric label-mapping fault
- dose: `1/4`
- trajectory: `0`
- changed slots: `66` total (`33` per direction)
- candidate release SHA-256: `7ba6b58723c30d17bdcc424058cf81c45a600f4959386329976565e64a6101d8`
- changed-slot IDs SHA-256: `48a27a97c1d74ff0ff4daa6738c8b45aa315ce3c110174f5f8bdc683e494a127`
- slot schedule SHA-256: `ff01ec410cac07baffd445f7b480efc9ef0723b4d7fca7bfcf8766108010c376`
- initial model state SHA-256: `8dca3fa0e91dff441b7b52e92be0b92d3f2f3c01acbf0e7e77434dab2fcf0ddc`

The candidate matched its paired clean trajectory on both initial model state and slot schedule.

## Development-evaluation result

Clean paired trajectory-0 target macro recall: `0.970588`

Candidate target macro recall: `0.776654`

Target regression:

`G_target = 0.193934`

Clean protected macro recall: `0.876446`

Candidate protected macro recall: `0.876830`

Protected regression:

`G_protected = -0.000384`

Per-target recall:

- `Refund_not_showing_up`: `1.000000` → `0.906250`
- `request_refund`: `0.941176` → `0.647059`

Candidate overall metrics:

- accuracy: `0.884384`
- macro recall: `0.874228`
- elapsed minutes: `153.03`

## Predeclared Stage-1 screen

Requirements frozen before corrupted-model training:

- target drop must be at least `0.10`
- protected macro-recall drop must be at most `0.02`

Observed:

- target gate: **PASS** (`0.193934 >= 0.10`)
- protected gate: **PASS** (`-0.000384 <= 0.02`)
- Stage-1 result: **PASS**

Interpretation: the `1/4` symmetric label fault produced a material localized regression while protected behavior remained effectively unchanged. Under the frozen pilot rule, dose escalation stops at `1/4` provisionally.

Per the frozen pilot rule, the next action is to replicate dose `1/4` on trajectories `1` and `2` before treating it as the selected pilot corruption strength.

## Integrity status

- confirmatory result: `NO`
- official Banking77 test split loaded: `NO`
- no post-result change was made to the target pair, dose thresholds, clean model configuration, or Stage-1 selection rule.
