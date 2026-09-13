# Experiment 009 — Pilot dose 1/8 result

Status: development pilot result — not confirmatory.

## Context

This result records the first intentionally corrupted Banking77 model trained under the predeclared Exp009 pilot dose-selection rule.

- target pair: `Refund_not_showing_up` ↔ `request_refund`
- root mechanism: symmetric label-mapping fault
- dose: `1/8`
- trajectory: `0`
- changed slots: `32` total (`16` per direction)
- candidate release SHA-256: `b5ac035247e1821497ead185ba4551dc4f63ed36bbea763dd3dc26c4719c7342`
- changed-slot IDs SHA-256: `40f267a751f54f2850e9ac40b032412cd28cc4d80cd449bc3e3c1caf5e1952df`
- slot schedule SHA-256: `ff01ec410cac07baffd445f7b480efc9ef0723b4d7fca7bfcf8766108010c376`
- initial model state SHA-256: `8dca3fa0e91dff441b7b52e92be0b92d3f2f3c01acbf0e7e77434dab2fcf0ddc`

The candidate matched its paired clean trajectory on both initial model state and slot schedule.

## Development-evaluation result

Clean paired trajectory-0 target macro recall: `0.970588`

Candidate target macro recall: `0.925551`

Target regression:

`G_target = 0.045037`

Clean protected macro recall: `0.876446`

Candidate protected macro recall: `0.877425`

Protected regression:

`G_protected = -0.000979`

Per-target recall:

- `Refund_not_showing_up`: `1.000000` → `0.968750`
- `request_refund`: `0.941176` → `0.882353`

Candidate overall metrics:

- accuracy: `0.889890`
- macro recall: `0.878675`

## Predeclared Stage-1 screen

Requirements frozen before corrupted-model training:

- target drop must be at least `0.10`
- protected macro-recall drop must be at most `0.02`

Observed:

- target gate: **FAIL** (`0.045037 < 0.10`)
- protected gate: **PASS** (`-0.000979 <= 0.02`)
- Stage-1 result: **NO PASS**

Interpretation: the `1/8` symmetric label fault produced a localized directional regression, but its magnitude was below the predeclared materiality threshold. This is an insufficient-dose pilot result, not a confirmatory failure.

Per the frozen pilot rule, the next action is to advance mechanically to dose `1/4`, trajectory `0`.

## Integrity status

- confirmatory result: `NO`
- official Banking77 test split loaded: `NO`
- no post-result change was made to the target pair, dose thresholds, clean model configuration, or Stage-1 selection rule.
