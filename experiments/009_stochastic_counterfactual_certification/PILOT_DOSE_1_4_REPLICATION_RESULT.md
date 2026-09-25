# Experiment 009 — Pilot dose 1/4 replication result

Status: **development pilot result — not confirmatory**.

This record applies the prospectively frozen Stage-2 replication rule in `PILOT_DOSE_SELECTION.md` to the three completed `1/4` candidate trajectories.

## Fixed pilot setup

- target pair: `Refund_not_showing_up` ↔ `request_refund`
- root mechanism: symmetric label-mapping fault
- dose: `1/4`
- changed slots: `66` total (`33` per direction)
- candidate release SHA-256: `7ba6b58723c30d17bdcc424058cf81c45a600f4959386329976565e64a6101d8`
- changed-slot IDs SHA-256: `48a27a97c1d74ff0ff4daa6738c8b45aa315ce3c110174f5f8bdc683e494a127`
- trajectories: `0`, `1`, `2`
- official Banking77 test split: untouched

All three candidates matched their paired clean trajectory on both classifier initialization and slot schedule.

## Per-trajectory results

| Trajectory | Clean target | Candidate target | `G_target` | Clean protected | Candidate protected | `G_protected` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.970588 | 0.776654 | 0.193934 | 0.876446 | 0.876830 | -0.000384 |
| 1 | 0.954963 | 0.836397 | 0.118566 | 0.886711 | 0.884843 | 0.001869 |
| 2 | 0.955882 | 0.867647 | 0.088235 | 0.883754 | 0.880619 | 0.003135 |

Aggregate target regression:

- mean `G_target = 0.133578`
- sample SD `= 0.054425`
- minimum `= 0.088235`
- maximum `= 0.193934`

Aggregate protected regression:

- mean `G_protected = 0.001540`
- sample SD `= 0.001783`
- minimum `= -0.000384`
- maximum `= 0.003135`

## Frozen Stage-2 replication rule

`PILOT_DOSE_SELECTION.md` requires the provisional dose to satisfy all four conditions across trajectories 0, 1, and 2:

1. mean `G_target >= 0.10`;
2. every trajectory has `G_target >= 0.05`;
3. mean `G_protected <= 0.02`;
4. every trajectory has `G_protected <= 0.03`.

Observed:

- mean target gate: **PASS** (`0.133578 >= 0.10`)
- per-trajectory target floor: **PASS** (minimum `0.088235 >= 0.05`)
- mean protected gate: **PASS** (`0.001540 <= 0.02`)
- per-trajectory protected ceiling: **PASS** (maximum `0.003135 <= 0.03`)

Therefore:

**PILOT_DOSE_1_4_STAGE2_REPLICATION = PASS**

The `1/4` dose is accepted for later Exp009 pilot-world development under the prospectively frozen rule. Dose escalation to `3/8` or `1/2` is not permitted or needed after this acceptance.

Trajectory 2 did not individually meet the Stage-1 trajectory-0 screening margin of `0.10`; that is not a Stage-2 failure because the frozen replication rule prospectively uses a `0.05` per-trajectory floor together with a `0.10` mean target-regression requirement.

## Integrity status

- confirmatory result: `NO`
- official Banking77 test split loaded: `NO`
- target pair unchanged after candidate outcomes
- dose grid unchanged after candidate outcomes
- Stage-2 rule applied exactly as frozen before the first corrupted-model result
- no post-hoc threshold change or dose escalation used to rescue the result

## Next step

Proceed from dose calibration to Exp009 pilot-world development: define the four matched nuisance changes, freeze the five-change world construction and diagnostic inputs, and then evaluate root-versus-nuisance counterfactual recovery under repeated paired trajectories before any confirmatory protocol is frozen.
