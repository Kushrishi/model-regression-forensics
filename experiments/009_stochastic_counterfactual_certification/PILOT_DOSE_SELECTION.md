# Experiment 009 — Pilot Corruption-Dose Selection

Status: **development pilot rule — frozen before the first corrupted-model training result**.

This document governs only the Banking77 development pilot used to choose a practical symmetric label-mapping corruption dose. It is not the Experiment 009 confirmatory protocol.

## Fixed pilot substrate

The pilot uses the already-selected clean model configuration and target pair:

- model: `distilbert/distilbert-base-uncased`;
- revision: `12040accade4e8a0f71eabdb258fecc2e7e948be`;
- epochs: `7`;
- batch size: `32`;
- learning rate: `2e-5`;
- weight decay: `0.01`;
- warmup ratio: `0.10`;
- maximum sequence length: `128`;
- target A: `Refund_not_showing_up`;
- target B: `request_refund`;
- root mechanism: symmetric A↔B label-mapping fault;
- official Banking77 test split: embargoed and not used.

The development partition SHA-256 is:

`61ecbdd9224cf2bbeafcf5ceb9164cfc3fc74eb449129136198646385aaa8e88`

The clean baseline release SHA-256 is:

`cb83232c055c4c55ca50f2fbd86627d59dc806ab33a861abc7402990ddb7e20c`

## Pre-corruption clean variability

The three clean 7-epoch trajectories produced target-pair macro recall values of:

- trajectory 0: `0.970588`;
- trajectory 1: `0.954963`;
- trajectory 2: `0.955882`.

The resulting clean target-pair mean is `0.960478` with population SD `0.007159`.

Protected-intent macro recall values were:

- trajectory 0: `0.876446`;
- trajectory 1: `0.886711`;
- trajectory 2: `0.883754`.

The resulting protected mean is `0.882304` with population SD `0.004315`.

These measurements were observed before any corrupted-model training result.

## Ordered corruption grid

The allowed pilot doses are fixed in ascending order:

1. `1/8`: 16 A→B plus 16 B→A relabels, 32 changed slots total;
2. `1/4`: 33 A→B plus 33 B→A relabels, 66 changed slots total;
3. `3/8`: 49 A→B plus 49 B→A relabels, 98 changed slots total;
4. `1/2`: 65 A→B plus 65 B→A relabels, 130 changed slots total.

The changed-slot sets are deterministic and strictly nested. No dose outside this grid may be introduced merely because the observed pilot result is inconvenient.

## Paired metrics

For trajectory `t` and dose `d`:

```text
G_target(t,d) = clean_target_macro_recall(t) - candidate_target_macro_recall(t,d)
G_protected(t,d) = clean_protected_macro_recall(t) - candidate_protected_macro_recall(t,d)
```

Positive `G_target` means the candidate regressed on the intended target pair.
Positive `G_protected` means protected behavior also degraded.

## Stage 1 — trajectory-0 screening

Doses are trained on trajectory 0 in ascending order.

A dose becomes the provisional pilot dose when both conditions hold:

1. `G_target(0,d) >= 0.10`;
2. `G_protected(0,d) <= 0.02`.

The 0.10 target margin is intentionally much larger than the observed clean target variability and represents a practically visible localized regression rather than ordinary retraining noise.

The 0.02 protected margin is also larger than ordinary clean protected variability while still requiring the planted fault to remain substantially localized.

If a dose fails either condition, proceed mechanically to the next larger dose.

## Stage 2 — replication on trajectories 1 and 2

The first trajectory-0 dose that passes Stage 1 is then trained on trajectories 1 and 2 using their paired clean baselines.

The provisional dose is accepted for later Exp009 pilot-world development only if all of the following hold across trajectories 0, 1, and 2:

1. mean `G_target >= 0.10`;
2. every trajectory has `G_target >= 0.05`;
3. mean `G_protected <= 0.02`;
4. every trajectory has `G_protected <= 0.03`.

If the provisional dose fails this replication rule, move to the next larger dose and evaluate that dose on all three trajectories. Continue in ascending order.

If the `1/2` dose also fails, this symmetric target-pair mechanism has no acceptable dose under the current pilot rule. The pilot must record that outcome rather than extending the dose grid ad hoc.

## Secondary protected-behavior audit

Per-intent protected recall and the worst protected-intent movement must be recorded for every trained candidate. They are development diagnostics at this stage, not dose-selection gates, because a per-intent equivalence margin has not yet been calibrated prospectively.

Those observations may inform the later confirmatory protocol, but they may not be used retrospectively to substitute a different pilot dose after the rule above has selected one.

## Integrity constraints

- All candidate runs remain development-only and non-confirmatory.
- The official Banking77 test split remains untouched.
- Candidate and paired clean siblings must share trajectory ID, classifier initialization, optimizer/scheduler settings, epoch count, batch size, fixed maximum length, slot schedule, backend regime, and software environment.
- The target pair may not be replaced after observing candidate outcomes.
- The corruption grid may not be expanded after observing candidate outcomes merely to force a successful regression.
- A technical failure may be rerun with the same trajectory, release, and configuration; an unfavorable scientific result must stand.
