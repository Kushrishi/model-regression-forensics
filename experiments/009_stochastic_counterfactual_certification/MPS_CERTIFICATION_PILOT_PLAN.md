# Experiment 009 — MPS Certification Pilot Execution Plan

Status: **FROZEN DEVELOPMENT EXECUTION PLAN**
Date: **2026-09-20**
Backend regime: **Apple Silicon / MPS only**

This plan governs the first result-bearing Exp009 nuisance-v2 certification pilot on the qualified M3 Max environment.

It is development-only and not the Experiment 009 confirmatory protocol.

## 1. Purpose

The pilot asks whether the accepted planted root produces a restoration effect that is distinguishable from four legitimate non-root restorations when all siblings are trained under matched stochastic trajectories.

The pilot does not test a strong blinded-localization claim because the current root/nuisance diff structures are observably different.

## 2. Fixed scientific state

The following are already frozen:

- Banking77 development partition;
- DistilBERT model/revision;
- seven-epoch training configuration;
- pilot target pair;
- accepted 1/4 root dose;
- nuisance rule v2;
- four nuisance pairs;
- nuisance update mechanism;
- trajectories 0, 1, and 2 for the development pilot.

The official Banking77 test split remains embargoed.

## 3. Backend rule

All result-bearing siblings in this pilot must be trained on the same Mac/MPS software/backend regime.

Historical Windows/CPU runs remain valid development evidence for:

- clean model selection;
- target selection;
- dose selection;
- nuisance-rule construction.

They must not be used as paired baseline/candidate/restoration siblings in the MPS certification analysis.

## 4. Seven states per trajectory

For each trajectory `t in {0,1,2}`, define:

1. `B_t` — clean baseline;
2. `C_t` — composite candidate containing:
   - accepted root;
   - nuisance 1;
   - nuisance 2;
   - nuisance 3;
   - nuisance 4;
3. `R_root,t` — composite with only the root restored;
4. `R_n1,t` — composite with only nuisance 1 restored;
5. `R_n2,t` — composite with only nuisance 2 restored;
6. `R_n3,t` — composite with only nuisance 3 restored;
7. `R_n4,t` — composite with only nuisance 4 restored.

Every state uses the same frozen training configuration.

Within trajectory `t`, every sibling must share:

- pretrained checkpoint;
- classifier-head initialization;
- Python/framework/dropout seed family;
- slot schedule;
- optimizer and scheduler;
- epoch count;
- batch size;
- maximum sequence length;
- MPS backend/software environment.

## 5. Stage A — composite-regression qualification

Train only:

- `B_0, B_1, B_2`;
- `C_0, C_1, C_2`.

Total Stage-A training runs: **6**.

For trajectory `t`:

```text
G_target(t) =
    target_macro_recall(B_t)
    - target_macro_recall(C_t)

G_protected(t) =
    protected_macro_recall(B_t)
    - protected_macro_recall(C_t)
```

### Frozen Stage-A gate

The composite advances only if all four conditions hold:

1. mean `G_target >= 0.10`;
2. every trajectory has `G_target >= 0.05`;
3. mean `G_protected <= 0.02`;
4. every trajectory has `G_protected <= 0.03`.

These are reused directly from the already-frozen Exp009 Stage-2 dose-replication gate.

They are not selected from the MPS composite outcomes.

Per-intent protected recall and worst-protected-intent movement must also be recorded, but they are not Stage-A gates.

### Stage-A stop rule

If the composite fails the frozen gate:

- record the MPS composite construction failure;
- do not train restoration siblings;
- do not change the root dose;
- do not manually swap nuisance pairs;
- do not loosen the gate post hoc.

The failure then informs redesign of the benchmark construction rather than being rescued.

## 6. Stage B — exhaustive restoration pilot

Stage B begins only if Stage A passes.

Train all five restoration siblings for each of the three trajectories:

- 3 root restorations;
- 12 nuisance restorations.

Total Stage-B training runs: **15**.

Cumulative maximum pilot training runs: **21**.

For candidate change `j`:

```text
Delta_j,t =
    target_macro_recall(R_j,t)
    - target_macro_recall(C_t)
```

For each nuisance `j`:

```text
D_j,t =
    Delta_root,t - Delta_j,t
```

The pilot must preserve all per-trajectory values, not only means.

## 7. Stage-B interpretation

Stage B is primarily a variance/effect-structure pilot.

It must report:

- root recovery by trajectory;
- each nuisance recovery by trajectory;
- root-vs-nuisance paired margins;
- target per-intent behavior;
- protected aggregate movement;
- worst protected-intent movement;
- runtime;
- release hashes;
- initial-model-state hashes;
- slot-schedule hashes.

With only three trajectories, this pilot must not be presented as final confirmatory statistical certification.

Its purpose is to determine:

- whether root recovery is directionally and materially separated from nuisances;
- the empirical variability of `D_j,t`;
- whether a confirmatory certification protocol is feasible;
- how many trajectories a future confirmatory study would require.

## 8. No adaptive run deletion or substitution

Once Stage B begins, all 15 restorations must be executed unless a technical failure prevents completion.

An unfavorable scientific result is not a reason to skip, replace, or rerun a sibling.

A technical retry must preserve:

- trajectory ID;
- release state;
- configuration;
- backend regime;
- run identity policy.

## 9. Output isolation

Use a new output namespace:

`artifacts/exp009/mps_certification_pilot_v2/runs`

Historical Windows/CPU artifacts must not be overwritten.

Run IDs must encode:

- trajectory ID;
- state.

Suggested state names:

- `baseline`;
- `composite`;
- `restore_root`;
- `restore_n1`;
- `restore_n2`;
- `restore_n3`;
- `restore_n4`.

## 10. Checkpoints and portable evidence

Training code may retain checkpoints during the pilot.

After analysis, the repository should commit only portable scientific evidence unless a checkpoint is specifically required:

- train summaries;
- development predictions where needed;
- release/state manifests;
- analysis summaries;
- checksums.

Large model checkpoints should remain local/reproducible artifacts rather than ordinary Git history.

## 11. Compute-efficiency rule

Do not parallelize result-bearing sibling training merely to save wall-clock time.

Run siblings sequentially so:

- resource contention does not differ across siblings;
- MPS memory pressure is stable;
- failures are easier to attribute;
- runtime metadata remains interpretable.

The primary efficiency mechanism is the Stage-A stop gate, not concurrent training.

## 12. Order-only control

The development protocol identifies an order-only control as useful mechanistic analysis.

It is **not part of the 21-run primary matrix**.

Consider it only after the primary restoration pilot is complete, and do not use it as a rescue for failed causal specificity.

## 13. Decision after Stage B

After all Stage-B results exist, choose among:

### A. Proceed toward confirmatory Exp009

If the effect structure is interpretable and root-vs-nuisance separation appears practically meaningful.

Next step:

- estimate variance of paired `D_j,t`;
- choose a practical separation margin;
- choose multiplicity-aware uncertainty procedure;
- calculate/simulate confirmatory trajectory count;
- freeze confirmatory protocol before touching the official test.

### B. Redesign benchmark construction

If nuisance/restoration effects reveal that the current certification case is not scientifically discriminative.

Preserve the negative result.

### C. Stop Exp009 certification

If root-vs-nuisance separation is fundamentally uninformative and further redesign would amount to post-hoc rescue.

MRF-Bench / MRF-Certify may still proceed using a different benchmark case.

## 14. Training authorization

This plan authorizes only the staged **development MPS pilot** after implementation/preflight checks pass.

It does **not** authorize:

- official-test evaluation;
- confirmatory claims;
- Exp010;
- changes to frozen v2 nuisance selection.
