# Experiment 009 — Hosted Stage-B execution amendment

Status: **FROZEN before Stage-B result-bearing training**  
Frozen: **2026-09-25**  
Evidence class: **development pilot only**

This amendment governs hosted execution of Stage B in the frozen Exp009
nuisance-v2 development pilot. It supplements, rather than rewrites,
`MPS_CERTIFICATION_PILOT_PLAN.md`.

At freeze time:

- the hosted Stage-A gate had passed;
- the Stage-A Grad-Dot development localization result had been recorded;
- no Stage-B restoration model had been trained;
- the official Banking77 test split remained untouched.

## 1. Reason for the amendment

The original pilot plan defines five restoration siblings per trajectory and
computes each restoration effect relative to the corresponding composite
`C_t`.

Hosted Stage A trained its composite siblings on separate ephemeral GitHub
runners. The scientific quantity of interest in Stage B is a *difference*
between a restoration and its composite reference. Reusing a Stage-A composite
would therefore mix the intended restoration effect with any runner/session
variation between the earlier composite job and the later restoration job.

Stage B will instead train a fresh composite anchor `C'_t` in the same hosted
job as its five restoration siblings. This is an execution-control amendment,
not a new trajectory and not a change to the benchmark construction.

The Stage-A composite remains valuable as a descriptive repeatability reference.
It is not the denominator for Stage-B restoration effects.

## 2. Unchanged scientific state

This amendment does not change:

- Banking77 development partition;
- official-test embargo;
- target pair;
- accepted 1/4 root dose;
- nuisance rule v2 or its four selected pairs;
- any release construction;
- trajectories 0, 1, and 2;
- seven-epoch training configuration;
- model checkpoint or tokenizer;
- target/protected metric definitions;
- the completed Stage-A result;
- the development-only claim boundary.

No nuisance may be removed, replaced, or redefined after Stage-B outcomes.

## 3. Hosted trajectory sessions

Stage B consists of three independent hosted trajectory sessions.

Each session trains exactly six states on one `macos-15` runner:

1. a fresh composite `C'_t`;
2. root restoration;
3. nuisance restoration 1;
4. nuisance restoration 2;
5. nuisance restoration 3;
6. nuisance restoration 4.

Therefore:

- fresh Stage-B composite anchors: **3**;
- restoration runs: **15**;
- Stage-B result-bearing runs: **18**;
- Stage-A + Stage-B result-bearing runs: **24**.

The three fresh composites do **not** increase the statistical trajectory count.
The pilot still has exactly three trajectory IDs.

## 4. Frozen within-session order

The composite anchor is always first. Restoration order is prospectively
rotated so the root restoration is early, late, and middle across the three
sessions:

- trajectory 0:
  `composite, restore_root, restore_n1, restore_n2, restore_n3, restore_n4`;
- trajectory 1:
  `composite, restore_n1, restore_n2, restore_n3, restore_n4, restore_root`;
- trajectory 2:
  `composite, restore_n3, restore_n4, restore_root, restore_n1, restore_n2`.

Order is fixed before Stage-B outcomes and may not be changed because an
intermediate result is favorable or unfavorable.

## 5. Pairing contract

Within trajectory `t`, all six states must share:

- trajectory seed family;
- initial model-state hash;
- slot-schedule hash;
- development-partition hash;
- model artifact/revision;
- optimizer/training configuration;
- source Git revision;
- hosted runner session/provenance.

Across trajectories, each named release state must reproduce the same release
hash and changed-slot-ID hash. The analysis checkout must match the source Git
revision recorded by every result-bearing Stage-B summary.

The fresh composite and restorations are separate training invocations, but the
training code resets the frozen trajectory seeds for every invocation.

A primary Stage-B trajectory may not splice states from different hosted jobs.

## 6. Technical failure rule

Once a trajectory session begins, an unfavorable scientific result is never a
reason to rerun, omit, or substitute a state.

If a genuine infrastructure failure prevents a complete six-state session:

1. preserve the failed job and any partial portable evidence;
2. remove transient model checkpoints before artifact upload;
3. do not use the partial session in the primary Stage-B analysis;
4. rerun the **entire six-state trajectory session** at the same source revision
   and configuration;
5. do not combine a replacement restoration with an earlier composite anchor.

This preserves the within-session comparison contract.

## 7. Frozen Stage-B estimands

Let `C'_t` be the fresh same-session composite and `R_j,t` one restoration.

Primary target recovery:

```text
Delta_target(j,t) =
    target_macro_recall(R_j,t)
    - target_macro_recall(C'_t)
```

For nuisance `k`, the root-vs-nuisance paired contrast is:

```text
D_k,t =
    Delta_target(root,t)
    - Delta_target(nuisance_k,t)
```

A trajectory-level worst-nuisance contrast is also reported descriptively:

```text
D_max,t =
    Delta_target(root,t)
    - max_k Delta_target(nuisance_k,t)
```

Protected behavior is reported as:

- signed protected-macro-recall movement from `C'_t`;
- maximum absolute per-protected-intent recall movement from `C'_t`;
- the intent or intents attaining that maximum absolute movement;
- worst protected-intent recall after each restoration.

Target-intent recall movements are reported separately for both target intents.

## 8. Stage-A composite repeatability check

For each trajectory, the new `C'_t` is compared descriptively with the
recorded Stage-A `C_t` on:

- target macro recall;
- protected macro recall.

This comparison is **not** an eligibility gate. No Stage-B session is discarded
because the fresh composite differs from the Stage-A composite.

## 9. Statistical interpretation

This is a three-trajectory development pilot.

The primary record must preserve every trajectory-level restoration effect and
all four root-vs-nuisance contrasts. It summarizes them with means, medians,
ranges, sample standard deviations, sign counts, and the trajectory-level
worst-nuisance contrast.

No p-value, retrospective significance threshold, or binary causal-certification
criterion is introduced for this pilot.

Because no family of hypothesis tests is performed, no multiplicity correction
is applied at this stage. A later confirmatory protocol, if justified, must
prospectively specify:

- a practically meaningful causal-separation margin;
- trajectory count / power or precision rationale;
- multiplicity-aware inference for the four nuisance contrasts;
- abstention and stopping rules.

## 10. Evidence package

Each trajectory session must retain portable evidence sufficient to verify:

- state order;
- runner provenance;
- source revision;
- trajectory seeds;
- release hashes and changed-slot counts;
- changed-slot-ID hashes;
- initial model-state hash;
- slot-schedule hash;
- training configuration;
- development metrics;
- per-label recalls;
- evaluation logits/predictions checksums.

Large model checkpoints are deleted after each state has emitted its portable
evidence. A cleanup step also removes any checkpoint left by an interrupted
training state before partial evidence is uploaded.

The aggregate analysis must be machine-readable and must fail on integrity
mismatches. An unfavorable scientific effect structure is a valid successful
workflow result, not an infrastructure failure.

## 11. Authorization boundary

Merging this amendment and its implementation does **not** itself authorize
result-bearing Stage-B training.

Stage B may start only after:

1. CI and Research CI are green;
2. the six-state preflight path is verified;
3. the Stage-B analysis code/tests are green;
4. a separate `STAGE_B_RUN_REQUEST.json` is committed to `main` with
   `authorized=true`.

The authorization request must pin the completed Stage-A evidence and the
prepared `main` revision. The Stage-B workflow verifies that the authorization
commit's parent is the pinned preparation revision.

The authorization request may not change the frozen design above.

## Frozen markers

`EXP009_HOSTED_STAGE_B_AMENDMENT=FROZEN`

`EXP009_STAGE_B_RESULT_OBSERVED_AT_FREEZE=NO`

`OFFICIAL_BANKING77_TEST_LOADED_AT_FREEZE=NO`
