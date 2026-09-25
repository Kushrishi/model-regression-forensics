# Exp009 hosted Stage-B development result

**Status:** complete  
**Evidence class:** development  
**Workflow run:** `36154597887`  
**Source revision:** `310d638211364ad6a9eb3b53c654bf6f261f0ad6`  
**Analysis artifact:** `10883935289`  
**Downloaded analysis SHA-256:** `9592bac6831605970f1440787eeafeb3be0156411a31f533649d4ce979684a33`  
**Official Banking77 test split loaded:** no

Stage B is a descriptive restoration-effect pilot. It does not apply a causal
certification threshold, perform hypothesis tests, report p-values, or convert
three development trajectories into confirmatory evidence.

## Frozen execution

The authorized design completed exactly as frozen:

- trajectories 0, 1, and 2;
- one fresh composite plus root and all four nuisance restorations per
  trajectory;
- six states executed sequentially within each hosted `macos-15` session;
- prospectively rotated state order;
- 18 result-bearing Stage-B trainings;
- every restoration effect computed against its same-session fresh composite;
- all four nuisance contrasts retained;
- official test split embargo preserved.

## Per-trajectory target recovery

| Trajectory | Root | N1 | N2 | N3 | N4 | Strongest nuisance | Root margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.163603 | -0.029412 | -0.014706 | +0.014706 | 0.000000 | +0.014706 | +0.148897 |
| 1 | +0.091912 | -0.014706 | -0.015625 | 0.000000 | 0.000000 | 0.000000 | +0.091912 |
| 2 | +0.117647 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | +0.117647 |

The planted-root restoration was strictly larger than every nuisance
restoration in **3 / 3** trajectories.

## Aggregate effect structure

Root target recovery:

- mean: **+0.124387**
- median: **+0.117647**
- range: **+0.091912 to +0.163603**
- sample standard deviation: **0.036318**

Root minus strongest nuisance:

- mean: **+0.119485**
- median: **+0.117647**
- range: **+0.091912 to +0.148897**
- positive / zero / negative: **3 / 0 / 0**
- sample standard deviation: **0.028537**

Mean nuisance target recovery:

- N1: **-0.014706**
- N2: **-0.010110**
- N3: **+0.004902**
- N4: **0.000000**

All four root-vs-nuisance margins were positive in all three trajectories.

## Protected behavior

Root-restoration protected-macro changes relative to each fresh composite were:

- trajectory 0: **+0.003530**
- trajectory 1: **+0.002861**
- trajectory 2: **+0.001146**

The largest single protected-intent recall movement observed under the root
restoration was `0.090909` in trajectory 1. Protected effects are reported as
development diagnostics, not as a frozen confirmatory locality gate.

## Fresh-composite repeatability

Relative to the historical Stage-A composites:

- trajectory 0 target: **+0.014706**; protected: **-0.000880**
- trajectory 1 target: **0.000000**; protected: **0.000000**
- trajectory 2 target: **0.000000**; protected: **0.000000**

These differences are descriptive only and were not used to discard, rerun, or
select trajectories.

## Development decision

The Stage-B effect structure is informative enough to justify continuing to a
stronger, structurally matched benchmark:

- the root restoration produced material positive target recovery in every
  trajectory;
- every nuisance restoration was smaller than the root within its paired
  trajectory;
- the root-vs-strongest-nuisance margin remained positive in all three
  trajectories;
- the official test split remained untouched.

This is **not** a retrospective Stage-B “pass threshold.” No such threshold was
frozen. It is the project-level development decision required by the roadmap:
the observed effect structure warrants investing in the next benchmark rather
than stopping the hypothesis here.

## Claim boundary

Stage B does **not** establish:

- confirmatory causal specificity;
- a causal-certification threshold;
- a statistically powered root-vs-nuisance inference;
- a structurally matched blind-localization benchmark;
- superiority over modern attribution/influence methods;
- cross-model or cross-dataset generalization.

The current nuisance-v2 construction still has a material structural mismatch:
the root changes labels only, while nuisances change both text and labels.
Therefore Stage A + Stage B cannot be promoted into the paper-grade blinded
benchmark.

The next active milestone is a prospectively frozen **structurally matched
benchmark**.

## Machine-readable evidence

See `STAGE_B_RESULT.json` for the compact permanent evidence record. The full
hosted analysis artifact is identified above by workflow/artifact IDs and
SHA-256.
