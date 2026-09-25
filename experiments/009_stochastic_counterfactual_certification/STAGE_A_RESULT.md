# Experiment 009 — Hosted Stage-A development result

Status: **COMPLETE — frozen Stage-A gate passed**  
Date: **2026-09-25**  
Evidence class: **development only**  
Source revision: `cb508c845157a3212a05cc5592d0e384e220877f`  
GitHub Actions run: `36139384603`

## Question

Stage A asks whether the frozen five-change composite produces a localized
regression on the target pair while leaving aggregate protected behavior within
the prospectively declared limits.

It also executes the frozen target-faithful last-layer Grad-Dot baseline under
truth isolation. The localization result is diagnostic development evidence,
not a structurally matched confirmatory benchmark.

## Frozen gate

The thresholds were fixed before this hosted run:

- mean target regression >= 0.10;
- each trajectory target regression >= 0.05;
- mean protected regression <= 0.02;
- each trajectory protected regression <= 0.03.

| Trajectory | Baseline target | Composite target | Target regression | Baseline protected | Composite protected | Protected regression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.970588 | 0.792279 | 0.178309 | 0.881419 | 0.877817 | 0.003602 |
| 1 | 0.970588 | 0.878676 | 0.091912 | 0.884779 | 0.881504 | 0.003275 |
| 2 | 0.970588 | 0.852941 | 0.117647 | 0.879861 | 0.878604 | 0.001257 |

Aggregate:

- mean target regression: **0.129289**;
- mean protected regression: **0.002711**;
- mean-target gate: **PASS**;
- per-trajectory target gate: **PASS**;
- mean-protected gate: **PASS**;
- per-trajectory protected gate: **PASS**;
- **STAGE_A_GATE=PASS**.

The official Banking77 test split was not loaded.

## Truth-isolated Grad-Dot localization

The debugger-facing localization step used opaque candidate identities. The
aggregate ranking was written before benchmark truth was loaded.

Mean candidate scores:

| Rank | Candidate | Mean score |
| ---: | --- | ---: |
| 1 | `candidate_243b5f64c58c` | 145.185678 |
| 2 | `candidate_438ace7a3c4c` | 0.637005 |
| 3 | `candidate_048aa2e31cdf` | 0.467031 |
| 4 | `candidate_b2ac5506d40e` | 0.059255 |
| 5 | `candidate_7eebe55b682f` | -0.003911 |

`candidate_243b5f64c58c` ranked first in trajectories 0, 1, and 2.

After the aggregate ranking existed, the benchmark truth scorer identified
`candidate_243b5f64c58c` as the planted root:

- root rank: **1**;
- top-1 correct: **true**;
- truth loaded after ranking: **true**.

## Interpretation boundary

This result establishes, for the frozen **development** construction, that:

1. the composite regression satisfied the prospective Stage-A behavioral gate;
2. the target-faithful Grad-Dot baseline localized the planted root at rank 1
   across all three trajectories under the implemented truth-isolation
   procedure.

It does **not** establish:

- causal specificity;
- a successful restoration result;
- a strong blinded-localization benchmark under structurally matched changes;
- superiority over modern attribution methods;
- confirmatory evidence;
- generalization beyond this Banking77/DistilBERT development substrate.

The nuisance-v2 structural mismatch remains material: the root changes labels
without changing text, whereas each nuisance changes both text and labels.

## Evidence package

The workflow emitted:

- `stage_a_analysis.json`;
- `graddot_aggregate.json`;
- `graddot_truth_evaluation.json`;
- six portable per-state training artifacts.

The Stage-A analysis artifact is retained under
`exp009-stage-a-analysis` from workflow run `36139384603`.

## Next decision

Stage B may now be considered because the frozen Stage-A prerequisite passed.
No Stage-B result is implied by this record. Stage-B execution architecture and
effect analysis must be reviewed and frozen before restoration training begins.
