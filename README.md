# Model Regression Forensics

Model Regression Forensics (MRF) studies a narrow model-debugging question:

> When a model regresses after a versioned training change, what evidence is
> sufficient to identify a responsible change rather than a merely correlated
> one?

The project separates three ideas that are easy to conflate:

1. **localization** — a diagnostic ranks a change as suspicious;
2. **restorative influence** — reverting that change improves the failed behavior;
3. **causal specificity** — its effect is distinguishable from plausible
   non-root interventions and ordinary retraining variability.

MRF is active independent research. It is not a production debugging product
and it does not currently claim a general solution to training-data attribution.

## Current research state

Experiments 000-008 are completed development history. Experiment 009 is the
active research program.

Exp009 moves from the retired synthetic task family to Banking77 with a pinned
DistilBERT classifier, deterministic versioned training releases, and repeated
paired stochastic trajectories.

Current development evidence:

- the clean development substrate and model configuration are fixed;
- the pilot target pair is `Refund_not_showing_up` ↔ `request_refund`;
- a 1/4 symmetric label-mapping fault changes 66 stable training slots;
- that fault passed the frozen development replication gate on three paired
  trajectories;
- the first prospectively frozen nuisance-selection rule was infeasible and was
  retained as a negative design result;
- a second development nuisance rule was frozen before nuisance-model training
  and selected four non-root changes;
- those nuisance changes are suitable for an intervention-effect pilot but are
  **not structurally matched well enough for a strong blinded-localization
  claim**;
- the hosted Stage-A composite-regression gate passed across all three frozen
  trajectories;
- the target-faithful last-layer Grad-Dot baseline ranked the planted root first
  in all three trajectories under truth-isolated scoring;
- Stage-B root/nuisance restoration evidence has not yet been generated;
- the official Banking77 test split remains untouched.

See [research/STATE.md](research/STATE.md) for the canonical short-form state and
[research/CLAIMS.md](research/CLAIMS.md) for claim boundaries.

## Why intervention matters

A high attribution score is not causal evidence by itself.

Earlier experiments repeatedly exposed this distinction. In Experiment 008,
for example, the planted root was localized correctly and restoring it recovered
the target behavior, yet non-root restorations also produced material recovery.
The frozen causal-specificity criterion therefore failed.

MRF treats that failure as evidence, not as a threshold-tuning problem.

## Exp009 development design

The current development pilot compares a clean baseline, a composite regressed
release, root restoration, and four nuisance restorations under matched
training trajectories.

The pilot is staged:

- **Stage A:** verify that the composite release produces the intended localized
  regression under the frozen gate;
- **Stage B:** only if Stage A passes, run exhaustive root and nuisance
  restorations and measure paired recovery effects.

The pilot is intended to estimate effect structure and stochastic variability.
Three trajectories are not treated as final confirmatory statistical evidence.

The exact frozen protocol is recorded under
[`experiments/009_stochastic_counterfactual_certification/`](experiments/009_stochastic_counterfactual_certification/).

## Research discipline

The repository uses several rules to reduce post-hoc reasoning:

- clean behavior is checked before root-cause analysis;
- planted truth is isolated from diagnostic inputs;
- benchmark construction rules are frozen before result-bearing runs;
- development and confirmatory evidence are kept separate;
- negative gates and failed constructions are retained;
- generated evidence includes hashes and runtime provenance;
- claims are bounded to what the current experiment actually tests.

A refreshed literature and baseline audit is required before the next
result-bearing Exp009 training stage.

## Repository map

```text
configs/        frozen configurations for historical experiments
experiments/    experiment-specific protocols and result records
research/       current state, claims, decisions, literature, and design records
scripts/        preparation, training, evaluation, and audit entry points
src/            reusable research implementation
tests/          software and scientific-invariant tests
```

Historical experiments remain in the repository because they document how later
benchmark controls were motivated. They are not all part of the current method.

## Reproducibility

Lightweight development checks:

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The lightweight environment skips tests that require optional ML dependencies.

Full research checks:

```bash
uv sync --extra dev --extra research
uv run pytest
```

Training and inference protocols record their own pinned model/configuration
requirements. Large generated checkpoints are not committed as ordinary source
files.

## Scope

The current evidence is conditional on the evaluated tasks, models, and frozen
protocols. MRF does not currently establish:

- state-of-the-art training-data attribution;
- general causal identification for arbitrary ML incidents;
- cross-model or cross-dataset generalization;
- confirmatory Exp009 causal certification; or
- a non-trivial blinded-localization result for the current nuisance-v2 pilot.

Those boundaries are intentional.
