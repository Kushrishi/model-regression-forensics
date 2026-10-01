# Model Regression Forensics

Model Regression Forensics (MRF) studies a narrow model-debugging question:

> When a model regresses after a versioned training change, what evidence is sufficient to identify a responsible change rather than a merely correlated one?

The project separates three ideas that are easy to conflate:

1. **localization** — a diagnostic ranks a change as suspicious;
2. **restorative influence** — reverting that change improves the failed behavior;
3. **causal specificity** — its effect is distinguishable from plausible non-root interventions and ordinary retraining variability.

MRF is active independent research. It is not a production debugging product and it does not currently claim a general solution to training-data attribution.

## Current research state

Experiments 000-008 are completed development history. Experiment 009 is the active research program on Banking77 with a pinned DistilBERT classifier, deterministic versioned training releases, and repeated paired stochastic trajectories.

Completed development evidence includes:

- a 1/4 symmetric planted label-mapping fault that changes 66 stable training slots and passed the frozen localized-regression replication gate across three paired trajectories;
- a truth-isolated last-layer Grad-Dot Stage-A baseline that ranked the planted root first in all three trajectories;
- a frozen Stage-B intervention pilot in which all 18 trainings completed and root restoration exceeded every nuisance restoration in all three paired trajectories, with mean root recovery **+0.1244** and mean root-minus-strongest-nuisance margin **+0.1195**;
- retention of the Stage-A/Stage-B limitation that the nuisance-v2 candidate changes were structurally distinguishable from the planted root.

That limitation motivated M3 rather than being hidden.

## Structurally matched benchmark

**M3 is complete.**

The new benchmark prospectively matches the observable structure of every candidate change:

- 66 changed stable slots per candidate;
- 33 label swaps in each direction;
- zero text changes;
- preserved aggregate label mass;
- two touched labels per candidate;
- identical debugger-facing schema;
- pairwise-disjoint changed slots and intent labels;
- deterministic opaque candidate IDs;
- exact restoration and composite-order audits.

The original M3 protocol requested three five-candidate worlds. Before any matched-benchmark model training, a clean-only capacity preflight showed that the frozen eligibility graph could support at most 13 globally disjoint candidate pairs, fewer than the 15 required. The project did not weaken the eligibility rule to force the preferred design. A prospective amendment reduced the benchmark to **two complete worlds**.

The amended hosted structural preflight passed for **2 worlds, 10 candidates, and 20 unique candidate-touched intents**. The official Banking77 test split remains untouched, and M3 performed no matched-benchmark model training.

This establishes matched benchmark construction only. It does **not** establish localization success or causal specificity on the new benchmark.

## Current milestone

**M4 — competitive localization baselines is active.**

The next question is how much localization signal remains after structural mismatch is removed. The benchmark intentionally does not hide the known target behavior from the debugger, so simple semantic/change-overlap information may be strong.

M4 therefore requires comparison against:

- a seeded random reference;
- a simple semantic/lexical or changed-record-overlap baseline where applicable;
- target-faithful last-layer Grad-Dot;
- modern influence/data-attribution methods only where their objective can be implemented faithfully for the frozen target.

If a simple baseline solves localization, that is a valid result rather than a reason to redesign the benchmark after observing it.

The eventual research contribution remains the later question of whether counterfactual retraining can support a prospectively defined **certify/abstain** decision that distinguishes a responsible change from plausible alternatives under retraining stochasticity.

See [research/STATE.md](research/STATE.md) for the canonical short-form state, [research/CLAIMS.md](research/CLAIMS.md) for claim boundaries, and [research/ROADMAP.md](research/ROADMAP.md) for the milestone plan.

## Why intervention matters

A high attribution score is not causal evidence by itself.

Earlier experiments repeatedly exposed this distinction. In Experiment 008, for example, the planted root was localized correctly and restoring it recovered the target behavior, yet non-root restorations also produced material recovery. The frozen causal-specificity criterion therefore failed.

MRF treats that failure as evidence, not as a threshold-tuning problem.

## Research discipline

The repository uses several rules to reduce post-hoc reasoning:

- clean behavior is checked before root-cause analysis;
- planted truth is isolated from diagnostic inputs;
- benchmark construction rules are frozen before result-bearing runs;
- feasibility amendments are recorded before affected model outcomes;
- development and confirmatory evidence are kept separate;
- negative gates and failed constructions are retained;
- generated evidence includes hashes and runtime provenance;
- claims are bounded to what the current experiment actually tests.

## Repository map

```text
configs/        frozen configurations for historical experiments
experiments/    experiment-specific protocols and result records
research/       current state, claims, decisions, literature, and design records
scripts/        preparation, training, evaluation, and audit entry points
src/            reusable research implementation
tests/          software and scientific-invariant tests
```

Historical experiments remain in the repository because they document how later benchmark controls were motivated. They are not all part of the current method.

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

Training and inference protocols record their own pinned model/configuration requirements. Large generated checkpoints are not committed as ordinary source files.

## Scope

The current evidence is conditional on the evaluated tasks, models, and frozen protocols. MRF does not currently establish:

- successful localization on the new matched benchmark;
- confirmatory Exp009 causal certification;
- state-of-the-art training-data attribution;
- superiority to modern attribution methods;
- general causal identification for arbitrary ML incidents;
- cross-model or cross-dataset generalization; or
- a completed paper or accepted publication.

Those boundaries are intentional.
