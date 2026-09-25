# Research roadmap

Last updated: 2026-09-24

Goal: finish Model Regression Forensics as a defensible research artifact
without sacrificing rigor for speed.

The project uses **serial scientific decisions with parallel mechanical work**:
only one unresolved scientific bottleneck controls the next experiment, while
testing, automation, literature extraction, documentation, and implementation
can proceed in parallel when they do not contaminate that decision.

## Quality gates

### G0 — Canonical-state integrity

Exit criteria:

- active scientific state is summarized in `RESEARCH_STATE.md`;
- stale public claims are identified;
- branch divergence is understood;
- tests/lint run automatically;
- experiment evidence and generated artifacts have explicit provenance.

### G1 — Research-gap audit

Exit criteria:

- current related work is refreshed;
- closest competing formulations are compared structurally;
- novelty/non-novelty boundaries are explicit;
- the paper question survives or is narrowed;
- required baselines are selected prospectively.

### G2 — Exp009 development benchmark

Exit criteria:

- nuisance construction is scientifically defensible;
- construction preflight passes without looking at nuisance training outcomes;
- candidate changes are non-trivial and do not expose the root by an avoidable
  structural shortcut;
- pilot root remains stable under the frozen development rules.

### G3 — Baseline/evaluation implementation

Exit criteria:

- selected baselines have tested implementations;
- compute/evaluation parity rules are documented;
- all methods consume only allowed diagnostic information;
- result schemas are machine-readable;
- primary aggregation/statistics code is tested.

### G4 — Confirmatory freeze

Exit criteria:

- official test split remains untouched;
- protocol, thresholds, seed schedule, candidate set, methods and failure rules
  are committed;
- a pre-confirmatory audit verifies no unresolved decision depends on hidden
  outcomes.

### G5 — Confirmatory execution

Exit criteria:

- all frozen runs are attempted;
- operational failures are retained;
- structured evidence is immutable/versioned;
- no post-hoc rescue threshold or candidate substitution occurs.

### G6 — Claims audit

Exit criteria:

- `CLAIMS_LEDGER.md` is updated from confirmatory evidence;
- effect uncertainty and failure cases are reported;
- every headline claim has direct supporting evidence;
- unsupported claims are removed rather than rationalized.

### G7 — Manuscript-quality package

Exit criteria:

- manuscript reads coherently without requiring repository archaeology;
- main figures/tables regenerate deterministically;
- reproducibility checklist is satisfied to the extent technically possible;
- limitations are explicit;
- external-facing README reflects the same scientific status.

### G8 — Adversarial review and release

Exit criteria:

- independent hostile-review pass completed;
- strongest anticipated reviewer objections have evidence-backed answers;
- anonymization/release plan matches target venue rules;
- preprint/submission is made only after all factual and authorship disclosures
  are correct.

## Daily operating loop

Every work session should end with one durable state transition. Examples:

- a literature question resolved;
- a protocol frozen;
- a failed construction recorded;
- an implementation landed with tests;
- a run completed with structured evidence;
- a claim upgraded/downgraded;
- a stale public statement corrected.

Avoid sessions whose only outcome is exploratory code with no recorded decision.

### Start of session

1. Read `RESEARCH_STATE.md`.
2. Identify the single current scientific bottleneck.
3. Separate decisions that require evidence from mechanical tasks that can run
   independently.
4. Do not launch a result-bearing experiment until its decision rule is written.

### End of session

1. Record result/evidence.
2. Update the decision log if the direction changed.
3. Update the claims ledger if evidence changed a claim.
4. Update `RESEARCH_STATE.md` if the canonical state changed.
5. Ensure CI passes.
6. Leave exactly one explicit next scientific decision.

## Efficiency rules

- No new experiment number unless the current paper question requires it.
- Prefer a small number of information-rich runs over broad parameter sweeps.
- Use cheap preflights before expensive training.
- Parallelize independent stochastic trajectories and baselines.
- Cache immutable datasets/model revisions where permitted.
- Never recompute a result that can be verified from hashed structured evidence.
- Generate plots/tables from result files; do not hand-copy numbers.
- Keep development and confirmatory evidence physically and semantically
  separate.
- Do not update website/LinkedIn from pilot evidence.
- Stop low-value work early.

## Priority order

1. G0 canonical-state integrity and CI
2. G1 literature + baseline audit
3. G2 nuisance design
4. G3 baseline/evaluation implementation
5. G4 confirmatory protocol
6. G5 confirmatory runs
7. G6 claims audit
8. G7 manuscript
9. G8 external release

This ordering is intentionally designed to prevent compute speed from outrunning
scientific validity.
