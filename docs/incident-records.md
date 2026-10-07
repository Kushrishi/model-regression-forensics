# Incident records (experimental schema 0.1)

This module adds a non-training ledger beside the existing exact-label comparator
and repair assessment. It does not replace the v1 machinery. It validates supplied
records, not the authenticity of runs, completeness of a release diff, scientific
adequacy of an estimand, or confidence in a diagnosis. No certification threshold,
training command, ranking method or automatic specific-diagnosis policy is added.

`Incident` requires good/bad release, environment, evaluation and visible evidence
artifact identities with SHA-256, categorized candidates, target functional,
protected behavior (or explicit non-applicability), intervention semantics,
training process, randomness treatment, restoration meaning and attempt budget.
Its `use` is discovery, synthetic software or retrospective software only. These
schema labels cannot convert an exploratory record into confirmatory evidence.

`Ledger` preserves completed, failed and timeout intervention attempts. Every
attempt counts against budget; repetitions have explicit randomness IDs. Each
completed attempt has finite target measurements; failures retain evidence and
cannot masquerade as usable scores. Group interventions are explicit change sets.
Duplicate run IDs, duplicate intervention/randomness pairs, unknown candidates,
unknown evidence references, nonfinite values and budget overruns are rejected.
A retry must have a distinct randomness/execution identity and consume budget;
there is no executor or automatic retry here.

`Decision` records a supplied policy artifact and rationale, evidence references,
and one hypothesis, several hypotheses, insufficient evidence or abstention. Each
hypothesis is a candidate change set, so interactions are representable. This is
an output contract, **not a policy that certifies uniqueness**. A single observed
successful repair does not identify a historical cause or imply untested repairs
failed. Partial evidence is preserved; the caller must justify any commitment.

`Truth` is a separate evaluator-only record bound to the incident digest. It
contains responsibility sets **under the declared intervention process**, with
provenance; never put historical fix metadata into the engineer-visible manifest.
Omitted candidates may appear in evaluator truth. The schema does not magically
seal files or stop a user from embedding answers in free-text evidence; future
benchmarks need an explicit information-access audit.

`summarize` retains every outcome and cost without converting repetitions into
independent incidents or selecting a restoration threshold. Semantic JSON hashes
identify exact supplied records; they do not verify the referenced payloads. Input
ordering remains part of the digest, allowing exact replay. Summed worker wall
time is accounting, not elapsed parallel campaign time. No distributional
confidence interval is inferred from a handful of repeats.

`decision_metrics` accepts validated `(Ledger, Truth)` pairs, one per incident.
It reports wrong-specific rate over all incidents, commitment coverage, conditional
risk (null if zero commitments), and intervention attempts. Naming one alternative
when evaluator truth contains several equally responsible sets counts as unjustified
specificity, even if that alternative restores behavior. Per-incident output also
reports any/all responsibility-set containment and returned hypothesis/candidate
counts. Historical cause is always `not_inferred`. Top-k/risk–coverage curves,
restoration/collateral distributions and oracle regret require future appropriately
specified data; no fabricated value is emitted.

## Run the retrospective software fixture

```bash
uv sync --extra dev
uv run python examples/incident_ledger.py --out /tmp/mrf-incident-fixture
uv run python -m model_forensics.incidents \
  /tmp/mrf-incident-fixture/ledger.json \
  --truth /tmp/mrf-incident-fixture/truth.json \
  --out /tmp/mrf-incident-replay.json
```

Use fresh output names. The example directory and CLI report never clobber existing
artifacts. CLI JSON inputs are bounded to 8 MiB; invalid input or output collision
exits 2 with an error. `--help` exits 0. No data/model downloads or training occur.
The retained ambiguous-repair example supplies arithmetic prediction outcomes;
its hypothetical intervention units are a software representation, not a newly
qualified external incident. Fixture truth is generated separately after the
ledger and is never consumed by its repair assessment. Retrospective metric
validation is not confirmatory scientific evidence or future threshold selection.

The example writes zero cost placeholders because its retained arithmetic evidence
did not measure worker timing. These zeros are not performance observations.
