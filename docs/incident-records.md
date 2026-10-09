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
One artifact identity must have one SHA-256 throughout the incident and ledger,
including run evidence and the decision policy. Conflicts fail validation even
without optional payload verification; repeated references to identical artifacts
remain valid. This checks reference consistency, not payload authenticity.
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

## Optional local payload verification

The replay command can additionally check bytes using both `--artifact-map paths.json`
and `--artifact-root evidence`. The map is an explicit JSON object from artifact
identity to a canonical relative POSIX path, for example `{"release-good":"good.json"}`.
Supply every referenced identity exactly once and no unrelated entries. Identities
are never interpreted as paths or download URLs. A conflicting hash for the same
identity is rejected. Evaluator provenance is included only when `--truth` is
explicitly supplied; this option does not discover or unseal evaluator records.

Checks use bounded streaming SHA-256 over regular local files only: 64 MiB per
artifact, 256 MiB aggregate, and 8 MiB for the map. Absolute paths, traversal,
symlinks, nonregular files, missing files, mismatches and observed in-read changes
fail closed. Descriptor-based traversal requires POSIX support; unsupported
platforms fail rather than silently using a weaker fallback. The root's parent
directories are trusted. This is not protection against arbitrary hostile
concurrent filesystem mutation. Do not map protected outcomes into a debugging
manifest. Nothing extracts archives, previews contents, downloads or trains.

The optional `payload_verification/0.1` report records identities, digests and byte
counts, not local paths or payload contents. It certifies only that the bytes read
matched the references, not that a run actually occurred, an artifact is authentic,
a diff is complete, or a diagnosis is scientifically justified. Without these flags,
the original report and digest semantics are unchanged. Output remains non-clobbering.
