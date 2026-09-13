# Experiment 009 — Development Data Substrate

Status: **frozen for development as of 2026-09-12**.

This record freezes the train-only Banking77 substrate used for Experiment 009
pilot development. It does **not** freeze the later model, hyperparameters,
benchmark worlds, corruption dose, confirmatory trajectory count, or statistical
procedure.

## Pinned source

- repository: `PolyAI-LDN/task-specific-datasets`
- revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- source path: `banking_data/train.csv`
- source SHA-256: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`
- raw source rows: `10003`
- intents: `77`

The official Banking77 test split was not loaded during this audit and remains
embargoed during development.

## Canonical duplicate policy

The pinned training source contains four exact canonical duplicate groups. Each
group contains two rows with the same canonical label and canonical text.

Before development partitioning, exact canonical `(label, text)` duplicates are
collapsed by SHA-256 content identity. Source-row position is not used to decide
which duplicate survives.

Frozen duplicate profile:

- canonical duplicate groups: `4`
- repeated occurrences removed: `4`
- unique canonical records after collapse: `9999`

This policy prevents identical examples from appearing in both development
training and development evaluation and avoids duplicate weighting caused by
source-file repetition.

## Development partition

The unique canonical records are partitioned independently within each intent.
Records are ordered by the SHA-256 content identity of canonical label/text.
The nearest-integer 20 percent of each intent is assigned to
`development_eval`; the remainder is assigned to `development_train`.

Frozen counts:

- `development_train`: `8001`
- `development_eval`: `1998`
- total unique records: `9999`

Frozen partition manifest SHA-256:

`61ecbdd9224cf2bbeafcf5ceb9164cfc3fc74eb449129136198646385aaa8e88`

## Validation state

At the freeze point:

- source audit: passed;
- targeted Exp009 data tests: passed;
- full repository tests: `169 passed`;
- Ruff lint: passed;
- Ruff format check: passed;
- `git diff --check`: passed;
- official test split loaded: **NO**.

Any future code that regenerates the development substrate should reproduce the
same counts and partition-manifest hash. A mismatch is a protocol event that
must be investigated before using the resulting data for pilot measurements.
