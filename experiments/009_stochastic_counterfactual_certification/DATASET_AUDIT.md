# Experiment 009 — Banking77 dataset audit

Status: pre-pilot source audit.

## Pinned source

- repository: `PolyAI-LDN/task-specific-datasets`
- revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- file: `banking_data/train.csv`
- SHA-256: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`
- raw training rows: 10,003
- official test split loaded during this audit: no

## Canonical duplicate finding

Before any Experiment 009 pilot training or model-result inspection, the pinned
training source was audited using canonical `(label, text)` identity after outer
whitespace trimming and Unicode NFC normalization.

The source contains:

- 9,999 unique canonical records;
- 4 duplicate canonical groups;
- 4 duplicate occurrences beyond the first copy.

Each duplicate group has multiplicity two. Within every group, both copies have
the same Banking77 label and exactly the same canonical text.

The affected labels are:

- `card_not_working`;
- `atm_support`;
- `change_pin`;
- `pin_blocked`.

## Frozen development policy

Exact canonical duplicates are collapsed to one unique record **before** the
stratified development train/eval partition is constructed.

This policy is chosen prospectively, before any pilot model result, because it:

1. prevents an identical example from being represented in both development
   training and development evaluation;
2. prevents duplicate rows from receiving accidental extra training weight;
3. avoids allowing source-row position to decide which duplicate copy survives;
4. preserves one representative of every unique canonical `(label, text)` pair;
5. keeps the transformation fully auditable against the untouched 10,003-row
   source file.

The development partition therefore operates on 9,999 unique canonical records,
not 10,003 raw rows.

The audit code records both raw and unique counts and requires the pinned source
to retain the exact observed duplicate profile. Any future mismatch is a hard
source-audit failure rather than an implicit repartition.

## Scientific boundary

This is a source-cleaning decision, not a model-selection or benchmark-tuning
decision. It was made before classifier training, target-pair selection,
corruption-dose calibration, nuisance calibration, trajectory-count selection,
or confirmatory evaluation.

The official Banking77 test split remains embargoed for development.
