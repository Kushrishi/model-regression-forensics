# Experiment 008 Pre-Model Freeze

Status: frozen before Experiment 008 model training.

No Experiment 008 model had been trained when this freeze record was created.

## Frozen provenance

- Public frozen-world manifest SHA256: `d9f0ea89edd180aca89f617f8feb71fee6c92f93a3aedddff6f9b1fcc98368a8`
- Clean baseline JSONL SHA256: `45b2899f22b162ae209948340307bff311349222a6c5c53934b924d773259d8c`
- Clean baseline SFT SHA256: `6d2a73ef3f0101a1836ff6cb4e4fc65eb6007e5b2fe1fd9997e89682d301d150`
- Frozen worlds: 2
- Explicit root truth in public frozen manifest: no
- Exp008 clean training input equivalent to Exp007 clean baseline: yes
- Prior adapter reuse authorized from equivalence alone: no

## Scientific construction

Both frozen worlds passed the complete static construction contract.

The baseline and candidate each contain 288 unique prompts.

The candidate prompt multiset is identical to the baseline prompt multiset.

Every nuisance intervention preserves its correct prompt/response-pair multiset.

Exactly 36 policy-inconsistent training examples exist in each candidate world,
and every one is on `triangle_large`.

Protected slices receive zero policy-inconsistent supervision.

## Truth isolation

Experiment 008 does not claim investigator blinding because benchmark-private
root identities were visible during protocol-construction validation.

Diagnostic evaluation remains truth-isolated: the diagnostic receives the
redacted lineage and permitted regression evidence, freezes its ranking, and is
scored against private benchmark truth afterward.

## Portability

The scientific protocol is hardware-independent.

Primary sibling comparisons must use matching runtime/backend provenance.

Windows WSL configuration is external to the repository.

## Validation

- 160 passed in 47.95s
- Ruff formatting: pass
- Ruff lint: pass
- git diff --check: pass
- Frozen-manifest byte regeneration: pass
- Public frozen manifest truth-free check: pass
- World 0 preparation: pass
- World 1 preparation: pass
- Exp007/Exp008 clean-baseline equivalence: pass
