# Experiment 009 — Pilot Nuisance Rule Outcome

Status: **development rule infeasible; no nuisance-model training performed**.

The first pilot nuisance-construction rule was frozen before any nuisance-model
training result was observed.

On September 15, 2026, the deterministic pre-training audit stopped because
the frozen clean-only eligibility and greedy intent-disjointness rule produced
fewer than the required four nuisance pairs.

## Outcome

- eligible pairs before disjoint selection: **0**
- required disjoint nuisance pairs: **4**
- greedily selectable disjoint nuisance pairs: **0**
- nuisance-model training performed: **NO**
- official Banking77 test split loaded: **NO**

The disjoint pairs selected before the rule exhausted were:

- none

## Interpretation

This is a development-benchmark construction result, not a model-training
failure and not a confirmatory Experiment 009 result.

The frozen rule explicitly required nuisance construction to stop if fewer than
four disjoint pairs satisfied the rule. That stopping condition was honored.

The rule must not be silently relaxed or individual nuisance pairs manually
substituted. A revised development nuisance rule may be designed from this
failure, documented, and frozen before any nuisance-model training begins.

The accepted `1/4` root corruption dose and its three-trajectory pilot evidence
are unchanged.
