# Decision log

## 2026-08 — Scope refinement

### Rejected

"pytest for ML robustness" and generic failure discovery are too crowded.

### Rejected

"discover failures → generate data → post-train → evaluate → repeat" overlaps too strongly with existing automated red-teaming and improvement loops.

### Current framing

**Causal model-regression forensics**

1. observe a behavioral regression;
2. represent training lineage between baseline and candidate;
3. rank candidate causes;
4. intervene on a suspected cause;
5. accept the diagnosis only if behavior recovers under controlled conditions.

### Novelty status

**Not established.**

Goodfire's 2026 predictive data debugging raises the bar substantially. The project must distinguish itself through post-hoc diagnosis, heterogeneous lineage, benchmark design, or another defensible gap.

## 2026-08-29 — Zero-shot metric refinement

The pinned untouched `SmolLM2-360M-Instruct` reference produced a target strict-exact score of 0.00 and an unrelated strict-exact score of 0.05. Raw generations included both `REJECT` and `Reject`, showing that strict exact match mixed the behavioral question with capitalization.

This was treated as a pilot measurement issue, not as an experiment result to optimize around. Before any SFT baseline, candidate, or recovery run:

- primary behavior metric: one-token `ACCEPT`/`REJECT` label accuracy, ignoring outer whitespace and letter case only;
- secondary format metric: strict exact match after outer-whitespace trimming;
- explanations or other text remain incorrect for both protocol purposes;
- the original zero-shot strict scores remain recorded rather than being retroactively reinterpreted.

The zero-shot model is therefore a reference control. Experiment 000 requires a clean SFT baseline checkpoint before introducing the planted regression.
