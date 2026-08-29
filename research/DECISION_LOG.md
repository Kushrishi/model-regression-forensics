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
