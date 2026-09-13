# Experiment 008 — Results

## Outcome

Experiment 008 completed the prospectively frozen two-world evaluation.

The experiment successfully established target-localized candidate regressions in both frozen worlds and the truth-isolated `selected_role_overlap` diagnostic uniquely ranked the planted root cause first in both worlds.

Selective restoration of the planted root also produced full target recovery in both worlds while preserving all protected behaviors.

However, the prospectively frozen unique-causal-certification criterion failed in both worlds because one or more non-root restorations produced target recovery above the allowed nuisance ceiling. Experiment 008 therefore does **not** establish unique causal certification.

The primary outcome is:

> **Localization succeeded and planted-root restoration was strongly restorative, but single-run counterfactual restoration was not uniquely specific under the frozen training regime.**

The alternative-order robustness control was not run because the frozen protocol reserved that control for a successful primary causal certification. It is not used as a rescue analysis.

## Frozen criteria

The relevant prospectively declared thresholds were:

- clean baseline accuracy >= 0.95 on every required split;
- baseline-to-candidate `triangle_large` regression >= 0.15;
- candidate protected-slice drift <= 0.05;
- planted-root target recovery >= 0.15;
- planted-root protected-slice drift <= 0.05;
- every non-root target recovery <= 0.05;
- every non-root protected-slice drift <= 0.05;
- exactly one restoration may satisfy the target-recovery threshold.

No threshold was changed after observing model behavior.

## Clean baseline

The shared clean baseline scored 1.0000 on every required evaluation split:

| Split | Accuracy |
| --- | ---: |
| `triangle_large` | 1.0000 |
| `circle_small` | 1.0000 |
| `circle_large` | 1.0000 |
| `square_small` | 1.0000 |
| `square_large` | 1.0000 |
| `triangle_small` | 1.0000 |
| `all` | 1.0000 |

The clean-baseline gate passed.

## Candidate regression gate

Both frozen candidate worlds produced a material target regression with zero protected-slice drift.

| World | Planted root | Candidate target | Target regression | Max protected drift | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| 00 | `shard_selective_05` | 0.0000 | 1.0000 | 0.0000 | PASS |
| 01 | `shard_selective_02` | 0.1875 | 0.8125 | 0.0000 | PASS |

The two-world candidate prerequisite therefore passed.

## Truth-isolated diagnosis

Diagnostic rankings were frozen before comparison with private benchmark truth.

The `selected_role_overlap` diagnostic uniquely ranked the planted root first in both frozen worlds:

| World | Planted root | Root rank | Uniquely Top-1 |
| --- | --- | ---: | --- |
| 00 | `shard_selective_05` | 1 | Yes |
| 01 | `shard_selective_02` | 1 | Yes |

The lexical baselines did not uniquely solve the task. Their five candidate scores tied, while the deterministic random ranking did not consistently identify the root.

This establishes successful localization under the frozen benchmark. It does not by itself establish causality.

## Primary counterfactual restoration

All five candidate changes were restored independently in each frozen world, producing ten fresh primary restoration models.

### World 00

Candidate `triangle_large` accuracy was 0.0000.

| Restoration | Role | Target accuracy | Recovery | Max protected drift | Recovery gate | Protected gate |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `shard_selective_01` | non-root | 0.1875 | +0.1875 | 0.0000 | FAIL | PASS |
| `shard_selective_02` | non-root | 0.0000 | +0.0000 | 0.0000 | PASS | PASS |
| `shard_selective_03` | non-root | 0.1875 | +0.1875 | 0.0000 | FAIL | PASS |
| `shard_selective_04` | non-root | 0.1875 | +0.1875 | 0.0000 | FAIL | PASS |
| `shard_selective_05` | planted root | 1.0000 | +1.0000 | 0.0000 | PASS | PASS |

The planted root fully restored the target with no protected drift.

However, three non-root restorations also exceeded the maximum allowed non-root recovery of 0.05. Four restorations satisfied the general >= 0.15 recovery threshold.

World 00 therefore failed non-root recovery specificity and unique recovery.

### World 01

Candidate `triangle_large` accuracy was 0.1875.

| Restoration | Role | Target accuracy | Recovery | Max protected drift | Recovery gate | Protected gate |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `shard_selective_01` | non-root | 0.1875 | +0.0000 | 0.0000 | PASS | PASS |
| `shard_selective_02` | planted root | 1.0000 | +0.8125 | 0.0000 | PASS | PASS |
| `shard_selective_03` | non-root | 0.1875 | +0.0000 | 0.0000 | PASS | PASS |
| `shard_selective_04` | non-root | 1.0000 | +0.8125 | 0.3125 | FAIL | FAIL |
| `shard_selective_05` | non-root | 0.1875 | +0.0000 | 0.0000 | PASS | PASS |

The planted root again fully restored the target with no protected drift.

However, non-root `shard_selective_04` produced the same +0.8125 target recovery and also caused 0.3125 maximum protected-slice drift. Both the planted root and this nuisance exceeded the general recovery threshold.

World 01 therefore failed non-root recovery, non-root protected locality, and unique recovery.

## Certification summary

| Criterion | World 00 | World 01 |
| --- | --- | --- |
| Candidate localized-regression gate | PASS | PASS |
| Diagnostic root uniquely Top-1 | PASS | PASS |
| Planted-root recovery | PASS | PASS |
| Planted-root protected locality | PASS | PASS |
| Non-root recovery specificity | FAIL | FAIL |
| Non-root protected locality | PASS | FAIL |
| Unique recovery | FAIL | FAIL |
| Primary causal certification | **FAIL** | **FAIL** |

Overall:

**EXP008_PRIMARY_CAUSAL_CERTIFICATION = FAIL**

## Interpretation

Experiment 008 separates three claims that must not be conflated.

### 1. The benchmark successfully produced localized regressions

Both candidate worlds strongly regressed `triangle_large` while all protected slices remained unchanged.

This repairs the materiality/locality problem encountered in Experiments 005–007.

### 2. The diagnostic successfully localized the planted training change

The truth-isolated semantic diagnostic uniquely ranked the planted root first in both frozen worlds.

This is strong localization evidence under this controlled benchmark.

### 3. Counterfactual restoration did not uniquely identify the cause

Reversing the planted root produced complete target recovery with zero protected drift in both worlds.

However, some policy-correct nuisance restorations also changed target behavior materially. Therefore, one fresh retraining result cannot be treated as a uniquely identifying causal intervention under this setup.

A plausible hypothesis is that modifying model-facing training order or content placement changes the optimization trajectory enough to affect held-out target behavior even when the restored candidate is not the planted label-corruption root.

Experiment 008 does not prove that mechanism. It establishes only the observed intervention instability. Distinguishing a true intervention effect from ordinary retraining-path variability is a target for the next research phase.

## Stopping decision

The frozen Experiment 008 protocol declared this the final major benchmark-design iteration on the current synthetic shape substrate.

Because unique causal certification failed, the project will not create an Experiment 009 whose purpose is to tune corruption dose, thresholds, world seeds, or nuisance allocation on the same substrate.

The shape substrate is retired as the primary research benchmark.

The next phase will instead move toward:

- stable reusable `model_forensics` research APIs;
- a more realistic natural-language regression substrate;
- explicit measurement of training/retraining variability;
- paired repeated counterfactual interventions;
- stronger attribution baselines;
- later cross-task or cross-model replication.

## Claim boundary

Experiment 008 supports the narrow statement:

> Under this frozen controlled setup, behavioral evidence and debugger-visible lineage were sufficient for a task-aware diagnostic to localize the planted training change in both worlds, and restoring that planted change fully repaired the target behavior without collateral protected-slice degradation.

It does **not** support the stronger statement that the intervention procedure uniquely established causal responsibility.

The principal methodological lesson is therefore:

> **Regression localization, restorative influence, and uniquely specific causal verification are distinct requirements and should be evaluated separately.**
