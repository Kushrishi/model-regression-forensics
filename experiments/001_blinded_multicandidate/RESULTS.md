# Experiment 001 — Results

Date: 2026-08-29

## Scope

This file freezes the model-validation phase of Experiment 001 before blinded
root-cause ranking begins. The selective intervention uses benchmark-owned oracle
ground truth and therefore validates the planted causal structure; it is **not**
a diagnostic result.

## Frozen protocol

All three runs were fresh sibling LoRA SFT runs from:

- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- revision: `a10cc1512eabd3dde888204e902eca88bddb4951`
- seed: `42`
- epochs: `10`
- batch size: `8`
- learning rate: `5e-4`
- warmup ratio: `0.05`
- max length: `192`
- LoRA rank / alpha / dropout: `16 / 32 / 0.0`
- trainable parameters: `8,683,520 / 370,504,640`
- optimizer steps: `360`

The intended manipulated variable was the prepared training split.

## Training inputs

| Run | Split | JSONL file SHA256 |
| --- | --- | --- |
| Baseline | `baseline_train` | `4fc71884e17d4d4c4a32287f76a14a4bb0575c7f43ab78b6909f977a1255818e` |
| Candidate | `candidate_train` | `8c73c1c9e3a783545e12debf25e8d95ef9b86bd22e918cc2b1fe038a41f15c47` |
| Intervention | `intervention_train` | `abaa50c330007e3582461cf7bdf1f0f008c587c541ea24e72857b14a11958c9b` |

Candidate training changes five equal-sized 48-record shards. The oracle
intervention restores only `shard_delta_04`, leaving the other four changed
shards (192 changed records) in place.

## Held-out results

Primary metric: `label_accuracy`. Strict exact match produced the same scores.

| Split | Baseline | Candidate | Intervention |
| --- | ---: | ---: | ---: |
| Target: `triangle_large` (16) | 1.000 | 0.000 | 1.000 |
| Control: `square_small` (16) | 1.000 | 1.000 | 1.000 |
| Full held-out set (96) | 1.000 | 0.1667 | 0.3333 |

Derived protocol quantities:

- baseline target score: `1.000`
- baseline-to-candidate target regression: `1.000`
- candidate-to-intervention target recovery: `1.000`
- maximum observed control drift: `0.000`

Predeclared gates from `configs/exp001.yaml`:

| Gate | Threshold | Observed | Result |
| --- | ---: | ---: | --- |
| Baseline target | >= 0.80 | 1.000 | PASS |
| Target regression | >= 0.15 | 1.000 | PASS |
| Target recovery | >= 0.10 | 1.000 | PASS |
| Control drift | <= 0.05 | 0.000 | PASS |

## Behavioral interpretation

The candidate failed exactly 80 of 96 held-out cases while the unchanged
`square_small` control remained perfect. This matches the construction: five of
six shape/size slices were flipped in candidate training, so 5 x 16 = 80 held-out
cases disagree with the clean canonical policy.

After restoring only the target-causal shard, the intervention scored 32/96. The
restored `triangle_large` target (16 cases) and unchanged `square_small` control
(16 cases) were correct, while the other four deliberately changed slices
remained incorrect.

Therefore the selective intervention behaved exactly as predicted: repairing the
target-causal shard restored the specified regression without reverting the four
other candidate changes.

## What this establishes

Experiment 001 now has a valid causal regression instance with:

1. a perfect clean baseline;
2. multiple simultaneous observable training changes;
3. a complete held-out target regression;
4. a stable negative-control slice; and
5. exact target recovery after a selective oracle intervention.

This validates the benchmark's planted causal structure.

## Blinded diagnosis results

The RCA implementation was committed before rankings were generated. Diagnosis
used only the debugger-visible diagnostic lineage, changed-shard artifacts, and
observed baseline/candidate target generations. The resulting ranking JSON files
were committed and pushed before the benchmark manifest was used for scoring.

Hidden root cause: `shard_delta_04`.

| Method | Root-cause rank | Top-1 | Top-3 | Reciprocal rank |
| --- | ---: | --- | --- | ---: |
| Seeded random (`42`) | 1 / 5 | True | True | 1.000 |
| Lexical overlap | 1 / 5 | True | True | 1.000 |

Five-candidate chance references:

- Top-1 accuracy: `0.20`
- Top-3 recall: `0.60`
- expected reciprocal rank: `0.4567`
- possible permutations: `120`

The lexical ranking was:

1. `shard_delta_04`: `0.9091`
2. `shard_delta_02`: `0.8261`
3. `shard_delta_03`: `0.8261`
4. `shard_delta_05`: `0.8261`
5. `shard_delta_01`: `0.7500`

The seeded-random baseline also happened to place the hidden cause first. That is
a chance success on one five-way trial, not evidence that random ranking is
competitive. Likewise, one Top-1 lexical result does not establish statistical
superiority or general RCA capability.

## Final interpretation

Experiment 001 establishes that the project can:

1. construct multiple observable training changes with hidden benchmark truth;
2. induce a controlled target regression while preserving a negative control;
3. selectively restore only the causal shard and recover the target behavior;
4. run a ground-truth-free ranking procedure from committed code;
5. freeze ranking artifacts before revealing benchmark truth; and
6. score the frozen rankings using standard localization metrics.

The experiment also exposes its own main limitation: each changed shard maps
cleanly to one behavioral slice, so lexical overlap can identify the causal shard
without deeper model-based attribution. Experiment 001 is therefore a successful
end-to-end blinded RCA sanity check, **not** evidence of difficult or general
root-cause localization.

Experiment 002 must remove this shortcut by making several visible candidate
shards similarly target-relevant while preserving a single hidden causal shard.
Later benchmark sweeps must also repeat across hidden causes and seeds so method
performance is measured over many trials rather than one instance.
