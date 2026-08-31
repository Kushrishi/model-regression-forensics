# Experiment 004 — Training-order audit

## Status

**POST-HOC / EXPLORATORY / DESCRIPTIVE**

No model training was performed for this audit.

All Experiment 004 candidate shards are strongly interleaved across the
288-example training sequence. The simple hypothesis that `shard_rca_04`
produced the only measurable restoration effect because it occurred unusually
late in training is not supported.

| Shard | Mean position | Median | Q4 | Last 64 | Last 32 | Last 16 | Distinct batches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `shard_rca_01` | 152.33 | 167.00 | 10 | 9 | 5 | 4 | 23 |
| `shard_rca_02` | 145.92 | 115.50 | 10 | 10 | 6 | 2 | 24 |
| `shard_rca_03` | 127.39 | 109.00 | 9 | 7 | 2 | 0 | 25 |
| `shard_rca_04` | 140.53 | 136.00 | 7 | 7 | 3 | 2 | 25 |
| `shard_rca_05` | 154.33 | 154.50 | 9 | 9 | 5 | 3 | 24 |

The candidate's approximately 71% `REJECT` label skew is also distributed
throughout all four training quartiles rather than concentrated at the end.

Gross shard position therefore does not explain the observed restoration
asymmetry.

Training nevertheless uses `batch_size=8`, `shuffle=False`, and 10 repetitions
of the same exact sequence. A subtler mini-batch/order-dependent optimization
effect remains possible.

The next post-hoc study therefore applies one alternative deterministic
permutation identically to the baseline, candidate, and all five restoration
datasets while holding all other training and evaluation variables fixed.

This exploratory analysis cannot alter the frozen Experiment 004 conclusion.
