# Hosted M4 execution

**Status:** completed under the separate source-pinned request; accepted result in `M4_RESULT.json`

This implementation executes the frozen M4 localization protocol and Amendment 1. It adds no method, target, candidate, trajectory, or causal-certification claim.

## Execution boundary

The dedicated workflow starts only when a separate `research/M4_EXECUTION_REQUEST.json` reaches main. The request must pin the complete implementation source commit, successful CI and Research CI for that commit, and the exact protocol and lockfile digests. Every job rejects tracked differences from that source except the request itself.

Preparation reconstructs the M3 benchmark from historical clean evidence and verifies every frozen M3 artifact digest. Training jobs receive only the blind release bundles. Benchmark truth is supplied to a separate job after the complete blind ranking has been written, hashed, and uploaded.

The matrix contains three clean trainings and six composite trainings on hosted macOS MPS runners. Each composite captures all seven epochs, the complete tensor-state digest, optimizer-step count, and final update's producing learning rate. The initial warmup update may use zero learning rate. Epoch-producing learning rates must be positive; the scheduler's next-update rate is not substituted.

The scoring path evaluates the frozen B0 through B4 definitions. B3 uses the unweighted final-checkpoint weight-and-bias influence. B4 sums all seven contributions using their producing learning rates. All target examples are retained. B0 integer scores remain integers during aggregation so rounding cannot alter its ordering.

Portable evidence is uploaded before job outcome enforcement. Incomplete checkpoint bytes are retained separately for seven days when a training or scoring job fails. Complete checkpoint hashes, portable run records, blind scores, and final analysis are retained for 90 days.

## Failure and retry policy

There is no automatic scientific retry or parameter tuning. A failure is classified from logs and input/provenance evidence before any retry. An unchanged-source retry of a failed job is permitted only for an operational failure, with the original run and attempt retained. Completed successful jobs are not selected or replaced using numerical outcomes. A code correction requires a new implementation commit, green CI, and a separate source-pinned request before execution.

Blind aggregation requires all nine trainings and all six composite scoring records. Partial rankings are not eligible for milestone reporting. The official Banking77 test split and M5 restoration training remain unauthorized.

## Validation

Unit tests check the source gate, protocol identities, scope restrictions, blind-bundle schema, evidence preservation, and checkpoint capture. A tiny synthetic classifier verifies that zero warmup does not abort training, all seven checkpoints match their saved tensor hashes, optimizer-step counts increase, and the final producing learning rate remains positive even when the scheduler's next-update rate is zero.

No matched-world model training is performed by this preparation.
