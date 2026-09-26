# Exp009 structurally matched benchmark preflight result

**Status:** PASS after prospective world-count amendment  
**Evidence class:** clean-only structural construction  
**Successful workflow run:** `36211417215`  
**Source revision:** `8d7dc743134d2a4658c44f70695db2609d232af7`  
**Artifact:** `10895363438`  
**Matched-benchmark model training performed:** no  
**Official Banking77 test split loaded:** no

## Initial three-world feasibility result

The base M3 protocol requested three worlds with five globally intent-disjoint
candidate pairs per world.

Hosted preflight `36211223915` failed before model training because the third
world could not be filled.

A clean-only graph diagnostic `36211324469` established:

- eligible pair edges: **81**;
- eligible intent vertices: **27**;
- rank-greedy disjoint pairs: **11**;
- maximum-cardinality matching: **13**;
- pairs required for three worlds: **15**.

Therefore three complete worlds are impossible under the frozen clean
eligibility criteria.

The project did not lower recall/count/adjacency criteria to force the preferred
world count. Amendment 1 prospectively reduced M3 to **two complete worlds**
before matched-benchmark model training.

## Successful two-world construction

The amended preflight selected:

- worlds: **2**;
- candidates per world: **5**;
- candidate changes: **10**;
- unique candidate-touched intents: **20**;
- changed slots per candidate: **66**;
- symmetric swaps: **33 / 33**;
- text changes per candidate: **0**;
- aggregate label-count delta: **0**.

Every candidate within both worlds has the same structural footprint.

### World 0

Selected pairs, in construction order:

1. `get_physical_card` ↔ `pin_blocked`
2. `Refund_not_showing_up` ↔ `request_refund`
3. `card_payment_not_recognised` ↔ `extra_charge_on_statement`
4. `activate_my_card` ↔ `card_not_working`
5. `card_about_to_expire` ↔ `getting_spare_card`

Deterministic root position: **4** (zero-based)

Root pair:

`card_about_to_expire` ↔ `getting_spare_card`

Composite release SHA-256:

`8e951a2e1aa1d234f3fd8f8aefd1afecfc8baca0124666d99b201593b9bf4f29`

### World 1

Selected pairs, in construction order:

1. `card_payment_wrong_exchange_rate` ↔ `exchange_charge`
2. `cash_withdrawal_charge` ↔ `cash_withdrawal_not_recognised`
3. `top_up_by_card_charge` ↔ `verify_top_up`
4. `cancel_transfer` ↔ `transfer_timing`
5. `card_arrival` ↔ `getting_virtual_card`

Deterministic root position: **4** (zero-based)

Root pair:

`card_arrival` ↔ `getting_virtual_card`

Composite release SHA-256:

`a67802d6b8ab190557812a9b572adf33881a33187c2497ac280f72b978719d2e`

## Structural audit

Both worlds passed all frozen structural checks:

- all five candidate structures identical;
- changed-slot sets pairwise disjoint;
- intent labels pairwise disjoint;
- exactly 330 composite changed slots;
- composite independent of candidate-application order;
- every standalone candidate restores exactly to baseline;
- restoring one candidate from the composite leaves exactly four candidate
  changes;
- diagnostic candidate IDs are opaque and role-free.

Frozen development partition:

`61ecbdd9224cf2bbeafcf5ceb9164cfc3fc74eb449129136198646385aaa8e88`

Clean baseline release:

`cb83232c055c4c55ca50f2fbd86627d59dc806ab33a861abc7402990ddb7e20c`

## Artifact integrity

The hosted artifact contains the deterministic diagnostic manifest, truth
manifest, and structural audit for each world.

Key SHA-256 values:

- summary:
  `321c1c5d1756e8fdee4a433c11ef0914ebf31616053e72c644c772296a25c344`
- world 0 diagnostic:
  `2861f10e35d40d28a94890bf6d49d2ff96c04161937d653bac11276846e1c0f4`
- world 0 structural audit:
  `9347e94ef0b74805c7c60a29cbc0b7c1469c1960bfbaddb469d8f28ae455ae3b`
- world 0 truth:
  `4ad2e5c331072457edd7db8bf8ec39e0fc263f22465c50ba5d9e18ca77eb3ef8`
- world 1 diagnostic:
  `9d1b10a29fc4ee0babfb839172fec368a16360470d74b89a2a91f06cb1cc7ebd`
- world 1 structural audit:
  `c0429b80f9ebd77d01bbeff54da7144d51af6cd61cfa22eb6de4e2c27e15f67e`
- world 1 truth:
  `ada221f35aa6ba1a50c19413f383e6b5b97476db1803d9dff4ac8f907ca368b7`

## Interpretation boundary

This result establishes the **construction** of a structurally matched
candidate benchmark.

It does not establish:

- that either composite produces a material target regression;
- localization success;
- localization difficulty;
- causal specificity;
- a certification/abstention rule;
- superiority over any attribution baseline;
- confirmatory evidence.

The target behavior is known to the debugger. A simple semantic/change-overlap
baseline may therefore rank the responsible pair highly. That is legitimate
release-forensics information and must be reported rather than designed away.

The next active milestone is M4: train/evaluate the matched development worlds
under a prospectively controlled preflight and compare simple overlap/random
references with target-faithful attribution baselines. The official Banking77
test split remains embargoed.
