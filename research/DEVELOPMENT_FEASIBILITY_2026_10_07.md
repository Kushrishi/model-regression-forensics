# Development feasibility audit — 7 October 2026

Owner-authorized development pilot, not M5 or confirmatory evidence. See the
[governance amendment](GOVERNANCE_AMENDMENT_2026_10_07.md). No independent review.

## Stage 0 boundary

A new attribution algorithm is premature. First ask whether a compact instruction
model can learn a structured overlapping-intent task, sustain unaffected behavior,
and show a repeatable deliberately corrupted training-release regression. This is
substrate feasibility, not a realistic incident benchmark or novelty evidence.

Model/tokenizer: `HuggingFaceTB/SmolLM2-135M-Instruct`, immutable revision
`12fd25f77366fa6b3b4b768ec3050bf629380bac`, Apache-2.0 model card. Transformers
4.47.1, PEFT 0.14.0, PyTorch 2.5.1+cpu, Accelerate 1.2.1. Artifact hashes retained
locally; no model binaries in Git. CPU only, no accelerator or paid API.

Task: only the official Banking77 **train.csv** at PolyAI-LDN/task-specific-datasets
revision `57ec275d8078af65b7731c2a98be812d844a6d6b`; official test.csv is never fetched.
Select four related card intents (arrival, delivery estimate, physical-card ordering,
linking). First 16 source-order rows per intent train; next 8 per intent are
in-train development evaluation. They are not untouched official test data.
Output is an exact intent string; malformed/unlisted generations count incorrect.
No subjective grading. Source text/diffs stay available to all later baselines.

Two paired development seeds, 11 and 23. LoRA rank 4, alpha 8, zero dropout,
q_proj/v_proj only; frozen base model. AdamW learning rate 0.003, batch 4,
three passes over 64 records (48 updates), context bound 256. This defines a
parameter-efficient counterfactual process, not full-model retraining. No saved
base-model copies or checkpoint history. Greedy bounded generation evaluates all
32 development examples in source order. CPU threads 2.

Initial release probe: symmetric target-label swap between card_arrival and
card_delivery_estimate, leaving the other two classes protected. This is explicitly
an obvious corruption fixture, not a claim of realistic ambiguity. Its direct diff
is legitimate evidence and must never be hidden to make localization difficult.
If even this cannot sustain a useful baseline/regression, stop before incidents.

Useful feasibility floor chosen before training: overall clean development accuracy
≥0.60, each target/protected aggregate ≥0.50 for both seeds. A measured target drop
must be positive for both seeds while protected accuracy stays above 0.50. Report
all outcomes/variance; a two-seed result supplies no general stochastic guarantee.
Failure stops this substrate attempt; escalation to another model needs a demonstrated
capacity explanation, not merely a desired score.

Compute cap for this stage: four small fits, 192 optimizer updates total; initially
measure one update and one fit. Hard 600-second limit per fit, maximum 2,400-second
wall envelope, two CPU threads, no GPU. Estimate ≤1.34 nominal CPU-hours if all
four hit the cap (startup/evaluation included); memory below the 8-GiB host limit.
No incident interventions or Stage 4 method are launched under this stage.
If runtime/memory make this envelope impractical, preserve partial evidence and stop.

## Incident-design audit required if feasibility passes

All candidate edits must plausibly affect the same failed behavior, expose the
complete release diff, and have counterfactual truth defined over the declared
LoRA process. Direct diff, lexical/semantic inspection, output similarity and
rank-then-intervene receive equal information/budget. An obvious label swap cannot
serve as the incident set. Interactions or ambiguity introduced solely to defeat
inspection are rejection criteria. Until such an incident survives this audit,
no diagnosis-performance metric or contribution is claimed.

### Pre-training source-schema correction

The first invocation failed before model loading/training because two provisional
label names were absent from the pinned CSV. The corrected exact classes are
`card_arrival`, `card_delivery_estimate`, `order_physical_card`, `card_linking`.
The failed output directory/log are preserved. This correction used label
metadata, not model outcomes. The CSV SHA-256 is
`b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`.
