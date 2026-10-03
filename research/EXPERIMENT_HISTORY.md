# Experiment history

This guide explains how Model Regression Forensics reached its completed matched localization study. Each linked result preserves the original protocol and outcome. Development failures are part of the record; later experiments do not overwrite them.

## Synthetic shape experiments

The initial substrate used SmolLM2-360M-Instruct with LoRA fine-tuning on a controlled shape-decision task. These experiments developed the debugging workflow and exposed problems with task capability, shortcut baselines, and restoration specificity.

| Experiment | Question and outcome |
| --- | --- |
| [000](../experiments/000_planted_regression/RESULTS.md) | Could the pipeline induce and repair a regression? Yes, with one visible change. Diagnosis was deliberately trivial. |
| [001](../experiments/001_blinded_multicandidate/RESULTS.md) | Could diagnosis operate without benchmark truth across five candidates? Yes, but lexical overlap exposed the root. |
| [002](../experiments/002_entangled_distractors/RESULTS.md) | Did removing whole-artifact lexical differences make diagnosis difficult? No: changed-record lexical overlap still localized the root. Restoration also affected other slices. |
| [003](../experiments/003_role_binding_confounders/RESULTS.md) | Could a construction neutralize known shortcuts while retaining a learnable clean task? Construction passed, but the clean model failed; candidate and intervention training stopped. |
| [003-B](../experiments/003b_balanced_loss/RESULTS.md) | Would balancing response loss rescue the clean task? No. |
| [003-C](../experiments/003c_selected_slot_lookup/RESULTS.md) | Could the model perform selected-slot lookup alone? Yes. This isolated a capability without establishing a root-cause result. |
| [003-D](../experiments/003d_explicit_policy_role_binding/RESULTS.md) | Would explicitly providing the decision policy rescue role binding? Yes. The result narrowed the capability problem without proving an internal mechanism. |
| [004](../experiments/004_explicit_policy_entangled_rca/RESULTS.md) | Could the learnable task support localization and causal verification? Localization succeeded; verification failed. The linked exploratory postmortems remain separate evidence. |
| [005](../experiments/005_causally_certified_rca/RESULTS.md) | Would balanced label counts produce the required localized regression? None of five worlds qualified. No world was certified. |
| [006](../experiments/006_semantic_balanced_causal_rca/RESULTS.md) | Would semantic-balanced corruption solve that construction problem? None of five worlds qualified. |
| [007](../experiments/007_sensitivity_calibrated_causal_rca/RESULTS.md) | Could a development dose create a material regression while preserving protected behavior? Regressions were material but insufficiently local; certification was not opened. |
| [008](../experiments/008_selective_causal_rca/RESULTS.md) | Could selective corruption yield localized regressions and unique restoration? Localization and root repair succeeded in both worlds, but nuisance restorations also changed target behavior. Unique certification failed. |

Experiment 008 was the final major iteration on the shape substrate. The project retired it rather than continuing to tune doses, thresholds, or world selection.

## Banking77 development

Experiment 009 moved to a pinned DistilBERT classifier on Banking77 and repeated paired training trajectories. Its official test split remained untouched.

[Stage A](../experiments/009_stochastic_counterfactual_certification/STAGE_A_RESULT.md) passed the frozen localized-regression gate across three trajectories, with mean target regression 0.129289 and mean protected regression 0.002711. The truth-isolated Grad-Dot baseline ranked the planted root first in each trajectory.

[Stage B](../experiments/009_stochastic_counterfactual_certification/STAGE_B_RESULT.md) completed eighteen trainings. Root restoration exceeded every nuisance restoration in all three trajectories; mean root recovery was +0.124387. This was a development pilot without confirmatory inference.

The nuisance changes were structurally distinguishable from the root. Those results motivated a matched benchmark; they are not pooled with its outcomes.

## Completed matched study

M3 prospectively required identical 66-slot symmetric label swaps, unchanged text, preserved aggregate label counts, opaque IDs, and disjoint candidate slots and labels. The original three-world construction exceeded the clean-only eligibility graph's capacity. A [pre-training amendment](../experiments/009_stochastic_counterfactual_certification/STRUCTURALLY_MATCHED_BENCHMARK_PROTOCOL_AMENDMENT_1.md) reduced it to two worlds without weakening eligibility.

M4 then trained three clean and six composite models across those worlds and three paired trajectories. Blind scoring preceded truth scoring. Simple label-overlap and lexical baselines ranked the root first in both worlds; Grad-Dot and TracIn ranked it first in one and last in the other.

Matching change structure did not remove the semantic shortcut: known target labels identify a disjoint candidate pair. M4 supports a bounded comparison in this construction, not difficult general localization or causal certification. The [technical report](M4_TECHNICAL_REPORT.md) presents the complete results.

## Disposition

The matched study is accepted and retained. Further certification protocol development and execution remain held under the [continuation gate](M4_CONTINUATION_GATE.md). A future study needs independent scientific justification; completing this history or replaying artifacts does not clear that gate.
