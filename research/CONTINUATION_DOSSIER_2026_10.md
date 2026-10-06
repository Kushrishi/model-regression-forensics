# Continuation dossier — October 2026

**Review date:** 2026-10-06. **Status:** author-side decision support, NOT independent approval.
**Recommendation:** NO-GO for result-bearing continuation now; conditional case for independent review of a distinct decision problem.

This document is a literature/contribution assessment. It is not a new frozen experiment, M5 protocol, certification rule, or authorization to train. The [continuation gate](M4_CONTINUATION_GATE.md) remains active. No new model training, intervention run, or official Banking77 test access occurred in this review.

## 1. Evidence at the boundary

The accepted M4 record contains two constructed Banking77 worlds with three paired trajectories per world. Target-label overlap and lexical Jaccard each rank the root 1/1 across worlds; final-checkpoint Grad-Dot and checkpoint TracIn rank it 1/5. Those are two benchmark worlds, not six independent incidents. Disjoint candidate/target labels make the target-root association unusually visible to simple diagnostics. This closes the current localization comparison, not training-data attribution generally.

The source-pinned technical report, blind aggregation/replay, candidate construction and failed nuisance/capacity rules remain the record. The ambiguous-repair example shows that changing the input interpretation or changing classifier weights can both restore a prediction. That separates recovery from unique historical explanation; it is not evidence that a general ambiguity detector works.

MRF v1 contributes a reproducible, bounded release-debugging case study, competitive simple baselines and an explicit negative result. It does not establish a new causal-identification method, modern instruction-model benchmark, or superiority over attribution literature.

## 2. Closest-work matrix

Sources are primary papers/author artifacts. The newest September/June 2026 items below were checked at abstract/landing-page level, not exhaustively reproduced. Missing methodological details are deliberately marked rather than inferred. This matrix is not an exhaustive novelty search.

| Work / source | Problem, unit, available evidence | Counterfactual / intervention / method / retraining | Output, abstention, evaluation scale, limit and overlap |
| --- | --- | --- | --- |
| [Influence Functions](https://proceedings.mlr.press/v70/koh17a.html), Koh & Liang 2017 | Effect of a training example on test loss; trained model, losses, gradients/Hessian | Approximate infinitesimal upweighting via inverse-Hessian products; avoids a full refit per example | Example influence/ranking; not release-level ambiguity certification. Neural/classical demonstrations; local approximation assumptions constrain use. MRF cannot claim influence attribution as new. |
| [TracIn](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html), 2020 | Training examples contributing to target loss; saved checkpoints and gradients | Gradient inner products over checkpoints approximate training-path contribution; checkpoint availability required, not exact leave-one-out retraining | Scores/rankings; no general identified/ambiguous guarantee. Image-model experiments. It is a baseline, not an MRF method invention. |
| [Datamodels](https://arxiv.org/abs/2202.00622), 2022 | Dataset-subset effects on model outputs; many subset-trained models | Fit a surrogate mapping inclusion indicators to target behavior; substantial retraining amortized across queries | Predicts behavior under data changes; no inherent release-cause abstention. Deep image classification; costly subset training. MRF must distinguish a practical release decision from another surrogate attribution exercise. |
| [TRAK](https://arxiv.org/abs/2303.14186), 2023 | Scalable example attribution to model predictions; models, projected gradients, training data | After-kernel approximation, ensembles of trained models; less costly than fitting datamodels but still needs specified model evidence | Example attribution and counterfactual predictive evaluation across model types/scales; not arbitrary pipeline-cause identification. A future comparison must supply comparable evidence/budget. |
| [Datascope](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ccdd1961dcd0245d9dadceccead0fad1-Abstract-Conference.html), ICLR 2024 | Debug data through preparation/feature pipelines, rather than isolated predictors | Shapley importance over relational provenance; efficient cases exploiting nearest-neighbor structure rather than generic exhaustive retraining | Data importance/repair prioritization, not a release-level uniqueness certificate. Pipeline experiments; computational assumptions matter. Pipeline awareness alone is already established. |
| [Distributional Training Data Attribution](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e8909cae8248c98279f6cd82074aa6d-Abstract-Conference.html), 2025 | Effects of data on distributions of model outputs across stochastic training | Distributional rather than single-realization influence; training randomness is part of the estimand | Distributional attribution; exact abstention and task/model details need full-paper verification before design adoption. MRF cannot claim novelty for accounting for seeds. |
| [Final-model-only attribution](https://proceedings.neurips.cc/paper_files/paper/2025/hash/99d7326032bbed26de1b244beaff6a84-Abstract-Conference.html), Wei et al. 2025 | Attribution with only final-model access, without checkpoint history | Access boundary differs from TracIn; verify exact assumptions before implementing | Bibliographic entry carried from the existing related-work record; full page could not be retrieved in this pass. This review does not assert full method/scale details. Access constraints must be explicit in any MRF comparison. |
| [DeMix](https://arxiv.org/abs/2606.11616), KDD 2026 / preprint | Locate erroneous examples AND error types: labels, features, spurious correlation; influence vectors over validation samples | Intervention-invariance perspective and classification of influence patterns; exact retraining cost requires full paper/code audit | Joint sample/type diagnosis; reported tabular, recommendation, LLM-alignment tasks. Abstention/uniqueness not established by abstract. Mixed-error diagnosis is not an unoccupied contribution space. |
| [Which Influence Are We Estimating?](https://arxiv.org/abs/2609.31214), September 2026 preprint | Disagreement between influence rankings; behavior, intervention and training-process specification | Distinguishes estimand mismatch from approximation error; controlled specified counterfactuals | Ranking/specification analysis, not automatic historical responsibility. Abstract-level review only. MRF must specify its estimand and cannot count specification itself as novelty. |
| [DataInf](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e84a0f233611a1dc8fb794dc52415a3-Abstract-Conference.html), ICLR 2024 | Efficient influence for LoRA-tuned LLMs/diffusion models; fine-tuning gradients | Inverse-Hessian approximation designed for parameter-efficient settings; no exact refit per query | Influence scores with modern-model applications; not release-level causal uniqueness. A LoRA substrate does not itself make MRF novel. |
| [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a), 2019; [selective-classification evaluation](https://arxiv.org/abs/2407.01032), 2024 | Prediction with reject option; coverage versus errors among committed predictions | Joint selective predictor or evaluate acceptance rules; this is not a data-change intervention | Risk/coverage and undetected-failure evaluation; does not validate cause labels or historical identifiability. MRF needs an incident-level diagnostic target, not just a renamed confidence threshold. |

The final-model-only entry needs full-text revalidation before publication; no conclusion here depends on its unverified details. [OpenAI frontier evaluations](https://openai.com/careers/research-engineer-frontier-evals-and-environments-san-francisco/) and [Anthropic post-training evaluations](https://job-boards.greenhouse.io/anthropic/jobs/5198255008-62) reinforce a practical need for reliable measurements and regression investigation; hiring descriptions are motivation, not scientific novelty evidence.

## 3. Distinct question that may survive

**When does available evidence justify committing to one release-level explanation, versus keeping several candidates or gathering more evidence?**

An engineer has baseline/regressed releases, a behavior regression, a bounded candidate set and a limited intervention budget. The practical decision is which candidate(s) to investigate or revert next, or whether evidence is insufficient. Outputs could be IDENTIFIED, AMBIGUOUS and INSUFFICIENT EVIDENCE. These are proposed decision categories, not an implemented method.

Historical cause, restorative effect and model sensitivity are different targets. If two edits both repair behavior, a successful repair cannot by itself uniquely identify which edit historically caused the incident. Even uniqueness among a declared candidate set does not exclude omitted causes. A dossier must allow set-valued explanations, interaction causes and an incomplete-candidate condition.

The potential contribution is decision improvement at a stated incorrect-specific-diagnosis risk and evidence budget. It is NOT ranking, checkpoint gradients, stochastic attribution, counterfactual specification, disagreement detection, or adding a threshold to existing scores.

## 4. Conceptual benchmark requirements — NOT a protocol

Independent review must establish realistic incidents before any prospective design is executed. Relevant classes include overlapping semantic data edits, composition changes, pipeline/preprocessing edits, label/reference edits, duplication/removal, interacting edits, multiple restorative interventions, omitted candidates and stochastic retraining effects. These cannot be retrofitted after seeing outcomes.

Candidate information must not encode the answer through filenames, edit size, target labels or semantic disjointness. The engineer-facing release diff remains legitimate evidence; artificial hiding is not a solution to a trivial benchmark. A benchmark is unsuitable if direct inspection already solves every incident. Conversely, manufactured ambiguity with no plausible engineering decision is also unsuitable.

The inferential unit would be an incident, not its prompts, seeds or candidates. Splits must separate incident construction/model development from final evaluation. Seed coupling, candidate ordering, truth seals, training artifacts and intervention availability need prospectively fixed provenance. No official Banking77 test outcomes are needed for this conceptual assessment.

## 5. Equal-evidence baseline requirements

At minimum compare random prioritization, release metadata/direct diff, target-label overlap where applicable, lexical/semantic relevance, output-behavior similarity and **simple prioritization plus the same paired interventions**. Grad-Dot/TracIn are relevant only with their required access. TRAK/DataInf are feasibility-dependent, not mandatory badges.

Controlled removal/restoration and exact retraining are evidence sources, not privileged information reserved for the proposed method. Each comparison must disclose which outputs/checkpoints/interventions it receives and how much compute it spends. If a simple workflow using equivalent evidence makes equally safe decisions, the proposed research benefit fails.

## 6. Metric assessment

| Candidate metric | Decision relevance | Failure to prevent |
| --- | --- | --- |
| Incorrect unique-attribution rate across all incidents | Penalizes unsafe specificity | Distinguish this from conditional risk; abstaining everywhere trivially lowers it. |
| Coverage and selective accuracy/risk among committed incidents | Measures usefulness versus safety | Report both jointly; no high accuracy headline at negligible coverage. |
| Incident-level risk–coverage curve | Compares commitment policies | Ranking ties, calibration-set selection and coverage grid must be fixed later, not fitted to final truth. |
| Set-valued containment and set size | Preserves plausible alternatives | Returning all candidates always contains truth but has no diagnostic value. Omitted/interaction causes require an explicit target representation. |
| Intervention count/cost until safe useful decision | Measures evidence efficiency | Include failed/ambiguous incidents, wall time and intervention parity, not only successful cases. |
| Wrong unique decisions at a matched useful coverage/budget | Direct benefit over simple workflow | No safety guarantee follows from a small descriptive benchmark alone. |

These are options for independent assessment. No numerical acceptance threshold, calibration rule or experiment size is approved here.

## 7. Modern-model substrate comparison

| Candidate | Access / license / reproducibility | Fit and risks |
| --- | --- | --- |
| Existing DistilBERT Banking77 classifier | Existing training code/artifacts; direct gradients, paired refits | Cheapest continuity/control; cannot defeat the current benchmark shortcut merely by retraining it. Not modern instruction-model evidence. |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | Apache-2.0 model card; compact open weights, standard Transformers; pin revision/tokenizer/chat template | Preferred **candidate for later feasibility**, structured intent/slot responses with exact machine-checkable output. Small capacity can confound a regression with task floor effects. Must compare pre/post target and unaffected behavior. No artifact downloaded or experiment begun. |
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | Apache-2.0 model card and accessible weights; immutable revision must be captured before use | Higher-capacity second option, higher repeated-run/storage cost. Do not choose it for prestige. No exact artifact or tuning recipe selected. |

Recommend independent review of a small instruction model plus structured-response behavior, NOT free-form subjective answer grading as the initial causal substrate. The choice is provisional until task usefulness, data terms, baseline solvability and intervention semantics survive review. API-only models are unsuitable for exact training counterfactuals.

## 8. Compute and storage envelope

No compute has been authorized or consumed for new MRF training. A useful planning calculation is R = S × (2 + K), with S paired seeds and K intervention trainings per incident. For illustration only, S=3 and K=5 gives 21 fits/incident; four incidents would require 84 fits. That is NOT a proposed experiment size.

At an assumed 10/30/60 minutes per fit, 84 fits imply 14/42/84 accelerator-hours before attribution or repeats. Runtime is unmeasured; a later authorized feasibility pilot must replace these assumptions. For 135M parameters, raw fp32 weights alone are about 0.54 GB per saved final model, or ~45 GB across 84; optimizer/checkpoint history can multiply storage several-fold. LoRA can lower trainable-state storage but changes the estimand and cannot be presented as full-model retraining. Compute must be justified by the incident decision benefit, not by GPU availability.

## 9. Kill criteria and decision

Close the research continuation if the benefit is only a known attribution idea, a confidence threshold, an answer encoded in the diff, or a simple equal-evidence intervention workflow matches it. Also stop if realistic ambiguity cannot be obtained without benchmark tricks or repeated trainings cost more than the question warrants. Do not broaden the claim to rescue a weak result.

Reasons to review: the repair/explanation distinction matters; incomplete evidence and overlapping plausible changes are realistic; provenance and the existing comparator provide a useful engineering substrate. Reasons not to continue: closest work already covers most ingredients, M4 supplies no difficult ambiguity benchmark, and a distinct safe decision rule is not established.

**Next decision:** independent scientific contribution assessment of the bounded decision problem and closest work, with a genuine option to conclude MRF as a completed negative case study. **NO-GO for M5 training/protocol execution now.** The independent review must not be relabeled from this author-side document. Existing STATE/CLAIMS/ROADMAP remain valid; their gate is not loosened.
