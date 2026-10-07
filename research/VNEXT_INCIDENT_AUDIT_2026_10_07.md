# External incident screen — 7 October 2026

Discovery only; no new training/retraining, model acquisition, official test access
or diagnosis benchmark execution. **Zero qualified incidents.** Two source leads
remain useful for future provenance inspection, not an approved experiment.

## Method and source identities

Use the owner's A–L screen: reproducible good/bad versions; behavioral regression;
predefinable metric; genuine plausible candidate changes; complete equally visible
diff; no answer encoding; executable interventions; measurable stochasticity;
independent-project cost; adequate license; separate fix provenance. A collection
calling a bug “real” or “reproducible” does not establish these requirements.

Read the full **770-record metadata index**, then screened six selected behavior/
accuracy/training-related leads in depth, plus two pinned DeepFD source pairs,
one DEFault pair and three source/task collections. Selection is discovery triage,
not an incident sample or prevalence estimate. Most index records were not given
an individual qualification decision. Original issue bodies were fetched for four
leads; metadata comments are supporting reported evidence, not new experiments.

[AIFaultBench](https://huggingface.co/datasets/mehilshah/AIFaultBench) revision
`62e11d6dfe4b1637f38a45d8fbc1c2b9c3c971ff`: complete `index.json` is 2,666,853
bytes, SHA-256 `9c1c86018d9098f1a22b27bb232d092d6e05fa0f91a7aa6a27080d3849dba95a`.
README SHA-256 `4800206c302c3bcded84144114c524e4d84aa45b213795d86b00b856295a744d`.
Its card licenses packaging/metadata CC BY 4.0, with original sources retaining
upstream licenses. No bulk code/data archive was downloaded or setup script run.
An initial 2-MB bounded index read was incomplete; only the subsequently complete,
JSON-validated index was screened. The hosted dataset viewer has a card YAML error;
reading the pinned index bypassed it without modifying the upstream dataset.

Pinned GitHub source trees / README blobs inspected:

| Source | Revision | README blob | Declared repository license |
| --- | --- | --- | --- |
| ArabelaTso/DeepFD | `9fe089510cf10af35b0c2cba024d9732a1048df4` | `1b6c6deb225b9c472a5dda1517bcd5d736741cdb` | Apache-2.0 |
| SigmaJahan/DEFault-DNN-Debugging | `9a1f12d8b7a355f07ca99cd34fd6322e0b8130c3` | `fa7084071020e457ea8f99b6992aae24d3bb6d76` | MIT |
| dlfaults/dl_faults | `7996f9dcef5947e87fc3924ca6f617ba5c9ecdbb` | `8fca7b17075687449217e211bdc63705a7cc408f` | CC BY 4.0 |
| stefan-grafberger/mlwhatif | `90bd5003c1e1ef0a51545455383d89e7e26a6d01` | `64353d31ada181682ae43e380d245f14bf2629dc` | Apache-2.0 |

License entries concern repository packaging, not automatic permission for every
underlying dataset or original Stack Overflow post. Upstream attribution and
licenses require incident-specific checks before reproduction/public payloads.

## Candidate dispositions

“Bad pin” below is the collection's reconstruction identity, not independent
verification that the original release was regressed. Reported observations are
not accepted experimental results. Unknown costs are deliberately not invented.

| Candidate / category | Good / bad / fix provenance and observed evidence | Visible changes, leakage and stochasticity | Cost / license / disposition |
| --- | --- | --- | --- |
| AIFault 001, [tensorflow/models #166](https://github.com/tensorflow/models/issues/166); training numerical behavior | Bad pin `05630a7578b25390f469b2f91f2c2326e5ed539b`; reported MNIST spatial-transform training NaNs. No good pin or fix in index. | No verified release diff or complete candidate set. Training randomness unbounded; NaN symptom alone does not specify comparative behavioral responsibility. | Model/data/environment reproduction cost unknown; Apache-2.0 upstream code, data terms separate. **Reject current incident**: no good/bad pair or defined intervention oracle. |
| AIFault 080, [vit-pytorch #311](https://github.com/lucidrains/vit-pytorch/issues/311); augmentation | Bad pin `90be7233a3f55c29692a72da6ee4dcb5aab267d4`; validation accuracy exceeds training accuracy. Comment attributes this to random train augmentations and notes reduced performance without them. No good/fix pin. | This may be expected train/eval behavior rather than a release regression. No complete change set; explanation supplied directly. Training stochasticity exists but is not quantified. | MIT upstream; original image dataset terms/cost unknown. **Reject**: no demonstrated known-good/regressed releases, counterfactual target or diagnosis gap. |
| AIFault 084, [vit-pytorch #257](https://github.com/lucidrains/vit-pytorch/issues/257); labels / preprocessing | Bad pin `5699ed7d139062020d1394f0e85a07f706c87c09`; reported 100% first-epoch train/validation accuracy. Comment describes path-derived labels collapsing due to wrong string comparison. No good/fix commit. | Direct data-label inspection can reveal collapse; do not hide it. Root comment cannot be debugger input when evaluating an unassisted diagnosis, but complete code/diff remains legitimate. No release candidate set established. | MIT code, image terms unresolved; tiny label parser check could be cheap, full fit not justified. **Reject** as release benchmark: good pair/multiple changes missing and simple evidence appears decisive. |
| AIFault 435, [diffusers #13930](https://github.com/huggingface/diffusers/issues/13930); inference token layout | Bad pin `41add3410424cc33d748a7fd3409132d2f6b4ad2`; issue identifies introduction PR13564 (`ebaa1871`) after v0.38.0. Reports reversed token/register layout and connector correlation changes. | Issue supplies exact offending flip and reference toy layout. Cheap list arithmetic below confirms that stated layout discrepancy only. Not a training-release intervention or stochastic retraining incident. | Apache-2.0 diffusers; LTX checkpoint terms not inspected. No weights acquired. **Reject research incident; retain cheap software example** of direct inspection settling a mismatch. |
| AIFault 562, [timm #2463](https://github.com/huggingface/pytorch-image-models/issues/2463); inference preprocessing/export | Bad pin `e44f14d7d2f557b9f3add82ee4f1ed2beefbb30d`; issue reports PyTorch 81.374% vs ONNX 74.190%, comment points to normalization/crop config lost on export. Good PyTorch/ONNX artifact hashes absent. | Multiple workflow differences possible, but public suggestion is direct preprocessing inspection. Inference conversion, not versioned training. No retraining responsibility process. | Apache-2.0 code; ImageNet access/license and full validation cost unresolved. **Reject current training incident**; no corpus downloaded. |
| AIFault 608, [transformers #46710](https://github.com/huggingface/transformers/issues/46710); tokenizer/inference dependency | Version4.55 vs5.9.0, bad pin `b4b5244c9c7cdb80d0aaafdb8f35244612788532`; garbled decoded response. Comments identify tokenizer fallback and link PR46091. | Complete library diff potentially large, but visible tokenizer behavior/simple decode check is strong. Sampled generation is stochastic; no training release or quantified metric across prompts. | Apache-2.0 library; 8B fp16 weights alone ~16GB exceed this 8GiB host; model/data permissions separate. **Reject**: wrong scope, no training counterfactual; no model download. |
| DeepFD subject31556268; loss/configuration/sampling | At pinned source, `origin.py` blob `0e91cde221ecf16a394ca223184746275d733c99`, `repair.py` blob `4b67f921f8fb930ec2bee31e4b329c219349461a`. Four XOR points, original random train/test split; repair changes MAE→MSE, SGD→Adam,1000→50000 epochs **and removes split**. | Multiple genuine proposed repairs, but collection does not establish a prior good release, complete historical change set or stochastic responsibility distribution. Directory names encode loss/epoch/lr. Removing names alone would not fix the absent release provenance. | Apache-2.0 packaging, original post attribution needed; tiny model but50000 epochs, old Keras API; cost unmeasured. **Retain provenance lead only**, not qualified or approved for fitting. |
| DeepFD subject59282996; configuration | `origin.py` blob `63955f1fa2984ae32aac2d1642d16759a2b53483`; repair blob `b322c2c38f089a57af804fd31ac3907de7bf440b`; additional repair variants exist. Epoch-labelled case, old Keras source. | Existence of repairs is not measured equal restoration. Historical good version, complete release diff and independently fixed metric remain unverified. Fault-labelled paths leak target. | Apache-2.0 packaging, dataset terms/cost unknown. **Retain source lead only** pending provenance; no training. Source-pair existence, not a qualification claim. |
| DEFault PixelCNN; loss/configuration/architecture | Correct blob `92bc3cedcd67f8009402b64cd8538256162293f9`; buggy blob `5d2e85dc63c2b6ec027367fa2395027f0704d46b`. Defaults75→10 epochs,64→32 hidden width,6→3 blocks, bits-per-dim→cross-entropy. | Named variants/comments explicitly label faults. These are supplied fault variants, not verified historical good/regressed releases with alternative plausible changes. Training and random sampling unmeasured; exceptions return0, so crash vs behavioral failure needs separation. | MIT code, TFDS data terms separate;75-epoch convolutional fit/sweep cost unmeasured and material. **Reject as current grounded release benchmark**, keep overlap evidence for existing diagnosis systems. |
| [DataPerf vision cleaning](https://www.dataperf.org/training-set-cleaning-vision); labels/data | Externally defined training-data cleaning task, not an identified version pair in this screen. | Useful data-debugging benchmark; no complete release-change candidates or release-responsibility oracle established. Do not fabricate a release around method-favorable corruptions. | Dataset-specific terms and intervention costs require checks. **Reject current release qualification**, not the value of DataPerf. |
| dlfaults taxonomy collection; mixed training pipeline | Pinned taxonomy annotates public GitHub/Stack Overflow artifacts, not an executable good/bad release for every row. | Taxonomy labels are truth/provenance, not equally visible diagnostic evidence. No interview/participant files accessed. | CC BY4 packaging; each upstream separately licensed. **Source pool only**, no incident accepted. |
| [DLFaultBench Zenodo record21422606](https://zenodo.org/records/21422606) | Search index identifies a collection, but live fetch returned410 Gone. No immutable per-case source pair or license verified. | Cannot establish A–L from a collection title or stale search snippet. AIFaultBench card points instead to record21782307; equivalence not assumed. | No archive downloaded. **Unverified source; not retained as qualified**. |

## Cheap non-training check

The diffusers issue's eight-slot arithmetic was independently checked with Python
lists: reference `[1,2,3,3,0,1,2,3]`, masked-write/reverse
`[3,2,1,0,3,2,1,0]`. They differ, confirming the **stated toy layout** without
Torch, weights or generation. This does not reproduce connector correlations,
video quality, upstream model behavior or stochastic retraining. Direct inspection
already identifies the toy discrepancy; no new MRF diagnosis contribution follows.

## Gate decision and cost

0 qualified, 2 DeepFD provenance leads, other rows rejected/unverified/source-only.
No incident meets the complete approval gate; the two leads are not counted as
survivors. No defensible three-incident benchmark or nontrivial complete-diff
training case exists yet. Source inspection consumed no training/accelerator hours.

No prospective result-bearing protocol or fit budget is proposed. A future cost
estimate must measure/establish per-fit feasibility and give seeds × interventions
× incidents, including failed attempts and oracle sweeps. Current unknown costs
cannot justify owner compute approval. Continue only with new public provenance
or a passive submission; do not manufacture distractors or escalate models.
