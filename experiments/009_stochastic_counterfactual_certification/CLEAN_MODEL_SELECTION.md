# Experiment 009 — Clean Model Selection

Status: **development selection recorded — confirmatory protocol not frozen**.

This record closes the clean-model selection phase for Experiment 009. It is based only on the frozen Banking77 development partition. The official Banking77 test split was not loaded or used.

## Selected clean configuration

- model: `distilbert/distilbert-base-uncased`
- pretrained revision: `12040accade4e8a0f71eabdb258fecc2e7e948be`
- pretrained `model.safetensors` SHA-256: `5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063`
- epochs: `7`
- batch size: `32`
- learning rate: `2e-5`
- weight decay: `0.01`
- warmup ratio: `0.10`
- maximum sequence length: `128`
- maximum gradient norm: `1.0`
- development partition SHA-256: `61ecbdd9224cf2bbeafcf5ceb9164cfc3fc74eb449129136198646385aaa8e88`

The seven-epoch configuration is selected because it improved materially over five epochs while remaining stable across three independent clean training trajectories. No planted regression, root identity, nuisance construction, restoration outcome, or official-test result was available during this choice.

## Clean development evidence

Three independent trajectories were run at both five and seven epochs.

### Five epochs

- mean accuracy: `0.862362`
- accuracy population SD: `0.003539`
- mean macro recall: `0.838890`
- macro-recall population SD: `0.006845`

### Seven epochs

- trajectory 0 accuracy / macro recall: `0.890390 / 0.878891`
- trajectory 1 accuracy / macro recall: `0.898899 / 0.888484`
- trajectory 2 accuracy / macro recall: `0.898398 / 0.885627`
- mean accuracy: `0.895896`
- accuracy population SD: `0.003898`
- mean macro recall: `0.884334`
- macro-recall population SD: `0.004022`

The seven-epoch models also showed strong prediction agreement on the 1,998-example development-evaluation split:

- trajectory 0 vs 1 agreement: `0.935435`
- trajectory 0 vs 2 agreement: `0.939940`
- trajectory 1 vs 2 agreement: `0.941441`
- all-three prediction agreement: `0.913914`
- all-three correct rate: `0.862863`

These measurements support using seven epochs as a strong, stable development substrate rather than continuing an open-ended epoch search.

## Important limits

This record does **not** freeze the final confirmatory protocol. In particular, it does not yet freeze:

- confirmatory target-pair/world selection;
- root corruption dose;
- nuisance construction;
- causal-certification margins;
- statistical procedure;
- number of confirmatory trajectories `K`;
- confirmatory backend/environment;
- confirmatory worlds.

The selected clean configuration should not be changed merely because a later planted regression, diagnostic ranking, or root-vs-nuisance result is unfavorable. A later change would require a documented technical or scientific defect discovered independently of confirmatory outcomes.

## Next development phase

The next phase is deterministic target-pair and pilot-world construction using only `development_train` and `development_eval`. Candidate target pairs should be selected prospectively from clean-development evidence, with sufficient clean per-intent reliability and a rule for semantic/confusion relevance. Weak clean intents must not be chosen simply because they are easy to regress.

`official_test_split_loaded=NO`

`regression_constructed=NO`

`confirmatory_result=NO`
