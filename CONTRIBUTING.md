# Contributing

## Software changes

Use Python 3.12 and run `uv sync --extra dev`, `uv run ruff check .`,
`uv run ruff format --check .` and `uv run pytest`. Optional numerical modules
are also checked by research CI. Explain the user-visible problem and include
a regression test for changed behavior. Keep the saved-prediction path free of
model-training dependencies.

See the [architecture](docs/architecture.md) before changing shared modules.
Preserve frozen protocols and result records; present a new experiment as a new
design rather than rewriting old evidence. Public source visibility does not
grant a distribution license.

## Reproducible regression cases

Public, licensed evidence is welcome through the issue template. Please provide:

- immutable known-good and regressed release identities;
- a measurable behavioral regression and its evaluation definition;
- the complete engineer-visible candidate changes and their categories;
- environment/dependency identities and reproduction commands;
- whether changes can be reverted or intervened on, and approximate compute cost;
- training randomness/repeats and any protected behavior;
- source/data licenses and permission to publish reproduction artifacts;
- whether a fix is known, with fix provenance clearly separate from diagnostic evidence.

Do not submit private, proprietary, patient, credential-bearing or unlicensed data.
A crash alone or a single obvious diff may be useful software evidence but may not
qualify for the proposed research question. Submission does not promise
collaboration, publication or acceptance into a benchmark. A submitted case must satisfy the prospective study design before it can
support a new comparative claim.
