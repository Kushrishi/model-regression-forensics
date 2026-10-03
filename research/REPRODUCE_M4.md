# Reproduce the accepted M4 aggregation

The twelve retained files in [M4_RESULT_ARTIFACTS](M4_RESULT_ARTIFACTS/) allow replay of blind ranking aggregation and subsequent truth scoring without model training, a GPU, or dataset/model downloads. Python package installation may require network access.

This reproduces processing of already-computed scoring records. It does not regenerate checkpoints or validate the scoring methods by rerunning training. The original execution source is required because the scripts enforce source and authorization identities.

## Verify and prepare retained evidence

Start from the root of a full clone containing the accepted result files. Use Python 3.12 or later. The following creates a new sibling directory and refuses to overwrite an existing one:

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
import json
import shutil

record = json.loads(Path("research/M4_RESULT.json").read_text())
for name, expected in record["artifacts"].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Artifact digest mismatch: {name}")

retained = Path("research/M4_RESULT_ARTIFACTS")
data = Path("../mrf-m4-replay-data")
data.mkdir()
(data / "scores").mkdir()
(data / "truth").mkdir()
(data / "expected").mkdir()
for world in ("world_00", "world_01"):
    for trajectory in range(3):
        name = f"{world}_t{trajectory}.json"
        shutil.copyfile(retained / name, data / "scores" / name)
    shutil.copyfile(
        retained / f"truth_{world}.json",
        data / "truth" / f"{world}.json",
    )
for name in ("blind_aggregate.json", "truth_evaluation.json"):
    shutil.copyfile(retained / name, data / "expected" / name)
print("All twelve retained artifact digests match the accepted record.")
PY
```

## Run the original source

Create an isolated worktree at the accepted execution commit. This leaves the current checkout unchanged. If that commit is absent in a shallow clone, retrieve the recorded commit before continuing.

```bash
git worktree add --detach ../mrf-m4-replay a5ec718e6081624893fed406eedb1dd5e1405892
cd ../mrf-m4-replay
uv sync --extra dev
uv run python scripts/analyze_exp009_m4_localization.py \
  --input-root ../mrf-m4-replay-data/scores \
  --output ../mrf-m4-replay-data/output/blind_aggregate.json
uv run python scripts/score_exp009_m4_truth.py \
  --blind-aggregate ../mrf-m4-replay-data/output/blind_aggregate.json \
  --blind-sha256-file ../mrf-m4-replay-data/output/blind_aggregate.json.sha256 \
  --truth-root ../mrf-m4-replay-data/truth \
  --output ../mrf-m4-replay-data/output/truth_evaluation.json
```

Aggregation reads only the six score records. Truth scoring runs afterward against its finalized digest. The scripts reject an incorrect source, incomplete record sets, identity drift, and existing output files.

## Compare exact bytes

While still in the replay worktree:

```bash
python3 - <<'PY'
from pathlib import Path

data = Path("../mrf-m4-replay-data")
for name in ("blind_aggregate.json", "truth_evaluation.json"):
    if (data / "output" / name).read_bytes() != (data / "expected" / name).read_bytes():
        raise ValueError(f"Replay differs from accepted evidence: {name}")
    print(f"Byte-identical replay: {name}")
PY
```

Both files must match. The retained records reproduce random root ranks 3/3, label-overlap and lexical ranks 1/1, and Grad-Dot and TracIn ranks 1/5 across the two worlds.

This procedure was verified on October 3, 2026 using the accepted source and retained records. It acquires no official-test data and authorizes no new research execution.
