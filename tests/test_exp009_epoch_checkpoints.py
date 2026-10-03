from __future__ import annotations

import json

import pytest
from test_exp009_release_training import _partition

from model_forensics.exp009_release import build_clean_release_slots
from model_forensics.exp009_release_training import (
    _prepare_epoch_checkpoint_root,
    train_versioned_classifier_pilot,
)


def test_checkpoint_root_rejects_existing_evidence(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "evidence.json").write_text("preserve")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _prepare_epoch_checkpoint_root(root, epochs=7)
    assert (root / "evidence.json").read_text() == "preserve"


def test_epoch_capture_accepts_zero_warmup_and_keeps_final_producing_lr(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from model_forensics import exp009_release_training as training
    from model_forensics.exp009_classifier import Exp009ClassifierPilotConfig, _tensor_state_sha256
    from model_forensics.exp009_m4 import M4CheckpointRecord, validate_m4_checkpoint_records

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(1, 77)

        def forward(self, input_ids, attention_mask, labels=None):
            del attention_mask
            logits = self.classifier(input_ids.to(torch.float32))
            loss = None if labels is None else torch.nn.functional.cross_entropy(logits, labels)
            return type("Output", (), {"logits": logits, "loss": loss})()

        def save_pretrained(self, root, safe_serialization=True):
            del safe_serialization
            root.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), root / "state.pt")

    class ToyTokenizer:
        def save_pretrained(self, root):
            (root / "tokenizer.json").write_text("{}")

    monkeypatch.setattr(training, "validate_frozen_development_partition", lambda p: "frozen")
    monkeypatch.setattr(training, "verify_pinned_model_artifact", lambda: tmp_path / "weights")
    monkeypatch.setattr(training, "load_pilot_tokenizer", ToyTokenizer)
    monkeypatch.setattr(training, "_build_model", lambda **kwargs: (ToyModel(), "initial"))
    monkeypatch.setattr(training, "select_device", lambda t: "cpu")
    monkeypatch.setattr(training, "_source_git_sha", lambda: "source")
    monkeypatch.setattr(training, "_resolved_commit_hash", lambda m: "revision")
    monkeypatch.setattr(
        training,
        "_encode_fixed",
        lambda tokenizer, texts, **kwargs: {
            "input_ids": torch.ones((len(texts), 1), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 1), dtype=torch.long),
        },
    )
    partition = _partition()
    summary = train_versioned_classifier_pilot(
        partition,
        release_slots=build_clean_release_slots(partition.development_train),
        trajectory_id=0,
        config=Exp009ClassifierPilotConfig(epochs=7, batch_size=32),
        target_labels=("intent_00", "intent_01"),
        run_id="toy",
        output_root=tmp_path / "runs",
        epoch_checkpoint_root=tmp_path / "epochs",
    )
    rows = summary["epoch_checkpoints"]
    validate_m4_checkpoint_records(
        tuple(
            M4CheckpointRecord(
                **{
                    k: row[k]
                    for k in (
                        "epoch",
                        "model_state_sha256",
                        "optimizer_step_count",
                        "producing_learning_rate",
                    )
                }
            )
            for row in rows
        )
    )
    assert [r["optimizer_step_count"] for r in rows] == [3, 6, 9, 12, 15, 18, 21]
    assert summary["optimization"]["final_learning_rate"] == 0.0
    assert rows[-1]["producing_learning_rate"] > 0.0
    for row in rows:
        state = torch.load(
            tmp_path / "epochs" / f"epoch_{row['epoch']:02d}" / "state.pt", weights_only=True
        )
        assert _tensor_state_sha256(state) == row["model_state_sha256"]
    saved = json.loads((tmp_path / "runs" / "toy" / "train_summary.json").read_text())
    assert saved["epoch_checkpoints"] == rows
