"""Data isolation checks; no training or external data needed."""

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "incremental_example",
    Path(__file__).resolve().parents[1] / "examples/incremental_intent_investigation.py",
)
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


def test_normalized_duplicates_and_conflicts_never_cross_split():
    rows = [[f"intent {label} example {i}", label] for label in ("a", "b") for i in range(10)]
    rows += [["  INTENT a example 0  ", "a"], ["ambiguous", "a"], [" AMBIGUOUS ", "b"]]
    train, evaluation, counts = example.partition(rows)
    assert len(train) == 16
    assert len(evaluation) == 4
    assert not {r["id"] for r in train} & {r["id"] for r in evaluation}
    assert all(r["text"].strip().casefold() != "ambiguous" for r in train + evaluation)
    assert counts["conflicting_texts_excluded"] == 1
    assert counts["duplicate_rows_collapsed"] == 2
    assert example.partition(list(reversed(rows)))[2] == counts


def test_class_partition_is_input_order_independent():
    labels = [f"intent-{i}" for i in range(150)]
    for seed in (0, 1, 2):
        order = example.class_order(labels, seed)
        assert order == example.class_order(list(reversed(labels)), seed)
        assert len(set(order[:30])) == 30
        assert not set(order[:30]) & set(order[30:])
