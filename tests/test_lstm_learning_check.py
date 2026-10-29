import importlib
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_learning_check_uses_100_balanced_complete_distinct_training_examples(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    check = importlib.import_module("check_lstm_learning")
    trainer = importlib.import_module("train_lstm")
    examples = [
        trainer.Example(
            f"{split}_{label}_{index}.cpp", f"f{index}", [split, str(label), str(index)], label, split, split
        )
        for split in ("train", "validation")
        for label in (0, 1)
        for index in range(60)
    ]
    # Opposite labels for identical inputs, duplicates and overlong inputs are excluded.
    examples.extend(
        [
            trainer.Example("conflict0.cpp", "f", ["conflict"], 0, "train", "a"),
            trainer.Example("conflict1.cpp", "f", ["conflict"], 1, "train", "a"),
            trainer.Example("long.cpp", "f", ["long"] * 20, 1, "train", "a"),
            examples[0],
        ]
    )
    selected = check.select_examples(examples, seed=42, max_length=3)
    assert Counter((e.split, e.label) for e in selected) == {
        ("train", 0): 50,
        ("train", 1): 50,
        ("validation", 0): 10,
        ("validation", 1): 10,
    }
    assert len({tuple(e.tokens) for e in selected}) == 120
    assert all(len(e.tokens) <= 3 and e.tokens != ["conflict"] for e in selected)
    assert selected == check.select_examples(examples, seed=42, max_length=3)
