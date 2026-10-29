import argparse
import importlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="Install the lstm extra to test training.")


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("train_lstm")


@pytest.fixture
def training_data():
    records = []
    labels = []
    for split in ("train", "validation", "test"):
        for label, token in ((0, "SAFE_PATTERN"), (1, "UNSAFE_PATTERN")):
            function = f"function_{label}"
            records.append(
                {
                    "source": f"/input/code/{split}.c",
                    "function": function,
                    "representation": "joern-ast-preorder-vars-v2",
                    "tokens": ["METHOD", token, token, split, "END"],
                }
            )
            labels.append(
                {"source": f"{split}.c", "function": function, "label": label, "split": split, "group": split}
            )
    return records, labels


def test_training_learns_and_saves_an_inference_compatible_checkpoint(trainer, training_data, tmp_path):
    from salomon.adapters.lstm.checkpoint import load_checkpoint
    from salomon.adapters.lstm.worker import predict_records

    records, labels = training_data
    examples = trainer.labeled_examples(records, labels)
    args = argparse.Namespace(
        output=tmp_path / "trained.pt",
        seed=42,
        threads=1,
        embedding_dim=8,
        hidden_dim=8,
        learning_rate=0.03,
        epochs=30,
        batch_size=2,
        max_length=32,
    )
    metrics = trainer.train(examples, args)
    _, checkpoint = load_checkpoint(args.output)
    assert checkpoint["trained"] is True
    assert "validation" not in checkpoint["vocab"]
    assert "test" not in checkpoint["vocab"]
    assert metrics["history"][-1]["train_loss"] < metrics["history"][0]["train_loss"]
    assert metrics["test"]["samples"] == 2
    assert args.output.with_suffix(".metrics.json").is_file()
    result = predict_records(records[:2], args.output)
    assert result["model_status"] == "trained"
    assert [item["prediction"] for item in result["predictions"]] == ["safe", "vulnerable"]
    with pytest.raises(ValueError, match="already exists"):
        trainer.train(examples, args)


@pytest.mark.parametrize("problem", ["group", "duplicate_sequence", "missing_function"])
def test_training_rejects_data_leakage_and_missing_labels(trainer, training_data, problem):
    records, labels = training_data
    if problem == "group":
        labels[2]["group"] = labels[0]["group"]
        message = "span multiple splits"
    elif problem == "duplicate_sequence":
        records[2]["tokens"] = records[0]["tokens"]
        message = "Identical AST sequences"
    else:
        labels[0]["function"] = "missing"
        message = "found 0"
    with pytest.raises(ValueError, match=message):
        trainer.labeled_examples(records, labels)
