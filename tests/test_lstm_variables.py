import importlib
import os
from pathlib import Path

import pytest

pytest.importorskip("torch")


@pytest.mark.skipif(os.environ.get("SALOMON_TEST_DOCKER") != "1", reason="Opt-in Joern/Docker integration test")
def test_variable_identity_renaming_and_shadowing(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    trainer = importlib.import_module("train_lstm")
    source = Path(__file__).parent / "fixtures" / "lstm_variables.cpp"
    records = trainer.export_with_docker(source, "salomon-lstm:latest", 600)
    tokens = {r["function"]: r["tokens"] for r in records}
    assert tokens["original"] == tokens["renamed"]
    assert tokens["original"] != tokens["different"]
    assert "LOCAL:VAR_3" in tokens["original"]
    assert [t for t in tokens["original"] if t.startswith("IDENTIFIER:")] == [
        "IDENTIFIER:VAR_3",
        "IDENTIFIER:VAR_1",
        "IDENTIFIER:VAR_3",
        "IDENTIFIER:VAR_2",
        "IDENTIFIER:VAR_3",
    ]
    # Inner 'value' is a separate declaration; returning outside uses the parameter.
    identifiers = [t for t in tokens["shadowed"] if t.startswith("IDENTIFIER:")]
    assert "LOCAL:VAR_3" in tokens["shadowed"]
    assert identifiers[-1] == "IDENTIFIER:VAR_1"
    assert "IDENTIFIER:VAR_3" in identifiers
