import importlib
from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_dataset_groups_related_names_and_identical_asts_transitively(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    prepare = importlib.import_module("prepare_vulnllm_sample")
    trainer = importlib.import_module("train_lstm")
    rows, records = [], []
    for index in range(20):
        rows.append({"row_index": index, "target": index % 2})
        records.append(
            {
                "source": f"/input/sample_{index}.cpp",
                "function": f"function_{index}",
                "representation": "joern-ast-preorder-vars-v2",
                "tokens": ["METHOD", str(index), "END"],
            }
        )
    records[1]["function"] = records[0]["function"]
    records[2]["tokens"] = records[1]["tokens"]
    records[3]["tokens"] = [*records[2]["tokens"], "TRUNCATED_SUFFIX"]
    # Missing or ambiguous ASTs must not silently inherit a function label.
    rows.extend([{"row_index": 100, "target": 1}, {"row_index": 101, "target": 0}])
    records.extend(
        [
            {
                "source": "/input/sample_101.cpp",
                "function": name,
                "representation": "joern-ast-preorder-vars-v2",
                "tokens": [name],
            }
            for name in ("outer", "inner")
        ]
    )
    labels, report = prepare.prepare_labels(rows, records, seed=42, max_length=3)
    assert (
        len(
            {
                entry["split"]
                for entry in labels
                if entry["source"] in {"sample_0.cpp", "sample_1.cpp", "sample_2.cpp", "sample_3.cpp"}
            }
        )
        == 1
    )
    assert report["usable"] == 20
    assert len(report["skipped"]) == 2
    assert len(trainer.labeled_examples(records, labels)) == 20
    assert prepare.prepare_labels(rows, records, seed=42, max_length=3) == (labels, report)
