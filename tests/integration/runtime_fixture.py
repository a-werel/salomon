"""TEST-ONLY models and inference checks, executed inside the VulCNN image.

These weights exercise plumbing. They have no vulnerability detection accuracy.
"""

import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/opt/VulCNN")
from model import TextCNN


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "TEST_ONLY.txt").write_text("Test fixtures only; not trained for vulnerability detection.\n")
    corpus = directory / "corpus.txt"
    corpus.write_text(("int FUN1 VAR1 return VAR2 addition METHOD PARAM RETURN\n" * 50), encoding="utf-8")
    subprocess.run(
        [
            "fasttext",
            "sent2vec",
            "-input",
            str(corpus),
            "-output",
            str(directory / "sent2vec"),
            "-dim",
            "128",
            "-epoch",
            "1",
            "-minCount",
            "1",
            "-minCountLabel",
            "1",
            "-bucket",
            "1000",
            "-thread",
            "1",
            "-neg",
            "2",
            "-wordNgrams",
            "1",
        ],
        check=True,
    )
    model = TextCNN(hidden_size=128)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc.bias[1] = 1
    torch.save(model.state_dict(), directory / "cnn.pt")


def check_classification(directory):
    spec = importlib.util.spec_from_file_location("vulcnn_cli", "/opt/salomon-vulcnn/cli.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    run = directory / "classifier-check"
    run.mkdir()
    functions = []
    for index, (name, value) in enumerate((("danger_one", 1), ("safe", 0), ("danger_two", 1))):
        features = run / f"{index}.npy"
        np.save(features, np.full((3, 12, 128), value, dtype=np.float32))
        functions.append(
            {"id": str(index), "name": name, "full_name": name, "line": index + 1, "features": str(features)}
        )
    cli.write_json(run / "result.json", {"filename": "fixture.c", "embedding_dim": 128, "functions": functions})
    model = TextCNN(hidden_size=128)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.convs[0].weight[0, 0, 0, 0] = 1
        model.fc.weight[1, 0] = 1
        model.fc.bias[1] = -0.5
    weights = run / "mixed.pt"
    torch.save(model.state_dict(), weights)
    cli.classify_functions(run, weights, 100)
    with (run / "predictions.pkl").open("rb") as stream:
        predictions = pickle.load(stream)
    assert [label for _, label in predictions] == [1, 0, 1]
    assert all(array.shape == (3, 100, 128) for array, _ in predictions)
    report = (run / "report.txt").read_text()
    assert "danger_one" in report and "danger_two" in report and "- safe" not in report
    rows = json.loads((run / "predictions.json").read_text())
    assert [row["name"] for row in rows] == ["danger_one", "safe", "danger_two"]
    # Dimension mismatches must fail without a successful report.
    model = TextCNN(hidden_size=64)
    torch.save(model.state_dict(), run / "wrong.pt")
    try:
        cli.classify_functions(run, run / "wrong.pt", 100)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Mismatched model dimensions were accepted")


if __name__ == "__main__":
    directory = Path(sys.argv[2])
    if sys.argv[1] == "generate":
        generate(directory)
    else:
        check_classification(directory)
