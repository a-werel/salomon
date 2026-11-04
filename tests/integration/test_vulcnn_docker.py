"""Opt-in integration tests: RUN_VULCNN_DOCKER_TESTS=1 python -m pytest tests/integration."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from salomon.adapters.vulcnn.adapter import VulCNNAdapter

pytestmark = pytest.mark.skipif(os.environ.get("RUN_VULCNN_DOCKER_TESTS") != "1", reason="Requires built VulCNN image")
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def runtime():
    base = REPO / "outputs/vulcnn"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="integration ", dir=base) as directory:
        root = Path(directory)
        models = root / "models"
        models.mkdir()

        def fixture_command(action):
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--init",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "--mount",
                    f"type=bind,src={models},dst=/fixtures",
                    "--mount",
                    f"type=bind,src={REPO / 'tests/integration'},dst=/checks,readonly",
                    "--entrypoint",
                    "python",
                    "salomon-vulcnn:local",
                    "/checks/runtime_fixture.py",
                    action,
                    "/fixtures",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

        fixture_command("generate")
        adapter = VulCNNAdapter(
            workspace=root / "workspace",
            models=models,
            embedding_model="sent2vec.bin",
            checkpoint="cnn.pt",
        )
        yield adapter, fixture_command


def test_complete_workflow_maps_real_functions_and_preserves_utf8_source(runtime):
    adapter, _ = runtime
    source = "// zażółć\n/* comment\n   across lines */\nint first(int a) { return a + 1; }\n\nint second(int b) { return b - 1; }\n"
    try:
        report = adapter.analyze_code(source.encode("utf-8"), filename="two functions.c")
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stderr)
    text = report.read_text()
    assert "- first " in text and "- second " in text
    assert "<global>" not in text and "FUN1" not in text
    rows = json.loads((report.parent / "predictions.json").read_text())
    assert [row["name"] for row in rows] == ["first", "second"]
    assert all(row["label"] == 1 for row in rows)
    assert (report.parent / "predictions.pkl").is_file()
    assert (report.parent / "original/two functions.c").read_text() == source


def test_pdg_only_cpp_and_multiple_functions_on_one_line(runtime):
    adapter, _ = runtime
    source = "int first(int a) { return a; } int second(int b) { return b; }\n"
    try:
        pdgs = adapter.create_pdg(source, filename="sample.cpp")
    except subprocess.CalledProcessError as error:
        pytest.fail(error.stderr)
    assert len(pdgs) == 2 and all(path.is_file() for path in pdgs)
    manifest = json.loads((pdgs[0].parent.parent / "result.json").read_text())
    assert [row["name"] for row in manifest["functions"]] == ["first", "second"]


def test_real_classifier_mixed_labels_and_tensor_output(runtime):
    _, fixture_command = runtime
    fixture_command("check")
