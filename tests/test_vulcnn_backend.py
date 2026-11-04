import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from salomon.adapters.vulcnn.adapter import VulCNNAdapter
from salomon.adapters.vulcnn.backend import VulCNNBackend


def test_utf8_bytes_are_staged_without_modifying_the_input(tmp_path):
    backend = VulCNNBackend(workspace=tmp_path)
    source = "// zażółć\nint example(void) { return 1; }\n".encode()
    run = backend.prepare_source(source, "example.c")
    assert (run / "original/example.c").read_bytes() == source


@pytest.mark.parametrize(
    "source, filename", [(b"\xff", "bad.c"), (" ", "a.c"), ("int f() {}", "../a.c"), ("x", "x.py")]
)
def test_invalid_input_fails_before_creating_runs(tmp_path, source, filename):
    backend = VulCNNBackend(workspace=tmp_path)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        backend.prepare_source(source, filename)
    assert list(tmp_path.iterdir()) == []


def test_pdg_api_returns_persistent_host_paths(tmp_path):
    adapter = VulCNNAdapter(workspace=tmp_path)

    def execute(stage, **parameters):
        if stage == "export":
            return {"functions": [{"pdg": f"{parameters['run']}/pdgs/f.dot"}]}
        return {}

    with patch.object(adapter.backend, "run_vulcnn_command", side_effect=execute) as command:
        graphs = adapter.create_pdg("int f(void) { return 1; }")
    assert [call.args[0] for call in command.call_args_list] == ["inspect", "normalize", "parse", "export"]
    assert graphs[0].is_relative_to(tmp_path)


def test_full_workflow_uses_same_helper_for_every_stage_and_returns_report(tmp_path):
    adapter = VulCNNAdapter(workspace=tmp_path / "outputs", models=tmp_path / "models")
    for name in ("sent2vec.bin", "cnn.pt"):
        path = tmp_path / "models/vulcnn" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def execute(stage, **parameters):
        return {"report": f"{parameters['run']}/report.txt"} if stage == "classify" else {}

    with patch.object(adapter.backend, "run_vulcnn_command", side_effect=execute) as command:
        report = adapter.analyze_code("int f(void) { return 1; }")
    assert [call.args[0] for call in command.call_args_list] == [
        "inspect",
        "normalize",
        "parse",
        "export",
        "embed",
        "classify",
    ]
    assert report.name == "report.txt"


def test_missing_models_fail_before_docker_or_artifact_creation(tmp_path):
    adapter = VulCNNAdapter(workspace=tmp_path / "outputs", models=tmp_path / "models")
    with (
        patch.object(adapter.backend, "run_vulcnn_command") as command,
        pytest.raises(FileNotFoundError, match="Required trained model"),
    ):
        adapter.analyze_code("int f() {}")
    command.assert_not_called()
    assert not (tmp_path / "outputs").exists()


def test_helper_delegates_to_cli_and_propagates_failures(tmp_path):
    backend = VulCNNBackend(workspace=tmp_path)
    with patch("salomon.adapters.vulcnn.backend.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0, json.dumps({"ok": True}), "")
        assert backend.run_vulcnn_command("parse", run="/workspace/a b") == {"ok": True}
        arguments = run.call_args.args[0]
        assert arguments[-3:] == ["parse", "--run", "/workspace/a b"]
        assert Path(arguments[1]).name == "cli.py"
        run.side_effect = subprocess.CalledProcessError(7, "cli", stderr="Joern failed")
        with pytest.raises(subprocess.CalledProcessError):
            backend.run_vulcnn_command("parse", run="/workspace/a b")


def test_paths_in_results_cannot_escape_workspace(tmp_path):
    backend = VulCNNBackend(workspace=tmp_path)
    with pytest.raises(ValueError):
        backend.host_path("/workspace/../../etc/passwd")
