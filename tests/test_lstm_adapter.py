import json
import subprocess

import pytest

from salomon.adapters.lstm import LSTMAdapter


def test_adapter_mounts_source_and_external_checkpoint_readonly(tmp_path, monkeypatch):
    source = tmp_path / "source with spaces.c"
    source.write_text("int example(void) { return 0; }", encoding="utf-8")
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"external checkpoint")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        response = {
            "model_status": "untrained",
            "assessment_available": False,
            "predictions": [{"source": f"/input/{source.name}", "function": "example", "prediction": None}],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(response), "Joern logs")

    monkeypatch.setattr(subprocess, "run", run)
    result = LSTMAdapter(checkpoint=checkpoint).analyze_code(source)
    command, options = calls[0]
    assert command[:3] == ["docker", "run", "--rm"]
    assert command.count("--mount") == 2
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert all(mount.endswith(",readonly") for mount in mounts)
    assert "--checkpoint" in command
    assert options["check"] is True
    assert result["assessment_available"] is False
    assert result["predictions"][0]["prediction"] is None
    assert result["predictions"][0]["source"] == str(source)


def test_adapter_propagates_container_failure(tmp_path, monkeypatch):
    source = tmp_path / "example.c"
    source.touch()

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="Unsupported checkpoint format")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="Unsupported checkpoint format"):
        LSTMAdapter().analyze_code(source)


def test_adapter_rejects_non_json_response(tmp_path, monkeypatch):
    source = tmp_path / "example.c"
    source.touch()
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "not JSON", "")
    )
    with pytest.raises(RuntimeError, match="valid JSON"):
        LSTMAdapter().analyze_code(source)
