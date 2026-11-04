"""Verify Docker transport and function identity without heavy host dependencies."""

import argparse
import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

spec = importlib.util.spec_from_file_location("vulcnn_cli", Path(__file__).parents[1] / "docker/vulcnn/cli.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def docker_args(tmp_path, container=None):
    return argparse.Namespace(
        workspace=tmp_path / "source and outputs",
        models=tmp_path / "models",
        image="salomon-vulcnn:local",
        container=container,
        timeout=30,
        stage="parse",
        parameters=["--run", "/workspace/run with spaces"],
    )


def test_docker_arguments_preserve_spaces_and_mount_models_read_only(tmp_path):
    args = docker_args(tmp_path)
    with patch.object(
        cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, '{"ok": true}', "")
    ) as run:
        assert cli.run_docker_command(args) == {"ok": True}
    command = run.call_args.args[0]
    assert f"type=bind,src={args.workspace},dst=/workspace" in command
    assert f"type=bind,src={args.models},dst=/models,readonly" in command
    assert command[-3:] == ["parse", "--run", "/workspace/run with spaces"]
    assert run.call_args.kwargs.get("shell", False) is False


def test_compose_mode_executes_in_existing_container(tmp_path):
    args = docker_args(tmp_path, "salomon-vulcnn")
    with patch.object(cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "{}", "")) as run:
        cli.run_docker_command(args)
    command = run.call_args.args[0]
    assert command[:4] == ["docker", "exec", "salomon-vulcnn", "timeout"]
    assert "--mount" not in command


def test_timeout_removes_only_the_disposable_job(tmp_path):
    with (
        patch.object(cli.subprocess, "run", side_effect=[subprocess.TimeoutExpired("docker", 40), None]) as run,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        cli.run_docker_command(docker_args(tmp_path))
    cleanup = run.call_args_list[1].args[0]
    assert cleanup[:3] == ["docker", "rm", "--force"]
    assert cleanup[3].startswith("salomon-vulcnn-")


def test_timeout_does_not_remove_shared_container(tmp_path):
    with (
        patch.object(cli.subprocess, "run", side_effect=subprocess.TimeoutExpired("docker", 40)) as run,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        cli.run_docker_command(docker_args(tmp_path, "salomon-vulcnn"))
    assert run.call_count == 1


def test_names_follow_source_identity_not_pdg_filename_order(tmp_path):
    original = [{"id": "100", "name": "unsafe", "line": 3}, {"id": "200", "name": "safe", "line": 8}]
    normalized = [
        {"id": "1", "name": "FUN1", "line": 3, "pdg": "/workspace/run/pdgs/1.dot"},
        {"id": "2", "name": "FUN2", "line": 8, "pdg": "/workspace/run/pdgs/2.dot"},
    ]
    cli.write_json(tmp_path / "functions.json", original)
    cli.write_json(tmp_path / "source.json", {"filename": "sample.c"})
    with patch.object(cli, "describe_functions", return_value=normalized), patch.object(cli, "canonicalize_pdg"):
        result = cli.export_pdgs(tmp_path)
    assert [f["name"] for f in result["functions"]] == ["unsafe", "safe"]
    assert result["functions"][0]["pdg"] == "/workspace/run/pdgs/1.dot"
    assert result["functions"][0]["id"] == "100"


@pytest.mark.parametrize("normalized", [[], [{"line": 4}, {"line": 8}]])
def test_changed_function_mapping_does_not_publish_a_manifest(tmp_path, normalized):
    cli.write_json(tmp_path / "functions.json", [{"line": 3}, {"line": 8}])
    with (
        patch.object(cli, "describe_functions", return_value=normalized),
        pytest.raises(ValueError, match="safely map"),
    ):
        cli.export_pdgs(tmp_path)
    assert not (tmp_path / "result.json").exists()


def test_joern_html_labels_match_upstream_quoted_label_format(tmp_path):
    path = tmp_path / "example.dot"
    path.write_text(
        'digraph x {\n"7" [label = <(&lt;operator&gt;.assignment,a &lt; &quot;b&quot;)<SUB>4</SUB>> ]\n'
        '"7" -> "8" [label = "DDG: a"]\n}\n',
        encoding="utf-8",
    )
    cli.canonicalize_pdg(path)
    result = path.read_text()
    assert 'label = "(<operator>.assignment,a < \\"b\\")"' in result
    assert '"7" -> "8" [label = "DDG: a"]' in result
    assert "<SUB>" not in result
