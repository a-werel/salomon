import json
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4


class AbstractBackend(ABC):
    def __init__(
        self,
        image: str,
        nn_model: str,
        workspace: str | Path | None = None,
        models: str | Path | None = None,
        container: str | None = None,
        embedding_model: str = "",
        max_length: int = 100,
        timeout: int = 600,
        workspace_environment_variable: str = "SALOMON_WORKSPACE",
        models_environment_variable: str = "SALOMON_MODELS",
        container_environment_variable: str = "SALOMON_CONTAINER",
    ):
        self.workspace_absolute_path = Path(
            workspace
            or os.environ.get(workspace_environment_variable, "outputs/salomon")
        ).resolve()
        self.models_absolute_path = Path(
            models
            or os.environ.get(models_environment_variable, "models")
        ).resolve()
        self.image_name = image
        self.existing_container = container or os.environ.get(
            container_environment_variable,
            "",
        )

        self.nn_model_relative_path = self._get_model_relative_path(nn_model)
        self.embedding_model_relative_path = (
            self._get_model_relative_path(embedding_model) if embedding_model else ""
        )

        if max_length < 10 or timeout <= 0:
            raise ValueError("max_length must be at least 10 and timeout must be positive.")

        self.max_length = max_length
        self.timeout = timeout
        self.cli_absolute_path = Path(__file__).resolve().parents[2] / "cli.py"
        if not self.cli_absolute_path.is_file():
            self.cli_absolute_path = (
                Path(__file__).resolve().parents[4] / "docker/example/cli.py"
            )

    @staticmethod
    def _get_model_relative_path(value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Model paths must be relative to the models directory.")
        return path.as_posix()

    @abstractmethod
    def check_if_models_are_present_in_project_dir(self) -> None:
        """Check whether every required model file is present in the models directory."""

    def _decode_source_code(self, source_code: str | bytes) -> str:
        if isinstance(source_code, bytes):
            source_code = source_code.decode("utf-8", errors="strict")
        if not isinstance(source_code, str) or source_code.strip() == "":
            raise ValueError("source code must contain UTF-8 C/C++ source code.")
        return source_code

    def _validate_filename(self, filename: str) -> None:
        if Path(filename).name != filename or "\\" in filename or "//" in filename:
            raise ValueError("filename must be a filename, without directory components.")
        if Path(filename).suffix.lower() not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}:
            raise ValueError("filename must have a C/C++ extension.")

    def prepare_original_source_code(self, source_code: str | bytes, filename: str) -> Path:
        source_code = self._decode_source_code(source_code)
        self._validate_filename(filename)

        run = self.workspace_absolute_path / uuid4().hex
        original = run / "original_source_code"
        original.mkdir(parents=True)
        (original / filename).write_text(source_code, encoding="utf-8")
        (run / "source.json").write_text(
            json.dumps({"filename": filename}),
            encoding="utf-8",
        )
        return run

    def host_to_container_workspace_path(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.workspace_absolute_path)
        return str(Path("/workspace") / relative)

    def container_to_host_workspace_path(self, path: str) -> Path:
        relative = Path(path).relative_to("/workspace")
        result = (self.workspace_absolute_path / relative).resolve()
        if not result.is_relative_to(self.workspace_absolute_path):
            raise ValueError(f"{path} escaped the workspace.")
        return result

    @abstractmethod
    def create_model_representation(self, run: Path) -> dict:
        """Return a model-specific result manifest as a dict."""

    def run_model_command(self, command: str, **parameters) -> dict:
        arguments = [
            sys.executable,
            str(self.cli_absolute_path),
            "docker",
            "--workspace",
            str(self.workspace_absolute_path),
            "--models",
            str(self.models_absolute_path),
            "--image",
            self.image_name,
            "--timeout",
            str(self.timeout),
        ]

        if self.existing_container:
            arguments.extend(["--container", self.existing_container])

        arguments.append(command)
        for key, value in parameters.items():
            arguments.extend([f"--{key.replace('_', '-')}", str(value)])

        completed = subprocess.run(
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout + 30,
        )
        return json.loads(completed.stdout)
